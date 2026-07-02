from pathlib import Path

from parsing_agent.evaluation import TABLE_ISSUE_MERGED_CELL_LOSS, TABLE_ISSUE_MISSING_HEADER
from parsing_agent.models import DocumentSource, EvaluationMetrics, JudgeResult, ParseCandidate, RepairAction
from parsing_agent.repair import HeuristicRepairer, RepairTarget
import fitz

from parsing_agent.visual_repair import (
    OpenAIVisualTableRecoverer,
    VisualTableRecovery,
    VisualRepairTask,
    _issue_specific_prompt_guidance,
    _normalize_recovered_table_markup,
    _parse_recovery_payload,
    extract_issue_page_numbers,
    extract_table_labels,
    replace_page_table_block,
    replace_table_block,
    replace_table_block_by_global_index,
)


def test_repairer_routes_detected_issues_into_targeted_repairs() -> None:
    source = DocumentSource(
        path=Path("sample.md"),
        media_type="text/markdown",
        size_bytes=0,
        run_id="run-1",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.6,
        normalized_similarity=0.6,
        structure_retention=0.2,
        table_preservation=0.2,
        empty_block_penalty=0.1,
        repetition_penalty=0.2,
        total_score=0.0,
    )
    candidate = ParseCandidate(
        parser_name="mock",
        content=(
            "# Title\n\n"
            "# Title\n\n"
            "| a | b |\n"
            "| --- | --- |\n"
            "| 1 | ![image](<x.png>) |\n\n"
            "footer\n\n"
            "footer\n"
            "wrapped\n"
            "line\n"
        ),
        format_name="md",
        source_path=Path("sample.md"),
    )

    repaired, actions = HeuristicRepairer().repair(source, candidate, metrics)

    assert repaired.metadata["repair_issue_types"] == [
        "image_link_noise",
        "table_layout_noise",
        "structure_heading_noise",
        "blank_line_noise",
        "boundary_repetition_noise",
        "wrapped_line_noise",
    ]
    assert repaired.metadata["repair_routes"] == [
        "remove_image_noise",
        "normalize_table_layout",
        "deduplicate_headings",
        "collapse_blank_runs",
        "deduplicate_boundaries",
        "merge_wrapped_lines",
    ]
    assert "![image]" not in repaired.content
    assert repaired.content.count("# Title") == 1
    assert "wrapped line" in repaired.content
    assert len(actions) == 6
    assert actions[0].issue_type == "image_link_noise"
    assert actions[0].route_name == "remove_image_noise"


def test_repairer_recovers_key_value_table_text_blocks() -> None:
    source = DocumentSource(
        path=Path("sample.md"),
        media_type="text/markdown",
        size_bytes=0,
        run_id="run-2",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.6,
        normalized_similarity=0.6,
        structure_retention=0.6,
        table_preservation=0.2,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
    )
    candidate = ParseCandidate(
        parser_name="mock",
        content=(
            "표 4.2-2\n"
            "면적  512.3\n"
            "답  63.0\n"
            "임야  299.2\n"
            "기타  60.6\n"
        ),
        format_name="md",
        source_path=Path("sample.md"),
    )

    repaired, actions = HeuristicRepairer().repair(source, candidate, metrics)

    assert "| 항목 | 값 |" in repaired.content
    assert "| 면적 | 512.3 |" in repaired.content
    assert "| 기타 | 60.6 |" in repaired.content
    assert any(action.issue_type == "table_text_block_recovery" for action in actions)
    assert "reconstruct_table_blocks" in repaired.metadata["repair_routes"]


def test_repairer_recovers_missing_structured_source_lines_when_text_coverage_is_low() -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-coverage-repair",
        extracted_text=(
            "1. 사업개요\n"
            "본 사업은 광양항 일원에서 수행된다.\n"
            "표 4.2-2 총괄\n"
            "세부 내용\n"
        ),
    )
    metrics = EvaluationMetrics(
        text_coverage=0.5,
        normalized_similarity=0.7,
        structure_retention=0.5,
        table_preservation=0.8,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
    )
    candidate = ParseCandidate(
        parser_name="mock",
        content="본 사업은 광양항 일원에서 수행된다.\n세부 내용\n",
        format_name="md",
        source_path=source.path,
    )

    repaired, actions = HeuristicRepairer().repair_heuristics(source, candidate, metrics)

    assert "1. 사업개요" in repaired.content
    assert "표 4.2-2 총괄" in repaired.content
    assert any(action.route_name == "recover_missing_source_lines" for action in actions)


def test_repairer_removes_repeated_pdf_headers_and_footers() -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-pdf-dedupe",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.8,
        normalized_similarity=0.8,
        structure_retention=0.6,
        table_preservation=0.8,
        empty_block_penalty=0.0,
        repetition_penalty=0.2,
        total_score=0.0,
    )
    candidate = ParseCandidate(
        parser_name="mock",
        content=(
            "<!-- page 1 -->\n"
            "환경영향평가서\n"
            "1. 사업개요\n"
            "본문 1\n"
            "환경영향평가서\n"
            "<!-- page 2 -->\n"
            "환경영향평가서\n"
            "2. 지역현황\n"
            "본문 2\n"
            "환경영향평가서\n"
        ),
        format_name="md",
        source_path=source.path,
    )

    repaired, actions = HeuristicRepairer().repair_heuristics(source, candidate, metrics)

    assert "환경영향평가서" not in repaired.content
    assert any(action.route_name == "deduplicate_pdf_headers" for action in actions)


def test_repairer_does_not_merge_page_footers_or_pdf_sections() -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-pdf-merge-guards",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.8,
        normalized_similarity=0.8,
        structure_retention=1.0,
        table_preservation=0.8,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
    )
    candidate = ParseCandidate(
        parser_name="mock",
        content=(
            "-21-\n"
            "第 4 章 環境影響要素 및 環境因子行列式對照表\n"
            "가. 建設段階\n"
            "(1) 海岸, 海底地形의 變形\n"
            "wrapped English line\n"
            "continues here\n"
        ),
        format_name="md",
        source_path=source.path,
    )

    repaired, actions = HeuristicRepairer().repair_heuristics(source, candidate, metrics)

    assert "-21-\n第 4 章" in repaired.content
    assert "가. 建設段階\n(1)" in repaired.content
    assert "wrapped English line continues here" in repaired.content
    assert any(action.route_name == "merge_wrapped_lines" for action in actions)


def test_repairer_restores_pdf_boundaries_around_headings_and_lists() -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-pdf-structure",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.8,
        normalized_similarity=0.8,
        structure_retention=0.5,
        table_preservation=0.8,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
    )
    candidate = ParseCandidate(
        parser_name="mock",
        content=(
            "<!-- page 1 -->\n"
            "1. 사업개요\n"
            "본문 첫줄\n"
            "가. 세부항목\n"
            "세부 설명\n"
        ),
        format_name="md",
        source_path=source.path,
    )

    repaired, actions = HeuristicRepairer().repair_heuristics(source, candidate, metrics)

    assert "1. 사업개요\n\n본문 첫줄" in repaired.content
    assert "가. 세부항목\n\n세부 설명" in repaired.content
    assert any(action.route_name == "restore_pdf_boundaries" for action in actions)


def test_extract_table_labels_deduplicates_judge_issues() -> None:
    labels = extract_table_labels(
        [
            "표 품질: 표 4.2-2가 깨졌고 표 4.7-3도 단위가 어긋남",
            "추가로 표 4.2-2의 숫자 일부가 손상됨",
        ]
    )

    assert labels == ["표 4.2-2", "표 4.7-3"]


def test_extract_issue_page_numbers_deduplicates_page_references() -> None:
    pages = extract_issue_page_numbers(
        [
            "p.6 표 헤더가 누락됨",
            "p.7 표 셀 병합이 깨짐, p.6의 일부 행도 중복됨",
        ]
    )

    assert pages == [6, 7]


def test_parse_recovery_payload_accepts_markdown_table_fallback() -> None:
    payload = _parse_recovery_payload(
        "**<표 4.2-2> 지목별 토지이용 현황**\n\n| 항목 | 값 |\n| --- | --- |\n| 전 | 63.0 |",
        "표 4.2-2",
        4,
    )

    assert payload["table_label"] == "표 4.2-2"
    assert payload["page_number"] == 4
    assert payload["confidence"] == 0.7
    assert "| 항목 | 값 |" in payload["markdown"]


def test_parse_recovery_payload_accepts_table_field_from_json_response() -> None:
    payload = _parse_recovery_payload(
        """```json
{
  "table": "<table><tr><th>a</th></tr><tr><td>b</td></tr></table>"
}
```""",
        "??4.2-2",
        4,
    )

    assert payload["table_label"] == "??4.2-2"
    assert payload["page_number"] == 4
    assert payload["confidence"] == 0.7
    assert payload["markdown"] == "<table><tr><th>a</th></tr><tr><td>b</td></tr></table>"


def test_parse_recovery_payload_accepts_raw_html_table_fallback() -> None:
    payload = _parse_recovery_payload(
        "<table><tr><th>a</th></tr><tr><td>b</td></tr></table>",
        "??4.2-2",
        4,
    )

    assert payload["table_label"] == "??4.2-2"
    assert payload["page_number"] == 4
    assert payload["confidence"] == 0.7
    assert payload["markdown"] == "<table><tr><th>a</th></tr><tr><td>b</td></tr></table>"


def test_normalize_recovered_table_markup_fixes_units_and_decimal_commas() -> None:
    normalized = _normalize_recovered_table_markup(
        "<table><tr><th>총매립용량 (㎡)</th><th>잔여 매립 가능량(㎡)</th><th>면적(km²)</th></tr>"
        "<tr><td>748</td><td>443,1</td><td>11.5</td></tr></table>"
    )

    assert "총매립용량 (㎥)" in normalized
    assert "잔여 매립 가능량(㎥)" in normalized
    assert "면적(㎢)" in normalized
    assert "443.1" in normalized


def test_normalize_recovered_table_markup_converts_html_spans_to_markdown() -> None:
    normalized = _normalize_recovered_table_markup(
        "<table>"
        "<tr><th rowspan=\"2\">group</th><th colspan=\"2\">plan</th></tr>"
        "<tr><th>area</th><th>ratio</th></tr>"
        "<tr><td>total</td><td>10</td><td>100</td></tr>"
        "</table>"
    )

    assert "<table" not in normalized
    assert "| group | plan / area | plan / ratio |" in normalized
    assert "| total | 10 | 100 |" in normalized


def test_replace_table_block_replaces_broken_section_after_caption() -> None:
    content = (
        "표 4.2-2 지목별 토지이용 현황\n"
        "전 답 63.0\n"
        "임야 299.2\n"
        "\n"
        "## 다음 절\n"
    )

    updated = replace_table_block(
        content,
        "표 4.2-2",
        "| 항목 | 값 |\n| --- | --- |\n| 전 | 63.0 |\n| 답 | 299.2 |",
    )

    assert "| 항목 | 값 |" in updated
    assert "전 답 63.0" not in updated
    assert "## 다음 절" in updated


def test_replace_table_block_replaces_interleaved_plain_text_table_block() -> None:
    content = (
        "Table 4.2-2 Land use\n"
        "\n"
        "Area  512.3  63.0  38.2\n"
        "\n"
        "Yeosu\n"
        "\n"
        "Share  100.0  12.3  7.5\n"
        "\n"
        "Gwangyang\n"
        "\n"
        "Area  464.3  18.6  44.9\n"
        "\n"
        "## Next Section\n"
    )

    updated = replace_table_block(
        content,
        "Table 4.2-2",
        "| Region | Total |\n| --- | --- |\n| Yeosu | 512.3 |\n| Gwangyang | 464.3 |",
    )

    before_next_heading = updated.split("## Next Section")[0]
    assert "| Region | Total |" in updated
    assert "Area  512.3  63.0  38.2" not in updated
    assert "Share  100.0  12.3  7.5" not in updated
    assert "\nYeosu\n" not in before_next_heading
    assert "\nGwangyang\n" not in before_next_heading
    assert "## Next Section" in updated

def test_replace_table_block_falls_back_to_support_order_when_label_is_missing() -> None:
    content = (
        "| first | table |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n\n"
        "| second | table |\n"
        "| --- | --- |\n"
        "| 3 | 4 |\n"
    )

    updated = replace_table_block(
        content,
        "표 4.2-3",
        "| patched | second |\n| --- | --- |\n| 30 | 40 |",
        candidate_metadata={
            "support_parser_metadata": {
                "layout-first-pdf": {
                    "table_label_positions": {
                        "4.2-2": {"page": 4, "region_index": 1, "global_index": 1},
                        "표 4.2-2": {"page": 4, "region_index": 1, "global_index": 1},
                        "4.2-3": {"page": 5, "region_index": 1, "global_index": 2},
                        "표 4.2-3": {"page": 5, "region_index": 1, "global_index": 2},
                    }
                }
            }
        },
    )

    assert "| first | table |" in updated
    assert "| patched | second |" in updated
    assert "| second | table |" not in updated


def test_replace_table_block_by_global_index_targets_nth_table() -> None:
    content = (
        "| first | table |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n\n"
        "| second | table |\n"
        "| --- | --- |\n"
        "| 3 | 4 |\n"
    )

    updated = replace_table_block_by_global_index(
        content,
        2,
        "| new | second |\n| --- | --- |\n| 30 | 40 |",
    )

    assert "| first | table |" in updated
    assert "| new | second |" in updated
    assert "| second | table |" not in updated


def test_replace_page_table_block_replaces_first_table_inside_page_section() -> None:
    content = (
        "<!-- page 6 -->\n"
        "intro\n"
        "| old | table |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "\n"
        "tail\n"
        "<!-- page 7 -->\n"
        "next page\n"
    )

    updated = replace_page_table_block(
        content,
        6,
        "| new | table |\n| --- | --- |\n| 3 | 4 |",
    )

    assert "| new | table |" in updated
    assert "| old | table |" not in updated
    assert "<!-- page 7 -->" in updated


def test_replace_page_table_block_can_target_second_table_inside_page_section() -> None:
    content = (
        "<!-- page 6 -->\n"
        "intro\n"
        "| old | first |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "\n"
        "| old | second |\n"
        "| --- | --- |\n"
        "| 3 | 4 |\n"
        "\n"
        "tail\n"
        "<!-- page 7 -->\n"
    )

    updated = replace_page_table_block(
        content,
        6,
        "| new | second |\n| --- | --- |\n| 30 | 40 |",
        table_index=2,
    )

    assert "| old | first |" in updated
    assert "| new | second |" in updated
    assert "| old | second |" not in updated


def test_replace_page_table_block_replaces_plain_text_table_block_inside_page_section() -> None:
    content = (
        "<!-- page 6 -->\n"
        "intro\n"
        "item  value  amount\n"
        "alpha  10  20\n"
        "beta  30  40\n"
        "\n"
        "tail\n"
        "<!-- page 7 -->\n"
        "next page\n"
    )

    updated = replace_page_table_block(
        content,
        6,
        "| new | table |\n| --- | --- |\n| 3 | 4 |",
    )

    assert "| new | table |" in updated
    assert "alpha  10  20" not in updated
    assert "tail" in updated


def test_replace_page_table_block_replaces_html_table_block_inside_page_section() -> None:
    content = (
        "<!-- page 6 -->\n"
        "intro\n"
        "<table>\n"
        "  <tr><td>old</td><td>table</td></tr>\n"
        "</table>\n"
        "\n"
        "tail\n"
        "<!-- page 7 -->\n"
        "next page\n"
    )

    updated = replace_page_table_block(
        content,
        6,
        "| new | table |\n| --- | --- |\n| 3 | 4 |",
    )

    assert "| new | table |" in updated
    assert "<tr><td>old</td><td>table</td></tr>" not in updated
    assert "tail" in updated


class _FakeVisualTableRecoverer:
    def repair(
        self,
        source: DocumentSource,
        content: str,
        metrics: EvaluationMetrics,
    ) -> tuple[str, list[RepairAction]]:
        del source, metrics
        updated = replace_table_block(
            content,
            "표 4.2-2",
            "| 항목 | 값 |\n| --- | --- |\n| 전 | 63.0 |\n| 답 | 299.2 |",
        )
        if updated == content:
            return content, []
        return (
            updated,
            [
                RepairAction(
                    action_name="recover_table_from_pdf_image",
                    description="Recover a broken table from the source PDF image.",
                    before_excerpt="표 4.2-2\n전 답 63.0",
                    after_excerpt="표 4.2-2\n| 항목 | 값 |",
                    issue_type="table_visual_recovery",
                    route_name="recover_tables_from_pdf_image",
                )
            ],
        )


def test_repairer_applies_visual_table_recovery_after_heuristics() -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-3",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.9,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        judge_result=JudgeResult(
            overall_score=0.7,
            issues=["표 4.2-2가 표 형태로 복구되지 않음"],
        ),
    )
    candidate = ParseCandidate(
        parser_name="mock",
        content="표 4.2-2 지목별 토지이용 현황\n전 답 63.0\n임야 299.2\n",
        format_name="md",
        source_path=Path("sample.pdf"),
    )

    repaired, actions = HeuristicRepairer(visual_table_recoverer=_FakeVisualTableRecoverer()).repair(
        source,
        candidate,
        metrics,
    )

    assert "| 항목 | 값 |" in repaired.content
    assert repaired.metadata["repair_issue_types"] == ["table_visual_recovery"]
    assert repaired.metadata["repair_routes"] == ["recover_tables_from_pdf_image"]
    assert len(actions) == 1


def test_pdf_table_heuristics_do_not_reconstruct_complex_tables_before_visual_repair() -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-complex-table",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MERGED_CELL_LOSS],
        judge_result=JudgeResult(overall_score=0.8, issues=["Table 4.2-2 merged cells were flattened."]),
    )
    candidate = ParseCandidate(
        parser_name="layout-first-pdf",
        content="표 4.2-2\n항목  63.0\n세부  299.2\n",
        format_name="md",
        source_path=Path("sample.pdf"),
        metadata={
            "table_format": "html",
            "table_regions": [
                {"table_id": "p1-t1", "page": 1, "extraction_mode": "reference"},
            ],
        },
    )

    repaired, actions = HeuristicRepairer().repair_heuristics(source, candidate, metrics)

    assert repaired is candidate
    assert actions == []


def test_visual_repair_tasks_prefer_html_for_complex_table_issues(monkeypatch) -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-task-format",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MERGED_CELL_LOSS],
        judge_result=JudgeResult(overall_score=0.8, issues=["Table 4.2-2 on p.4 needs visual repair."]),
    )
    candidate = ParseCandidate(
        parser_name="layout-first-pdf",
        content="표 4.2-2\nbroken table\n",
        format_name="md",
        metadata={
            "table_format": "html",
            "table_regions": [{"table_id": "p1-t1", "page": 1, "extraction_mode": "reference"}],
        },
    )
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test")
    monkeypatch.setattr(recoverer, "_find_page_number", lambda path, table_label: 4)

    tasks = HeuristicRepairer(visual_table_recoverer=recoverer).plan_chunk_repairs(
        source,
        candidate,
        metrics,
        max_tasks=2,
    )

    assert len(tasks) == 1
    assert tasks[0].issue_types == (TABLE_ISSUE_MERGED_CELL_LOSS,)
    assert tasks[0].preferred_output_format == "html"


def test_visual_repair_tasks_use_support_label_page_hints_when_pdf_lookup_fails(monkeypatch) -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-task-page-hint",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MERGED_CELL_LOSS],
        judge_result=JudgeResult(overall_score=0.8, issues=["표 4.2-2 구조 보존이 미흡함"]),
    )
    candidate_metadata = {
        "support_parser_metadata": {
            "layout-first-pdf": {
                "table_label_pages": {"4.2-2": 6, "표 4.2-2": 6},
            }
        }
    }
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test")
    monkeypatch.setattr(recoverer, "_find_page_number", lambda path, table_label: None)

    tasks = recoverer.plan_tasks(source, "표 4.2-2\nbroken table", metrics, candidate_metadata=candidate_metadata)

    assert len(tasks) == 1
    assert tasks[0].table_label == "표 4.2-2"
    assert tasks[0].page_number == 6


def test_visual_repair_tasks_prefer_structured_table_findings_over_issue_text(monkeypatch) -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-task-structured-findings",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MISSING_HEADER],
        judge_result=JudgeResult(
            overall_score=0.8,
            issues=["ambiguous free-form issue text"],
            table_findings=[
                {"issue_type": TABLE_ISSUE_MISSING_HEADER, "table_label": "표 4.2-2", "page_number": 4}
            ],
        ),
    )
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test")
    monkeypatch.setattr(
        recoverer,
        "_find_page_number",
        lambda path, table_label: (_ for _ in ()).throw(AssertionError("structured page_number should be used first")),
    )

    tasks = recoverer.plan_tasks(source, "표 4.2-2\nbroken table", metrics, candidate_metadata={})

    assert len(tasks) == 1
    assert tasks[0].table_label == "표 4.2-2"
    assert tasks[0].page_number == 4


def test_visual_repair_tasks_prefer_structured_repair_targets_over_judge_findings(monkeypatch) -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-task-structured-targets",
        extracted_text="source text",
        page_count=12,
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MISSING_HEADER],
        judge_result=JudgeResult(
            overall_score=0.8,
            issues=["free-form issue text"],
            table_findings=[
                {"issue_type": TABLE_ISSUE_MISSING_HEADER, "table_label": "표 4.2-2", "page_number": 9}
            ],
        ),
    )
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test")
    monkeypatch.setattr(
        recoverer,
        "_find_page_number",
        lambda path, table_label: (_ for _ in ()).throw(AssertionError("repair target page should be used first")),
    )

    tasks = recoverer.plan_tasks(
        source,
        "표 4.2-2\nbroken table",
        metrics,
        candidate_metadata={},
        repair_targets=[
            RepairTarget(
                target_kind="table",
                issue_type=TABLE_ISSUE_MISSING_HEADER,
                route_name="recover_tables_from_pdf_image",
                description="recover table",
                table_label="표 4.2-2",
                page_number=4,
                source_name="judge_table_finding",
            )
        ],
    )

    assert len(tasks) == 1
    assert tasks[0].table_label == "표 4.2-2"
    assert tasks[0].page_number == 4


def test_visual_repair_tasks_use_page_scoped_synthetic_targets_when_label_is_missing() -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-task-synthetic-targets",
        extracted_text="source text",
        page_count=12,
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MISSING_HEADER],
    )
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test")

    tasks = recoverer.plan_tasks(
        source,
        "broken table",
        metrics,
        candidate_metadata={},
        repair_targets=[
            RepairTarget(
                target_kind="table",
                issue_type=TABLE_ISSUE_MISSING_HEADER,
                route_name="recover_tables_from_pdf_image",
                description="recover table",
                table_label=None,
                page_number=4,
                source_name="parser_table_region",
            )
        ],
    )

    assert len(tasks) == 1
    assert tasks[0].page_number == 4
    assert tasks[0].table_label.startswith("__page_table__:")


def test_visual_repair_tasks_correct_invalid_structured_page_number_with_metadata() -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-task-structured-page-fix",
        extracted_text="source text",
        page_count=12,
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MISSING_HEADER],
    )
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test")

    tasks = recoverer.plan_tasks(
        source,
        "표 4.2-2\nbroken table",
        metrics,
        candidate_metadata={"table_label_pages": {"표 4.2-2": 4}},
        repair_targets=[
            RepairTarget(
                target_kind="table",
                issue_type=TABLE_ISSUE_MISSING_HEADER,
                route_name="recover_tables_from_pdf_image",
                description="recover table",
                table_label="표 4.2-2",
                page_number=107,
                source_name="judge_table_finding",
            )
        ],
    )

    assert len(tasks) == 1
    assert tasks[0].table_label == "표 4.2-2"
    assert tasks[0].page_number == 4


def test_visual_repair_prioritizes_flattened_table_block_over_slot_only_issue(monkeypatch) -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-task-priority",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=["missing_header", "numeric_token_break"],
        judge_result=JudgeResult(
            overall_score=0.8,
            issues=["표 4.1-1, 표 4.2-2 구조 보존 미흡"],
        ),
    )
    candidate_metadata = {
        "table_slots": [
            {
                "label": "표 4.1-1",
                "placeholder": "<!-- table-slot: page=3 region=1 global=1 label=표 4.1-1 parser=layout-first-pdf -->",
            },
            {
                "label": "표 4.2-2",
                "placeholder": "<!-- table-slot: page=4 region=2 global=2 label=표 4.2-2 parser=layout-first-pdf -->",
                "original_text": "구 분 소 계 전 답 임 야 대 지 도 로 하 천 기 타\n여수시 512.3 63.0 38.2 299.2",
            },
        ],
        "support_parser_metadata": {
            "layout-first-pdf": {
                "table_label_pages": {"표 4.1-1": 3, "표 4.2-2": 4},
            }
        },
    }
    content = (
        "표 4.1-1\n"
        "<!-- table-slot: page=3 region=1 global=1 label=표 4.1-1 parser=layout-first-pdf -->\n\n"
        "표 4.2-2\n"
        "구 분 소 계 전 답 임 야 대 지 도 로 하 천 기 타\n"
        "여수시 512.3 63.0 38.2 299.2 24.9 23.7 2.7 60.6\n"
    )
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test")
    monkeypatch.setattr(
        recoverer,
        "_find_page_number",
        lambda path, table_label: {"표 4.1-1": 3, "표 4.2-2": 4}.get(table_label),
    )

    tasks = recoverer.plan_tasks(source, content, metrics, candidate_metadata=candidate_metadata, max_tasks=1)

    assert len(tasks) == 1
    assert tasks[0].table_label == "표 4.2-2"


def test_visual_repair_prioritizes_structural_collapse_issue_text(monkeypatch) -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-task-priority-issue-text",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=["missing_header", "split_multipage_table", "numeric_token_break"],
        judge_result=JudgeResult(
            overall_score=0.8,
            issues=[
                "표 4.1-1, 표 4.2-1, 표 4.2-3, 표 4.2-4 는 table-slot 주석만 남아 있음",
                "표 4.2-2에서 행/열 구조가 붕괴되고 핵심 행이 누락됨",
            ],
        ),
    )
    candidate_metadata = {
        "table_slots": [
            {
                "label": "표 4.1-1",
                "placeholder": "<!-- table-slot: page=3 region=1 global=1 label=표 4.1-1 parser=layout-first-pdf -->",
                "original_text": "경도와 위도의 극점",
            },
            {
                "label": "표 4.2-2",
                "placeholder": "<!-- table-slot: page=4 region=2 global=2 label=표 4.2-2 parser=layout-first-pdf -->",
                "original_text": "구 분 소 계 전 답 임 야 대 지 도 로 하 천 기 타\n여수시 512.3 63.0 38.2 299.2",
            },
        ],
        "support_parser_metadata": {
            "layout-first-pdf": {
                "table_label_pages": {"표 4.1-1": 3, "표 4.2-2": 4},
            }
        },
    }
    content = (
        "표 4.1-1\n"
        "경도와 위도의 극점\n\n"
        "표 4.2-2\n"
        "구 분 소 계 전 답 임 야 대 지 도 로 하 천 기 타\n"
        "여수시 512.3 63.0 38.2 299.2 24.9 23.7 2.7 60.6\n"
    )
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test")
    monkeypatch.setattr(
        recoverer,
        "_find_page_number",
        lambda path, table_label: {"표 4.1-1": 3, "표 4.2-2": 4}.get(table_label),
    )

    tasks = recoverer.plan_tasks(source, content, metrics, candidate_metadata=candidate_metadata, max_tasks=1)

    assert len(tasks) == 1
    assert tasks[0].table_label == "표 4.2-2"


def test_issue_specific_prompt_guidance_includes_table_issue_hints() -> None:
    guidance = _issue_specific_prompt_guidance(
        (
            "missing_header",
            "numeric_token_break",
            "split_multipage_table",
        ),
        "html",
    )

    assert any("header row" in line for line in guidance)
    assert any("decimal points" in line for line in guidance)
    assert any("continue across pages" in line for line in guidance)
    assert any("Return HTML" in line for line in guidance)


def test_visual_repair_tasks_expand_ambiguous_page_into_multiple_region_tasks() -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-task-ambiguous-page",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MERGED_CELL_LOSS],
        judge_result=JudgeResult(overall_score=0.8, issues=["p.6 표 구조 보존이 미흡함"]),
    )
    candidate_metadata = {
        "support_parser_metadata": {
            "layout-first-pdf": {
                "table_regions": [
                    {"table_id": "p6-t2", "page": 6, "bbox": [0.0, 120.0, 100.0, 200.0], "extraction_mode": "reference"},
                    {"table_id": "p6-t1", "page": 6, "bbox": [0.0, 20.0, 100.0, 100.0], "extraction_mode": "reference"},
                ]
            }
        }
    }
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test", max_tables_per_round=4)

    tasks = recoverer.plan_tasks(source, "broken page section", metrics, candidate_metadata=candidate_metadata)

    assert [task.table_label for task in tasks] == ["__page_table__:6:1", "__page_table__:6:2"]


def test_chunk_repair_planning_can_override_recoverer_default_task_cap() -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-task-cap-override",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MERGED_CELL_LOSS],
        judge_result=JudgeResult(
            overall_score=0.8,
            issues=["p.6 table needs visual repair.", "p.7 table needs visual repair."],
        ),
    )
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="broken table content",
        format_name="md",
        metadata={
            "support_parser_metadata": {
                "layout-first-pdf": {
                    "table_regions": [
                        {"table_id": "p6-t1", "page": 6, "bbox": [0.0, 10.0, 100.0, 100.0], "extraction_mode": "reference"},
                        {"table_id": "p7-t1", "page": 7, "bbox": [0.0, 10.0, 100.0, 100.0], "extraction_mode": "reference"},
                    ]
                }
            }
        },
    )
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test", max_tables_per_round=1)

    tasks = HeuristicRepairer(visual_table_recoverer=recoverer).plan_chunk_repairs(
        source,
        candidate,
        metrics,
        max_tasks=2,
    )

    assert [task.table_label for task in tasks] == ["__page_table__:6:1", "__page_table__:7:1"]


def test_direct_visual_repair_page_scoped_task_uses_page_replacement_fallback(monkeypatch) -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-page-fallback",
        extracted_text="source text",
    )
    content = (
        "<!-- page 6 -->\n"
        "intro\n"
        "item  value  amount\n"
        "alpha  10  20\n"
        "beta  30  40\n"
        "\n"
        "tail\n"
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MERGED_CELL_LOSS],
        judge_result=JudgeResult(overall_score=0.8, issues=["p.6 table needs visual repair."]),
    )
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test")
    monkeypatch.setattr(
        recoverer,
        "plan_tasks",
        lambda source_arg, content_arg, metrics_arg: [
            VisualRepairTask(task_id="page-6", table_label="__page_table__:6", page_number=6)
        ],
    )
    monkeypatch.setattr(
        recoverer,
        "recover_task",
        lambda source_arg, content_arg, task: VisualTableRecovery(
            table_label=task.table_label,
            page_number=task.page_number,
            confidence=0.9,
            markdown="| new | table |\n| --- | --- |\n| 3 | 4 |",
            notes=[],
            crop_method="full-page",
            bbox=None,
        ),
    )

    updated, actions = recoverer.repair(source, content, metrics)

    assert "| new | table |" in updated
    assert "alpha  10  20" not in updated
    assert len(actions) == 1


def test_direct_visual_repair_uses_candidate_metadata_for_html_preference(monkeypatch) -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-direct-html-preference",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MERGED_CELL_LOSS],
        judge_result=JudgeResult(overall_score=0.8, issues=["Table 4.2-2 on p.4 needs visual repair."]),
    )
    candidate = ParseCandidate(
        parser_name="layout-first-pdf",
        content="표 4.2-2\nbroken table\n",
        format_name="md",
        metadata={
            "table_format": "html",
            "table_regions": [{"table_id": "p1-t1", "page": 1, "extraction_mode": "reference"}],
        },
    )
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test")
    captured: dict[str, object] = {}

    def _fake_plan_tasks(source_arg, content_arg, metrics_arg, candidate_metadata=None):
        del source_arg, content_arg, metrics_arg
        captured["candidate_metadata"] = candidate_metadata
        return [
            VisualRepairTask(
                task_id="task-1",
                table_label="표 4.2-2",
                page_number=4,
                issue_types=(TABLE_ISSUE_MERGED_CELL_LOSS,),
                preferred_output_format="html" if candidate_metadata else "markdown",
            )
        ]

    monkeypatch.setattr(recoverer, "plan_tasks", _fake_plan_tasks)
    monkeypatch.setattr(
        recoverer,
        "recover_task",
        lambda source_arg, content_arg, task: VisualTableRecovery(
            table_label=task.table_label,
            page_number=task.page_number,
            confidence=0.9,
            markdown="<table>\n  <tr><th>a</th></tr>\n  <tr><td>b</td></tr>\n</table>",
            notes=[f"format={task.preferred_output_format}"],
            crop_method="full-page",
            bbox=None,
        ),
    )

    repaired, actions = HeuristicRepairer(visual_table_recoverer=recoverer).repair(source, candidate, metrics)

    assert isinstance(captured["candidate_metadata"], dict)
    assert captured["candidate_metadata"]["table_format"] == "html"
    assert captured["candidate_metadata"]["table_regions"] == candidate.metadata["table_regions"]
    assert "<table>" not in repaired.content
    assert "| a |" in repaired.content
    assert "| b |" in repaired.content
    assert any(action.issue_type == "table_visual_recovery" for action in actions)
    assert any("format=html" in action.description for action in actions)


class _FakeTable:
    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        self.bbox = bbox


class _FakeTableFinder:
    def __init__(self, tables: list[_FakeTable]) -> None:
        self.tables = tables


class _FakePage:
    rect = fitz.Rect(0, 0, 600, 800)

    def __init__(self) -> None:
        self._tables = [
            _FakeTable((40, 90, 560, 150)),
            _FakeTable((70, 210, 520, 330)),
            _FakeTable((70, 520, 520, 720)),
        ]

    def search_for(self, text: str):
        if "4.2-2" in text:
            return [fitz.Rect(80, 180, 160, 200)]
        return []

    def find_tables(self) -> _FakeTableFinder:
        return _FakeTableFinder(self._tables)


def test_visual_repair_crop_prefers_detected_table_bbox_near_label() -> None:
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test")
    anchor = fitz.Rect(80, 180, 160, 200)

    crop = recoverer._build_table_crop(_FakePage(), 4, "table 4.2-2")

    assert crop.method == "pymupdf"
    assert crop.bbox == (70, 210, 520, 330)

    selected = recoverer._detect_table_rect(_FakePage(), anchor)

    assert selected is not None
    assert (round(selected.x0), round(selected.y0), round(selected.x1), round(selected.y1)) == (70, 210, 520, 330)


def test_replace_table_block_prefers_explicit_table_slot_placeholder() -> None:
    placeholder = "<!-- table-slot: page=4 region=1 global=1 label=표 4.2-2 parser=layout-first-pdf -->"
    content = (
        "<!-- page 4 -->\n"
        "표 4.2-2 지목별 토지이용 현황\n"
        f"{placeholder}\n"
        "\n"
        "## 다음 절\n"
    )

    updated = replace_table_block(
        content,
        "표 4.2-2",
        "| 구분 | 면적 |\n| --- | --- |\n| 합계 | 63.0 |",
        candidate_metadata={
            "table_slots": [
                {
                    "placeholder": placeholder,
                    "page": 4,
                    "region_index": 1,
                    "global_index": 1,
                    "label": "표 4.2-2",
                    "support_parser": "layout-first-pdf",
                }
            ]
        },
    )

    assert placeholder not in updated
    assert "| 구분 | 면적 |" in updated
    assert "## 다음 절" in updated
