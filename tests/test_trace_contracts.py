"""노드 간 계약 구조화 검증: judge 문장 파싱 폴백 격하, 트레이스 요약 구조화."""

from pathlib import Path

from parsing_agent.evaluation import (
    TABLE_ISSUE_MERGED_CELL_LOSS,
    TABLE_ISSUE_MISSING_HEADER,
    classify_table_issues,
)
from parsing_agent.models import DocumentSource, JudgeResult, ParseCandidate, RepairAction
from parsing_agent.repair import RepairTarget
from parsing_agent.workflow import _summarize_langsmith_payload


def _pdf_source() -> DocumentSource:
    return DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="trace-contract",
        extracted_text="표 4.2-2 조사항목",
    )


def _candidate() -> ParseCandidate:
    return ParseCandidate(parser_name="opendataloader-pdf", content="표 4.2-2 조사항목", format_name="md")


# --- classify_table_issues: 구조화 채널 우선 ------------------------------------


def test_structured_findings_suppress_sentence_pattern_matching() -> None:
    judge_result = JudgeResult(
        overall_score=0.7,
        table_findings=[{"issue_type": TABLE_ISSUE_MERGED_CELL_LOSS, "table_label": "표 4.2-2"}],
        # 문장에는 "반복"(TEXT_DUPLICATION 패턴)과 "계속"(SPLIT_MULTIPAGE 패턴)이
        # 포함되지만, 구조화 findings가 있으므로 무시돼야 한다.
        issues=["표 내용이 반복되는 것처럼 보이며 다음 페이지로 계속 이어질 수 있음"],
    )

    issues = classify_table_issues(_pdf_source(), _candidate(), judge_result)

    assert issues == [TABLE_ISSUE_MERGED_CELL_LOSS]


def test_sentence_pattern_fallback_when_no_structured_findings() -> None:
    judge_result = JudgeResult(
        overall_score=0.7,
        table_findings=[],
        issues=["표의 머리글이 없습니다"],
    )

    issues = classify_table_issues(_pdf_source(), _candidate(), judge_result)

    assert TABLE_ISSUE_MISSING_HEADER in issues


def test_invalid_issue_type_in_findings_still_allows_fallback() -> None:
    judge_result = JudgeResult(
        overall_score=0.7,
        table_findings=[{"issue_type": "not_a_taxonomy_type"}],
        issues=["표의 머리글이 없습니다"],
    )

    issues = classify_table_issues(_pdf_source(), _candidate(), judge_result)

    assert TABLE_ISSUE_MISSING_HEADER in issues


# --- 트레이스 요약: 문장 없이 구조화 필드만 -------------------------------------


def test_trace_summary_exposes_repair_target_fields_without_sentences() -> None:
    targets = [
        RepairTarget(
            target_kind="table",
            issue_type=TABLE_ISSUE_MISSING_HEADER,
            route_name="recover_tables_from_pdf_image",
            description="이 문장은 트레이스에 나가면 안 된다",
            table_label="표 4.2-2",
            page_number=4,
            severity="high",
            confidence=0.8,
        )
    ]

    summary = _summarize_langsmith_payload({"repair_targets": targets})

    field = summary["fields"]["repair_targets"]
    assert field["type"] == "repair_targets"
    assert field["items"][0]["issue_type"] == TABLE_ISSUE_MISSING_HEADER
    assert field["items"][0]["route_name"] == "recover_tables_from_pdf_image"
    assert field["items"][0]["table_label"] == "표 4.2-2"
    assert "description" not in field["items"][0]


def test_trace_summary_exposes_repair_action_codes_without_excerpts() -> None:
    actions = [
        RepairAction(
            action_name="merge_wrapped_lines",
            description="문장 설명",
            before_excerpt="수정 전 원문 문장",
            after_excerpt="수정 후 원문 문장",
            issue_type="wrapped_line_noise",
            route_name="merge_wrapped_lines",
        )
    ]

    summary = _summarize_langsmith_payload({"repairs": actions})

    field = summary["fields"]["repairs"]
    assert field["type"] == "repair_actions"
    assert field["items"][0] == {
        "action_name": "merge_wrapped_lines",
        "issue_type": "wrapped_line_noise",
        "route_name": "merge_wrapped_lines",
    }


def test_trace_summary_keeps_plain_collections_as_counts() -> None:
    summary = _summarize_langsmith_payload({"failed_visual_task_keys": ["표 1|2|missing_header"]})

    assert summary["fields"]["failed_visual_task_keys"] == {"type": "collection", "count": 1}
