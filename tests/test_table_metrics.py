"""TEDS-lite 셀 단위 표 메트릭 검증."""

from pathlib import Path

from parsing_agent.config import WorkflowConfig
from parsing_agent.evaluation import DeterministicEvaluator
from parsing_agent.models import DocumentSource, ParseCandidate
from parsing_agent.table_metrics import (
    calculate_table_cell_similarity,
    parse_markdown_table_grid,
    teds_lite,
)

REFERENCE_GRID = [
    ["구분", "조사항목", "조사주기"],
    ["수질", "bod, cod", "분기 1회"],
    ["대기", "pm10, no2", "월 1회"],
]


def test_parse_markdown_table_grid_drops_separator_rows() -> None:
    grid = parse_markdown_table_grid(
        [
            "| 구분 | 조사항목 | 조사주기 |",
            "| --- | --- | --- |",
            "| 수질 | BOD, COD | 분기 1회 |",
        ]
    )
    assert grid == [["구분", "조사항목", "조사주기"], ["수질", "bod, cod", "분기 1회"]]


def test_teds_lite_identical_grids_score_one() -> None:
    assert teds_lite(REFERENCE_GRID, REFERENCE_GRID) == 1.0


def test_teds_lite_penalizes_lost_column() -> None:
    two_column = [row[:2] for row in REFERENCE_GRID]
    score = teds_lite(REFERENCE_GRID, two_column)
    assert score < 0.75


def test_teds_lite_penalizes_missing_rows() -> None:
    truncated = REFERENCE_GRID[:1]
    assert teds_lite(REFERENCE_GRID, truncated) < 0.5


def test_cell_similarity_matches_best_candidate_table() -> None:
    candidate_text = "\n".join(
        [
            "본문입니다.",
            "| 엉뚱 | 표 |",
            "| --- | --- |",
            "| x | y |",
            "",
            "| 구분 | 조사항목 | 조사주기 |",
            "| --- | --- | --- |",
            "| 수질 | BOD, COD | 분기 1회 |",
            "| 대기 | PM10, NO2 | 월 1회 |",
        ]
    )
    score = calculate_table_cell_similarity([REFERENCE_GRID], candidate_text)
    assert score is not None and score > 0.95


def test_cell_similarity_none_without_reference_and_zero_without_candidate_tables() -> None:
    assert calculate_table_cell_similarity([], "본문") is None
    assert calculate_table_cell_similarity([REFERENCE_GRID], "표 없는 본문") == 0.0


def test_report_payload_serializes_cell_similarity() -> None:
    from parsing_agent.models import EvaluationMetrics, WorkflowResult
    from parsing_agent.reporting import build_report_payload

    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.9,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.9,
        table_cell_similarity=0.42,
    )
    result = WorkflowResult(
        source=_dummy_source(),
        best_candidate=ParseCandidate(parser_name="p", content="본문", format_name="md"),
        metrics=metrics,
    )

    payload = build_report_payload(result)

    # 리포트 직렬화가 필드 하드코딩이라 새 메트릭이 빠지는 회귀가 실제로 있었다.
    assert payload["metrics"]["table_cell_similarity"] == 0.42


def _dummy_source() -> DocumentSource:
    return DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="report-serialization",
        extracted_text="원문",
    )


def test_evaluator_records_cell_similarity_for_pdf_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        "parsing_agent.evaluation.extract_pdf_table_grids",
        lambda path, max_pages=40: [REFERENCE_GRID],
    )
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="teds-eval",
        extracted_text="구분 조사항목 조사주기 수질 대기",
    )
    candidate = ParseCandidate(
        parser_name="p",
        content="| 구분 | 항목 |\n| --- | --- |\n| 수질 | BOD |",
        format_name="md",
    )

    metrics = DeterministicEvaluator(WorkflowConfig(judge_weight=0)).evaluate(source, candidate)

    assert metrics.table_cell_similarity is not None
    assert metrics.table_cell_similarity < 0.7
    assert any("teds_lite" in note for note in metrics.notes)
