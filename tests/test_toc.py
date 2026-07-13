"""PDF 북마크(TOC) 기반 구조 복원·평가의 회귀 테스트. 실제 fitz PDF로 검증한다."""

from pathlib import Path

import fitz

from parsing_agent.config import WorkflowConfig
from parsing_agent.evaluation import DeterministicEvaluator
from parsing_agent.models import DocumentSource, EvaluationMetrics, ParseCandidate
from parsing_agent.repair import _classify_repair_directives
from parsing_agent.toc import read_pdf_toc, restore_headings_from_toc, toc_title_coverage


def _metrics(coverage: float = 1.0) -> EvaluationMetrics:
    return EvaluationMetrics(
        text_coverage=coverage,
        normalized_similarity=coverage,
        structure_retention=1.0,
        table_preservation=1.0,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=coverage,
    )


def _pdf_with_toc(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "outlined.pdf"
    document = fitz.open()
    for _ in range(2):
        document.new_page()
    document.set_toc([[1, "1. General Guidance", 1], [2, "1.1 Data Access", 2]])
    document.save(pdf_path)
    return pdf_path


def _source(pdf_path: Path, text: str = "원문") -> DocumentSource:
    return DocumentSource(
        path=pdf_path, media_type="application/pdf", size_bytes=0,
        run_id="toc-test", extracted_text=text, page_count=2,
    )


def test_read_pdf_toc_returns_entries(tmp_path: Path) -> None:
    toc = read_pdf_toc(_pdf_with_toc(tmp_path))
    assert toc == [(1, "1. General Guidance", 1), (2, "1.1 Data Access", 2)]


def test_read_pdf_toc_empty_without_bookmarks(tmp_path: Path) -> None:
    pdf_path = tmp_path / "plain.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    assert read_pdf_toc(pdf_path) == []


def test_toc_title_coverage_counts_surviving_titles(tmp_path: Path) -> None:
    toc = read_pdf_toc(_pdf_with_toc(tmp_path))
    assert toc_title_coverage(toc, "1. General Guidance 본문...") == 0.5
    assert toc_title_coverage(toc, "무관한 내용") == 0.0
    assert toc_title_coverage([], "x") is None


def test_restore_promotes_existing_title_and_inserts_missing_one(tmp_path: Path) -> None:
    source = _source(_pdf_with_toc(tmp_path))
    content = "1. General Guidance\n\n본문이다.\n\n<!-- page 2 -->\n\n둘째 페이지 본문."

    restored = restore_headings_from_toc(source, content)

    assert "# 1. General Guidance" in restored          # 승격
    assert "## 1.1 Data Access" in restored             # 페이지 마커 뒤 복원
    assert restored.index("<!-- page 2 -->") < restored.index("## 1.1 Data Access")


def test_restore_is_noop_without_toc(tmp_path: Path) -> None:
    pdf_path = tmp_path / "plain.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    assert restore_headings_from_toc(_source(pdf_path), "본문") == "본문"


def test_structure_metric_uses_toc_coverage_as_floor(tmp_path: Path) -> None:
    source = _source(_pdf_with_toc(tmp_path), text="원문 텍스트 줄")
    # 정규식 cue가 전혀 없는 후보 — 기존 메트릭이면 구조 점수가 무너지지만,
    # TOC 제목이 전부 살아 있으므로 하한이 1.0이 된다.
    candidate = ParseCandidate(
        parser_name="p",
        content="# 1. General Guidance\n본문\n## 1.1 Data Access\n둘째 본문",
        format_name="md",
    )

    metrics = DeterministicEvaluator(config=WorkflowConfig(judge_weight=0)).evaluate(source, candidate)

    assert metrics.structure_retention == 1.0


def test_directive_triggers_on_weak_structure_for_pdf(tmp_path: Path) -> None:
    source = _source(_pdf_with_toc(tmp_path))
    candidate = ParseCandidate(parser_name="opendataloader-pdf", content="후보", format_name="md")
    weak = _metrics(coverage=1.0)
    weak.structure_retention = 0.3

    routes = {d.route_name for d in _classify_repair_directives(source, candidate, weak)}

    assert "restore_headings_from_toc" in routes
