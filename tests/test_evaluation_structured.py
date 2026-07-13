"""구조화 포맷(비-PDF) 콘텐츠 기반 평가의 회귀 테스트.

핵심 계약: 파서가 구조를 *추가*한 것은 감점이 아니고, 콘텐츠 손실은
여전히 잡혀야 한다. 플래그를 끄면 기존 형태 비교로 돌아간다.
"""

from pathlib import Path

from parsing_agent.config import WorkflowConfig
from parsing_agent.evaluation import (
    DeterministicEvaluator,
    _structured_structure_retention,
    _structured_table_preservation,
    calculate_content_similarity,
    strip_markdown_decorations,
)
from parsing_agent.models import DocumentSource, ParseCandidate


def _structured_source(tmp_path: Path, file_name: str, plain_text: str) -> DocumentSource:
    path = tmp_path / file_name
    path.write_text(plain_text, encoding="utf-8")
    return DocumentSource(
        path=path,
        media_type="application/octet-stream",
        size_bytes=path.stat().st_size,
        run_id="eval-structured",
        extracted_text=plain_text,
    )


def _candidate(content: str) -> ParseCandidate:
    return ParseCandidate(parser_name="p", content=content, format_name="md")


# ---------------------------------------------------------------------------
# strip_markdown_decorations
# ---------------------------------------------------------------------------


def test_strip_markdown_decorations_removes_parser_added_syntax() -> None:
    markdown = "\n".join(
        [
            "<!-- slide 1 -->",
            "## 사업 개요",
            "- **timeout:** 900",
            "| 지표 | 값 |",
            "| --- | --- |",
            "| 최대 파고 | 8.18 |",
            "",
            "본문 문단이다.",
        ]
    )

    stripped = strip_markdown_decorations(markdown)

    assert "<!--" not in stripped
    assert "#" not in stripped
    assert "|" not in stripped
    assert "**" not in stripped
    assert "---" not in stripped
    assert "사업 개요" in stripped
    assert "timeout: 900" in stripped
    assert "지표 값" in stripped
    assert "최대 파고 8.18" in stripped
    assert "본문 문단이다." in stripped


def test_content_similarity_ignores_markup_differences() -> None:
    source = '{"name": "collision", "months": 52}'
    candidate = "- **name:** collision\n- **months:** 52"

    assert calculate_content_similarity(source, strip_markdown_decorations(candidate)) == 1.0


def test_content_similarity_still_penalizes_missing_tokens() -> None:
    source = "첫 문장이다. 둘째 문장이다. 셋째 문장이다."
    candidate = "첫 문장이다."

    assert calculate_content_similarity(source, candidate) < 0.6


# ---------------------------------------------------------------------------
# 구조/표 보존의 중립 규칙
# ---------------------------------------------------------------------------


def test_structured_table_preservation_is_neutral_without_source_tables() -> None:
    assert _structured_table_preservation("평문 원문", "| a | b |\n| --- | --- |\n| 1 | 2 |") == 1.0


def test_structured_table_preservation_still_compares_when_source_has_tables() -> None:
    source = "| a | b |\n| 1 | 2 |"
    assert _structured_table_preservation(source, "표 없는 후보") == 0.0


def test_structured_structure_retention_is_neutral_without_source_markers() -> None:
    assert _structured_structure_retention("평문 한 줄", "제목\n본문 여러 줄\n더 많은 줄") == 1.0


# ---------------------------------------------------------------------------
# evaluator 통합
# ---------------------------------------------------------------------------


def test_perfect_structured_parse_scores_high(tmp_path: Path) -> None:
    plain = "사업 개요\n본문 문단이다.\n지표 값\n최대 파고 8.18"
    source = _structured_source(tmp_path, "report.docx", plain)
    candidate = _candidate(
        "# 사업 개요\n\n본문 문단이다.\n\n| 지표 | 값 |\n| --- | --- |\n| 최대 파고 | 8.18 |"
    )

    metrics = DeterministicEvaluator(config=WorkflowConfig(judge_weight=0)).evaluate(source, candidate)

    assert metrics.text_coverage == 1.0
    assert metrics.normalized_similarity > 0.95
    assert metrics.structure_retention == 1.0
    assert metrics.table_preservation == 1.0
    assert metrics.total_score > 0.95


def test_gutted_candidate_still_scores_low(tmp_path: Path) -> None:
    plain = "\n".join(f"{i}번째 문장의 본문 내용이다." for i in range(1, 11))
    source = _structured_source(tmp_path, "report.docx", plain)
    candidate = _candidate("# 1번째 문장의 본문 내용이다.")

    metrics = DeterministicEvaluator(config=WorkflowConfig(judge_weight=0)).evaluate(source, candidate)

    assert metrics.text_coverage < 0.3
    assert metrics.total_score < 0.6


def test_flag_off_restores_legacy_shape_comparison(tmp_path: Path) -> None:
    plain = "지표 값\n최대 파고 8.18"
    source = _structured_source(tmp_path, "data.csv", plain)
    candidate = _candidate("| 지표 | 값 |\n| --- | --- |\n| 최대 파고 | 8.18 |")

    on = DeterministicEvaluator(
        config=WorkflowConfig(judge_weight=0, structured_content_evaluation_enabled=True)
    ).evaluate(source, candidate)
    off = DeterministicEvaluator(
        config=WorkflowConfig(judge_weight=0, structured_content_evaluation_enabled=False)
    ).evaluate(source, candidate)

    assert on.table_preservation == 1.0
    assert off.table_preservation == 0.0  # 기존 형태 비교: 원문에 파이프 표가 없어 0
    assert on.total_score > off.total_score


def test_plain_text_sources_are_unaffected(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    plain = "그대로 통과하는 평문이다."
    path.write_text(plain, encoding="utf-8")
    source = DocumentSource(
        path=path, media_type="text/plain", size_bytes=0, run_id="plain", extracted_text=plain
    )

    metrics = DeterministicEvaluator(config=WorkflowConfig(judge_weight=0)).evaluate(
        source, _candidate(plain)
    )

    assert metrics.total_score == 1.0


def test_content_similarity_survives_large_repetitive_token_streams() -> None:
    """1,000+ 레코드 데이터에서 SequenceMatcher autojunk 붕괴(실측 0.001)의 회귀 가드."""
    rows = [f"2026-07-01 {i:04d} 1.{i % 100:02d} 0 0 p" for i in range(1500)]
    source = "\n".join(rows)
    candidate = "| t | v |\n| --- | --- |\n" + "\n".join(f"| {row} |" for row in rows)

    score = calculate_content_similarity(source, strip_markdown_decorations(candidate))

    assert score > 0.95


def test_content_similarity_large_stream_still_detects_loss() -> None:
    rows = [f"관측값 {i} 파고 {i % 9}.{i % 10}" for i in range(3000)]
    source = "\n".join(rows)
    candidate = "\n".join(rows[:900])  # 70% 손실

    assert calculate_content_similarity(source, candidate) < 0.6


def test_unlabeled_pdf_table_preservation_falls_back_to_teds(tmp_path: Path, monkeypatch) -> None:
    """라벨 없는 PDF에서 라벨 매칭 표 점수(실측 0.016)가 TEDS-lite로 대체되는지."""
    monkeypatch.setattr(
        "parsing_agent.evaluation.extract_pdf_table_grids",
        lambda path, max_pages=40: [[["구간", "연장"], ["북측 호안", "320"]]],
    )
    pdf_path = tmp_path / "spec.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    source = DocumentSource(
        path=pdf_path, media_type="application/pdf", size_bytes=0, run_id="teds",
        extracted_text="라벨 없는 본문. 구간 연장 북측 호안 320", page_count=1,
    )
    candidate = _candidate("라벨 없는 본문.\n\n| 구간 | 연장 |\n| --- | --- |\n| 북측 호안 | 320 |")

    metrics = DeterministicEvaluator(config=WorkflowConfig(judge_weight=0)).evaluate(source, candidate)

    assert metrics.table_cell_similarity == 1.0
    assert metrics.table_preservation == 1.0


def test_labeled_pdf_keeps_label_matching_metric(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "parsing_agent.evaluation.extract_pdf_table_grids",
        lambda path, max_pages=40: [[["구간", "연장"], ["북측 호안", "320"]]],
    )
    pdf_path = tmp_path / "labeled.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    source = DocumentSource(
        path=pdf_path, media_type="application/pdf", size_bytes=0, run_id="teds-label",
        extracted_text="표 4.2-1 설치 내역. 구간 연장", page_count=1,
    )
    # 라벨은 있는데 후보가 라벨을 잃음 → 라벨 매칭 점수가 그대로 작동해야 한다
    candidate = _candidate("| 구간 | 연장 |\n| --- | --- |\n| 북측 호안 | 320 |")

    metrics = DeterministicEvaluator(config=WorkflowConfig(judge_weight=0)).evaluate(source, candidate)

    assert metrics.table_preservation < 1.0  # TEDS로 덮어쓰지 않는다


def test_pdf_similarity_is_not_penalized_by_table_markup(tmp_path: Path) -> None:
    """표가 좋아질수록 유사도가 나빠지던 PDF 경로의 역설(실측 0.76→0.46) 회귀 가드."""
    pdf_path = tmp_path / "tables.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    plain = "구간 연장 형식\n북측 호안 320 고정식\n남측 개구부 180 이동식"
    source = DocumentSource(
        path=pdf_path, media_type="application/pdf", size_bytes=0, run_id="pdf-sim",
        extracted_text=plain, page_count=1,
    )
    tabled = _candidate(
        "| 구간 | 연장 | 형식 |\n| --- | --- | --- |\n| 북측 호안 | 320 | 고정식 |\n| 남측 개구부 | 180 | 이동식 |"
    )

    on = DeterministicEvaluator(config=WorkflowConfig(judge_weight=0)).evaluate(source, tabled)
    off = DeterministicEvaluator(
        config=WorkflowConfig(judge_weight=0, pdf_content_similarity_enabled=False)
    ).evaluate(source, tabled)

    assert on.normalized_similarity > 0.9   # 장식 제거 후엔 사실상 동일 콘텐츠
    assert on.normalized_similarity > off.normalized_similarity
