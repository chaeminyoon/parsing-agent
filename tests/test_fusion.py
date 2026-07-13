"""교차 파서 융합(본문 복구·표 단위 베스트 선택)의 회귀 테스트."""

from pathlib import Path

from parsing_agent.config import WorkflowConfig
from parsing_agent.fusion import fuse_missing_body_lines, fuse_tables_from_alternate
from parsing_agent.models import DocumentSource, EvaluationMetrics, ParseCandidate
from parsing_agent.repair import _classify_repair_directives


def _pdf_source(tmp_path: Path, extracted_text: str) -> DocumentSource:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    return DocumentSource(
        path=pdf_path,
        media_type="application/pdf",
        size_bytes=0,
        run_id="fusion-test",
        extracted_text=extracted_text,
        page_count=1,
    )


def _metrics(coverage: float = 1.0, table: float = 1.0, cell: float | None = None) -> EvaluationMetrics:
    return EvaluationMetrics(
        text_coverage=coverage,
        normalized_similarity=coverage,
        structure_retention=1.0,
        table_preservation=table,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=coverage,
        table_cell_similarity=cell,
    )


# ---------------------------------------------------------------------------
# 본문 융합
# ---------------------------------------------------------------------------


def test_fuse_missing_body_restores_dropped_paragraph_in_position(tmp_path: Path) -> None:
    source = _pdf_source(
        tmp_path,
        "제1장 개요\n첫 번째 본문 문단은 사업의 배경을 설명한다.\n"
        "두 번째 본문 문단은 파서가 통째로 떨어뜨린 내용이다.\n제2장 저감방안",
    )
    content = "제1장 개요\n\n첫 번째 본문 문단은 사업의 배경을 설명한다.\n\n제2장 저감방안"

    fused = fuse_missing_body_lines(source, content)

    assert "두 번째 본문 문단은 파서가 통째로 떨어뜨린 내용이다." in fused
    # 원문 순서 보존: 첫 문단 뒤, 제2장 앞
    assert fused.index("첫 번째 본문") < fused.index("두 번째 본문") < fused.index("제2장")


def test_fuse_missing_body_skips_noise_and_covered_lines(tmp_path: Path) -> None:
    source = _pdf_source(tmp_path, "본문 내용이 표로 이미 들어가 있다\n- 3 -\n짧다")
    content = "| 본문 | 내용이 | 표로 | 이미 | 들어가 | 있다 |"

    fused = fuse_missing_body_lines(source, content)

    assert fused == content  # 커버된 줄·페이지번호·짧은 줄은 삽입하지 않는다


def test_fuse_missing_body_without_source_text_is_noop(tmp_path: Path) -> None:
    source = _pdf_source(tmp_path, "")
    assert fuse_missing_body_lines(source, "후보") == "후보"


# ---------------------------------------------------------------------------
# 표 융합 (reference_grids / alternate_content 주입)
# ---------------------------------------------------------------------------

_REFERENCE = [[["구간", "연장"], ["북측 호안", "320"], ["남측 개구부", "180"]][i] for i in range(3)]


def test_fuse_tables_swaps_in_better_alternate_table(tmp_path: Path) -> None:
    source = _pdf_source(tmp_path, "본문")
    # 현재 후보: 셀이 깨진 표
    content = "머리말\n\n| 구간 | 연장 |\n| --- | --- |\n| 북측호안320 |  |\n| 남측개구부180 |  |\n\n꼬리말"
    # 대체 후보: 정확한 표
    alternate = "| 구간 | 연장 |\n| --- | --- |\n| 북측 호안 | 320 |\n| 남측 개구부 | 180 |"

    fused = fuse_tables_from_alternate(
        source, content, current_parser="opendataloader-pdf",
        reference_grids=[_REFERENCE], alternate_content=alternate,
    )

    assert "| 북측 호안 | 320 |" in fused
    assert "북측호안320" not in fused
    assert fused.startswith("머리말") and fused.rstrip().endswith("꼬리말")


def test_fuse_tables_keeps_current_when_alternate_is_not_better(tmp_path: Path) -> None:
    source = _pdf_source(tmp_path, "본문")
    good = "| 구간 | 연장 |\n| --- | --- |\n| 북측 호안 | 320 |\n| 남측 개구부 | 180 |"
    worse_alternate = "| 구간 | 연장 |\n| --- | --- |\n| 엉뚱한 | 값 |\n| 다른 | 표 |"

    fused = fuse_tables_from_alternate(
        source, good, current_parser="opendataloader-pdf",
        reference_grids=[_REFERENCE], alternate_content=worse_alternate,
    )

    assert fused == good


def test_fuse_tables_without_reference_grids_is_noop(tmp_path: Path) -> None:
    source = _pdf_source(tmp_path, "본문")
    content = "| a | b |\n| --- | --- |\n| 1 | 2 |"

    assert fuse_tables_from_alternate(
        source, content, current_parser="opendataloader-pdf",
        reference_grids=[], alternate_content="| x |\n| --- |\n| y |",
    ) == content


def test_fuse_tables_inserts_when_current_lost_the_table_entirely(tmp_path: Path) -> None:
    """표를 통째로 잃은 후보(실측: CO-OPS PDF에서 opendataloader 표 0개)의 복구."""
    source = _pdf_source(tmp_path, "본문")
    content = "머리말 문단이다.\n\n측정 사양 절이다.\n\n꼬리말이다."
    alternate = "측정 사양 절이다.\n\n| 구간 | 연장 |\n| --- | --- |\n| 북측 호안 | 320 |\n| 남측 개구부 | 180 |"

    fused = fuse_tables_from_alternate(
        source, content, current_parser="opendataloader-pdf",
        reference_grids=[_REFERENCE], alternate_content=alternate,
    )

    assert "| 북측 호안 | 320 |" in fused
    # 앵커(공유 문맥 줄) 바로 뒤, 꼬리말 앞에 삽입된다
    assert fused.index("측정 사양") < fused.index("| 북측 호안") < fused.index("꼬리말")


def test_fuse_tables_appends_at_end_without_anchor(tmp_path: Path) -> None:
    source = _pdf_source(tmp_path, "본문")
    content = "완전히 다른 문서 내용."
    alternate = "공유되지 않는 문맥\n\n| 구간 | 연장 |\n| --- | --- |\n| 북측 호안 | 320 |\n| 남측 개구부 | 180 |"

    fused = fuse_tables_from_alternate(
        source, content, current_parser="opendataloader-pdf",
        reference_grids=[_REFERENCE], alternate_content=alternate,
    )

    assert fused.startswith("완전히 다른 문서 내용.")
    assert "| 북측 호안 | 320 |" in fused


# ---------------------------------------------------------------------------
# 디렉티브 트리거
# ---------------------------------------------------------------------------


def test_directives_include_fusion_routes_for_weak_pdf_candidate(tmp_path: Path) -> None:
    source = _pdf_source(tmp_path, "원문 본문 줄이다.")
    candidate = ParseCandidate(parser_name="opendataloader-pdf", content="후보", format_name="md")

    routes = {
        d.route_name
        for d in _classify_repair_directives(source, candidate, _metrics(coverage=0.85, table=0.5, cell=0.4))
    }

    assert "fuse_missing_body_from_source" in routes
    assert "fuse_tables_from_alternate_parser" in routes


def test_directives_skip_fusion_for_healthy_or_non_pdf(tmp_path: Path) -> None:
    healthy_pdf = ParseCandidate(parser_name="opendataloader-pdf", content="후보", format_name="md")
    routes_healthy = {
        d.route_name
        for d in _classify_repair_directives(_pdf_source(tmp_path, "원문"), healthy_pdf, _metrics())
    }
    assert not any(route.startswith("fuse_") for route in routes_healthy)

    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("원문", encoding="utf-8")
    txt_source = DocumentSource(
        path=txt_path, media_type="text/plain", size_bytes=0, run_id="t", extracted_text="원문"
    )
    weak_txt = ParseCandidate(parser_name="text-fallback", content="후보", format_name="md")
    routes_txt = {
        d.route_name
        for d in _classify_repair_directives(txt_source, weak_txt, _metrics(coverage=0.5, table=0.2))
    }
    assert not any(route.startswith("fuse_") for route in routes_txt)


def test_workflow_config_accepts_fusion_defaults() -> None:
    # 융합 라우트는 heuristic(무료)이라 별도 게이트 없이 항상 후보에 오른다.
    # 이 테스트는 config가 그 전제를 깨는 필드를 요구하지 않음을 고정한다.
    WorkflowConfig(judge_weight=0)


# ---------------------------------------------------------------------------
# 다중 대체 후보 (docling 풀 편입)
# ---------------------------------------------------------------------------


def test_fusion_pools_tables_across_multiple_alternates(tmp_path: Path, monkeypatch) -> None:
    """대체 후보가 여럿일 때 그리드 심판이 풀 전체에서 최고 표를 고른다."""
    import parsing_agent.fusion as fusion

    source = _pdf_source(tmp_path, "본문")
    content = "제목 문단\n\n| 구간 | 연장 |\n| --- | --- |\n| 깨진값 |  |\n| 또깨짐 |  |"
    weak_alt = "| 구간 | 연장 |\n| --- | --- |\n| 북측호안 320 |  |\n| 남측 |  |"
    strong_alt = "| 구간 | 연장 |\n| --- | --- |\n| 북측 호안 | 320 |\n| 남측 개구부 | 180 |"
    monkeypatch.setattr(fusion, "_alternate_candidate_contents", lambda s, p: [weak_alt, strong_alt])

    fused = fusion.fuse_tables_from_alternate(
        source, content, current_parser="opendataloader-pdf", reference_grids=[_REFERENCE],
    )

    assert "| 북측 호안 | 320 |" in fused  # 풀에서 strong_alt가 선택된다
    assert "깨진값" not in fused


def test_docling_adapter_degrades_gracefully_when_missing(tmp_path: Path) -> None:
    """docling 미설치 환경(CI)에서 어댑터는 빈 결과로 우아하게 저하한다."""
    from parsing_agent.docling_parser import DoclingPdfParserAdapter, docling_available

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    source = DocumentSource(
        path=pdf_path, media_type="application/pdf", size_bytes=0, run_id="docling",
    )

    candidates = DoclingPdfParserAdapter().parse(source, WorkflowConfig())

    if not docling_available():
        assert candidates == []
    # 설치돼 있으면 가짜 PDF라 변환 실패 → 역시 빈 결과 (예외 없이)
    else:
        assert candidates == []


def test_alternate_parser_names_exclude_current() -> None:
    from parsing_agent.fusion import _alternate_parser_names

    assert "opendataloader-pdf" not in _alternate_parser_names("opendataloader-pdf")
    assert "docling-pdf" in _alternate_parser_names("opendataloader-pdf")
    assert set(_alternate_parser_names("docling-pdf")) == {"opendataloader-pdf", "layout-first-pdf"}


def test_visual_repair_targets_skipped_when_grid_judge_says_tables_are_healthy(tmp_path: Path) -> None:
    """표셀 0.849 문서에 비전 2회·이득 0이던 실측의 회귀 가드 — TEDS 게이트."""
    from parsing_agent.repair import identify_repair_targets

    source = _pdf_source(tmp_path, "원문")
    candidate = ParseCandidate(parser_name="opendataloader-pdf", content="후보", format_name="md")
    healthy = _metrics(coverage=1.0, table=0.5, cell=0.85)  # 라벨 점수는 낮지만 그리드는 건강
    healthy.table_issues = ["missing_header"]

    routes = {t.route_name for t in identify_repair_targets(source, candidate, healthy)}
    assert "recover_tables_from_pdf_image" not in routes

    broken = _metrics(coverage=1.0, table=0.5, cell=0.4)
    broken.table_issues = ["missing_header"]
    routes_broken = {t.route_name for t in identify_repair_targets(source, candidate, broken)}
    assert "recover_tables_from_pdf_image" in routes_broken
