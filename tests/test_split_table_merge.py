"""다중페이지 분할 표 병합 + 오프셋 허용 TEDS의 회귀 테스트.

골든셋 파일럿에서 사람이 지목한 1순위 결함(CO-OPS: TEDS 0.85 vs 사람 표점수 2)의
수리 경로. per-page 그리드 심판의 사각과 병합 수리를 함께 검증한다.
"""

from parsing_agent.table_metrics import teds_lite, teds_lite_best_offset
from parsing_agent.visual_tables import merge_split_tables

_HEADER = "| 구분 | 값 |\n| --- | --- |"


def test_merges_adjacent_tables_with_repeated_header() -> None:
    content = f"머리말\n\n{_HEADER}\n| 가 | 1 |\n\n<!-- page 2 -->\n\n{_HEADER}\n| 나 | 2 |\n\n꼬리말"

    merged = merge_split_tables(content)

    assert merged.count("| --- | --- |") == 1        # 반복 헤더 제거
    assert merged.index("| 가 | 1 |") < merged.index("| 나 | 2 |")
    assert "<!-- page 2 -->" in merged               # 마커는 표 아래 보존
    assert merged.index("| 나 | 2 |") < merged.index("<!-- page 2 -->")


def test_merges_headerless_continuation_and_continued_line() -> None:
    content = f"{_HEADER}\n| 가 | 1 |\n\n(계속)\n\n| 나 | 2 |\n| 다 | 3 |"

    merged = merge_split_tables(content)

    assert "(계속)" not in merged.split("| 나 | 2 |")[0]  # 병합돼 이어짐
    assert merged.count("|") == content.count("|")        # 행 손실 없음


def test_demotes_fake_header_of_page_fragment() -> None:
    """실측(CO-OPS): 단편의 첫 데이터 행이 헤더로 오인된 경우 — 강등 병합."""
    content = f"{_HEADER}\n| 가 | 1 |\n\n| 나 | 2 |\n| --- | --- |\n| 다 | 3 |"

    merged = merge_split_tables(content)

    assert merged.count("| --- | --- |") == 1
    assert merged.index("| 가 | 1 |") < merged.index("| 나 | 2 |") < merged.index("| 다 | 3 |")


def test_does_not_merge_when_text_or_columns_differ() -> None:
    text_between = f"{_HEADER}\n| 가 | 1 |\n\n중간에 본문 설명이 있다.\n\n{_HEADER}\n| 나 | 2 |"
    assert merge_split_tables(text_between) == text_between

    column_mismatch = f"{_HEADER}\n| 가 | 1 |\n\n| 하나 | 둘 | 셋 |\n| 나 | 2 | 3 |"
    assert merge_split_tables(column_mismatch) == column_mismatch


def test_no_merge_returns_original_verbatim() -> None:
    content = "표 없는 본문\n줄 둘\n"
    assert merge_split_tables(content) is content  # 끝 개행까지 그대로


def test_teds_best_offset_scores_merged_table_fairly() -> None:
    """2페이지째 기준 그리드가 병합 표의 중간 행에서 시작해도 만점이 나와야 한다."""
    reference_page2 = [["나", "2"], ["다", "3"]]
    merged_candidate = [["구분", "값"], ["가", "1"], ["나", "2"], ["다", "3"]]

    assert teds_lite(reference_page2, merged_candidate) < 0.6      # 기존: 역감점
    assert teds_lite_best_offset(reference_page2, merged_candidate) == 1.0


def test_post_loop_normalization_merges_split_tables() -> None:
    """채점기가 병합에 중립이므로(오프셋 TEDS) 루프 밖 정규화가 병합을 확정한다."""
    from parsing_agent.repair import apply_table_normalizations

    content = f"{_HEADER}\n| 가 | 1 |\n\n{_HEADER}\n| 나 | 2 |"

    normalized, applied = apply_table_normalizations(content)

    assert "merge_split_multipage_tables" in applied
    assert normalized.count("| --- | --- |") == 1


# ---------------------------------------------------------------------------
# 평문 표 잔재 제거
# ---------------------------------------------------------------------------


def test_removes_plain_dump_of_table_content_above_table() -> None:
    """P3 실측(borderless): 삽입된 표 위에 남은 원문 평문 덤프 제거."""
    from parsing_agent.visual_tables import remove_plain_table_remnants

    content = (
        "관측소 관측일수 결측률 비고 덕적도 361 1.1% 정상 칠발도 349 4.4% 센서 교체\n\n"
        "| 관측소 | 관측일수 | 결측률 | 비고 |\n| --- | --- | --- | --- |\n"
        "| 덕적도 | 361 | 1.1% | 정상 |\n| 칠발도 | 349 | 4.4% | 센서교체 |"
    )

    cleaned = remove_plain_table_remnants(content)

    assert "관측소 관측일수" not in cleaned.splitlines()[0]  # 잔재 제거
    assert "| 덕적도 | 361 | 1.1% | 정상 |" in cleaned      # 표는 보존
    # 공백 어긋남("센서 교체" vs "센서교체")도 압축 매칭으로 잡힌다
    assert "센서 교체" not in cleaned


def test_keeps_captions_and_unrelated_text() -> None:
    from parsing_agent.visual_tables import remove_plain_table_remnants

    content = (
        "표 4.2-1 관측소별 실적 현황 요약본이다\n\n"
        "| 관측소 | 관측일수 |\n| --- | --- |\n| 덕적도 | 361 |\n\n"
        "다음 장에서는 결측 원인을 상세히 분석한다. 관측 체계의 한계와 개선 방향을 다룬다."
    )

    assert remove_plain_table_remnants(content) == content  # 캡션·본문 보존


def test_removes_remnant_below_table_too() -> None:
    from parsing_agent.visual_tables import remove_plain_table_remnants

    content = (
        "| 관측소 | 관측일수 | 결측률 |\n| --- | --- | --- |\n| 덕적도 | 361 | 1.1% |\n"
        "| 칠발도 | 349 | 4.4% |\n\n"
        "관측소 관측일수 결측률 덕적도 361 1.1% 칠발도 349 4.4%"
    )

    cleaned = remove_plain_table_remnants(content)

    assert cleaned.rstrip().endswith("| 칠발도 | 349 | 4.4% |")


def test_no_remnant_returns_original_verbatim() -> None:
    from parsing_agent.visual_tables import remove_plain_table_remnants

    content = "표 없는 본문이다.\n"
    assert remove_plain_table_remnants(content) is content
