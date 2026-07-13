"""셀 단위 표 구조 메트릭 (TEDS-lite).

PDF 자체에서 추출한 표 그리드(PyMuPDF find_tables)를 기준으로, 후보
마크다운의 표를 셀 단위로 비교한다. 진짜 TEDS(트리 편집 거리)와 달리
행 인덱스 정렬 기반의 근사라서 TEDS-lite라고 부른다 — 행 삽입/삭제에는
관대하지 않지만, "라벨은 남았는데 셀 내용·열 구조가 깨진 표"를 열 개수
일관성보다 훨씬 세밀하게 잡는다.

한계: PDF에 괘선이 없는 표는 find_tables가 못 찾으므로 기준 그리드가
없어 None을 반환한다. 이 메트릭은 total_score에 섞지 않고 진단용으로만
리포트에 남긴다 — 골든셋 라벨과의 상관이 확인되면 그때 가중치를 준다.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import fitz

Grid = list[list[str]]

_WS_RE = re.compile(r"\s+")


def _normalize_cell(text: str | None) -> str:
    return _WS_RE.sub(" ", (text or "").strip()).lower()


def extract_pdf_table_grids(pdf_path, *, max_pages: int = 40) -> list[Grid]:
    """PDF에서 괘선 기반 표 그리드를 추출한다. 표가 없으면 빈 목록."""
    grids: list[Grid] = []
    try:
        with fitz.open(pdf_path) as document:
            for page_index in range(min(document.page_count, max_pages)):
                page = document.load_page(page_index)
                try:
                    finder = page.find_tables()
                except Exception:  # noqa: BLE001 - 페이지 하나의 실패는 건너뛴다
                    continue
                for table in getattr(finder, "tables", []):
                    try:
                        rows = table.extract()
                    except Exception:  # noqa: BLE001
                        continue
                    grid = [[_normalize_cell(cell) for cell in row] for row in rows if row]
                    grid = [row for row in grid if any(row)]
                    if len(grid) >= 2 and max(len(row) for row in grid) >= 2:
                        grids.append(grid)
    except (OSError, RuntimeError):
        return []
    return grids


def parse_markdown_table_grid(table_lines: list[str]) -> Grid:
    """마크다운 표 블록을 그리드로 변환한다. 구분선(---) 행은 제외."""
    grid: Grid = []
    for line in table_lines:
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [_normalize_cell(cell) for cell in stripped.split("|")[1:-1]]
        if cells and all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        grid.append(cells)
    return [row for row in grid if any(row)]


def _cell_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def teds_lite(reference: Grid, candidate: Grid) -> float:
    """행 인덱스 정렬 기반 셀 단위 유사도. 0~1.

    행 수·열 수 차이는 빈 셀과의 비교로 자연스럽게 페널티가 된다
    (큰 쪽 그리드의 전체 셀 수로 나누므로 누락·과잉 모두 감점).
    """
    if not reference or not candidate:
        return 0.0
    row_count = max(len(reference), len(candidate))
    total_cells = 0
    score = 0.0
    for row_index in range(row_count):
        ref_row = reference[row_index] if row_index < len(reference) else []
        cand_row = candidate[row_index] if row_index < len(candidate) else []
        column_count = max(len(ref_row), len(cand_row))
        for column_index in range(column_count):
            ref_cell = ref_row[column_index] if column_index < len(ref_row) else ""
            cand_cell = cand_row[column_index] if column_index < len(cand_row) else ""
            total_cells += 1
            score += _cell_similarity(ref_cell, cand_cell)
    if total_cells == 0:
        return 0.0
    return score / total_cells


def teds_lite_best_offset(reference: Grid, candidate: Grid, *, max_offsets: int = 400) -> float:
    """행 오프셋을 허용한 TEDS-lite — 병합된 다중페이지 표를 공정하게 심판한다.

    페이지 경계로 잘렸던 표를 후보가 하나로 병합하면, 2페이지째 기준 그리드는
    병합 표의 중간 행부터 시작한다. 오프셋 0만 보는 기존 방식은 그 병합을
    역감점해서 (사람이 원하는) 수리를 롤백시킨다. 기준 그리드가 후보의 어느
    행에서 시작하든 최고 정렬을 취한다.
    """
    if not reference or not candidate:
        return 0.0
    limit = min(max(len(candidate) - 1, 0), max_offsets)
    best = 0.0
    for offset in range(0, limit + 1):
        window = candidate[offset : offset + len(reference)]
        if not window:
            break
        best = max(best, teds_lite(reference, window))
        if best >= 0.999:
            break
    return best


def _extract_candidate_grids(candidate_text: str) -> list[Grid]:
    grids: list[Grid] = []
    current: list[str] = []
    for line in [*candidate_text.splitlines(), ""]:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current.append(line)
            continue
        if current:
            grid = parse_markdown_table_grid(current)
            if len(grid) >= 2:
                grids.append(grid)
            current = []
    return grids


def calculate_table_cell_similarity(
    reference_grids: list[Grid],
    candidate_text: str,
) -> float | None:
    """PDF 기준 그리드 각각을 후보의 가장 비슷한 표와 매칭해 평균 낸다.

    기준 그리드가 없으면 None (메트릭 미적용). 후보에 표가 하나도 없으면
    0.0 — 원본에 괘선 표가 있는데 후보가 표를 전부 잃은 경우다.
    """
    if not reference_grids:
        return None
    candidate_grids = _extract_candidate_grids(candidate_text)
    if not candidate_grids:
        return 0.0
    total = 0.0
    for reference in reference_grids:
        total += max(teds_lite_best_offset(reference, candidate) for candidate in candidate_grids)
    return total / len(reference_grids)
