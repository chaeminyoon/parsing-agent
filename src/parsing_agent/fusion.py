"""교차 파서 융합 — 단일 파서의 한계를 후보 합집합으로 넘는다.

경쟁 엔진은 파서가 하나라 문서마다 강약이 갈린다. 이 파이프라인은 후보를
여러 개 만들고 같은 잣대로 평가할 수 있으므로, 게이트에 걸린 후보를
'부분 수리'한다:

- 본문 융합: 원문 추출 텍스트에 있는데 후보에 없는 문단을 토큰 커버리지로
  찾아, 마지막으로 커버된 원문 줄의 위치(앵커) 뒤에 삽입한다.
- 표 융합: PDF 괘선 그리드(TEDS-lite)를 심판으로 세워, 표마다 현재 후보와
  대체 파서 후보 중 더 정확한 쪽을 채택한다. 현재 후보가 표를 통째로 잃은
  경우에는 대체 후보의 문맥 앵커 위치에 삽입한다.

두 변환 모두 휴리스틱 수리 라우트로 실행되므로, 재채점에서 점수가 떨어지면
워크플로가 롤백한다 — 융합이 문서를 망치는 경우는 자동 방어된다.
"""
from __future__ import annotations

import re
from pathlib import Path

from parsing_agent.models import DocumentSource
from parsing_agent.table_metrics import (
    Grid,
    extract_pdf_table_grids,
    parse_markdown_table_grid,
    teds_lite,
)

_WORD_RE = re.compile(r"\w+")

# 본문 융합: 이보다 토큰이 적은 줄은 잡음(페이지 번호·구분선)일 가능성이 높다.
_MIN_LINE_TOKENS = 4
# 줄 토큰의 이 비율 이상이 후보에 있으면 '이미 커버됨'으로 본다.
_LINE_COVERED_RATIO = 0.7
# 한 번의 융합에서 삽입하는 줄 수 상한 — 원문 전체를 쏟아붓는 사고 방지.
_MAX_INSERTED_LINES = 200

# 표 융합: 대체 후보가 이 마진 이상 좋아야 교체한다 (동률 스왑 방지).
_TABLE_SWAP_MARGIN = 0.1
_TABLE_SWAP_FLOOR = 0.5
# 현재 후보의 표가 이 점수 미만이면 '대응 표 없음'으로 보고 삽입 경로를 탄다.
_TABLE_MATCH_FLOOR = 0.2

# 대체 파서 매핑 — 현재 후보를 만든 파서를 제외한 나머지 PDF 파서 전부.
# docling은 옵셔널이라 미설치면 어댑터가 빈 결과를 내고 자연히 빠진다.
_PDF_PARSER_POOL = ("opendataloader-pdf", "layout-first-pdf", "docling-pdf")


def _alternate_parser_names(current_parser: str) -> list[str]:
    return [name for name in _PDF_PARSER_POOL if name != (current_parser or "")]

# 대체 파서 실행은 비싸다(Java 서브프로세스/레이아웃 분석) — 경로별로 캐시.
_alternate_content_cache: dict[tuple[str, str], str | None] = {}


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _line_covered(line_tokens: list[str], candidate_tokens: set[str]) -> bool:
    if not line_tokens:
        return True
    hit = sum(1 for token in line_tokens if token in candidate_tokens)
    return hit / len(line_tokens) >= _LINE_COVERED_RATIO


def _normalize_line(line: str) -> str:
    return " ".join(line.split())


def fuse_missing_body_lines(source: DocumentSource, content: str) -> str:
    """원문에 있는데 후보에서 사라진 본문 줄을 앵커 위치에 복구한다."""
    source_text = source.extracted_text
    if not source_text:
        return content

    candidate_tokens = set(_tokens(content))
    candidate_lines = content.splitlines()
    # 정규화된 후보 줄 → 위치 (앵커 삽입 지점 탐색용)
    candidate_index = {_normalize_line(line): i for i, line in enumerate(candidate_lines) if line.strip()}

    insertions: list[tuple[int, str]] = []  # (후보 줄 인덱스 뒤, 원문 줄)
    anchor = -1  # 마지막으로 후보에서 위치를 확인한 원문 줄의 후보 인덱스
    inserted = 0
    for raw_line in source_text.splitlines():
        line = _normalize_line(raw_line)
        if not line:
            continue
        position = candidate_index.get(line)
        if position is not None:
            anchor = max(anchor, position)
            continue
        line_tokens = _tokens(line)
        if len(line_tokens) < _MIN_LINE_TOKENS:
            continue
        if _line_covered(line_tokens, candidate_tokens):
            continue
        if inserted >= _MAX_INSERTED_LINES:
            break
        insertions.append((anchor, line))
        inserted += 1

    if not insertions:
        return content

    result: list[str] = []
    pending: dict[int, list[str]] = {}
    for position, line in insertions:
        pending.setdefault(position, []).append(line)
    # 앵커가 없는(-1) 줄은 문서 맨 앞에 놓는다 — 원문 순서 보존.
    for line in pending.get(-1, []):
        result.extend([line, ""])
    for index, existing in enumerate(candidate_lines):
        result.append(existing)
        for line in pending.get(index, []):
            result.extend(["", line])
    return "\n".join(result)


def _candidate_table_blocks(content: str) -> list[tuple[Grid, int, int]]:
    """후보의 마크다운 표 블록을 (그리드, 시작 줄, 끝 줄)로 나열한다."""
    blocks: list[tuple[Grid, int, int]] = []
    lines = content.splitlines()
    start: int | None = None
    for index in range(len(lines) + 1):
        line = lines[index].strip() if index < len(lines) else ""
        is_row = line.startswith("|") and line.endswith("|")
        if is_row and start is None:
            start = index
        elif not is_row and start is not None:
            grid = parse_markdown_table_grid(lines[start:index])
            if len(grid) >= 2:
                blocks.append((grid, start, index - 1))
            start = None
    return blocks


def _grid_to_markdown(lines: list[str], start: int, end: int) -> list[str]:
    return lines[start : end + 1]


def _single_alternate_content(source: DocumentSource, alternate_name: str) -> str | None:
    cache_key = (str(source.path), alternate_name)
    if cache_key in _alternate_content_cache:
        return _alternate_content_cache[cache_key]
    content: str | None = None
    try:
        from parsing_agent.config import WorkflowConfig
        from parsing_agent.parsers import build_default_parser_registry

        config = WorkflowConfig(
            layout_first_image_captioning_enabled=False,
            langsmith_tracing=False,
        )
        candidates = build_default_parser_registry().get(alternate_name).parse(source, config)
        if candidates:
            content = candidates[0].content
    except Exception:  # noqa: BLE001 - 대체 파서 실패는 융합을 건너뛸 뿐이다
        content = None
    _alternate_content_cache[cache_key] = content
    return content


def _alternate_candidate_contents(source: DocumentSource, current_parser: str) -> list[str]:
    contents: list[str] = []
    for alternate_name in _alternate_parser_names(current_parser):
        content = _single_alternate_content(source, alternate_name)
        if content:
            contents.append(content)
    return contents


def fuse_tables_from_alternate(
    source: DocumentSource,
    content: str,
    *,
    current_parser: str,
    reference_grids: list[Grid] | None = None,
    alternate_content: str | None = None,
) -> str:
    """PDF 괘선 그리드를 심판으로, 표마다 더 정확한 후보의 표로 교체한다.

    ``reference_grids``/``alternate_content``는 테스트 주입용이며, 생략 시
    PDF에서 그리드를 추출하고 대체 파서를 실행한다.
    """
    if reference_grids is None:
        reference_grids = extract_pdf_table_grids(source.path)
    if not reference_grids:
        return content
    if alternate_content is not None:
        alternate_contents = [alternate_content]
    else:
        alternate_contents = _alternate_candidate_contents(source, current_parser)
    if not alternate_contents:
        return content

    current_lines = content.splitlines()
    current_blocks = _candidate_table_blocks(content)

    # 모든 대체 후보의 표 블록을 하나의 풀로 합친다 — docling·layout-first 등
    # 어느 엔진의 표든 그리드 심판 앞에서는 동등한 후보다.
    pooled_blocks: list[tuple[Grid, list[str], int, int]] = []
    for alternate in alternate_contents:
        alternate_lines = alternate.splitlines()
        for grid, start, end in _candidate_table_blocks(alternate):
            pooled_blocks.append((grid, alternate_lines, start, end))
    if not pooled_blocks:
        return content

    current_index = {_normalize_line(line): i for i, line in enumerate(current_lines) if line.strip()}

    replacements: dict[int, tuple[int, list[str]]] = {}  # 시작 줄 → (끝 줄, 새 블록)
    insertions: dict[int, list[list[str]]] = {}  # 후보 줄 인덱스 뒤 → 블록들
    used_alternates: set[int] = set()
    for reference in reference_grids:
        current_score, current_pick = -1.0, None
        for block_index, (grid, _, _) in enumerate(current_blocks):
            score = teds_lite(reference, grid)
            if score > current_score:
                current_score, current_pick = score, block_index
        alternate_scored = [
            (teds_lite(reference, grid), pool_index)
            for pool_index, (grid, _, _, _) in enumerate(pooled_blocks)
            if pool_index not in used_alternates
        ]
        if not alternate_scored:
            continue
        alternate_score, alternate_pick = max(alternate_scored)
        if alternate_score < _TABLE_SWAP_FLOOR:
            continue
        _, alt_lines, alt_start, alt_end = pooled_blocks[alternate_pick]
        new_block = _grid_to_markdown(alt_lines, alt_start, alt_end)
        if current_pick is not None and current_score >= _TABLE_MATCH_FLOOR:
            # 현재 후보에 대응 표가 있다 → 대체가 마진 이상 좋을 때만 교체.
            if alternate_score < current_score + _TABLE_SWAP_MARGIN:
                continue
            _, cur_start, cur_end = current_blocks[current_pick]
            if cur_start in replacements:
                continue
            replacements[cur_start] = (cur_end, new_block)
        else:
            # 현재 후보가 이 표를 통째로 잃었다 — 가장 큰 승리 케이스.
            # 대체 후보에서 표 직전 문맥 줄을 앵커로 삼아 같은 위치에 삽입한다.
            anchor = _insertion_anchor(alt_lines, alt_start, current_index)
            insertions.setdefault(anchor, []).append(new_block)
        used_alternates.add(alternate_pick)

    if not replacements and not insertions:
        return content

    result: list[str] = []
    # 앵커를 못 찾은(-1) 표는 문서 끝에 모아 붙인다 (원문 위치 불명).
    index = 0
    while index < len(current_lines):
        if index in replacements:
            end, new_block = replacements[index]
            result.extend(new_block)
            index = end + 1
            continue
        result.append(current_lines[index])
        for block in insertions.get(index, []):
            result.extend(["", *block, ""])
        index += 1
    for block in insertions.get(-1, []):
        result.extend(["", *block])
    return "\n".join(result)


def _insertion_anchor(
    alternate_lines: list[str],
    table_start: int,
    current_index: dict[str, int],
) -> int:
    """대체 후보에서 표 직전의 문맥 줄과 일치하는 현재 후보 줄을 찾는다."""
    for offset in range(table_start - 1, max(-1, table_start - 8), -1):
        line = _normalize_line(alternate_lines[offset]) if offset >= 0 else ""
        if not line or line.startswith("|"):
            continue
        position = current_index.get(line)
        if position is not None:
            return position
    return -1


def clear_fusion_cache(path: Path | str | None = None) -> None:
    """테스트/장기 실행에서 대체 후보 캐시를 비운다."""
    if path is None:
        _alternate_content_cache.clear()
        return
    key_path = str(path)
    for key in [k for k in _alternate_content_cache if k[0] == key_path]:
        del _alternate_content_cache[key]
