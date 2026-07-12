from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import re
from typing import Any

from parsing_agent.filetype import is_pdf
from parsing_agent.interfaces import CandidateRepairer
from parsing_agent.models import DocumentSource, EvaluationIssue, EvaluationMetrics, ParseCandidate, RepairAction
from parsing_agent.visual_repair import (
    VisualRepairTask,
    _normalize_recovered_table_markup,
    _page_table_selector_from_label,
    insert_table_after_anchor,
    replace_page_table_block,
    replace_table_block,
)

# 재구성 표의 셀이 crop 텍스트에 존재해야 하는 최소 비율 (환각 방지 게이트)
_MIN_RECOVERY_GROUNDING = 0.5

_PDF_SECTION_RE = re.compile(r"^\s*(?:제\s*\d+\s*장|\d+(?:\.\d+)*\.|[가-힣A-Z]\.)\s+")
_TABLE_CAPTION_RE = re.compile(r"^\s*(?:표|그림|table|figure)\s*\d", re.IGNORECASE)
_PAGE_MARKER_RE = re.compile(r"^<!-- page (\d+) -->$")
_PAGE_FOOTER_RE = re.compile(r"^-?\s*\d+\s*-?$")
_PAREN_LIST_RE = re.compile(r"^\(\d+\)\s+")


def _excerpt(text: str, limit: int = 120) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


_LIST_RE = re.compile(r"^([-*+]|\d+\.)\s+")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?:<[^>]+>|[^)]+)\)")
_HEADING_RE = re.compile(r"^\s*#+\s*(.+?)\s*$")
_KEY_VALUE_RE = re.compile(r"^\s*(?P<label>[^:：|]{1,80}?)(?:[:：]|\s{2,})(?P<value>.+?)\s*$")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class RepairDirective:
    issue_type: str
    route_name: str
    action_name: str
    description: str
    transform: callable


@dataclass(frozen=True, slots=True)
class RepairTarget:
    target_kind: str
    issue_type: str
    route_name: str
    description: str
    table_label: str | None = None
    page_number: int | None = None
    source_name: str | None = None
    severity: str = "medium"
    confidence: float = 0.5
    source_excerpt: str | None = None
    candidate_excerpt: str | None = None
    bbox: list[float] | None = None
    expected_gain: float = 0.0
    estimated_cost: float = 0.0
    risk_level: str = "low"
    repairability: str | None = None


def _join_lines(lines: list[str], original_text: str) -> str:
    normalized = "\n".join(lines)
    if original_text.endswith("\n"):
        normalized += "\n"
    return normalized


def _normalize_compare_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip()).lower()


def _is_structured_source_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _HEADING_RE.match(stripped):
        return True
    if _LIST_RE.match(stripped):
        return True
    if _PDF_SECTION_RE.match(stripped):
        return True
    if _TABLE_CAPTION_RE.match(stripped):
        return True
    if stripped.endswith(":") and len(stripped) <= 80:
        return True
    return False


def _find_present_candidate_line_indexes(lines: list[str]) -> dict[str, int]:
    indexes: dict[str, int] = {}
    for index, line in enumerate(lines):
        normalized = _normalize_compare_line(line)
        if not normalized or normalized in indexes:
            continue
        indexes[normalized] = index
    return indexes


def _missing_structured_source_lines(source: DocumentSource, text: str) -> list[tuple[str, str | None, str | None]]:
    source_text = source.extracted_text or ""
    if not source_text.strip():
        return []
    source_lines = source_text.splitlines()
    candidate_indexes = _find_present_candidate_line_indexes(text.splitlines())
    missing: list[tuple[str, str | None, str | None]] = []
    for index, raw_line in enumerate(source_lines):
        stripped = raw_line.strip()
        normalized = _normalize_compare_line(stripped)
        if not _is_structured_source_line(stripped) or not normalized or normalized in candidate_indexes:
            continue
        previous_anchor = None
        for back_index in range(index - 1, -1, -1):
            prev_normalized = _normalize_compare_line(source_lines[back_index])
            if prev_normalized and prev_normalized in candidate_indexes:
                previous_anchor = prev_normalized
                break
        next_anchor = None
        for forward_index in range(index + 1, len(source_lines)):
            next_normalized = _normalize_compare_line(source_lines[forward_index])
            if next_normalized and next_normalized in candidate_indexes:
                next_anchor = next_normalized
                break
        if previous_anchor is None and next_anchor is None:
            continue
        missing.append((stripped, previous_anchor, next_anchor))
    return missing


def _missing_table_caption_lines(source: DocumentSource, text: str) -> list[tuple[str, str | None, str | None]]:
    return [
        item
        for item in _missing_structured_source_lines(source, text)
        if _TABLE_CAPTION_RE.match(item[0].strip())
    ]


def _recover_missing_structured_source_lines(source: DocumentSource, text: str) -> str:
    lines = text.splitlines()
    candidate_indexes = _find_present_candidate_line_indexes(lines)
    missing_lines = _missing_structured_source_lines(source, text)
    if not missing_lines:
        return text
    result = list(lines)
    inserted: set[str] = set()
    for missing_line, previous_anchor, next_anchor in missing_lines:
        normalized_missing = _normalize_compare_line(missing_line)
        if normalized_missing in inserted or normalized_missing in candidate_indexes:
            continue
        if next_anchor is not None and next_anchor in candidate_indexes:
            insert_at = candidate_indexes[next_anchor]
        elif previous_anchor is not None and previous_anchor in candidate_indexes:
            insert_at = candidate_indexes[previous_anchor] + 1
        else:
            continue
        result.insert(insert_at, missing_line)
        candidate_indexes = _find_present_candidate_line_indexes(result)
        inserted.add(normalized_missing)
    return _join_lines(result, text)


def _recover_missing_table_caption_lines(source: DocumentSource, text: str) -> str:
    lines = text.splitlines()
    candidate_indexes = _find_present_candidate_line_indexes(lines)
    missing_lines = _missing_table_caption_lines(source, text)
    if not missing_lines:
        return text
    result = list(lines)
    inserted: set[str] = set()
    for missing_line, previous_anchor, next_anchor in missing_lines:
        normalized_missing = _normalize_compare_line(missing_line)
        if normalized_missing in inserted or normalized_missing in candidate_indexes:
            continue
        if next_anchor is not None and next_anchor in candidate_indexes:
            insert_at = candidate_indexes[next_anchor]
        elif previous_anchor is not None and previous_anchor in candidate_indexes:
            insert_at = candidate_indexes[previous_anchor] + 1
        else:
            continue
        result.insert(insert_at, missing_line)
        candidate_indexes = _find_present_candidate_line_indexes(result)
        inserted.add(normalized_missing)
    return _join_lines(result, text)


def _looks_like_recovered_table(markup: str) -> bool:
    stripped = markup.strip()
    if not stripped:
        return False
    if "<table" in stripped.lower() and "</table>" in stripped.lower():
        return True
    table_rows = [
        line
        for line in stripped.splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    ]
    if len(table_rows) < 2:
        return False
    return any(set(cell.strip()) <= {":", "-"} and cell.strip() for cell in table_rows[1].split("|") if cell.strip())


def _recovered_table_passes_sanity(
    *,
    recovered_markdown: str,
    issue_types: tuple[str, ...] | list[str],
) -> bool:
    if not _looks_like_recovered_table(recovered_markdown):
        return False
    if "numeric_token_break" in set(issue_types) and not _NUMBER_RE.search(recovered_markdown):
        return False
    non_empty_cells = [
        cell.strip()
        for line in recovered_markdown.splitlines()
        if "|" in line
        for cell in line.split("|")
        if cell.strip() and set(cell.strip()) - {":", "-"}
    ]
    return len(non_empty_cells) >= 2


def _collapse_blank_lines(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    result: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip():
            blank_run = 0
            result.append(line)
            continue
        if blank_run == 0:
            result.append("")
        blank_run += 1

    return _join_lines(result, text)


def _structured_table_findings(metrics: EvaluationMetrics) -> list[dict[str, Any]]:
    """judge가 반환한 구조화 표 finding을 inspect용 공통 형태로 정규화한다.

    `issue_type`은 필수로 보고, `table_label`, `page_number`는 있으면 함께
    보존한다. inspect 단계에서 이 정보가 RepairTarget으로 전달돼야 route와
    repair가 같은 근거를 공유할 수 있다.
    """
    judge_result = metrics.judge_result
    if judge_result is None:
        return []
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None]] = set()
    for item in judge_result.table_findings:
        if not isinstance(item, dict):
            continue
        issue_type = item.get("issue_type")
        if not isinstance(issue_type, str) or not issue_type.strip():
            continue
        table_label = item.get("table_label")
        normalized_label = table_label.strip() if isinstance(table_label, str) and table_label.strip() else None
        raw_page_number = item.get("page_number")
        page_number: int | None = None
        if raw_page_number is not None:
            try:
                page_number = int(raw_page_number)
            except (TypeError, ValueError):
                page_number = None
        dedupe_key = (issue_type.strip(), normalized_label or "", page_number)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        findings.append(
            {
                "issue_type": issue_type.strip(),
                "table_label": normalized_label,
                "page_number": page_number,
            }
        )
    return findings


def _contains_markdown_images(text: str) -> bool:
    return bool(_IMAGE_RE.search(text))


def _strip_markdown_images(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and _IMAGE_RE.fullmatch(stripped):
            continue
        cleaned = _IMAGE_RE.sub("", line)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).rstrip()
        result.append(cleaned)
    return _join_lines(result, text)


def _remove_repeated_lines(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    result: list[str] = []
    previous_non_empty: str | None = None
    for line in lines:
        normalized = line.strip().lower()
        if normalized and normalized == previous_non_empty:
            continue
        result.append(line)
        if normalized:
            previous_non_empty = normalized

    return _join_lines(result, text)


def _has_duplicate_headings(text: str) -> bool:
    seen: set[str] = set()
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match is None:
            continue
        normalized = match.group(1).strip().lower()
        if normalized in seen:
            return True
        seen.add(normalized)
    return False


def _remove_duplicate_headings(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    result: list[str] = []
    seen: set[str] = set()
    for line in lines:
        match = _HEADING_RE.match(line)
        if match is None:
            result.append(line)
            continue
        normalized = match.group(1).strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(line)
    return _join_lines(result, text)


def _is_structural_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if _LIST_RE.match(stripped):
        return True
    if _PAREN_LIST_RE.match(stripped):
        return True
    if _PDF_SECTION_RE.match(stripped):
        return True
    if _TABLE_CAPTION_RE.match(stripped):
        return True
    if _PAGE_MARKER_RE.match(stripped) or _PAGE_FOOTER_RE.match(stripped):
        return True
    return stripped.count("|") >= 2 and (stripped.startswith("|") or stripped.endswith("|"))


_HANGUL_CHAR_RE = re.compile(r"[가-힣]")
# 한국어 공문서 문장의 종결 어미: 이걸로 끝나는 줄은 문장 경계일 가능성이
# 높아 병합하지 않는다 ("~이다", "~있음", "~수행함", "~예상됨" 등).
_KOREAN_SENTENCE_TERMINAL_RE = re.compile(r"(?:다|요|음|함|됨|임|것)\s*$")
# 헤딩/표 캡션 같은 짧은 명사구 줄을 본문으로 오인 병합하지 않기 위한 최소 길이.
_KOREAN_MERGE_MIN_LINE_LENGTH = 25


def _is_korean_continuation(left: str, right: str) -> bool:
    """왼쪽 줄이 한국어 문장 중간에서 잘렸는지 판정한다.

    한글은 대소문자가 없어 기존 `first_char.islower()` 규칙으로는
    줄바꿈으로 잘린 한국어 문장을 병합하지 못한다. 왼쪽 줄이 한글로
    끝나면서 종결 어미가 아니고, 오른쪽 줄이 한글로 시작하면 이어진
    문장으로 본다.
    """
    if len(left) < _KOREAN_MERGE_MIN_LINE_LENGTH:
        return False
    if not _HANGUL_CHAR_RE.match(right[:1]):
        return False
    if not _HANGUL_CHAR_RE.match(left[-1:]):
        return False
    return not _KOREAN_SENTENCE_TERMINAL_RE.search(left)


def _should_merge_lines(current: str, next_line: str) -> bool:
    left = current.rstrip()
    right = next_line.lstrip()
    if not left or not right:
        return False
    if _is_structural_line(left) or _is_structural_line(right):
        return False
    if left.endswith((".", "!", "?", ":", ";")):
        return False
    if left.endswith("-") and not _PAGE_FOOTER_RE.match(left.strip()):
        return True
    if _is_korean_continuation(left, right):
        return True
    first_char = right[:1]
    return first_char.islower() or first_char.isdigit() or first_char in {'"', "'", "(", "["}


def _has_wrapped_line_sequences(text: str) -> bool:
    lines = text.splitlines()
    return any(_should_merge_lines(lines[index], lines[index + 1]) for index in range(len(lines) - 1))


def _merge_wrapped_lines(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 2:
        return text

    result: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        while index + 1 < len(lines) and _should_merge_lines(current, lines[index + 1]):
            next_line = lines[index + 1]
            if current.rstrip().endswith("-"):
                current = current.rstrip()[:-1] + next_line.lstrip()
            else:
                current = current.rstrip() + " " + next_line.lstrip()
            index += 1
        result.append(current)
        index += 1

    return _join_lines(result, text)


def _extract_strict_table_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """`|`로 시작하고 끝나는 연속 행 구간을 (start, end) 목록으로 반환한다."""
    blocks: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            start = index
            while index < len(lines):
                current = lines[index].strip()
                if not (current.startswith("|") and current.endswith("|")):
                    break
                index += 1
            blocks.append((start, index))
        else:
            index += 1
    return blocks


def _table_row_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().split("|")[1:-1]]


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(not cell or set(cell) <= {"-", ":"} for cell in cells)


def _block_content_cells(lines: list[str]) -> set[str]:
    cells: set[str] = set()
    for line in lines:
        row = _table_row_cells(line)
        if _is_separator_row(row):
            continue
        for cell in row:
            compact = re.sub(r"\s+", "", cell)
            if len(compact) >= 2:
                cells.add(compact)
    return cells


def _find_duplicate_table_block(lines: list[str]) -> tuple[int, int] | None:
    """다른 표 블록에 내용이 이미 포함된 잔재 블록의 범위를 찾는다.

    수리가 표를 교체·삽입할 때 깨진 원본 조각이 표 블록으로 인식되지
    않으면 그대로 남아 같은 내용이 두 번 등장한다 (골든 라벨 1·2호에서
    반복 확인). 셀 단위 포함 비율로 판정하며, 작은 쪽 블록을 잔재로 본다.
    """
    blocks = _extract_strict_table_blocks(lines)
    for i in range(len(blocks)):
        for j in range(len(blocks)):
            if i == j:
                continue
            small_start, small_end = blocks[i]
            large_start, large_end = blocks[j]
            if (small_end - small_start) > (large_end - large_start):
                continue
            small_cells = _block_content_cells(lines[small_start:small_end])
            if len(small_cells) < 3:
                continue
            large_text = re.sub(r"\s+", "", "".join(lines[large_start:large_end]))
            matched = sum(1 for cell in small_cells if cell in large_text)
            if matched / len(small_cells) >= 0.7:
                return blocks[i]
    return None


def _has_duplicate_table_blocks(text: str) -> bool:
    return _find_duplicate_table_block(text.splitlines()) is not None


def _remove_duplicate_table_blocks(text: str) -> str:
    lines = text.splitlines()
    for _ in range(10):
        duplicate = _find_duplicate_table_block(lines)
        if duplicate is None:
            break
        start, end = duplicate
        lines = lines[:start] + lines[end:]
    return _join_lines(lines, text)


def _has_merged_leading_gaps(text: str) -> bool:
    lines = text.splitlines()
    for start, end in _extract_strict_table_blocks(lines):
        previous: list[str] | None = None
        for index in range(start, end):
            cells = _table_row_cells(lines[index])
            if _is_separator_row(cells):
                continue
            if previous is not None and any(cells):
                for column in (0, 1):
                    if (
                        column < len(cells)
                        and column < len(previous)
                        and not cells[column]
                        and previous[column]
                    ):
                        return True
            previous = cells
    return False


def _fill_merged_leading_cells(text: str) -> str:
    """세로 병합이 풀리며 비어버린 선두 분류 열(0·1열)에 위 행의 값을 복제한다.

    "통합표는 개별 행에 값이 반영되어야 한다"는 라벨 피드백 대응.
    뒤쪽 열(비고 등)은 원래 비어 있는 경우가 많아 잘못 채울 위험이 크므로
    선두 두 열만 다룬다 — 나머지 병합 복원은 비전 수리(HTML rowspan 확장)가
    담당한다.
    """
    lines = text.splitlines()
    result = list(lines)
    for start, end in _extract_strict_table_blocks(lines):
        previous: list[str] | None = None
        for index in range(start, end):
            cells = _table_row_cells(lines[index])
            if _is_separator_row(cells):
                continue
            if previous is not None and any(cells):
                changed = False
                for column in (0, 1):
                    if (
                        column < len(cells)
                        and column < len(previous)
                        and not cells[column]
                        and previous[column]
                    ):
                        cells[column] = previous[column]
                        changed = True
                if changed:
                    result[index] = "| " + " | ".join(cells) + " |"
            previous = cells
    return _join_lines(result, text)


def _fused_table_split_points(lines: list[str], start: int, end: int) -> list[int]:
    """한 블록 안에서 별개 표가 시작되는 지점(두 번째 헤더/캡션 행)을 찾는다."""
    points: list[int] = []
    separator_seen = False
    for index in range(start, end):
        cells = _table_row_cells(lines[index])
        if _is_separator_row(cells):
            # 블록 끝의 떠돌이 구분선은 헤더가 아니므로 분할하지 않는다
            if separator_seen and index - 1 > start and index + 1 < end and (index - 1) not in points:
                # 블록 중간의 두 번째 구분선: 바로 위 행이 새 표의 헤더다
                points.append(index - 1)
            separator_seen = True
            continue
        non_empty = [cell for cell in cells if cell]
        if (
            len(non_empty) == 1
            and _TABLE_CAPTION_RE.match(non_empty[0].lstrip("<([").strip())
            and index > start
        ):
            # 표 안에 캡션 행이 끼어 있으면 그 지점부터 별개 표다 ("<표 2.4-2>" 꺾쇠 표기 포함)
            points.append(index)
    return points


def _has_fused_table_blocks(text: str) -> bool:
    lines = text.splitlines()
    return any(
        _fused_table_split_points(lines, start, end)
        for start, end in _extract_strict_table_blocks(lines)
    )


def _split_fused_tables(text: str) -> str:
    """파서가 하나로 붙여버린 통합표를 별개 표들로 분리한다.

    분리 지점: 블록 중간의 두 번째 헤더(구분선 직전 행), 또는 표 안에
    끼어든 캡션 행. 캡션 행은 파이프를 벗겨 일반 텍스트 줄로 꺼낸다.
    """
    lines = text.splitlines()
    result: list[str] = []
    consumed = 0
    for start, end in _extract_strict_table_blocks(lines):
        result.extend(lines[consumed:start])
        points = _fused_table_split_points(lines, start, end)
        if not points:
            result.extend(lines[start:end])
        else:
            cursor = start
            for point in points:
                result.extend(lines[cursor:point])
                cells = _table_row_cells(lines[point])
                non_empty = [cell for cell in cells if cell]
                if len(non_empty) == 1 and _TABLE_CAPTION_RE.match(non_empty[0].lstrip("<([").strip()):
                    result.extend(["", non_empty[0], ""])
                    cursor = point + 1
                else:
                    result.append("")
                    cursor = point
            result.extend(lines[cursor:end])
        consumed = end
    result.extend(lines[consumed:])
    return _join_lines(result, text)


def apply_table_normalizations(text: str) -> tuple[str, list[str]]:
    """채점 루프 밖에서 도는 후처리 표 정규화.

    골든 라벨이 결함으로 확정한 세 가지를 다룬다: 수리 잔재 중복 제거,
    병합 해제로 빈 선두 열 값 복제, 통합표 분리. 셀 포함 검사·값 복사·
    경계 삽입만 하므로 구조상 내용 무손실이고, 점수 루프에 넣지 않는
    이유는 현재 결정적 채점기가 이 정규화들을 감점하기 때문이다
    (라벨 기반 표 메트릭이 중복 잔재를 보상하는 오판 — 골든 라벨로 확인).
    적용된 변환 이름 목록을 함께 반환한다.
    """
    applied: list[str] = []
    if _has_duplicate_table_blocks(text):
        text = _remove_duplicate_table_blocks(text)
        applied.append("remove_duplicate_table_blocks")
    if _has_fused_table_blocks(text):
        text = _split_fused_tables(text)
        applied.append("split_fused_tables")
    if _has_merged_leading_gaps(text):
        text = _fill_merged_leading_cells(text)
        applied.append("fill_merged_leading_cells")
    return text, applied


def _looks_like_corrupted_table_line(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    if _IMAGE_RE.search(stripped):
        return True
    return stripped.count("|") >= 2 and not (stripped.startswith("|") or stripped.endswith("|"))


def _has_table_layout_noise(text: str) -> bool:
    return any(_looks_like_corrupted_table_line(line) for line in text.splitlines())


def _normalize_table_layout(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    result: list[str] = []
    for line in lines:
        if "|" not in line:
            result.append(line)
            continue
        cleaned = _IMAGE_RE.sub("", line)
        if cleaned.count("|") >= 2:
            cells = [cell.strip() for cell in cleaned.split("|")]
            if cells and not cells[0]:
                cells = cells[1:]
            if cells and not cells[-1]:
                cells = cells[:-1]
            normalized_cells = [cell if cell else " " for cell in cells]
            cleaned = "| " + " | ".join(normalized_cells) + " |"
        result.append(cleaned.rstrip())
    return _join_lines(result, text)


def _looks_like_table_text_row(line: str) -> bool:
    if "|" in line:
        return False
    match = _KEY_VALUE_RE.match(line)
    if match is None:
        return False
    label = match.group("label").strip()
    value = match.group("value").strip()
    if not label or not value:
        return False
    if len(value.split()) > 12 and not re.search(r"\d", value):
        return False
    return True


def _has_table_text_blocks(text: str) -> bool:
    lines = text.splitlines()
    run_length = 0
    for line in lines:
        if _looks_like_table_text_row(line):
            run_length += 1
            if run_length >= 2:
                return True
            continue
        run_length = 0
    return False


def _reconstruct_table_text_blocks(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 2:
        return text

    result: list[str] = []
    index = 0
    while index < len(lines):
        if not _looks_like_table_text_row(lines[index]):
            result.append(lines[index])
            index += 1
            continue

        block: list[tuple[str, str]] = []
        while index < len(lines) and _looks_like_table_text_row(lines[index]):
            match = _KEY_VALUE_RE.match(lines[index])
            assert match is not None
            block.append((match.group("label").strip(), match.group("value").strip()))
            index += 1

        if len(block) < 2:
            label, value = block[0]
            result.append(f"{label}: {value}")
            continue

        result.append("| 항목 | 값 |")
        result.append("| --- | --- |")
        for label, value in block:
            result.append(f"| {label} | {value} |")

    return _join_lines(result, text)


def _is_repeated_boundary_candidate(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _is_structural_line(stripped):
        return False
    if len(stripped) > 80:
        return False
    return len(stripped.split()) <= 8


def _remove_repeated_boundary_lines(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 3:
        return text

    boundary_counts: dict[str, int] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not _is_repeated_boundary_candidate(stripped):
            continue
        previous_blank = index == 0 or not lines[index - 1].strip()
        next_blank = index == len(lines) - 1 or not lines[index + 1].strip()
        if previous_blank or next_blank:
            normalized = stripped.lower()
            boundary_counts[normalized] = boundary_counts.get(normalized, 0) + 1

    repeated_boundaries = {line for line, count in boundary_counts.items() if count >= 2}
    if not repeated_boundaries:
        return text

    result: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        previous_blank = index == 0 or not lines[index - 1].strip()
        next_blank = index == len(lines) - 1 or not lines[index + 1].strip()
        if stripped.lower() in repeated_boundaries and (previous_blank or next_blank):
            continue
        result.append(line)

    normalized = _collapse_blank_lines(_join_lines(result, text)).strip("\n")
    if not normalized:
        return normalized
    if text.endswith("\n"):
        normalized += "\n"
    return normalized


def _page_segments(lines: list[str]) -> list[tuple[int, int]]:
    marker_indexes = [index for index, line in enumerate(lines) if _PAGE_MARKER_RE.match(line.strip())]
    if not marker_indexes:
        return []
    segments: list[tuple[int, int]] = []
    for index, marker_index in enumerate(marker_indexes):
        next_marker = marker_indexes[index + 1] if index + 1 < len(marker_indexes) else len(lines)
        segments.append((marker_index, next_marker))
    return segments


def _repeated_pdf_boundary_lines(text: str) -> set[str]:
    lines = text.splitlines()
    repeated: dict[str, int] = {}
    for start, end in _page_segments(lines):
        page_lines = lines[start + 1 : end]
        non_empty = [line.strip() for line in page_lines if line.strip()]
        if len(non_empty) < 2:
            continue
        first_line = non_empty[0]
        last_line = non_empty[-1]
        for candidate in (first_line, last_line):
            if len(candidate) > 40 or _is_structural_line(candidate):
                continue
            normalized = _normalize_compare_line(candidate)
            repeated[normalized] = repeated.get(normalized, 0) + 1
    return {line for line, count in repeated.items() if count >= 2}


def _remove_repeated_pdf_header_footer_lines(text: str) -> str:
    repeated_lines = _repeated_pdf_boundary_lines(text)
    if not repeated_lines:
        return text
    lines = text.splitlines()
    result: list[str] = []
    for start, end in _page_segments(lines):
        result.append(lines[start])
        page_lines = lines[start + 1 : end]
        non_empty_indexes = [index for index, line in enumerate(page_lines) if line.strip()]
        first_non_empty = non_empty_indexes[0] if non_empty_indexes else None
        last_non_empty = non_empty_indexes[-1] if non_empty_indexes else None
        for index, line in enumerate(page_lines):
            normalized = _normalize_compare_line(line)
            if normalized in repeated_lines and index in {first_non_empty, last_non_empty}:
                continue
            result.append(line)
    return _join_lines(result, text)


def _looks_like_pdf_boundary_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(_PDF_SECTION_RE.match(stripped) or _LIST_RE.match(stripped) or _TABLE_CAPTION_RE.match(stripped))


def _restore_pdf_boundaries(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    result: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if _looks_like_pdf_boundary_heading(stripped):
            previous = result[-1].strip() if result else ""
            if previous and not _PAGE_MARKER_RE.match(previous):
                result.append("")
        result.append(line)
        if not _looks_like_pdf_boundary_heading(stripped):
            continue
        next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if next_line and not _looks_like_pdf_boundary_heading(next_line):
            result.append("")
    return _collapse_blank_lines(_join_lines(result, text))


def _is_pdf_candidate(source: DocumentSource, candidate: ParseCandidate) -> bool:
    return is_pdf(source.media_type, candidate.source_path or source.path)


def _should_defer_table_rewrite(source: DocumentSource, candidate: ParseCandidate, metrics: EvaluationMetrics) -> bool:
    """PDF 표 이슈가 명확하면 heuristic 표 재작성보다 visual repair를 우선한다.

    PDF candidate에서 table issue가 이미 잡혔다면, 텍스트를 억지로
    다시 쓰는 것보다 원본 PDF 이미지를 다시 보는 경로가 더 적절하다고
    판단한다.
    """
    return _is_pdf_candidate(source, candidate) and bool(metrics.table_issues)


def _classify_repair_directives(
    source: DocumentSource,
    candidate: ParseCandidate,
    metrics: EvaluationMetrics,
) -> list[RepairDirective]:
    """metric과 콘텐츠 패턴을 heuristic repair 지시문으로 바꾼다.

    각 directive는 `issue_type`, `route_name`, 설명, 실제 변환 함수를 함께
    가진다. 기준은 의도적으로 단순하며, 대부분 `0.75` 미만 점수이거나
    직접적인 텍스트 패턴이 보일 때 발동한다.
    """
    content = candidate.content
    directives: list[RepairDirective] = []
    seen_issue_types: set[str] = set()
    defer_table_rewrite = _should_defer_table_rewrite(source, candidate, metrics)

    def add_directive(
        issue_type: str,
        route_name: str,
        action_name: str,
        description: str,
        transform,
    ) -> None:
        if issue_type in seen_issue_types:
            return
        seen_issue_types.add(issue_type)
        directives.append(
            RepairDirective(
                issue_type=issue_type,
                route_name=route_name,
                action_name=action_name,
                description=description,
                transform=transform,
            )
        )

    if _contains_markdown_images(content):
        add_directive(
            "image_link_noise",
            "remove_image_noise",
            "strip_markdown_images",
            "Remove markdown image links that add rendering noise or leak into parsed cells.",
            _strip_markdown_images,
        )
    if _missing_table_caption_lines(source, content):
        add_directive(
            "missing_table_caption",
            "recover_missing_table_captions",
            "recover_missing_table_caption_lines",
            "Recover missing source table or figure captions before table-specific repair runs.",
            lambda current: _recover_missing_table_caption_lines(source, current),
        )
    if metrics.text_coverage < 0.72 and _missing_structured_source_lines(source, content):
        add_directive(
            "text_coverage_missing_lines",
            "recover_missing_source_lines",
            "recover_missing_structured_source_lines",
            "Recover short structured source lines that disappeared from the parsed result.",
            lambda current: _recover_missing_structured_source_lines(source, current),
        )
    if not defer_table_rewrite and (metrics.table_preservation < 0.75 or _has_table_layout_noise(content)):
        add_directive(
            "table_layout_noise",
            "normalize_table_layout",
            "normalize_table_layout",
            "Normalize table rows and strip inline image fragments from table cells.",
            _normalize_table_layout,
        )
    if not defer_table_rewrite and metrics.table_preservation < 0.75 and _has_table_text_blocks(content):
        add_directive(
            "table_text_block_recovery",
            "reconstruct_table_blocks",
            "reconstruct_table_text_blocks",
            "Reconstruct repeated key/value text rows into a conservative two-column markdown table.",
            _reconstruct_table_text_blocks,
        )
    if metrics.structure_retention < 0.75 or _has_duplicate_headings(content):
        add_directive(
            "structure_heading_noise",
            "deduplicate_headings",
            "remove_duplicate_headings",
            "Remove duplicated markdown headings that distort the document hierarchy.",
            _remove_duplicate_headings,
        )
    if metrics.empty_block_penalty > 0:
        add_directive(
            "blank_line_noise",
            "collapse_blank_runs",
            "collapse_blank_lines",
            "Collapse oversized blank-line runs to a single blank line.",
            _collapse_blank_lines,
        )
    if metrics.repetition_penalty > 0:
        if _is_pdf_candidate(source, candidate) and _repeated_pdf_boundary_lines(content):
            add_directive(
                "pdf_header_footer_noise",
                "deduplicate_pdf_headers",
                "remove_repeated_pdf_header_footer_lines",
                "Remove repeated short header and footer lines that recur across PDF pages.",
                _remove_repeated_pdf_header_footer_lines,
            )
        add_directive(
            "boundary_repetition_noise",
            "deduplicate_boundaries",
            "remove_repeated_boundary_lines",
            "Remove repeated short boundary lines that look like headers or footers.",
            _remove_repeated_boundary_lines,
        )
        add_directive(
            "line_repetition_noise",
            "deduplicate_lines",
            "remove_repeated_lines",
            "Remove consecutive repeated non-empty lines.",
            _remove_repeated_lines,
        )
    if _has_wrapped_line_sequences(content):
        add_directive(
            "wrapped_line_noise",
            "merge_wrapped_lines",
            "merge_wrapped_lines",
            "Merge neighboring wrapped lines inside plain-text paragraphs.",
            _merge_wrapped_lines,
        )
    if _is_pdf_candidate(source, candidate) and metrics.structure_retention < 0.75:
        restored = _restore_pdf_boundaries(content)
        if restored != content:
            add_directive(
                "pdf_block_boundary_noise",
                "restore_pdf_boundaries",
                "restore_pdf_boundaries",
                "Restore blank-line boundaries around PDF headings, lists, and table captions.",
                _restore_pdf_boundaries,
            )
    return directives


def _repair_target_kind(issue_type: str) -> str:
    if issue_type.startswith("table_"):
        return "table"
    if issue_type.startswith("structure_") or issue_type.startswith("wrapped_"):
        return "text"
    return "document"


def _metric_issue_by_type(metrics: EvaluationMetrics) -> dict[str, EvaluationIssue]:
    by_type: dict[str, EvaluationIssue] = {}
    for issue in metrics.issues:
        by_type.setdefault(issue.issue_type, issue)
    return by_type


def _target_severity(issue: EvaluationIssue | None, default: str = "medium") -> str:
    if issue is None:
        return default
    return issue.severity


def _target_confidence(issue: EvaluationIssue | None, default: float = 0.55) -> float:
    if issue is None:
        return default
    return max(0.0, min(1.0, issue.confidence))


def _target_expected_gain(
    *,
    issue: EvaluationIssue | None,
    route_name: str,
    metrics: EvaluationMetrics,
) -> float:
    if issue is not None:
        metric_scores = {
            "text_coverage": metrics.text_coverage,
            "normalized_similarity": metrics.normalized_similarity,
            "structure_retention": metrics.structure_retention,
            "table_preservation": metrics.table_preservation,
            "empty_block_penalty": 1.0 - metrics.empty_block_penalty,
            "repetition_penalty": 1.0 - metrics.repetition_penalty,
        }
        metric_score = metric_scores.get(issue.metric_name)
        if metric_score is not None:
            return round(max(0.0, min(0.35, (1.0 - metric_score) * issue.confidence)), 4)
    if route_name == "recover_tables_from_pdf_image":
        return round(max(0.05, min(0.25, (1.0 - metrics.table_preservation) * 0.6)), 4)
    return 0.05


def _target_cost_and_risk(route_name: str, issue: EvaluationIssue | None) -> tuple[float, str]:
    if route_name == "recover_tables_from_pdf_image":
        confidence = 0.5 if issue is None else issue.confidence
        risk = "medium" if confidence >= 0.6 else "high"
        return 1.0, risk
    return 0.0, "low"


def _repairability_for_route(route_name: str, issue: EvaluationIssue | None = None) -> str:
    if issue is not None and issue.repairability:
        return issue.repairability
    if route_name == "recover_tables_from_pdf_image":
        return "visual"
    return "heuristic"


def _target_from_directive(
    *,
    directive: RepairDirective,
    issue: EvaluationIssue | None,
    metrics: EvaluationMetrics,
) -> RepairTarget:
    estimated_cost, risk_level = _target_cost_and_risk(directive.route_name, issue)
    return RepairTarget(
        target_kind=_repair_target_kind(directive.issue_type),
        issue_type=directive.issue_type,
        route_name=directive.route_name,
        description=directive.description,
        source_name="heuristic_directive",
        severity=_target_severity(issue),
        confidence=_target_confidence(issue),
        source_excerpt=None if issue is None else issue.source_excerpt,
        candidate_excerpt=None if issue is None else issue.candidate_excerpt,
        bbox=None if issue is None else issue.bbox,
        expected_gain=_target_expected_gain(
            issue=issue,
            route_name=directive.route_name,
            metrics=metrics,
        ),
        estimated_cost=estimated_cost,
        risk_level=risk_level,
        repairability=_repairability_for_route(directive.route_name, issue),
    )


def _target_from_table_finding(
    *,
    issue_type: str,
    route_name: str,
    description: str,
    source_name: str,
    metrics: EvaluationMetrics,
    issue: EvaluationIssue | None = None,
    table_label: str | None = None,
    page_number: int | None = None,
    bbox: list[float] | None = None,
) -> RepairTarget:
    estimated_cost, risk_level = _target_cost_and_risk(route_name, issue)
    return RepairTarget(
        target_kind="table",
        issue_type=issue_type,
        route_name=route_name,
        description=description,
        table_label=table_label if table_label is not None else (None if issue is None else issue.table_label),
        page_number=page_number if page_number is not None else (None if issue is None else issue.page_number),
        source_name=source_name,
        severity=_target_severity(issue),
        confidence=_target_confidence(issue, default=0.65),
        source_excerpt=None if issue is None else issue.source_excerpt,
        candidate_excerpt=None if issue is None else issue.candidate_excerpt,
        bbox=bbox if bbox is not None else (None if issue is None else issue.bbox),
        expected_gain=_target_expected_gain(
            issue=issue,
            route_name=route_name,
            metrics=metrics,
        ),
        estimated_cost=estimated_cost,
        risk_level=risk_level,
        repairability=_repairability_for_route(route_name, issue),
    )


def identify_repair_targets(
    source: DocumentSource,
    candidate: ParseCandidate,
    metrics: EvaluationMetrics,
) -> list[RepairTarget]:
    """inspect 단계의 최종 출력인 repair target 목록을 만든다.

    heuristic directive는 그대로 repair target이 되고, evaluation 단계에서
    잡힌 표 이슈는 항상 visual repair target으로 변환된다. route 노드는
    이 목록을 보고 heuristic 수리와 이미지 기반 수리를 나눈다.
    """
    issues_by_type = _metric_issue_by_type(metrics)
    targets = [
        _target_from_directive(
            directive=directive,
            issue=issues_by_type.get(directive.issue_type),
            metrics=metrics,
        )
        for directive in _classify_repair_directives(source, candidate, metrics)
    ]
    covered_table_issue_types: set[str] = set()
    for finding in _structured_table_findings(metrics):
        issue_type = str(finding["issue_type"])
        targets.append(
            _target_from_table_finding(
                issue_type=issue_type,
                route_name="recover_tables_from_pdf_image",
                description=f"Recover broken table regions affected by {issue_type}.",
                table_label=finding.get("table_label"),
                page_number=finding.get("page_number"),
                source_name="judge_table_finding",
                metrics=metrics,
                issue=issues_by_type.get(issue_type),
            )
        )
        covered_table_issue_types.add(issue_type)
    table_regions = candidate.metadata.get("table_regions")
    if (
        not covered_table_issue_types
        and isinstance(table_regions, list)
        and metrics.table_issues
    ):
        primary_issue_type = metrics.table_issues[0]
        seen_regions: set[tuple[str | None, int | None]] = set()
        for region in table_regions:
            if not isinstance(region, dict):
                continue
            table_label = region.get("label") or region.get("table_label")
            if isinstance(table_label, str):
                table_label = table_label.strip() or None
            else:
                table_label = None
            raw_page_number = region.get("page_number", region.get("page"))
            try:
                page_number = int(raw_page_number) if raw_page_number is not None else None
            except (TypeError, ValueError):
                page_number = None
            region_key = (table_label, page_number)
            if region_key in seen_regions or (table_label is None and page_number is None):
                continue
            seen_regions.add(region_key)
            bbox = region.get("bbox")
            targets.append(
                _target_from_table_finding(
                    issue_type=primary_issue_type,
                    route_name="recover_tables_from_pdf_image",
                    description=f"Recover broken table regions affected by {primary_issue_type}.",
                    table_label=table_label,
                    page_number=page_number,
                    source_name="parser_table_region",
                    metrics=metrics,
                    issue=issues_by_type.get(primary_issue_type),
                    bbox=bbox if isinstance(bbox, list) else None,
                )
            )
            covered_table_issue_types.add(primary_issue_type)
    for issue_type in metrics.table_issues:
        if issue_type in covered_table_issue_types:
            continue
        targets.append(
            _target_from_table_finding(
                issue_type=issue_type,
                route_name="recover_tables_from_pdf_image",
                description=f"Recover broken table regions affected by {issue_type}.",
                source_name="metrics_table_issue",
                metrics=metrics,
                issue=issues_by_type.get(issue_type),
            )
        )
    if metrics.text_coverage < 0.72:
        # 구조 라인 복구(heuristic)로도 커버리지가 낮으면 본문 내용 자체가
        # 누락된 것이다. 휴리스틱 transform으로는 복원 불가능하므로 LLM
        # 수리 전용 target을 만든다 (repairability="llm").
        coverage_issue = issues_by_type.get("text_coverage_missing_lines")
        targets.append(
            RepairTarget(
                target_kind="text",
                issue_type="text_coverage_missing_content",
                route_name="llm_restore_missing_content",
                description="Candidate is missing source body content that heuristics cannot restore.",
                source_name="coverage_metric",
                severity="high" if metrics.text_coverage < 0.55 else "medium",
                confidence=round(min(1.0, max(0.0, 1.0 - metrics.text_coverage)), 4),
                source_excerpt=None if coverage_issue is None else coverage_issue.source_excerpt,
                candidate_excerpt=None if coverage_issue is None else coverage_issue.candidate_excerpt,
                expected_gain=round(max(0.05, 0.75 - metrics.text_coverage), 4),
                estimated_cost=0.5,
                risk_level="medium",
                repairability="llm",
            )
        )
    return targets


class HeuristicRepairer(CandidateRepairer):
    def __init__(self, *, visual_table_recoverer=None, text_repairer=None) -> None:
        self._visual_table_recoverer = visual_table_recoverer
        self._text_repairer = text_repairer

    def repair_llm_targets(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
        metrics: EvaluationMetrics,
        targets: list[RepairTarget],
        max_targets: int = 3,
    ) -> tuple[ParseCandidate, list[RepairAction]]:
        """이슈 단위 LLM 텍스트 수리를 최대 `max_targets`개까지 적용한다.

        text repairer가 없거나 모든 이슈가 스킵되면 candidate를 그대로
        돌려준다. 개별 이슈 실패는 다음 이슈로 넘어간다.
        """
        del metrics
        if self._text_repairer is None or not targets:
            return candidate, []
        content = candidate.content
        actions: list[RepairAction] = []
        for target in targets[: max(0, max_targets)]:
            try:
                outcome = self._text_repairer.repair_target(source, content, target)
            except Exception:  # noqa: BLE001 - 이슈 하나의 실패가 나머지를 막으면 안 된다
                continue
            if outcome is None:
                continue
            content = outcome.content
            actions.append(outcome.action)
        if not actions:
            return candidate, []
        repaired = replace(
            candidate,
            content=content,
            repaired_from=candidate.repaired_from or candidate.parser_name,
            metadata={
                **candidate.metadata,
                "repair_actions": [
                    *candidate.metadata.get("repair_actions", []),
                    *[action.action_name for action in actions],
                ],
            },
        )
        return repaired, actions

    def repair_heuristics(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
        metrics: EvaluationMetrics,
        targets: list[RepairTarget] | None = None,
    ) -> tuple[ParseCandidate, list[RepairAction]]:
        """route에서 허용한 heuristic transform만 적용한다.

        `targets`가 있으면 `(issue_type, route_name)` 기준으로 directive를
        필터링한다. 즉 모든 heuristic을 한 번에 적용하는 것이 아니라,
        route 노드가 고른 전략만 실행한다.
        """
        updated = candidate.content
        actions: list[RepairAction] = []
        directives = _classify_repair_directives(source, candidate, metrics)
        allowed_routes = None
        if targets is not None:
            allowed_routes = {(target.issue_type, target.route_name) for target in targets}
        for directive in directives:
            if allowed_routes is not None and (directive.issue_type, directive.route_name) not in allowed_routes:
                continue
            transformed = directive.transform(updated)
            if transformed == updated:
                continue
            actions.append(
                RepairAction(
                    action_name=directive.action_name,
                    description=directive.description,
                    before_excerpt=_excerpt(updated),
                    after_excerpt=_excerpt(transformed),
                    issue_type=directive.issue_type,
                    route_name=directive.route_name,
                )
            )
            updated = transformed

        if not actions:
            return candidate, []

        repaired_from = candidate.repaired_from or candidate.parser_name
        repaired_candidate = replace(
            candidate,
            content=updated,
            repaired_from=repaired_from,
            metadata={
                **candidate.metadata,
                "repair_actions": [action.action_name for action in actions],
                "repair_issue_types": [action.issue_type for action in actions if action.issue_type is not None],
                "repair_routes": [action.route_name for action in actions if action.route_name is not None],
            },
        )
        return repaired_candidate, actions

    def plan_chunk_repairs(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
        metrics: EvaluationMetrics,
        max_tasks: int,
        targets: list[RepairTarget] | None = None,
    ) -> list[VisualRepairTask]:
        """어떤 표 영역을 visual repair 할지 task 목록만 계획한다.

        이 단계에서는 내용을 수정하지 않는다. visual recoverer가 없거나
        table issue가 없으면 빈 목록을 반환한다.
        """
        if self._visual_table_recoverer is None:
            return []
        if not metrics.table_issues:
            return []
        try:
            planned = self._visual_table_recoverer.plan_tasks(
                source,
                candidate.content,
                metrics,
                candidate_metadata=candidate.metadata,
                max_tasks=max_tasks,
                repair_targets=targets,
            )
        except TypeError:
            planned = self._visual_table_recoverer.plan_tasks(
                source,
                candidate.content,
                metrics,
                candidate_metadata=candidate.metadata,
                max_tasks=max_tasks,
            )
        return planned[:max_tasks]

    def apply_chunk_repair(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
        task: VisualRepairTask,
        rejection_sink: list[dict[str, object]] | None = None,
    ) -> tuple[ParseCandidate, RepairAction] | None:
        """계획된 visual table repair task 하나를 실제로 수행한다.

        visual recoverer가 빈 결과를 주거나 confidence가 `0.45` 미만이면
        버린다. 통과하면 candidate 안의 해당 표 블록을 교체하고, 교체할
        블록이 없으면 라벨/페이지 앵커 뒤 삽입으로 폴백한다. 거부 시
        `rejection_sink`에 사유를 기록한다.
        """

        def _reject(reason: str, **details: object) -> None:
            if rejection_sink is not None:
                rejection_sink.append(
                    {
                        "task_id": task.task_id,
                        "table_label": task.table_label,
                        "page_number": task.page_number,
                        "issue_types": list(task.issue_types),
                        "reason": reason,
                        **details,
                    }
                )

        if self._visual_table_recoverer is None:
            _reject("no_visual_recoverer")
            return None
        if source.page_count is not None and (task.page_number < 1 or task.page_number > source.page_count):
            _reject("page_out_of_range")
            return None
        try:
            recovery = self._visual_table_recoverer.recover_task(source, candidate.content, task)
        except Exception as exc:  # noqa: BLE001 - 사유 기록 후 해당 task만 버린다
            _reject("recover_exception", error=f"{type(exc).__name__}: {exc}")
            return None
        if recovery is None or not recovery.markdown.strip():
            _reject("empty_recovery")
            return None
        if recovery.confidence < 0.45:
            _reject("low_confidence", confidence=round(recovery.confidence, 4))
            return None
        if recovery.grounding is not None and recovery.grounding < _MIN_RECOVERY_GROUNDING:
            # 재구성된 셀 내용이 crop 영역의 실제 텍스트와 겹치지 않는다.
            # 잘못된 페이지/crop에서 만들어진 환각 표가 라벨 자리에 들어가는
            # 사고를 막는 게이트 (골든 라벨 1호가 잡아낸 실제 결함).
            _reject("content_mismatch", grounding=round(recovery.grounding, 4))
            return None
        recovered_markdown = _normalize_recovered_table_markup(recovery.markdown)
        if not recovered_markdown:
            _reject("normalize_failed")
            return None
        if not _recovered_table_passes_sanity(
            recovered_markdown=recovered_markdown,
            issue_types=task.issue_types,
        ):
            _reject("sanity_check_failed")
            return None
        transformed = replace_table_block(
            candidate.content,
            task.table_label,
            recovered_markdown,
            candidate_metadata=candidate.metadata,
        )
        page_number = None
        if transformed == candidate.content and task.table_label.startswith("__page_table__:"):
            page_number, table_index = _page_table_selector_from_label(task.table_label)
            if page_number is not None:
                transformed = replace_page_table_block(
                    candidate.content,
                    page_number,
                    recovered_markdown,
                    table_index=table_index,
                )
        patch_mode = "replace"
        if transformed == candidate.content:
            # 파서가 표를 표 블록으로 렌더링하지 못한 경우: 교체 대상이
            # 없으므로 라벨/페이지 앵커 뒤에 복구된 표를 삽입한다.
            transformed = insert_table_after_anchor(
                candidate.content,
                task.table_label,
                recovered_markdown,
                page_number=page_number if page_number is not None else task.page_number,
            )
            patch_mode = "insert_after_anchor"
        if transformed == candidate.content:
            _reject("patch_target_not_found")
            return None
        note_suffix = ""
        if recovery.notes:
            note_suffix = f" Notes: {'; '.join(recovery.notes[:2])}"
        crop_suffix = ""
        if recovery.bbox is not None:
            crop_suffix = f" Crop: {recovery.crop_method} bbox={recovery.bbox}."
        patch_suffix = "" if patch_mode == "replace" else " Patched by inserting after the label/page anchor (no table block to replace)."
        action = RepairAction(
            action_name="recover_table_from_pdf_image",
            description=(
                f"Recover {recovery.table_label} from the source PDF page image and replace the broken parsed block."
                f"{patch_suffix}"
                f"{crop_suffix}"
                f"{note_suffix}"
            ),
            before_excerpt=_excerpt(candidate.content),
            after_excerpt=_excerpt(transformed),
            issue_type="table_visual_recovery",
            route_name="recover_tables_from_pdf_image",
        )
        return (
            replace(
                candidate,
                content=transformed,
                repaired_from=candidate.repaired_from or candidate.parser_name,
                metadata={
                    **candidate.metadata,
                    "repair_chunk_table_label": task.table_label,
                    "repair_chunk_markdown": recovered_markdown,
                    "repair_chunk_issue_types": list(task.issue_types),
                    "repair_chunk_output_format": task.preferred_output_format,
                },
            ),
            action,
        )

    def repair(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
        metrics: EvaluationMetrics,
    ) -> tuple[ParseCandidate, list[RepairAction]]:
        """heuristic 수리와 visual repair를 함께 쓰는 범용 repair 진입점이다.

        현재 workflow는 주로 route 기반 repair를 사용하지만, 이 메서드는
        adapter 수준의 기본 경로로 남아 있다. 먼저 heuristic 수리를 하고,
        필요하면 visual table recoverer를 추가로 적용한다.
        """
        repaired_candidate, actions = self.repair_heuristics(source, candidate, metrics)
        current_candidate = repaired_candidate
        all_actions = list(actions)

        if self._visual_table_recoverer is not None:
            try:
                visual_content, visual_actions = self._visual_table_recoverer.repair(
                    source,
                    current_candidate.content,
                    metrics,
                    candidate_metadata=current_candidate.metadata,
                )
            except TypeError:
                visual_content, visual_actions = self._visual_table_recoverer.repair(
                    source,
                    current_candidate.content,
                    metrics,
                )
            if visual_actions:
                current_candidate = replace(
                    current_candidate,
                    content=visual_content,
                    repaired_from=current_candidate.repaired_from or current_candidate.parser_name,
                )
                all_actions.extend(visual_actions)

        if not all_actions:
            return candidate, []

        final_candidate = replace(
            current_candidate,
            metadata={
                **candidate.metadata,
                "repair_actions": [action.action_name for action in all_actions],
                "repair_issue_types": [action.issue_type for action in all_actions if action.issue_type is not None],
                "repair_routes": [action.route_name for action in all_actions if action.route_name is not None],
            },
        )
        return final_candidate, all_actions
