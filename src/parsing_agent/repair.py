from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import re
from typing import Any

from parsing_agent.interfaces import CandidateRepairer
from parsing_agent.models import DocumentSource, EvaluationIssue, EvaluationMetrics, ParseCandidate, RepairAction
from parsing_agent.visual_repair import (
    VisualRepairTask,
    _normalize_recovered_table_markup,
    _page_table_selector_from_label,
    replace_page_table_block,
    replace_table_block,
)

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
    source_path = candidate.source_path or source.path
    return source.media_type == "application/pdf" or source_path.suffix.lower() == ".pdf"


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
    ) -> tuple[ParseCandidate, RepairAction] | None:
        """계획된 visual table repair task 하나를 실제로 수행한다.

        visual recoverer가 빈 결과를 주거나 confidence가 `0.45` 미만이면
        버린다. 통과한 경우에만 candidate 안의 해당 표 블록을 교체한다.
        """
        if self._visual_table_recoverer is None:
            return None
        if source.page_count is not None and (task.page_number < 1 or task.page_number > source.page_count):
            return None
        try:
            recovery = self._visual_table_recoverer.recover_task(source, candidate.content, task)
        except Exception:
            return None
        if recovery is None or recovery.confidence < 0.45 or not recovery.markdown.strip():
            return None
        recovered_markdown = _normalize_recovered_table_markup(recovery.markdown)
        if not recovered_markdown:
            return None
        if not _recovered_table_passes_sanity(
            recovered_markdown=recovered_markdown,
            issue_types=task.issue_types,
        ):
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
        if transformed == candidate.content:
            return None
        note_suffix = ""
        if recovery.notes:
            note_suffix = f" Notes: {'; '.join(recovery.notes[:2])}"
        crop_suffix = ""
        if recovery.bbox is not None:
            crop_suffix = f" Crop: {recovery.crop_method} bbox={recovery.bbox}."
        action = RepairAction(
            action_name="recover_table_from_pdf_image",
            description=(
                f"Recover {recovery.table_label} from the source PDF page image and replace the broken parsed block."
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
