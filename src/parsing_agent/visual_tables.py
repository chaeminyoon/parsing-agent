"""표 텍스트 프리미티브 — 라벨 정규화, HTML→마크다운 변환, 평문 표 스캔, 페이지/블록 치환.

visual_repair.py 분할(모듈이 2천 줄에 육박해 계층별로 나눔). 이 모듈은 최하층:
태스크·메트릭·LLM을 모르는 순수 텍스트 조작만 둔다.
"""
from __future__ import annotations

import html
from html.parser import HTMLParser
import re
from typing import Any




_TABLE_PREFIX = "\uD45C"


_TABLE_LABEL_RE = re.compile(rf"(?:{_TABLE_PREFIX}|table)\s*(?:<\s*)?(\d+(?:\.\d+)?-\d+)(?:\s*>)?", re.IGNORECASE)


_TABLE_CAPTION_RE = re.compile(rf"^\s*{_TABLE_PREFIX}\s*(\d+(?:\.\d+)?-\d+)")


_HEADING_RE = re.compile(r"^\s*#+\s+")


_PAGE_REF_RE = re.compile(r"\bp\.?\s*(\d+)\b", re.IGNORECASE)


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


_PAGE_SCOPED_TABLE_PREFIX = "__page_table__:"


_HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)


_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


_GENERIC_HTML_TAG_RE = re.compile(r"</?(?:[A-Za-z][A-Za-z0-9:-]*)(?:\s+[^<>]*?)?>", re.IGNORECASE)


def _page_scoped_table_label(page_number: int) -> str:
    return f"{_PAGE_SCOPED_TABLE_PREFIX}{page_number}"


def _indexed_page_scoped_table_label(page_number: int, table_index: int) -> str:
    return f"{_PAGE_SCOPED_TABLE_PREFIX}{page_number}:{table_index}"


def _page_table_selector_from_label(label: str) -> tuple[int | None, int | None]:
    if not label.startswith(_PAGE_SCOPED_TABLE_PREFIX):
        return None, None
    suffix = label[len(_PAGE_SCOPED_TABLE_PREFIX) :]
    if ":" not in suffix:
        return (int(suffix), None) if suffix.isdigit() else (None, None)
    page_suffix, table_suffix = suffix.split(":", 1)
    if not page_suffix.isdigit():
        return None, None
    page_number = int(page_suffix)
    table_index = int(table_suffix) if table_suffix.isdigit() else None
    return page_number, table_index


def _page_number_from_scoped_label(label: str) -> int | None:
    page_number, _table_index = _page_table_selector_from_label(label)
    return page_number


def _replace_table_slot_placeholder(content: str, placeholder: str, markdown: str) -> str:
    replacement_lines = [line.rstrip() for line in markdown.splitlines() if line.strip()]
    if len(replacement_lines) < 2 or placeholder not in content:
        return content
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != placeholder:
            continue
        replacement = list(lines[:index])
        if replacement and replacement[-1].strip():
            replacement.append("")
        replacement.extend(replacement_lines)
        replacement.extend(lines[index + 1 :])
        normalized = "\n".join(replacement)
        if content.endswith("\n"):
            normalized += "\n"
        return normalized
    return content


def _extract_markdown_table(text: str) -> str:
    lines = text.splitlines()
    table_lines: list[str] = []
    for line in lines:
        if line.lstrip().startswith("|"):
            table_lines.append(line.rstrip())
            continue
        if len(table_lines) >= 2:
            break
        table_lines.clear()
    if len(table_lines) >= 2:
        return "\n".join(table_lines)
    return ""


def _extract_html_table(text: str) -> str:
    match = re.search(r"(<table\b.*?</table>)", text, re.IGNORECASE | re.DOTALL)
    if match is None:
        return ""
    return match.group(1).strip()


class _RecoveredHtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[dict[str, Any]]] = []
        self.current_row: list[dict[str, Any]] | None = None
        self.current_cell: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = dict(attrs)
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            self.current_cell = {
                "is_header": tag == "th",
                "rowspan": _safe_span(attrs_dict.get("rowspan")),
                "colspan": _safe_span(attrs_dict.get("colspan")),
                "parts": [],
            }
        elif tag == "br" and self.current_cell is not None:
            self.current_cell["parts"].append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.current_cell is not None and self.current_row is not None:
            self.current_row.append(
                {
                    "text": _normalize_recovered_cell_text("".join(self.current_cell["parts"])),
                    "is_header": self.current_cell["is_header"],
                    "rowspan": self.current_cell["rowspan"],
                    "colspan": self.current_cell["colspan"],
                }
            )
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell["parts"].append(data)


def _safe_span(value: str | None) -> int:
    try:
        return max(int(value or "1"), 1)
    except ValueError:
        return 1


def _normalize_recovered_cell_text(text: str) -> str:
    normalized = html.unescape(text).replace("\xa0", " ")
    normalized = re.sub(r"\s*\n\s*", "<br>", normalized.strip())
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized.strip()


def _pad_rows(rows: list[list[str]], width: int | None = None) -> list[list[str]]:
    target = width if width is not None else max((len(row) for row in rows), default=0)
    return [row + [""] * (target - len(row)) for row in rows]


def _trim_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return rows
    width = len(rows[0])
    keep_indices = [col for col in range(width) if any(row[col].strip() for row in rows)]
    if not keep_indices:
        return rows
    return [[row[col] for col in keep_indices] for row in rows]


def _expand_recovered_html_rows(
    rows: list[list[dict[str, Any]]],
    *,
    trim_empty: bool = True,
) -> tuple[list[list[str]], int]:
    expanded: list[list[str]] = []
    active_rowspans: dict[int, dict[str, Any]] = {}
    header_rows = 0

    for row in rows:
        out_row: list[str] = []
        col = 0
        row_is_header = bool(row) and all(bool(cell["is_header"]) for cell in row)

        def consume_active() -> None:
            nonlocal col
            while col in active_rowspans:
                span = active_rowspans[col]
                out_row.append(str(span["text"]))
                span["remaining"] -= 1
                if span["remaining"] <= 0:
                    del active_rowspans[col]
                col += 1

        if not row_is_header:
            active_rowspans = {
                col_idx: span for col_idx, span in active_rowspans.items() if not bool(span.get("from_header"))
            }

        consume_active()
        if row_is_header and len(expanded) == header_rows:
            header_rows += 1

        for cell in row:
            text = str(cell["text"])
            colspan = max(int(cell["colspan"]), 1)
            rowspan = max(int(cell["rowspan"]), 1)
            for span_idx in range(colspan):
                out_row.append(text if span_idx == 0 else "")
                if rowspan > 1:
                    active_rowspans[col] = {
                        "remaining": rowspan - 1,
                        "text": text,
                        "from_header": row_is_header,
                    }
                col += 1
            consume_active()

        expanded.append(out_row)

    expanded = _pad_rows(expanded)
    if trim_empty:
        expanded = _trim_empty_columns(expanded)
    return expanded, max(header_rows, 1 if expanded else 0)


def _fill_header_blanks(header_rows: list[list[str]]) -> list[list[str]]:
    filled: list[list[str]] = []
    for row in header_rows:
        current = ""
        output_row: list[str] = []
        for cell in row:
            value = cell.strip()
            if value:
                current = value
            output_row.append(value or current)
        filled.append(output_row)
    return filled


def _collapse_header_rows(header_rows: list[list[str]]) -> list[str]:
    if not header_rows:
        return []
    filled_rows = _fill_header_blanks(header_rows)
    width = len(filled_rows[0])
    collapsed: list[str] = []
    for col in range(width):
        parts: list[str] = []
        for row in filled_rows:
            value = row[col].strip()
            if value and (not parts or parts[-1] != value):
                parts.append(value)
        collapsed.append(" / ".join(parts))
    return collapsed


def _escape_markdown_cell(text: str) -> str:
    return text.replace("|", r"\|").strip()


def _html_table_to_markdown(table_html: str) -> str:
    parser = _RecoveredHtmlTableParser()
    parser.feed(table_html)
    source_rows = parser.rows
    if not source_rows:
        return ""

    header_source_rows: list[list[dict[str, Any]]] = []
    body_source_rows: list[list[dict[str, Any]]] = []
    header_done = False
    for row in source_rows:
        row_is_header = bool(row) and all(bool(cell["is_header"]) for cell in row)
        if not header_done and row_is_header:
            header_source_rows.append(row)
            continue
        header_done = True
        body_source_rows.append(row)

    if header_source_rows:
        header_grid, _ = _expand_recovered_html_rows(header_source_rows, trim_empty=False)
        body_rows, _ = _expand_recovered_html_rows(body_source_rows, trim_empty=False)
    else:
        all_rows, header_count = _expand_recovered_html_rows(source_rows)
        header_count = max(header_count, 1)
        header_grid = all_rows[:header_count]
        body_rows = all_rows[header_count:]

    target_width = max(
        max((len(row) for row in header_grid), default=0),
        max((len(row) for row in body_rows), default=0),
    )
    if target_width:
        header_grid = _pad_rows(header_grid, target_width)
        body_rows = _pad_rows(body_rows, target_width)

    combined_rows = _trim_empty_columns(header_grid + body_rows)
    header_grid = combined_rows[: len(header_grid)]
    body_rows = combined_rows[len(header_grid) :]
    header_grid = [[cell.strip() for cell in row] for row in header_grid]
    body_rows = [[cell.strip() for cell in row] for row in body_rows]
    header = _collapse_header_rows(header_grid)
    if not header:
        return ""

    lines = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body_rows:
        if any(cell.strip() for cell in row):
            lines.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _replace_html_tables_with_markdown(markup: str) -> str:
    return _HTML_TABLE_RE.sub(lambda match: _html_table_to_markdown(match.group(0)) or match.group(0), markup)


def _normalize_recovered_table_markup(markup: str) -> str:
    normalized = _replace_html_tables_with_markdown(markup)
    normalized = _MARKDOWN_IMAGE_RE.sub("", normalized)
    normalized = re.sub(r"<br\s*/?>", " ", normalized, flags=re.IGNORECASE)
    normalized = _GENERIC_HTML_TAG_RE.sub("", normalized)
    normalized = normalized.replace("km²", "㎢")
    normalized = re.sub(r"((?:총|기)?매립\s*용량\s*)\(\s*㎡\s*\)", r"\1(㎥)", normalized)
    normalized = re.sub(r"(잔여\s*매립\s*가능량\s*)\(\s*㎡\s*\)", r"\1(㎥)", normalized)
    normalized = re.sub(r"(?<!\d)(\d{1,3}),(\d)(?!\d)", r"\1.\2", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


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
    return any(
        set(cell.strip()) <= {":", "-"} and cell.strip()
        for cell in table_rows[1].split("|")
        if cell.strip()
    )


def _normalize_table_label(label: str) -> str:
    match = _TABLE_LABEL_RE.search(label)
    if match is None:
        return label.strip()
    return f"{_TABLE_PREFIX} {match.group(1)}"


def _label_number(label: str) -> str | None:
    match = _TABLE_LABEL_RE.search(label)
    if match is None:
        return None
    return match.group(1)


def _candidate_excerpt(content: str, table_label: str, context_lines: int = 14) -> str:
    page_number = _page_number_from_scoped_label(table_label)
    if page_number is not None:
        page_lines = _extract_page_section(content, page_number)
        return "\n".join(page_lines[:context_lines]).strip()
    label_number = _label_number(table_label)
    if label_number is None:
        return ""
    pattern = re.compile(rf"(?:{_TABLE_PREFIX}\s*)?{re.escape(label_number)}")
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if pattern.search(line):
            start = index
            end = min(len(lines), index + context_lines)
            return "\n".join(lines[start:end]).strip()
    return ""


def _extract_page_section(content: str, page_number: int) -> list[str]:
    lines = content.splitlines()
    marker = f"<!-- page {page_number} -->"
    next_marker_re = re.compile(r"^<!-- page \d+ -->$")
    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == marker:
            start_index = index + 1
            break
    if start_index is None:
        return []
    end_index = len(lines)
    for index in range(start_index, len(lines)):
        if next_marker_re.match(lines[index].strip()):
            end_index = index
            break
    return lines[start_index:end_index]


def _looks_like_plain_text_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("<!--") or _HEADING_RE.match(stripped):
        return False
    if stripped.startswith("|") or stripped.lower().startswith("<table") or stripped.lower().startswith("</table"):
        return False
    number_count = len(_NUMBER_RE.findall(stripped))
    has_spaced_columns = "  " in line
    return has_spaced_columns or number_count >= 2


def _looks_like_plain_text_table_header_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("<!--") or _HEADING_RE.match(stripped):
        return False
    if stripped.startswith("|") or stripped.lower().startswith("<table") or stripped.lower().startswith("</table"):
        return False
    if _TABLE_CAPTION_RE.match(stripped):
        return False
    if re.search(r"[.!?:;]", stripped):
        return False
    tokens = stripped.split()
    if len(tokens) < 4 or len(tokens) > 12:
        return False
    return all(len(token) <= 8 for token in tokens)


def _looks_like_plain_text_table_stub_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("<!--") or _HEADING_RE.match(stripped):
        return False
    if stripped.startswith("|") or stripped.lower().startswith("<table") or stripped.lower().startswith("</table"):
        return False
    if _TABLE_CAPTION_RE.match(stripped):
        return False
    if len(stripped) > 48:
        return False
    if len(stripped.split()) > 6:
        return False
    if any(marker in stripped for marker in (".", "?", "!", ":", ";")):
        return False
    return True


def _has_future_plain_text_table_line(lines: list[str], start_index: int, end_index: int) -> bool:
    index = start_index
    while index < end_index:
        stripped = lines[index].strip()
        if stripped and (_HEADING_RE.match(stripped) or _TABLE_CAPTION_RE.match(stripped)):
            return False
        if _looks_like_plain_text_table_line(lines[index]):
            return True
        if stripped and not (_looks_like_plain_text_table_header_line(lines[index]) or _looks_like_plain_text_table_stub_line(lines[index])):
            return False
        index += 1
    return False


def _scan_plain_text_table_block(lines: list[str], start_index: int, end_index: int) -> tuple[int, int] | None:
    if start_index >= end_index or not (
        _looks_like_plain_text_table_line(lines[start_index]) or _looks_like_plain_text_table_header_line(lines[start_index])
    ):
        return None
    line_count = 0
    actual_table_line_count = 0
    index = start_index
    while index < end_index:
        current_line = lines[index]
        stripped = current_line.strip()
        if stripped and (_HEADING_RE.match(stripped) or _TABLE_CAPTION_RE.match(stripped)):
            break
        if _looks_like_plain_text_table_line(current_line):
            line_count += 1
            actual_table_line_count += 1
            index += 1
            continue
        if _looks_like_plain_text_table_header_line(current_line):
            line_count += 1
            index += 1
            continue
        if _looks_like_plain_text_table_stub_line(current_line):
            if actual_table_line_count > 0 or _has_future_plain_text_table_line(lines, index + 1, end_index):
                line_count += 1
                index += 1
                continue
            break
        if not stripped:
            lookahead = index + 1
            while lookahead < end_index and not lines[lookahead].strip():
                lookahead += 1
            if lookahead >= end_index:
                break
            next_line = lines[lookahead]
            if _looks_like_plain_text_table_line(next_line) or _looks_like_plain_text_table_header_line(next_line):
                index = lookahead
                continue
            if _looks_like_plain_text_table_stub_line(next_line) and _has_future_plain_text_table_line(
                lines, lookahead + 1, end_index
            ):
                index = lookahead
                continue
        break
    if actual_table_line_count < 1 or line_count < 2:
        return None
    return start_index, index


def _find_page_table_block_bounds(
    lines: list[str],
    page_start: int,
    page_end: int,
    table_index: int | None = None,
) -> tuple[int, int] | None:
    block_bounds: list[tuple[int, int]] = []

    index = page_start
    while index < page_end:
        if lines[index].strip().startswith("|"):
            table_end = index + 1
            while table_end < page_end and lines[table_end].strip().startswith("|"):
                table_end += 1
            block_bounds.append((index, table_end))
            index = table_end
            continue
        index += 1

    index = page_start
    while index < page_end:
        if not lines[index].strip().lower().startswith("<table"):
            index += 1
            continue
        table_end = index + 1
        while table_end < page_end:
            if "</table>" in lines[table_end - 1].lower():
                break
            table_end += 1
        block_bounds.append((index, page_end if table_end >= page_end else table_end))
        index = max(index + 1, table_end)

    index = page_start
    while index < page_end:
        table_bounds = _scan_plain_text_table_block(lines, index, page_end)
        if table_bounds is None:
            index += 1
            continue
        block_bounds.append(table_bounds)
        index = table_bounds[1]

    if not block_bounds:
        return None
    block_bounds.sort(key=lambda bounds: bounds[0])
    if table_index is None or table_index <= 1:
        return block_bounds[0]
    if table_index > len(block_bounds):
        return None
    return block_bounds[table_index - 1]


def _table_block_bounds_by_scan(lines: list[str]) -> list[tuple[int, int]]:
    block_bounds: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        if lines[index].strip().startswith("|"):
            table_end = index + 1
            while table_end < len(lines) and lines[table_end].strip().startswith("|"):
                table_end += 1
            block_bounds.append((index, table_end))
            index = table_end
            continue
        if lines[index].strip().lower().startswith("<table"):
            table_end = index + 1
            while table_end < len(lines):
                if "</table>" in lines[table_end - 1].lower():
                    break
                table_end += 1
            block_bounds.append((index, min(table_end, len(lines))))
            index = max(index + 1, table_end)
            continue
        table_bounds = _scan_plain_text_table_block(lines, index, len(lines))
        if table_bounds is not None:
            block_bounds.append(table_bounds)
            index = table_bounds[1]
            continue
        index += 1
    return block_bounds


def replace_table_block_by_global_index(content: str, table_index: int, markdown: str) -> str:
    replacement_lines = [line.rstrip() for line in markdown.splitlines() if line.strip()]
    if len(replacement_lines) < 2 or table_index <= 0:
        return content

    lines = content.splitlines()
    block_bounds = _table_block_bounds_by_scan(lines)
    if table_index > len(block_bounds):
        return content
    table_start, table_end = block_bounds[table_index - 1]
    replacement = list(lines[:table_start])
    if replacement and replacement[-1].strip():
        replacement.append("")
    replacement.extend(replacement_lines)
    if table_end < len(lines) and lines[table_end].strip():
        replacement.append("")
    replacement.extend(lines[table_end:])
    normalized = "\n".join(replacement)
    if content.endswith("\n"):
        normalized += "\n"
    return normalized


def replace_page_table_block(content: str, page_number: int, markdown: str, table_index: int | None = None) -> str:
    replacement_lines = [line.rstrip() for line in markdown.splitlines() if line.strip()]
    if len(replacement_lines) < 2:
        return content

    lines = content.splitlines()
    marker = f"<!-- page {page_number} -->"
    next_marker_re = re.compile(r"^<!-- page \d+ -->$")
    page_start: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == marker:
            page_start = index + 1
            break
    if page_start is None:
        return content
    page_end = len(lines)
    for index in range(page_start, len(lines)):
        if next_marker_re.match(lines[index].strip()):
            page_end = index
            break

    table_bounds = _find_page_table_block_bounds(lines, page_start, page_end, table_index=table_index)
    if table_bounds is None:
        return content
    table_start, table_end = table_bounds

    replacement = list(lines[:table_start])
    if replacement and replacement[-1].strip():
        replacement.append("")
    replacement.extend(replacement_lines)
    if table_end < len(lines) and lines[table_end].strip():
        replacement.append("")
    replacement.extend(lines[table_end:])
    normalized = "\n".join(replacement)
    if content.endswith("\n"):
        normalized += "\n"
    return normalized


_HTML_TEXT_RE = re.compile(r">([^<>]+)<")


def _recovery_grounding_ratio(reference_text: str, markup: str) -> float | None:
    """재구성된 표의 셀들이 crop 영역 실제 텍스트에 존재하는 비율.

    vision 모델이 엉뚱한 표를 재구성하면(잘못된 crop, 환각) 셀 내용이
    crop 텍스트에 없으므로 비율이 낮게 나온다. 공백 차이를 무시하기 위해
    양쪽 모두 공백을 제거하고 부분 문자열 포함으로 판정한다.
    스캔 페이지처럼 참조 텍스트가 거의 없으면 판정 불가(None)로 둔다.
    """
    reference = re.sub(r"\s+", "", reference_text or "")
    if len(reference) < 30:
        return None
    cells: list[str] = []
    if "<table" in markup.lower():
        cells = [match.strip() for match in _HTML_TEXT_RE.findall(markup)]
    else:
        for line in markup.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("|") and stripped.endswith("|")):
                continue
            cells.extend(cell.strip() for cell in stripped.split("|")[1:-1])
    normalized_cells = []
    for cell in cells:
        compact = re.sub(r"\s+", "", cell)
        if len(compact) >= 2 and set(compact) - {"-", ":"}:
            normalized_cells.append(compact)
    if not normalized_cells:
        return None
    matched = sum(1 for cell in normalized_cells if cell in reference)
    return matched / len(normalized_cells)


def _find_text_label_line_index(lines: list[str], label: str) -> int | None:
    """번호 없는 표 라벨을 공백 무시 부분 일치로 찾는다.

    같은 라벨이 목차와 본문에 모두 등장할 수 있다. 표 유사 내용이 바로
    뒤따르는 등장을 우선하고, 없으면 마지막 등장(본문 쪽)을 쓴다 —
    첫 등장을 그대로 쓰면 목차 줄 뒤에 표가 삽입되는 사고가 난다.
    """
    needle = re.sub(r"\s+", "", label).strip()
    if len(needle) < 4:
        return None
    matches: list[int] = []
    for index, line in enumerate(lines):
        haystack = re.sub(r"\s+", "", line)
        if len(haystack) < 4:
            continue
        if needle in haystack or haystack in needle:
            matches.append(index)
    if not matches:
        return None
    for index in matches:
        window = lines[index + 1 : index + 6]
        if any(entry.strip().count("|") >= 2 for entry in window):
            return index
    return matches[-1]


def insert_table_after_anchor(
    content: str,
    table_label: str,
    markdown: str,
    *,
    page_number: int | None = None,
) -> str:
    """교체할 표 블록이 없을 때의 폴백: 앵커 다음에 복구된 표를 삽입한다.

    앵커 우선순위는 라벨 번호 → 라벨 텍스트 → 페이지 마커 순서이고,
    아무 앵커도 없으면 content를 그대로 반환한다 (삽입 위치를 추측으로
    정해 문서를 오염시키지 않는다).
    """
    markdown_lines = [line.rstrip() for line in markdown.splitlines() if line.strip()]
    if len(markdown_lines) < 2:
        return content
    lines = content.splitlines()
    anchor: int | None = None
    label_number = _label_number(_normalize_table_label(table_label))
    if label_number is not None:
        pattern = re.compile(rf"(?:{_TABLE_PREFIX}\s*)?{re.escape(label_number)}")
        anchor = next((index for index, line in enumerate(lines) if pattern.search(line)), None)
    if anchor is None:
        anchor = _find_text_label_line_index(lines, table_label)
    if anchor is None and page_number is not None:
        marker = f"<!-- page {page_number} -->"
        anchor = next((index for index, line in enumerate(lines) if line.strip() == marker), None)
    if anchor is None:
        return content
    replacement = [*lines[: anchor + 1], "", *markdown_lines, "", *lines[anchor + 1 :]]
    normalized = "\n".join(replacement)
    if content.endswith("\n"):
        normalized += "\n"
    return normalized
