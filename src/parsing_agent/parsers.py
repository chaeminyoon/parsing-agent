from __future__ import annotations

from collections import Counter
import base64
import html
import json
import mimetypes
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Iterable
import urllib.error
import urllib.request

import fitz
from pypdf import PdfReader

try:
    import opendataloader_pdf
except ImportError:  # pragma: no cover - optional dependency guard
    opendataloader_pdf = None

from parsing_agent.config import WorkflowConfig
from parsing_agent.filetype import TEXT_SUFFIXES, is_pdf_source as _is_pdf_source, is_text_like_source
from parsing_agent.format_parsers import (
    CsvParserAdapter,
    DataParserAdapter,
    DocxParserAdapter,
    HtmlParserAdapter,
    OdtParserAdapter,
    PptxParserAdapter,
    XlsxParserAdapter,
    XmlParserAdapter,
)
from parsing_agent.interfaces import ParserAdapter
from parsing_agent.models import DocumentSource, ParseCandidate
from parsing_agent.textutil import (
    clean_table_cell as _clean_cell,
    read_text_with_fallback as _read_text_with_fallback,
    rows_to_markdown as _rows_to_markdown,
)

_WORD_RE = re.compile(r"\w+")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?P<target><[^>]+>|[^)]+)\)")
_TABLE_LABEL_LINE_RE = re.compile(
    r"^\s*(?:table|\uD45C)\s*(?:<\s*)?\d+(?:\.\d+)*(?:-\d+)?(?:\s*>)?",
    re.IGNORECASE,
)
_TABLE_LABEL_CAPTURE_RE = re.compile(
    r"^\s*((?:table|\uD45C)\s*(?:<\s*)?(\d+(?:\.\d+)*(?:-\d+)?)(?:\s*>)?)",
    re.IGNORECASE,
)
_PAGE_MARKER_RE = re.compile(r"^<!-- page (\d+) -->$")


def _extract_pdf_page_texts(source: DocumentSource) -> list[str]:
    if not _is_pdf_source(source):
        return []
    reader = PdfReader(str(source.path))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _clean_html_cell(value: object) -> str:
    if value is None:
        return ""
    return html.escape(" ".join(str(value).replace("\n", " ").split()), quote=False)


def _compact_text(text: str) -> str:
    return "".join(str(text).split())


def _token_coverage(source_text: str, candidate_text: str) -> float:
    source_tokens = Counter(_WORD_RE.findall(source_text.lower()))
    if not source_tokens:
        return 1.0 if not candidate_text.strip() else 0.0
    candidate_tokens = Counter(_WORD_RE.findall(candidate_text.lower()))
    matched_tokens = sum(min(count, candidate_tokens[token]) for token, count in source_tokens.items())
    return matched_tokens / sum(source_tokens.values())


def _extract_page_words(page: fitz.Page) -> list[tuple[fitz.Rect, str, int, int, float, float]]:
    try:
        raw_words = page.get_text("words")
    except Exception:
        return []

    words: list[tuple[fitz.Rect, str, int, int, float, float]] = []
    for raw_word in raw_words:
        if len(raw_word) < 5:
            continue
        text = str(raw_word[4]).strip()
        if not text:
            continue
        try:
            rect = fitz.Rect(raw_word[:4])
        except Exception:
            continue
        if rect.is_empty:
            continue
        line_no = int(raw_word[6]) if len(raw_word) > 6 else 0
        word_no = int(raw_word[7]) if len(raw_word) > 7 else 0
        words.append((rect, text, line_no, word_no, rect.y0, rect.x0))
    return words


def _rebuild_block_text_from_words(
    page_words: list[tuple[fitz.Rect, str, int, int, float, float]],
    rect: fitz.Rect,
    raw_block_text: str,
) -> str:
    if not page_words:
        return raw_block_text

    selected_words = [
        (text, line_no, word_no, y0, x0)
        for word_rect, text, line_no, word_no, y0, x0 in page_words
        if _rect_overlap_ratio(word_rect, rect) >= 0.8
    ]
    if not selected_words:
        return raw_block_text

    selected_words.sort(key=lambda item: (item[1], item[3], item[4], item[2]))
    lines: list[list[str]] = []
    current_line_key: tuple[int, float] | None = None
    current_line: list[str] = []
    for text, line_no, _word_no, y0, _x0 in selected_words:
        line_key = (line_no, round(y0, 1))
        if current_line_key is None or line_key == current_line_key:
            current_line.append(text)
            current_line_key = line_key
            continue
        lines.append(current_line)
        current_line = [text]
        current_line_key = line_key
    if current_line:
        lines.append(current_line)

    rebuilt_text = "\n".join(" ".join(line) for line in lines if line)
    if not rebuilt_text:
        return raw_block_text
    if len(_compact_text(rebuilt_text)) < len(_compact_text(raw_block_text)) * 0.8:
        return raw_block_text
    return rebuilt_text


def _clean_target_path(target: str) -> str:
    cleaned = target.strip()
    if cleaned.startswith("<") and cleaned.endswith(">"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _resolve_relative_asset_path(target: str, markdown_path: Path) -> Path | None:
    cleaned = _clean_target_path(target)
    if not cleaned or "://" in cleaned or cleaned.startswith("data:"):
        return None
    asset_path = (markdown_path.parent / cleaned).resolve()
    if not asset_path.exists():
        return None
    return asset_path


def _image_file_to_data_url(image_path: Path) -> str | None:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        return None
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_markdown_image_data_urls(markdown_text: str, markdown_path: Path) -> dict[str, str]:
    embedded_images: dict[str, str] = {}
    for match in _MARKDOWN_IMAGE_RE.finditer(markdown_text):
        cleaned_target = _clean_target_path(match.group("target"))
        if cleaned_target in embedded_images:
            continue
        asset_path = _resolve_relative_asset_path(cleaned_target, markdown_path)
        if asset_path is None:
            continue
        data_url = _image_file_to_data_url(asset_path)
        if data_url is None:
            continue
        embedded_images[cleaned_target] = data_url
    return embedded_images


def _split_markdown_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_markdown_table_separator(line: str) -> bool:
    cells = _split_markdown_table_cells(line)
    if not cells:
        return False
    return all(cell and set(cell) <= {":", "-"} for cell in cells)


def _is_markdown_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.count("|") >= 2 and (stripped.startswith("|") or stripped.endswith("|"))


def _extract_table_label_parts(line: str) -> tuple[str, str] | None:
    match = _TABLE_LABEL_CAPTURE_RE.match(line.strip())
    if match is None:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _extract_page_label_candidates(page: fitz.Page) -> list[tuple[fitz.Rect, str, str]]:
    candidates: list[tuple[fitz.Rect, str, str]] = []
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return candidates
    for block in blocks:
        rect = fitz.Rect(block[:4])
        block_text = str(block[4] or "").strip()
        block_type = block[6] if len(block) > 6 else 0
        if block_type != 0 or not block_text:
            continue
        for line in block_text.splitlines():
            label_parts = _extract_table_label_parts(line.strip())
            if label_parts is None:
                continue
            candidates.append((rect, label_parts[0], label_parts[1]))
    return candidates


def _resolve_table_label_for_rect(
    table_rect: fitz.Rect,
    label_candidates: list[tuple[fitz.Rect, str, str]],
) -> tuple[str, str] | None:
    best_match: tuple[float, tuple[str, str]] | None = None
    for candidate_rect, label_text, label_number in label_candidates:
        if candidate_rect.y1 > table_rect.y0 + 40:
            continue
        vertical_gap = table_rect.y0 - candidate_rect.y1
        if vertical_gap < 0 or vertical_gap > 180:
            continue
        horizontal_overlap = max(0.0, min(table_rect.x1, candidate_rect.x1) - max(table_rect.x0, candidate_rect.x0))
        if horizontal_overlap <= 0:
            continue
        score = vertical_gap - min(horizontal_overlap, 200.0) / 1000.0
        if best_match is None or score < best_match[0]:
            best_match = (score, (label_text, label_number))
    return None if best_match is None else best_match[1]


def _extract_markdown_table_grounding(markdown_text: str, page_count: int | None) -> dict[str, object]:
    lines = markdown_text.splitlines()
    current_page = 1 if page_count else None
    page_table_counts: dict[int, int] = {}
    table_regions: list[dict[str, object]] = []
    table_label_pages: dict[str, int] = {}
    table_label_positions: dict[str, dict[str, int]] = {}
    global_index = 0
    pending_label: tuple[str, str] | None = None
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        stripped = raw_line.strip()
        page_match = _PAGE_MARKER_RE.match(stripped)
        if page_match is not None:
            current_page = int(page_match.group(1))
            pending_label = None
            index += 1
            continue
        label_parts = _extract_table_label_parts(stripped)
        if label_parts is not None:
            pending_label = label_parts
            index += 1
            continue
        if index + 1 < len(lines) and _is_markdown_table_row(stripped) and _is_markdown_table_separator(lines[index + 1]):
            table_lines = [lines[index]]
            row_count = 1
            col_count = len(_split_markdown_table_cells(lines[index]))
            cursor = index + 2
            while cursor < len(lines) and _is_markdown_table_row(lines[cursor].strip()):
                table_lines.append(lines[cursor])
                row_count += 1
                cursor += 1
            page_number = current_page if current_page is not None else 1
            page_table_counts[page_number] = page_table_counts.get(page_number, 0) + 1
            region_index = page_table_counts[page_number]
            global_index += 1
            label_text = pending_label[0] if pending_label is not None else None
            label_number = pending_label[1] if pending_label is not None else None
            table_region: dict[str, object] = {
                "table_id": f"p{page_number}-t{region_index}",
                "page": page_number,
                "row_count": row_count,
                "col_count": col_count,
                "extraction_mode": "markdown",
            }
            if label_text is not None:
                table_region["label"] = label_text
                table_label_pages.setdefault(label_text, page_number)
                table_label_positions.setdefault(
                    label_text,
                    {"page": page_number, "region_index": region_index, "global_index": global_index},
                )
            if label_number is not None:
                table_label_pages.setdefault(label_number, page_number)
                table_label_positions.setdefault(
                    label_number,
                    {"page": page_number, "region_index": region_index, "global_index": global_index},
                )
            table_regions.append(table_region)
            pending_label = None
            index = cursor
            continue
        if stripped:
            pending_label = None
        index += 1
    return {
        "table_regions": table_regions,
        "table_label_pages": table_label_pages,
        "table_label_positions": table_label_positions,
        "table_format": "markdown",
    }


def _extract_pdf_table_grounding_metadata(source: DocumentSource) -> dict[str, object]:
    if not _is_pdf_source(source):
        return {
            "table_regions": [],
            "table_label_pages": {},
            "table_label_positions": {},
        }
    table_regions: list[dict[str, object]] = []
    table_label_pages: dict[str, int] = {}
    table_label_positions: dict[str, dict[str, int]] = {}
    global_index = 0
    try:
        with fitz.open(source.path) as document:
            for page_index, page in enumerate(document, start=1):
                label_candidates = _extract_page_label_candidates(page)
                page_table_index = 0
                if not hasattr(page, "find_tables"):
                    continue
                try:
                    table_finder = page.find_tables()
                except Exception:
                    continue
                for table in getattr(table_finder, "tables", []) if table_finder is not None else []:
                    if not getattr(table, "bbox", None):
                        continue
                    rows = table.extract() or []
                    if not rows:
                        continue
                    rect = fitz.Rect(table.bbox)
                    page_table_index += 1
                    global_index += 1
                    col_count = getattr(table, "col_count", None) or max(len(row) for row in rows)
                    row_count = len(rows)
                    table_region: dict[str, object] = {
                        "table_id": f"p{page_index}-t{page_table_index}",
                        "page": page_index,
                        "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                        "row_count": row_count,
                        "col_count": col_count,
                        "extraction_mode": "grounding",
                    }
                    label_parts = _resolve_table_label_for_rect(rect, label_candidates)
                    if label_parts is not None:
                        label_text, label_number = label_parts
                        table_region["label"] = label_text
                        table_label_pages.setdefault(label_text, page_index)
                        table_label_pages.setdefault(label_number, page_index)
                        position_payload = {
                            "page": page_index,
                            "region_index": page_table_index,
                            "global_index": global_index,
                        }
                        table_label_positions.setdefault(label_text, position_payload)
                        table_label_positions.setdefault(label_number, position_payload)
                    table_regions.append(table_region)
    except Exception:
        return {
            "table_regions": [],
            "table_label_pages": {},
            "table_label_positions": {},
        }
    return {
        "table_regions": table_regions,
        "table_label_pages": table_label_pages,
        "table_label_positions": table_label_positions,
    }


def _merge_table_grounding_metadata(
    primary: dict[str, object],
    secondary: dict[str, object],
) -> dict[str, object]:
    merged_regions = list(primary.get("table_regions", []))
    if not merged_regions:
        merged_regions = list(secondary.get("table_regions", []))
    merged_label_pages = dict(secondary.get("table_label_pages", {}))
    merged_label_pages.update(primary.get("table_label_pages", {}))
    merged_label_positions = dict(secondary.get("table_label_positions", {}))
    merged_label_positions.update(primary.get("table_label_positions", {}))
    merged: dict[str, object] = {
        "table_regions": merged_regions,
        "table_label_pages": merged_label_pages,
        "table_label_positions": merged_label_positions,
    }
    if "table_format" in primary:
        merged["table_format"] = primary["table_format"]
    elif "table_format" in secondary:
        merged["table_format"] = secondary["table_format"]
    return merged


def _normalized_cell_key(cell: object) -> tuple[float, float, float, float] | None:
    if cell is None:
        return None
    try:
        rect = fitz.Rect(cell)
    except Exception:
        return None
    if rect.is_empty:
        return None
    return (round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2))


def _cell_span_map(cells: list[object], row_count: int, col_count: int) -> dict[tuple[int, int], tuple[int, int] | None]:
    if row_count <= 0 or col_count <= 0 or len(cells) != row_count * col_count:
        return {}

    grid = [
        [_normalized_cell_key(cells[row_index * col_count + col_index]) for col_index in range(col_count)]
        for row_index in range(row_count)
    ]
    spans: dict[tuple[int, int], tuple[int, int] | None] = {}
    for row_index, row in enumerate(grid):
        for col_index, key in enumerate(row):
            if key is None or (row_index, col_index) in spans:
                continue
            coordinates = [
                (candidate_row, candidate_col)
                for candidate_row, candidate in enumerate(grid)
                for candidate_col, candidate_key in enumerate(candidate)
                if candidate_key == key
            ]
            min_row = min(candidate_row for candidate_row, _ in coordinates)
            max_row = max(candidate_row for candidate_row, _ in coordinates)
            min_col = min(candidate_col for _, candidate_col in coordinates)
            max_col = max(candidate_col for _, candidate_col in coordinates)
            for candidate_row, candidate_col in coordinates:
                spans[(candidate_row, candidate_col)] = None
            spans[(min_row, min_col)] = (max_row - min_row + 1, max_col - min_col + 1)
    return spans


def _has_spanning_cells(cells: list[object], row_count: int, col_count: int) -> bool:
    spans = _cell_span_map(cells, row_count, col_count)
    return any(span not in {None, (1, 1)} for span in spans.values())


def _is_simple_table_candidate(
    rows: list[list[object]],
    *,
    cells: list[object],
    row_count: int,
    col_count: int,
    is_continuation: bool,
) -> bool:
    if not rows or row_count <= 0 or col_count <= 0:
        return False
    if is_continuation:
        return False
    if any(len(row) != col_count for row in rows):
        return False
    if _has_spanning_cells(cells, row_count, col_count):
        return False
    if col_count > 2 and any(not _clean_cell(cell) for cell in rows[0]):
        return False
    return True


def _rows_to_html(
    rows: list[list[object]],
    *,
    cells: list[object] | None = None,
    row_count: int | None = None,
    col_count: int | None = None,
    table_attributes: str = "",
) -> str:
    if not rows or not any(any(_clean_html_cell(cell) for cell in row) for row in rows):
        return ""

    resolved_row_count = row_count or len(rows)
    resolved_col_count = col_count or max(len(row) for row in rows)
    normalized_rows = [
        [_clean_html_cell(cell) for cell in row[:resolved_col_count]] + [""] * max(resolved_col_count - len(row), 0)
        for row in rows[:resolved_row_count]
    ]
    spans = _cell_span_map(cells or [], resolved_row_count, resolved_col_count)
    lines = [f"<table{table_attributes}>"]
    for row_index, row in enumerate(normalized_rows):
        lines.append("  <tr>")
        tag = "th" if row_index == 0 else "td"
        for col_index, cell in enumerate(row):
            span = spans.get((row_index, col_index), (1, 1))
            if span is None:
                continue
            rowspan, colspan = span
            attributes = ""
            if rowspan > 1:
                attributes += f' rowspan="{rowspan}"'
            if colspan > 1:
                attributes += f' colspan="{colspan}"'
            lines.append(f"    <{tag}{attributes}>{cell}</{tag}>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _table_to_html(
    table: object,
    rows: list[list[object]] | None = None,
    *,
    continuation: bool = False,
    continued_from_page: int | None = None,
) -> str:
    table_rows = table.extract() if rows is None else rows
    row_count = len(table_rows)
    col_count = getattr(table, "col_count", None) or (max(len(row) for row in table_rows) if table_rows else 0)
    attributes = ""
    if continuation:
        attributes = ' data-continuation="true"'
        if continued_from_page is not None:
            attributes += f' data-continued-from-page="{continued_from_page}"'
    return _rows_to_html(
        table_rows,
        cells=list(getattr(table, "cells", []) or []),
        row_count=row_count,
        col_count=col_count,
        table_attributes=attributes,
    )


def _rect_overlap_ratio(rect: fitz.Rect, other: fitz.Rect) -> float:
    intersection = rect & other
    if intersection.is_empty or rect.is_empty:
        return 0.0
    return intersection.get_area() / max(rect.get_area(), 1.0)


def _rect_vertical_overlap_ratio(rect: fitz.Rect, other: fitz.Rect) -> float:
    intersection = rect & other
    if intersection.is_empty or rect.is_empty:
        return 0.0
    return max(intersection.y1 - intersection.y0, 0.0) / max(rect.y1 - rect.y0, 1.0)


def _table_overlap_metrics(rect: fitz.Rect, table_rects: list[fitz.Rect]) -> tuple[float, float]:
    max_area_ratio = 0.0
    max_vertical_ratio = 0.0
    for table_rect in table_rects:
        max_area_ratio = max(max_area_ratio, _rect_overlap_ratio(rect, table_rect))
        max_vertical_ratio = max(max_vertical_ratio, _rect_vertical_overlap_ratio(rect, table_rect))
    return max_area_ratio, max_vertical_ratio


def _should_drop_text_block_for_table_overlap(rect: fitz.Rect, table_rects: list[fitz.Rect]) -> tuple[bool, float, float]:
    max_area_ratio, max_vertical_ratio = _table_overlap_metrics(rect, table_rects)
    should_drop = max_area_ratio >= 0.6 or max_vertical_ratio >= 0.85
    return should_drop, max_area_ratio, max_vertical_ratio


def _render_source_page_with_layout_elements(
    source_page_text: str,
    table_contents: list[str],
    image_contents: list[str],
) -> str:
    if not source_page_text.strip():
        return "\n\n".join(part for part in [*table_contents, *image_contents] if part.strip())

    lines = source_page_text.splitlines()
    remaining_tables = list(table_contents)
    rendered_lines: list[str] = []
    for line in lines:
        rendered_lines.append(line.rstrip())
        if remaining_tables and _TABLE_LABEL_LINE_RE.search(line):
            rendered_lines.append("")
            rendered_lines.append(remaining_tables.pop(0))
            rendered_lines.append("")

    trailing_blocks = [*remaining_tables, *image_contents]
    if trailing_blocks:
        if rendered_lines and rendered_lines[-1].strip():
            rendered_lines.append("")
        for block in trailing_blocks:
            rendered_lines.append(block)
            rendered_lines.append("")
    return "\n".join(rendered_lines).strip()


def _render_table_reference_block(table_region: dict[str, object]) -> str:
    bbox = table_region.get("bbox")
    bbox_text = ",".join(str(value) for value in bbox) if isinstance(bbox, list) else "unknown"
    return (
        f"[Table reference: id={table_region.get('table_id', 'unknown')} "
        f"page={table_region.get('page', 'unknown')} "
        f"bbox={bbox_text} "
        f"rows={table_region.get('row_count', 'unknown')} "
        f"cols={table_region.get('col_count', 'unknown')}]"
    )


def _looks_like_duplicate_table_header(block_text: str, table_rows: list[list[object]], rect: fitz.Rect, table_rects: list[fitz.Rect]) -> bool:
    normalized_block = _compact_text(block_text)
    if not normalized_block:
        return False
    for table_rect, table_rows_item in zip(table_rects, table_rows, strict=False):
        if rect.y0 > table_rect.y0:
            continue
        if table_rect.y0 - rect.y1 > 80:
            continue
        header_line = _compact_text(" ".join(_clean_cell(cell) for cell in table_rows_item[0] if _clean_cell(cell)))
        if header_line and normalized_block == header_line:
            return True
    return False


def _is_noise_margin_block(rect: fitz.Rect, page_rect: fitz.Rect, top_margin: float, bottom_margin: float) -> bool:
    if top_margin > 0 and rect.y1 <= page_rect.y0 + top_margin:
        return True
    if bottom_margin > 0 and rect.y0 >= page_rect.y1 - bottom_margin:
        return True
    return False


def _render_crop_data_url(page: fitz.Page, rect: fitz.Rect, padding: float) -> str:
    clip = fitz.Rect(rect)
    clip.x0 = max(page.rect.x0, clip.x0 - padding)
    clip.y0 = max(page.rect.y0, clip.y0 - padding)
    clip.x1 = min(page.rect.x1, clip.x1 + padding)
    clip.y1 = min(page.rect.y1, clip.y1 + padding)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
    encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_response_text(payload: dict[str, object]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()
    fragments: list[str] = []
    for item in payload.get("output", []) if isinstance(payload.get("output"), list) else []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                fragments.append(content["text"])
    return "\n".join(fragments).strip()


def _post_response(payload: dict[str, object], config: WorkflowConfig, timeout_seconds: float) -> dict[str, object]:
    request = urllib.request.Request(
        f"{config.judge_base_url.rstrip('/')}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.judge_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _caption_image_block(page: fitz.Page, rect: fitz.Rect, page_index: int, config: WorkflowConfig) -> str | None:
    if not config.judge_api_key or not config.layout_first_image_caption_model:
        return None
    image_url = _render_crop_data_url(page, rect, config.layout_first_image_crop_padding)
    prompt = (
        "이 PDF 페이지의 이미지/차트 영역만 보고 마크다운 문서에 들어갈 짧은 한국어 설명을 작성하세요. "
        "차트라면 축, 범례, 핵심 수치/추세를 우선 요약하고, 지도/도면이라면 위치와 표시 요소를 요약하세요. "
        "확실하지 않은 내용은 추측하지 말고 '확인 불가'라고 쓰세요."
    )
    payload = {
        "model": config.layout_first_image_caption_model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        "max_output_tokens": 220,
    }
    try:
        caption = _extract_response_text(
            _post_response(payload, config, config.layout_first_image_caption_timeout_seconds)
        )
    except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None
    if not caption:
        return None
    return f"**Figure (page {page_index}):** {caption}"


class LocalTextParserAdapter(ParserAdapter):
    name = "text-fallback"
    _TEXT_SUFFIXES = TEXT_SUFFIXES

    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        if not self._is_text_like(source):
            return []

        # 결정론적 로컬 폴백. 인코딩은 UTF-8 계열 → cp949/euc-kr 순으로 시도한다
        # (레거시 완성형 .txt가 전체 파서 실패로 이어지던 버그 수정).
        content = _read_text_with_fallback(source.path)
        return [
            ParseCandidate(
                parser_name=self.name,
                content=content,
                format_name=config.output_format,
                metadata={"media_type": source.media_type},
                source_path=source.path,
            )
        ]

    def _is_text_like(self, source: DocumentSource) -> bool:
        return is_text_like_source(source)


class ExtractedSourceTextParserAdapter(ParserAdapter):
    name = "source-text"

    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        if source.extracted_text is None:
            return []
        if is_text_like_source(source):
            return []
        if not source.extracted_text.strip():
            return []
        return [
            ParseCandidate(
                parser_name=self.name,
                content=source.extracted_text,
                format_name=config.output_format,
                metadata={"media_type": source.media_type, "page_count": source.page_count},
                source_path=source.path,
            )
        ]


class LayoutFirstPdfParserAdapter(ParserAdapter):
    name = "layout-first-pdf"

    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        if not _is_pdf_source(source):
            return []

        content_parts: list[str] = []
        table_regions: list[dict[str, object]] = []
        table_count = 0
        text_block_count = 0
        image_block_count = 0
        image_caption_count = 0
        margin_filtered_text_block_count = 0
        table_overlap_filtered_text_block_count = 0
        partial_table_overlap_retained_text_block_count = 0
        rebuilt_word_text_block_count = 0
        source_text_fallback_page_count = 0
        previous_table: dict[str, object] | None = None
        source_page_texts = _extract_pdf_page_texts(source)

        with fitz.open(source.path) as document:
            for page_index, page in enumerate(document, start=1):
                elements: list[tuple[float, float, str, str]] = []
                page_table_regions: list[dict[str, object]] = []
                structured_table_rects: list[fitz.Rect] = []
                structured_table_rows_for_dedupe: list[list[list[object]]] = []
                page_table_index = 0
                page_words = _extract_page_words(page)
                if hasattr(page, "find_tables"):
                    try:
                        table_finder = page.find_tables()
                    except Exception:
                        table_finder = None
                    for table in getattr(table_finder, "tables", []) if table_finder is not None else []:
                        if not getattr(table, "bbox", None):
                            continue
                        rows = table.extract()
                        if not rows:
                            continue
                        rect = fitz.Rect(table.bbox)
                        col_count = getattr(table, "col_count", None) or max(len(row) for row in rows)
                        row_count = len(rows)
                        previous_page = previous_table.get("page_index") if previous_table is not None else None
                        is_continuation = bool(
                            config.layout_first_merge_multipage_tables
                            and previous_table
                            and previous_page == page_index - 1
                            and previous_table.get("ended_near_bottom")
                            and previous_table.get("col_count") == col_count
                            and rect.y0 <= page.rect.y0 + page.rect.height * 0.25
                        )
                        page_table_index += 1
                        table_region: dict[str, object] = {
                            "table_id": f"p{page_index}-t{page_table_index}",
                            "page": page_index,
                            "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                            "row_count": row_count,
                            "col_count": col_count,
                        }
                        if is_continuation and isinstance(previous_page, int):
                            table_region["continued_from_page"] = previous_page
                        if _is_simple_table_candidate(
                            rows,
                            cells=list(getattr(table, "cells", []) or []),
                            row_count=row_count,
                            col_count=col_count,
                            is_continuation=is_continuation,
                        ):
                            if config.layout_first_table_format.lower() == "markdown":
                                table_content = _rows_to_markdown(rows)
                            else:
                                table_content = _table_to_html(table, rows)
                            if table_content:
                                structured_table_rects.append(rect)
                                structured_table_rows_for_dedupe.append(rows)
                                table_count += 1
                                elements.append((rect.y0, rect.x0, "table", table_content))
                                table_region["extraction_mode"] = "structured"
                            else:
                                table_region["extraction_mode"] = "reference"
                        else:
                            table_region["extraction_mode"] = "reference"
                        table_regions.append(table_region)
                        page_table_regions.append(table_region)
                        previous_table = {
                            "col_count": col_count,
                            "header": rows[0],
                            "page_index": page_index,
                            "ended_near_bottom": rect.y1 >= page.rect.y1 - page.rect.height * 0.15,
                        }

                for block in page.get_text("blocks"):
                    rect = fitz.Rect(block[:4])
                    block_text = str(block[4]).strip()
                    block_type = block[6] if len(block) > 6 else 0
                    if _is_noise_margin_block(
                        rect,
                        page.rect,
                        config.layout_first_skip_top_margin,
                        config.layout_first_skip_bottom_margin,
                    ):
                        if block_type == 0 and block_text:
                            margin_filtered_text_block_count += 1
                        continue
                    if block_type != 0:
                        image_block_count += 1
                        caption = None
                        if (
                            config.layout_first_image_captioning_enabled
                            and image_caption_count < max(config.layout_first_image_caption_max_blocks, 0)
                        ):
                            caption = _caption_image_block(page, rect, page_index, config)
                        if caption:
                            image_caption_count += 1
                        elements.append((rect.y0, rect.x0, "image", caption or f"[Image block omitted: page={page_index}]"))
                        continue
                    if not block_text:
                        continue
                    if _looks_like_duplicate_table_header(
                        block_text,
                        structured_table_rows_for_dedupe,
                        rect,
                        structured_table_rects,
                    ):
                        table_overlap_filtered_text_block_count += 1
                        continue
                    rebuilt_block_text = _rebuild_block_text_from_words(page_words, rect, block_text)
                    if rebuilt_block_text != block_text:
                        rebuilt_word_text_block_count += 1
                        block_text = rebuilt_block_text
                    should_drop_for_overlap, overlap_area_ratio, overlap_vertical_ratio = (
                        _should_drop_text_block_for_table_overlap(rect, structured_table_rects)
                    )
                    if should_drop_for_overlap:
                        table_overlap_filtered_text_block_count += 1
                        continue
                    if overlap_area_ratio > 0 or overlap_vertical_ratio > 0:
                        partial_table_overlap_retained_text_block_count += 1
                    text_block_count += 1
                    elements.append((rect.y0, rect.x0, "text", " ".join(block_text.split())))

                ordered_elements = sorted(elements, key=lambda item: (item[0], item[1])) if elements else []
                ordered_table_contents = [content for _, _, kind, content in ordered_elements if kind == "table"]
                ordered_image_contents = [content for _, _, kind, content in ordered_elements if kind == "image"]
                ordered_text_contents = [content for _, _, kind, content in ordered_elements if kind == "text"]
                page_text_from_layout = "\n\n".join(content for content in ordered_text_contents if content.strip())
                source_page_text = source_page_texts[page_index - 1] if page_index - 1 < len(source_page_texts) else ""
                use_source_text_fallback = bool(
                    source_page_text
                    and _token_coverage(source_page_text, page_text_from_layout) < 0.9
                )

                if use_source_text_fallback:
                    source_text_fallback_page_count += 1
                    content_parts.append(f"<!-- page {page_index} -->")
                    content_parts.append(
                        _render_source_page_with_layout_elements(
                            source_page_text,
                            ordered_table_contents,
                            ordered_image_contents,
                        )
                    )
                    continue

                if ordered_elements:
                    content_parts.append(f"<!-- page {page_index} -->")
                    content_parts.extend(content for _, _, _, content in ordered_elements)
                    continue

                reference_table_blocks = [
                    _render_table_reference_block(table_region)
                    for table_region in page_table_regions
                    if table_region.get("extraction_mode") == "reference"
                ]
                if reference_table_blocks:
                    content_parts.append(f"<!-- page {page_index} -->")
                    content_parts.extend(reference_table_blocks)

        content = "\n\n".join(part for part in content_parts if part.strip())
        if not content.strip():
            return []
        return [
            ParseCandidate(
                parser_name=self.name,
                content=content,
                format_name="md",
                metadata={
                    "page_count": source.page_count,
                    "table_count": table_count,
                    "text_block_count": text_block_count,
                    "image_block_count": image_block_count,
                    "image_caption_count": image_caption_count,
                    "margin_filtered_text_block_count": margin_filtered_text_block_count,
                    "table_overlap_filtered_text_block_count": table_overlap_filtered_text_block_count,
                    "partial_table_overlap_retained_text_block_count": partial_table_overlap_retained_text_block_count,
                    "rebuilt_word_text_block_count": rebuilt_word_text_block_count,
                    "source_text_fallback_page_count": source_text_fallback_page_count,
                    "table_format": config.layout_first_table_format,
                    "multipage_table_merge": config.layout_first_merge_multipage_tables,
                    "layout_engine": "pymupdf",
                    "table_regions": table_regions,
                },
                source_path=source.path,
            )
        ]


class OpenDataLoaderPdfParserAdapter(ParserAdapter):
    name = "opendataloader-pdf"

    def __init__(self, converter=None) -> None:
        self._converter = converter

    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        if not _is_pdf_source(source):
            return []
        converter = self._converter or self._resolve_converter()
        with TemporaryDirectory(prefix=f"opendataloader-{source.run_id}-") as temp_dir:
            output_dir = Path(temp_dir)
            converter(
                input_path=[str(source.path)],
                output_dir=str(output_dir),
                format="markdown",
                quiet=True,
                use_struct_tree=True,
            )
            markdown_paths = sorted(output_dir.rglob("*.md"))
            if not markdown_paths:
                raise ValueError(f"OpenDataLoader did not emit markdown output for {source.path}")
            return [
                (
                    lambda grounding: ParseCandidate(
                        parser_name=self.name,
                        content=markdown_text,
                        format_name="md",
                        metadata={
                            "candidate_index": index,
                            "emitted_name": markdown_path.name,
                            "embedded_image_data_urls": _extract_markdown_image_data_urls(markdown_text, markdown_path),
                            **grounding,
                        },
                        source_path=source.path,
                    )
                )(
                    _merge_table_grounding_metadata(
                        _extract_markdown_table_grounding(markdown_text, source.page_count),
                        _extract_pdf_table_grounding_metadata(source),
                    )
                )
                for index, markdown_path in enumerate(markdown_paths)
                for markdown_text in [_read_text_with_fallback(markdown_path)]
            ]

    def _resolve_converter(self):
        if opendataloader_pdf is None:
            raise RuntimeError("opendataloader-pdf is not installed.")
        return opendataloader_pdf.convert


class ParserRegistry:
    def __init__(self, adapters: Iterable[ParserAdapter] | None = None) -> None:
        self._adapters: dict[str, ParserAdapter] = {}
        for adapter in adapters or (LocalTextParserAdapter(),):
            self.register(adapter)

    def register(self, adapter: ParserAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"Parser adapter already registered: {adapter.name}")
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> ParserAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            raise KeyError(f"Unknown parser adapter: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._adapters

    def run(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        candidates: list[ParseCandidate] = []
        for name in config.parser_names:
            candidates.extend(self.get(name).parse(source, config))
        return candidates


def build_default_parser_registry() -> ParserRegistry:
    return ParserRegistry(
        [
            LayoutFirstPdfParserAdapter(),
            OpenDataLoaderPdfParserAdapter(),
            LocalTextParserAdapter(),
            ExtractedSourceTextParserAdapter(),
            DocxParserAdapter(),
            PptxParserAdapter(),
            XlsxParserAdapter(),
            OdtParserAdapter(),
            CsvParserAdapter(),
            HtmlParserAdapter(),
            DataParserAdapter(),
            XmlParserAdapter(),
        ]
    )
