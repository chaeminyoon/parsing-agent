from __future__ import annotations

import base64
from dataclasses import dataclass
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, Iterable
from urllib import request

import fitz
from langsmith import tracing_context

from parsing_agent.config import WorkflowConfig
from parsing_agent.evaluation import (
    TABLE_ISSUE_MERGED_CELL_LOSS,
    TABLE_ISSUE_MISSING_HEADER,
    TABLE_ISSUE_NUMERIC_TOKEN_BREAK,
    TABLE_ISSUE_SPLIT_MULTIPAGE,
    TABLE_ISSUE_TEXT_DUPLICATION,
)
from parsing_agent.models import DocumentSource, EvaluationMetrics, RepairAction

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
_SYSTEM_PROMPT = """You repair broken markdown tables extracted from PDFs.
Return strict JSON with this schema:
{
  "table_label": "표 4.2-2",
  "page_number": 12,
  "confidence": number between 0 and 1,
  "markdown": "markdown table or empty string",
  "notes": ["short note", "..."]
}
Rules:
- Reconstruct only the target table.
- Preserve visible Korean text, units, and numbers exactly when legible.
- Use a markdown table with a header row and separator row.
- Do not invent unreadable values; leave uncertain cells blank.
- If the target table is not visible enough to recover, return empty markdown and low confidence.
- Do not include prose outside the JSON object."""


def _is_pdf_source(source: DocumentSource) -> bool:
    return source.media_type == "application/pdf" or source.path.suffix.lower() == ".pdf"


@dataclass(frozen=True, slots=True)
class VisualTableRecovery:
    table_label: str
    page_number: int
    confidence: float
    markdown: str
    notes: list[str]
    crop_method: str
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True, slots=True)
class VisualRepairTask:
    task_id: str
    table_label: str
    page_number: int
    issue_types: tuple[str, ...] = ()
    preferred_output_format: str = "markdown"


@dataclass(frozen=True, slots=True)
class TableCrop:
    page_number: int
    clip: fitz.Rect
    method: str
    bbox: tuple[float, float, float, float] | None


def extract_table_labels(issues: Iterable[str]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for issue in issues:
        for match in _TABLE_LABEL_RE.finditer(issue):
            label = f"{_TABLE_PREFIX} {match.group(1)}"
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
    return labels


def extract_issue_page_numbers(issues: Iterable[str]) -> list[int]:
    page_numbers: list[int] = []
    seen: set[int] = set()
    for issue in issues:
        for match in _PAGE_REF_RE.finditer(issue):
            page_number = int(match.group(1))
            if page_number in seen:
                continue
            seen.add(page_number)
            page_numbers.append(page_number)
    return page_numbers


def _structured_table_findings(metrics: EvaluationMetrics) -> list[dict[str, Any]]:
    judge_result = metrics.judge_result
    if judge_result is None:
        return []
    findings: list[dict[str, Any]] = []
    for item in judge_result.table_findings:
        if not isinstance(item, dict):
            continue
        normalized: dict[str, Any] = {}
        issue_type = item.get("issue_type")
        if isinstance(issue_type, str) and issue_type:
            normalized["issue_type"] = issue_type
        table_label = item.get("table_label")
        if isinstance(table_label, str) and table_label.strip():
            normalized["table_label"] = table_label.strip()
        page_number = item.get("page_number")
        if page_number is not None:
            try:
                normalized["page_number"] = int(page_number)
            except (TypeError, ValueError):
                pass
        if normalized:
            findings.append(normalized)
    return findings


def _structured_table_findings_from_targets(repair_targets: Iterable[Any] | None) -> list[dict[str, Any]]:
    """route/repair가 전달한 structured table target을 visual repair 입력으로 정규화한다."""
    if repair_targets is None:
        return []
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None]] = set()
    for item in repair_targets:
        if getattr(item, "target_kind", None) != "table":
            continue
        issue_type = getattr(item, "issue_type", None)
        if not isinstance(issue_type, str) or not issue_type.strip():
            continue
        table_label = getattr(item, "table_label", None)
        normalized_label = table_label.strip() if isinstance(table_label, str) and table_label.strip() else None
        raw_page_number = getattr(item, "page_number", None)
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


def _prefers_html_table_output(
    issue_types: Iterable[str],
    candidate_metadata: dict[str, Any] | None = None,
) -> bool:
    if any(issue_type in {TABLE_ISSUE_MERGED_CELL_LOSS, TABLE_ISSUE_SPLIT_MULTIPAGE} for issue_type in issue_types):
        return True
    metadata = candidate_metadata or {}
    support_metadata = metadata.get("support_parser_metadata")
    metadata_views = [metadata]
    if isinstance(support_metadata, dict):
        metadata_views.extend(value for value in support_metadata.values() if isinstance(value, dict))
    for metadata_view in metadata_views:
        if str(metadata_view.get("table_format") or "").lower() == "html":
            return True
        table_regions = metadata_view.get("table_regions")
        if not isinstance(table_regions, list):
            continue
        if any(
            isinstance(region, dict)
            and (
                region.get("continued_from_page") is not None
                or region.get("extraction_mode") == "reference"
            )
            for region in table_regions
        ):
            return True
    return False


def _candidate_table_regions(candidate_metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    metadata = candidate_metadata or {}
    metadata_views = [metadata]
    support_metadata = metadata.get("support_parser_metadata")
    if isinstance(support_metadata, dict):
        metadata_views.extend(value for value in support_metadata.values() if isinstance(value, dict))

    regions: list[dict[str, Any]] = []
    for metadata_view in metadata_views:
        table_regions = metadata_view.get("table_regions")
        if not isinstance(table_regions, list):
            continue
        regions.extend(region for region in table_regions if isinstance(region, dict))
    return regions


def _candidate_table_label_pages(candidate_metadata: dict[str, Any] | None = None) -> dict[str, int]:
    metadata = candidate_metadata or {}
    metadata_views = [metadata]
    support_metadata = metadata.get("support_parser_metadata")
    if isinstance(support_metadata, dict):
        metadata_views.extend(value for value in support_metadata.values() if isinstance(value, dict))

    label_pages: dict[str, int] = {}
    for metadata_view in metadata_views:
        raw_mapping = metadata_view.get("table_label_pages")
        if not isinstance(raw_mapping, dict):
            continue
        for key, value in raw_mapping.items():
            try:
                page_number = int(value)
            except (TypeError, ValueError):
                continue
            label_pages.setdefault(str(key), page_number)
    return label_pages


def _candidate_table_label_positions(candidate_metadata: dict[str, Any] | None = None) -> dict[str, dict[str, int]]:
    metadata = candidate_metadata or {}
    metadata_views = [metadata]
    support_metadata = metadata.get("support_parser_metadata")
    if isinstance(support_metadata, dict):
        metadata_views.extend(value for value in support_metadata.values() if isinstance(value, dict))

    positions: dict[str, dict[str, int]] = {}
    for metadata_view in metadata_views:
        raw_mapping = metadata_view.get("table_label_positions")
        if not isinstance(raw_mapping, dict):
            continue
        for key, value in raw_mapping.items():
            if not isinstance(value, dict):
                continue
            page = value.get("page")
            region_index = value.get("region_index")
            global_index = value.get("global_index")
            if not isinstance(page, int) or not isinstance(region_index, int) or not isinstance(global_index, int):
                continue
            positions.setdefault(str(key), {"page": page, "region_index": region_index, "global_index": global_index})
    return positions


def _candidate_table_slots(candidate_metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    metadata = candidate_metadata or {}
    raw_slots = metadata.get("table_slots")
    if not isinstance(raw_slots, list):
        return []
    return [slot for slot in raw_slots if isinstance(slot, dict)]


def _resolve_table_slot(
    table_label: str,
    candidate_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    table_slots = _candidate_table_slots(candidate_metadata)
    if not table_slots:
        return None

    page_number, table_index = _page_table_selector_from_label(table_label)
    if page_number is not None:
        for slot in table_slots:
            if slot.get("page") == page_number and (table_index is None or slot.get("region_index") == table_index):
                return slot

    normalized_label = _normalize_table_label(table_label)
    position = _resolve_table_position_from_metadata(table_label, candidate_metadata)
    for slot in table_slots:
        slot_label = str(slot.get("label") or "")
        if slot_label == table_label or slot_label == normalized_label:
            return slot
        if position is not None and slot.get("global_index") == position.get("global_index"):
            return slot
    return None


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


def _has_ambiguous_page_scoped_table_regions(
    page_number: int,
    candidate_metadata: dict[str, Any] | None = None,
) -> bool:
    page_regions = [
        region
        for region in _candidate_table_regions(candidate_metadata)
        if region.get("page") == page_number
    ]
    return len(page_regions) > 1


def _resolve_page_number_from_metadata(table_label: str, candidate_metadata: dict[str, Any] | None = None) -> int | None:
    label_pages = _candidate_table_label_pages(candidate_metadata)
    normalized_label = _normalize_table_label(table_label)
    label_number = _label_number(normalized_label)
    for key in (normalized_label, normalized_label.replace(" ", ""), label_number or ""):
        if key and key in label_pages:
            return label_pages[key]
    return None


def _is_valid_page_number(page_number: int | None, source: DocumentSource) -> bool:
    if not isinstance(page_number, int):
        return False
    if page_number < 1:
        return False
    if source.page_count is not None and page_number > source.page_count:
        return False
    return True


def _resolve_structured_finding_page_number(
    source: DocumentSource,
    table_label: str | None,
    page_number: int | None,
    candidate_metadata: dict[str, Any] | None,
    find_page_number,
) -> int | None:
    """structured finding의 page 후보를 명시적 우선순위 규칙으로 보정한다.

    raw page_number가 유효하면 그대로 사용하고, 없거나 범위를 벗어난 경우에만
    metadata와 실제 PDF label 검색 결과로 보정한다.
    """
    if _is_valid_page_number(page_number, source):
        return page_number
    normalized_label = table_label.strip() if isinstance(table_label, str) and table_label.strip() else None
    if normalized_label:
        metadata_page = _resolve_page_number_from_metadata(normalized_label, candidate_metadata)
        if _is_valid_page_number(metadata_page, source):
            return metadata_page
        found_page = find_page_number(source.path, normalized_label)
        if _is_valid_page_number(found_page, source):
            return found_page
    return None


def _resolve_table_position_from_metadata(
    table_label: str,
    candidate_metadata: dict[str, Any] | None = None,
) -> dict[str, int] | None:
    positions = _candidate_table_label_positions(candidate_metadata)
    normalized_label = _normalize_table_label(table_label)
    label_number = _label_number(normalized_label)
    for key in (normalized_label, normalized_label.replace(" ", ""), label_number or ""):
        if key and key in positions:
            return positions[key]
    return None


def _sorted_page_regions(page_number: int, candidate_metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    page_regions = [
        region
        for region in _candidate_table_regions(candidate_metadata)
        if region.get("page") == page_number
    ]
    return sorted(
        page_regions,
        key=lambda region: (
            float((region.get("bbox") or [0.0, 0.0, 0.0, 0.0])[1]),
            float((region.get("bbox") or [0.0, 0.0, 0.0, 0.0])[0]),
        ),
    )


def _region_backed_fallback_tasks(
    candidate_metadata: dict[str, Any] | None,
    issue_types: tuple[str, ...],
    preferred_output_format: str,
    max_tasks: int,
) -> list[VisualRepairTask]:
    trouble_regions = [
        region
        for region in _candidate_table_regions(candidate_metadata)
        if region.get("extraction_mode") == "reference" or region.get("continued_from_page") is not None
    ]
    if not trouble_regions:
        return []

    tasks: list[VisualRepairTask] = []
    trouble_keys = {
        (
            int(region.get("page") or 0),
            tuple(float(value) for value in (region.get("bbox") or [])),
            str(region.get("table_id") or ""),
        )
        for region in trouble_regions
    }
    for page_number in sorted({int(region.get("page") or 0) for region in trouble_regions if region.get("page")}):
        page_regions = _sorted_page_regions(page_number, candidate_metadata)
        for region_index, region in enumerate(page_regions, start=1):
            region_key = (
                int(region.get("page") or 0),
                tuple(float(value) for value in (region.get("bbox") or [])),
                str(region.get("table_id") or ""),
            )
            if region_key not in trouble_keys:
                continue
            tasks.append(
                VisualRepairTask(
                    task_id=f"visual-region-table-{page_number}-{region_index}",
                    table_label=_indexed_page_scoped_table_label(page_number, region_index),
                    page_number=page_number,
                    issue_types=issue_types,
                    preferred_output_format=preferred_output_format,
                )
            )
            if len(tasks) >= max_tasks:
                return tasks
    return tasks


def _task_issue_types(metrics: EvaluationMetrics) -> tuple[str, ...]:
    return tuple(metrics.table_issues)


def _post_response(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with tracing_context(enabled=False):
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _extract_response_text(response_payload: dict[str, Any]) -> str:
    output_items = response_payload.get("output") or []
    text_parts: list[str] = []
    for item in output_items:
        if item.get("type") != "message":
            continue
        for content_item in item.get("content") or []:
            if content_item.get("type") == "output_text":
                text = content_item.get("text")
                if text:
                    text_parts.append(str(text))
    if text_parts:
        return "\n".join(text_parts)
    raise ValueError("Visual repair response did not include output_text content.")


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


def _parse_recovery_payload(response_text: str, table_label: str, page_number: int) -> dict[str, Any]:
    normalized_text = response_text.strip()
    if normalized_text.startswith("```"):
        normalized_text = re.sub(r"^```(?:json)?\s*", "", normalized_text)
        normalized_text = re.sub(r"\s*```$", "", normalized_text)
    try:
        payload = json.loads(normalized_text)
    except json.JSONDecodeError:
        markup = _extract_markdown_table(response_text) or _extract_html_table(response_text)
        if not markup:
            raise
        return {
            "table_label": table_label,
            "page_number": page_number,
            "confidence": 0.7,
            "markdown": markup,
            "notes": ["Parsed table markup fallback from a non-JSON vision response."],
        }
    if not isinstance(payload, dict):
        return payload

    normalized_payload = dict(payload)
    if not normalized_payload.get("markdown"):
        fallback_markup = normalized_payload.get("table") or normalized_payload.get("html")
        if fallback_markup:
            normalized_payload["markdown"] = fallback_markup
            normalized_payload.setdefault("notes", []).append(
                "Normalized table markup from a non-standard vision response field."
            )
    if normalized_payload.get("markdown"):
        normalized_payload.setdefault("table_label", table_label)
        normalized_payload.setdefault("page_number", page_number)
        normalized_payload.setdefault("confidence", 0.7)
    return normalized_payload


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


def _estimate_explicit_label_priority(
    table_label: str,
    content: str,
    issue_types: Iterable[str],
    judge_issues: Iterable[str] = (),
    candidate_metadata: dict[str, Any] | None = None,
) -> int:
    score = 0
    excerpt = _candidate_excerpt(content, table_label, context_lines=10)
    slot = _resolve_table_slot(table_label, candidate_metadata)

    if excerpt:
        score += 1
        excerpt_lines = [line for line in excerpt.splitlines() if line.strip()]
        flattened_lines = sum(1 for line in excerpt_lines if _looks_like_plain_text_table_line(line))
        score += min(flattened_lines, 4) * 2
        if "<!-- table-slot:" in excerpt:
            score -= 2
    if slot is not None:
        if str(slot.get("original_text") or "").strip():
            score += 4
        else:
            score -= 2
    if TABLE_ISSUE_NUMERIC_TOKEN_BREAK in issue_types:
        score += 3
    if TABLE_ISSUE_MISSING_HEADER in issue_types:
        score += 2
    if TABLE_ISSUE_SPLIT_MULTIPAGE in issue_types:
        score += 1
    if TABLE_ISSUE_MERGED_CELL_LOSS in issue_types:
        score += 1
    if TABLE_ISSUE_TEXT_DUPLICATION in issue_types:
        score += 1
    normalized_label = _normalize_table_label(table_label)
    related_issue_text = " ".join(
        issue for issue in judge_issues if _normalize_table_label(issue).find(normalized_label) != -1 or normalized_label in issue
    ).lower()
    if "table-slot" in related_issue_text:
        score -= 4
    for keyword in ("행/열", "붕괴", "누락", "뒤섞", "깨짐", "missing", "broken", "collapsed"):
        if keyword in related_issue_text:
            score += 2
    return score


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


def _issue_specific_prompt_guidance(issue_types: Iterable[str], preferred_output_format: str) -> list[str]:
    guidance: list[str] = []
    normalized_issue_types = set(issue_types)
    if TABLE_ISSUE_SPLIT_MULTIPAGE in normalized_issue_types:
        guidance.append(
            "This table may continue across pages. Preserve continuation rows, repeat the header only when the page image clearly shows it, and do not truncate the tail rows."
        )
    if TABLE_ISSUE_MERGED_CELL_LOSS in normalized_issue_types:
        guidance.append(
            "The table likely contains merged cells. Preserve rowspan/colspan structure when visible, and prefer a complete HTML <table> block if merged headers or grouped cells are present."
        )
    if TABLE_ISSUE_MISSING_HEADER in normalized_issue_types:
        guidance.append(
            "Recover the header row carefully. Use only visibly legible header cells from the image; if a header is unreadable, leave that header cell blank rather than inventing one."
        )
    if TABLE_ISSUE_NUMERIC_TOKEN_BREAK in normalized_issue_types:
        guidance.append(
            "Numbers or units may be broken. Preserve decimal points, commas, minus signs, percent signs, and Korean/metric units exactly as shown in the image."
        )
    if TABLE_ISSUE_TEXT_DUPLICATION in normalized_issue_types:
        guidance.append(
            "The parsed output may contain duplicated table text. Reconstruct one clean table only and exclude repeated lines outside the target table."
        )
    if preferred_output_format == "html":
        guidance.append(
            "Return HTML when that is the safer way to preserve header groups, merged cells, or continuation structure."
        )
    return guidance


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


def replace_table_block(
    content: str,
    table_label: str,
    markdown: str,
    *,
    candidate_metadata: dict[str, Any] | None = None,
) -> str:
    slot = _resolve_table_slot(table_label, candidate_metadata)
    if slot is not None:
        placeholder = str(slot.get("placeholder") or "")
        if placeholder:
            transformed = _replace_table_slot_placeholder(content, placeholder, markdown)
            if transformed != content:
                return transformed

    normalized_label = _normalize_table_label(table_label)
    label_number = _label_number(normalized_label)
    if label_number is None:
        return content

    markdown_lines = [line.rstrip() for line in markdown.splitlines() if line.strip()]
    if len(markdown_lines) < 2:
        return content

    lines = content.splitlines()
    label_pattern = re.compile(rf"(?:{_TABLE_PREFIX}\s*)?{re.escape(label_number)}")
    label_index: int | None = None
    for index, line in enumerate(lines):
        if label_pattern.search(line):
            label_index = index
            break
    if label_index is None:
        position = _resolve_table_position_from_metadata(table_label, candidate_metadata)
        if position is None:
            return content
        return replace_table_block_by_global_index(content, position["global_index"], markdown)

    start = label_index + 1
    while start < len(lines) and not lines[start].strip():
        start += 1
    relaxed_bounds = _scan_plain_text_table_block(lines, start, len(lines)) if start < len(lines) else None
    if relaxed_bounds is not None:
        start, end = relaxed_bounds
    else:
        end = start
        while end < len(lines):
            stripped = lines[end].strip()
            if end > start and (_HEADING_RE.match(stripped) or _TABLE_CAPTION_RE.match(stripped)):
                break
            if not stripped and end > start:
                break
            end += 1

    replacement = list(lines[:start])
    if start < len(lines) and lines[start].strip():
        replacement.append("")
    replacement.extend(markdown_lines)
    if end < len(lines) and lines[end].strip():
        replacement.append("")
    replacement.extend(lines[end:])
    normalized = "\n".join(replacement)
    if content.endswith("\n"):
        normalized += "\n"
    return normalized


class OpenAIVisualTableRecoverer:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 90.0,
        max_tables_per_round: int = 1,
        min_confidence: float = 0.45,
        detection_provider: str = "pymupdf",
        crop_padding: float = 8.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_tables_per_round = max_tables_per_round
        self._min_confidence = min_confidence
        self._detection_provider = detection_provider
        self._crop_padding = crop_padding

    def repair(
        self,
        source: DocumentSource,
        content: str,
        metrics: EvaluationMetrics,
        candidate_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, list[RepairAction]]:
        try:
            tasks = self.plan_tasks(source, content, metrics, candidate_metadata=candidate_metadata)
        except TypeError:
            tasks = self.plan_tasks(source, content, metrics)
        updated = content
        actions: list[RepairAction] = []
        for task in tasks[: self._max_tables_per_round]:
            try:
                recovery = self.recover_task(source, updated, task)
            except Exception:
                continue
            if recovery is None:
                continue
            if recovery.confidence < self._min_confidence or not recovery.markdown.strip():
                continue
            recovered_markdown = _normalize_recovered_table_markup(recovery.markdown)
            if not recovered_markdown:
                continue
            transformed = replace_table_block(
                updated,
                task.table_label,
                recovered_markdown,
                candidate_metadata=candidate_metadata,
            )
            if transformed == updated and task.table_label.startswith(_PAGE_SCOPED_TABLE_PREFIX):
                _page_number, table_index = _page_table_selector_from_label(task.table_label)
                transformed = replace_page_table_block(updated, task.page_number, recovered_markdown, table_index=table_index)
            if transformed == updated:
                continue
            note_suffix = ""
            if recovery.notes:
                note_suffix = f" Notes: {'; '.join(recovery.notes[:2])}"
            crop_suffix = ""
            if recovery.bbox is not None:
                crop_suffix = f" Crop: {recovery.crop_method} bbox={recovery.bbox}."
            actions.append(
                RepairAction(
                    action_name="recover_table_from_pdf_image",
                    description=(
                        f"Recover {recovery.table_label} from the source PDF page image and replace the broken parsed block."
                        f"{crop_suffix}"
                        f"{note_suffix}"
                    ),
                    before_excerpt=_candidate_excerpt(updated, task.table_label) or updated[:120].strip(),
                    after_excerpt=_candidate_excerpt(transformed, task.table_label) or transformed[:120].strip(),
                    issue_type="table_visual_recovery",
                    route_name="recover_tables_from_pdf_image",
                )
            )
            updated = transformed
        return updated, actions

    def plan_tasks(
        self,
        source: DocumentSource,
        content: str,
        metrics: EvaluationMetrics,
        candidate_metadata: dict[str, Any] | None = None,
        max_tasks: int | None = None,
        repair_targets: Iterable[Any] | None = None,
    ) -> list[VisualRepairTask]:
        if not _is_pdf_source(source) or not metrics.table_issues:
            return []
        effective_max_tasks = max_tasks if max_tasks is not None else self._max_tables_per_round
        tasks: list[VisualRepairTask] = []
        preferred_output_format = (
            "html" if _prefers_html_table_output(metrics.table_issues, candidate_metadata) else "markdown"
        )
        issue_types = _task_issue_types(metrics)
        judge_result = metrics.judge_result
        routed_findings = _structured_table_findings_from_targets(repair_targets)
        if routed_findings:
            for index, finding in enumerate(routed_findings[:effective_max_tasks]):
                table_label = str(finding.get("table_label") or "").strip()
                resolved_page_number = _resolve_structured_finding_page_number(
                    source,
                    table_label,
                    finding.get("page_number"),
                    candidate_metadata,
                    self._find_page_number,
                )
                if resolved_page_number is None:
                    continue
                task_label = table_label or _page_scoped_table_label(resolved_page_number)
                tasks.append(
                    VisualRepairTask(
                        task_id=f"visual-target-table-{resolved_page_number}-{index}",
                        table_label=task_label,
                        page_number=resolved_page_number,
                        issue_types=issue_types,
                        preferred_output_format=(
                            "html"
                            if _prefers_html_table_output(issue_types, candidate_metadata)
                            else preferred_output_format
                        ),
                    )
                )
            if tasks:
                return tasks
        if judge_result is not None:
            structured_findings = _structured_table_findings(metrics)
            if structured_findings:
                for index, finding in enumerate(structured_findings[:effective_max_tasks]):
                    table_label = str(finding.get("table_label") or "").strip()
                    resolved_page_number = _resolve_structured_finding_page_number(
                        source,
                        table_label,
                        finding.get("page_number"),
                        candidate_metadata,
                        self._find_page_number,
                    )
                    if resolved_page_number is None:
                        continue
                    task_label = table_label or _page_scoped_table_label(resolved_page_number)
                    tasks.append(
                        VisualRepairTask(
                            task_id=f"visual-structured-table-{resolved_page_number}-{index}",
                            table_label=task_label,
                            page_number=resolved_page_number,
                            issue_types=issue_types,
                            preferred_output_format=(
                                "html"
                                if _prefers_html_table_output(issue_types, candidate_metadata)
                                else preferred_output_format
                            ),
                        )
                    )
                if tasks:
                    return tasks

            explicit_labels = extract_table_labels(judge_result.issues)
            indexed_labels = list(enumerate(explicit_labels))
            indexed_labels.sort(
                key=lambda item: (
                    -_estimate_explicit_label_priority(
                        item[1],
                        content,
                        issue_types,
                        judge_result.issues,
                        candidate_metadata,
                    ),
                    item[0],
                )
            )
            explicit_labels = [label for _index, label in indexed_labels]
            for index, table_label in enumerate(explicit_labels[:effective_max_tasks]):
                page_number = _resolve_page_number_from_metadata(table_label, candidate_metadata) or self._find_page_number(
                    source.path,
                    table_label,
                )
                if page_number is None:
                    continue
                tasks.append(
                    VisualRepairTask(
                        task_id=f"visual-table-{page_number}-{index}",
                        table_label=table_label,
                        page_number=page_number,
                        issue_types=issue_types,
                        preferred_output_format=(
                            "html" if _prefers_html_table_output(issue_types, candidate_metadata) else preferred_output_format
                        ),
                    )
                )
            if tasks:
                return tasks

            if metrics.table_preservation < 0.9:
                task_index = 0
                for page_number in extract_issue_page_numbers(judge_result.issues):
                    if task_index >= effective_max_tasks:
                        break
                    page_regions = _sorted_page_regions(page_number, candidate_metadata)
                    if page_regions:
                        for region_index, _region in enumerate(page_regions, start=1):
                            tasks.append(
                                VisualRepairTask(
                                    task_id=f"visual-page-table-{page_number}-{region_index}",
                                    table_label=_indexed_page_scoped_table_label(page_number, region_index),
                                    page_number=page_number,
                                    issue_types=issue_types,
                                    preferred_output_format=(
                                        "html"
                                        if _prefers_html_table_output(issue_types, candidate_metadata)
                                        else preferred_output_format
                                    ),
                                )
                            )
                            task_index += 1
                            if task_index >= effective_max_tasks:
                                break
                        continue
                    tasks.append(
                        VisualRepairTask(
                            task_id=f"visual-page-table-{page_number}-{task_index}",
                            table_label=_page_scoped_table_label(page_number),
                            page_number=page_number,
                            issue_types=issue_types,
                            preferred_output_format=(
                                "html" if _prefers_html_table_output(issue_types, candidate_metadata) else preferred_output_format
                            ),
                        )
                    )
                    task_index += 1
                if tasks:
                    return tasks

        return _region_backed_fallback_tasks(
            candidate_metadata,
            issue_types,
            preferred_output_format=(
                "html" if _prefers_html_table_output(issue_types, candidate_metadata) else preferred_output_format
            ),
            max_tasks=effective_max_tasks,
        )

    def recover_task(
        self,
        source: DocumentSource,
        candidate_text: str,
        task: VisualRepairTask,
    ) -> VisualTableRecovery | None:
        if not _is_pdf_source(source):
            return None
        return self._recover_single_table(
            source.path,
            candidate_text,
            task.table_label,
            page_number=task.page_number,
            issue_types=task.issue_types,
            preferred_output_format=task.preferred_output_format,
        )

    def _recover_single_table(
        self,
        pdf_path: Path,
        candidate_text: str,
        table_label: str,
        page_number: int | None = None,
        issue_types: Iterable[str] = (),
        preferred_output_format: str = "markdown",
    ) -> VisualTableRecovery | None:
        page_number = page_number or self._find_page_number(pdf_path, table_label)
        if page_number is None:
            return None
        image_url, crop = self._render_table_region_data_url(pdf_path, page_number, table_label)
        prompt = self._build_prompt(
            table_label,
            page_number,
            candidate_text,
            issue_types=issue_types,
            preferred_output_format=preferred_output_format,
        )
        response_payload = _post_response(
            url=f"{self._base_url}/responses",
            api_key=self._api_key,
            payload={
                "model": self._model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_url, "detail": "high"},
                        ],
                    }
                ],
            },
            timeout_seconds=self._timeout_seconds,
        )
        payload = _parse_recovery_payload(_extract_response_text(response_payload), table_label, page_number)
        markdown = _normalize_recovered_table_markup(str(payload.get("markdown") or "").strip())
        notes = payload.get("notes") or []
        if isinstance(notes, str):
            notes = [notes]
        elif isinstance(notes, list):
            notes = [str(item) for item in notes]
        else:
            notes = [str(notes)]
        return VisualTableRecovery(
            table_label=_normalize_table_label(str(payload.get("table_label") or table_label)),
            page_number=int(payload.get("page_number") or page_number),
            confidence=float(payload.get("confidence") or 0.0),
            markdown=markdown,
            notes=notes,
            crop_method=crop.method,
            bbox=crop.bbox,
        )

    def _build_prompt(
        self,
        table_label: str,
        page_number: int,
        candidate_text: str,
        *,
        issue_types: Iterable[str] = (),
        preferred_output_format: str = "markdown",
    ) -> str:
        excerpt = _candidate_excerpt(candidate_text, table_label)
        target_descriptor = table_label
        if _page_number_from_scoped_label(table_label) is not None:
            target_descriptor = f"primary broken table on page {page_number}"
        normalized_issue_types = tuple(issue_types)
        rendered_issue_types = ", ".join(normalized_issue_types) if normalized_issue_types else "none provided"
        issue_guidance = _issue_specific_prompt_guidance(normalized_issue_types, preferred_output_format)
        issue_guidance_block = "\n".join(f"- {line}" for line in issue_guidance) or "- No extra issue-specific guidance."
        return (
            f"Target table label: {target_descriptor}\n"
            f"Source page number: {page_number}\n\n"
            f"Known table issues: {rendered_issue_types}\n"
            f"Preferred output format: {preferred_output_format}\n\n"
            f"Issue-specific repair guidance:\n{issue_guidance_block}\n\n"
            "Reconstruct only the target table from the page image.\n"
            "Return only the reconstructed table markup in the JSON field.\n"
            "If html is preferred, use a complete <table> block. Otherwise use markdown with a header row.\n"
            "If a cell is unreadable, leave it blank instead of inventing text.\n\n"
            f"Current parser excerpt around {target_descriptor}:\n{excerpt or '[no local excerpt found]'}"
        )

    def _find_page_number(self, pdf_path: Path, table_label: str) -> int | None:
        label_number = _label_number(table_label)
        target_variants = {table_label, table_label.replace(" ", "")}
        if label_number is not None:
            target_variants.add(label_number)
        with fitz.open(pdf_path) as document:
            for page_index in range(document.page_count):
                page_text = document.load_page(page_index).get_text("text")
                if any(variant in page_text for variant in target_variants):
                    return page_index + 1
        return None

    def _render_table_region_data_url(self, pdf_path: Path, page_number: int, table_label: str) -> tuple[str, TableCrop]:
        with fitz.open(pdf_path) as document:
            page = document.load_page(page_number - 1)
            crop = self._build_table_crop(page, page_number, table_label)
            clip = crop.clip
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), clip=clip, alpha=False)
        encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
        return f"data:image/png;base64,{encoded}", crop

    def _build_table_crop(self, page, page_number: int, table_label: str) -> TableCrop:
        scoped_page_number, scoped_table_index = _page_table_selector_from_label(table_label)
        if scoped_page_number is not None:
            try:
                table_finder = page.find_tables() if hasattr(page, "find_tables") else None
            except Exception:
                table_finder = None
            table_rects = [fitz.Rect(table.bbox) for table in getattr(table_finder, "tables", []) if getattr(table, "bbox", None)]
            if table_rects:
                sorted_rects = sorted(table_rects, key=lambda rect: (rect.y0, rect.x0))
                if scoped_table_index is not None and 1 <= scoped_table_index <= len(sorted_rects):
                    primary_rect = sorted_rects[scoped_table_index - 1]
                else:
                    primary_rect = max(sorted_rects, key=lambda rect: rect.get_area())
                expanded = fitz.Rect(
                    max(page.rect.x0, primary_rect.x0 - self._crop_padding),
                    max(page.rect.y0, primary_rect.y0 - max(self._crop_padding, 80.0)),
                    min(page.rect.x1, primary_rect.x1 + self._crop_padding),
                    min(page.rect.y1, primary_rect.y1 + max(self._crop_padding, 120.0)),
                )
                return TableCrop(
                    page_number=page_number,
                    clip=expanded,
                    method="page-table-window",
                    bbox=self._rect_tuple(primary_rect),
                )
            return TableCrop(page_number=page_number, clip=page.rect, method="full-page", bbox=self._rect_tuple(page.rect))
        label_anchor = self._find_label_anchor(page, table_label)
        table_rect = self._detect_table_rect(page, label_anchor)
        if table_rect is not None:
            clip = self._expand_rect(table_rect, page.rect, self._crop_padding)
            return TableCrop(
                page_number=page_number,
                clip=clip,
                method=self._detection_provider,
                bbox=self._rect_tuple(table_rect),
            )
        if label_anchor is not None:
            top = max(0.0, label_anchor.y0 - 24.0)
            bottom = min(page.rect.height, top + max(420.0, page.rect.height * 0.55))
            clip = fitz.Rect(0.0, top, page.rect.width, bottom)
            return TableCrop(
                page_number=page_number,
                clip=clip,
                method="label-window",
                bbox=self._rect_tuple(clip),
            )
        return TableCrop(page_number=page_number, clip=page.rect, method="full-page", bbox=self._rect_tuple(page.rect))

    def _find_label_anchor(self, page, table_label: str) -> fitz.Rect | None:
        matches = page.search_for(table_label)
        if not matches and " " in table_label:
            matches = page.search_for(table_label.replace(" ", ""))
        if not matches:
            label_number = _label_number(table_label)
            if label_number is not None:
                matches = page.search_for(label_number)
        return matches[0] if matches else None

    def _detect_table_rect(self, page, label_anchor: fitz.Rect | None) -> fitz.Rect | None:
        if self._detection_provider != "pymupdf" or not hasattr(page, "find_tables"):
            return None
        try:
            table_finder = page.find_tables()
        except Exception:
            return None
        table_rects = [fitz.Rect(table.bbox) for table in getattr(table_finder, "tables", []) if getattr(table, "bbox", None)]
        if not table_rects:
            return None
        if label_anchor is None:
            return max(table_rects, key=lambda rect: rect.get_area())
        below_or_overlapping = [
            rect for rect in table_rects if rect.y1 >= label_anchor.y0 and rect.y0 <= label_anchor.y1 + 260.0
        ]
        candidates = below_or_overlapping or table_rects
        return min(
            candidates,
            key=lambda rect: (
                0 if rect.y0 >= label_anchor.y0 else 1,
                abs(rect.y0 - label_anchor.y1),
                abs(rect.x0 - label_anchor.x0),
            ),
        )

    def _expand_rect(self, rect: fitz.Rect, page_rect: fitz.Rect, padding: float) -> fitz.Rect:
        return fitz.Rect(
            max(page_rect.x0, rect.x0 - padding),
            max(page_rect.y0, rect.y0 - padding),
            min(page_rect.x1, rect.x1 + padding),
            min(page_rect.y1, rect.y1 + padding),
        )

    def _rect_tuple(self, rect: fitz.Rect) -> tuple[float, float, float, float]:
        return (round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2))


def build_default_visual_table_recoverer(config: WorkflowConfig) -> OpenAIVisualTableRecoverer | None:
    if not config.visual_table_recovery_enabled:
        return None
    if not config.visual_table_recovery_model or not config.judge_api_key:
        return None
    return OpenAIVisualTableRecoverer(
        model=config.visual_table_recovery_model,
        api_key=config.judge_api_key,
        base_url=config.judge_base_url,
        timeout_seconds=config.visual_table_recovery_timeout_seconds,
        max_tables_per_round=config.visual_table_recovery_max_tables,
        detection_provider=config.visual_table_detection_provider,
        crop_padding=config.visual_table_crop_padding,
    )
