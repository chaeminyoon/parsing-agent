"""visual repair 태스크 구성 — 파인딩/라벨/candidate 메타데이터 해석.

visual_repair.py 분할의 중간층: visual_tables의 프리미티브를 써서 무엇을 고칠지
결정한다. 실제 LLM 호출과 패치 적용은 visual_repair(최상층)에 있다.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable

import fitz


from parsing_agent.evaluation import (
    TABLE_ISSUE_MERGED_CELL_LOSS,
    TABLE_ISSUE_NUMERIC_TOKEN_BREAK,
    TABLE_ISSUE_SPLIT_MULTIPAGE,
)
from parsing_agent.models import DocumentSource, EvaluationMetrics

from parsing_agent.visual_tables import (  # noqa: F401
    _NUMBER_RE,
    _PAGE_REF_RE,
    _TABLE_LABEL_RE,
    _TABLE_PREFIX,
    _extract_html_table,
    _extract_markdown_table,
    _indexed_page_scoped_table_label,
    _label_number,
    _looks_like_recovered_table,
    _normalize_table_label,
    _page_table_selector_from_label,
)


@dataclass(frozen=True, slots=True)
class VisualTableRecovery:
    table_label: str
    page_number: int
    confidence: float
    markdown: str
    notes: list[str]
    crop_method: str
    bbox: tuple[float, float, float, float] | None
    # 재구성된 셀들이 crop 영역의 실제 텍스트에 존재하는 비율.
    # 디지털 PDF에서만 계산되고, 스캔 페이지(텍스트 없음)는 None.
    grounding: float | None = None


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


def _recovered_table_passes_sanity(
    *,
    recovered_markdown: str,
    issue_types: tuple[str, ...] | list[str],
) -> bool:
    if not _looks_like_recovered_table(recovered_markdown):
        return False
    if TABLE_ISSUE_NUMERIC_TOKEN_BREAK in set(issue_types) and not _NUMBER_RE.search(recovered_markdown):
        return False
    non_empty_cells = [
        cell.strip()
        for line in recovered_markdown.splitlines()
        if "|" in line
        for cell in line.split("|")
        if cell.strip() and set(cell.strip()) - {":", "-"}
    ]
    return len(non_empty_cells) >= 2
