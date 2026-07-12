"""비전 표 재구성 — OpenAI 비전 호출, crop 전략, 패치 오케스트레이션.

3계층 분할의 최상층: 프리미티브는 visual_tables, 태스크 구성은 visual_tasks에
있고 이 모듈이 둘을 재수출하므로 기존 import 경로는 그대로 동작한다.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable
from urllib import request

import fitz
from langsmith import tracing_context

from parsing_agent.llm_usage import record_llm_call

from parsing_agent.config import WorkflowConfig
from parsing_agent.filetype import is_pdf_source as _is_pdf_source
from parsing_agent.evaluation import (
    TABLE_ISSUE_MERGED_CELL_LOSS,
    TABLE_ISSUE_MISSING_HEADER,
    TABLE_ISSUE_NUMERIC_TOKEN_BREAK,
    TABLE_ISSUE_SPLIT_MULTIPAGE,
    TABLE_ISSUE_TEXT_DUPLICATION,
)
from parsing_agent.models import DocumentSource, EvaluationMetrics, RepairAction

# 분할된 하위 계층 재수출 — 기존 import 경로(from parsing_agent.visual_repair import X) 유지
from parsing_agent.visual_tables import (  # noqa: F401
    _GENERIC_HTML_TAG_RE,
    _HEADING_RE,
    _HTML_TABLE_RE,
    _HTML_TEXT_RE,
    _MARKDOWN_IMAGE_RE,
    _NUMBER_RE,
    _PAGE_REF_RE,
    _PAGE_SCOPED_TABLE_PREFIX,
    _RecoveredHtmlTableParser,
    _TABLE_CAPTION_RE,
    _TABLE_LABEL_RE,
    _TABLE_PREFIX,
    _candidate_excerpt,
    _collapse_header_rows,
    _escape_markdown_cell,
    _expand_recovered_html_rows,
    _extract_html_table,
    _extract_markdown_table,
    _extract_page_section,
    _fill_header_blanks,
    _find_page_table_block_bounds,
    _find_text_label_line_index,
    _has_future_plain_text_table_line,
    _html_table_to_markdown,
    _indexed_page_scoped_table_label,
    _label_number,
    _looks_like_plain_text_table_header_line,
    _looks_like_plain_text_table_line,
    _looks_like_plain_text_table_stub_line,
    _looks_like_recovered_table,
    _normalize_recovered_cell_text,
    _normalize_recovered_table_markup,
    _normalize_table_label,
    _pad_rows,
    _page_number_from_scoped_label,
    _page_scoped_table_label,
    _page_table_selector_from_label,
    _recovery_grounding_ratio,
    _replace_html_tables_with_markdown,
    _replace_table_slot_placeholder,
    _safe_span,
    _scan_plain_text_table_block,
    _table_block_bounds_by_scan,
    _trim_empty_columns,
    insert_table_after_anchor,
    replace_page_table_block,
    replace_table_block_by_global_index,
)
from parsing_agent.visual_tasks import (  # noqa: F401
    TableCrop,
    VisualRepairTask,
    VisualTableRecovery,
    _candidate_table_label_pages,
    _candidate_table_label_positions,
    _candidate_table_regions,
    _candidate_table_slots,
    _has_ambiguous_page_scoped_table_regions,
    _is_valid_page_number,
    _parse_recovery_payload,
    _prefers_html_table_output,
    _recovered_table_passes_sanity,
    _region_backed_fallback_tasks,
    _resolve_page_number_from_metadata,
    _resolve_structured_finding_page_number,
    _resolve_table_position_from_metadata,
    _resolve_table_slot,
    _sorted_page_regions,
    _structured_table_findings,
    _structured_table_findings_from_targets,
    _task_issue_types,
    extract_issue_page_numbers,
    extract_table_labels,
)


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
    started = time.monotonic()
    try:
        with tracing_context(enabled=False):
            with request.urlopen(req, timeout=timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        record_llm_call(
            stage="visual_table_recovery",
            model=payload.get("model"),
            duration_ms=int((time.monotonic() - started) * 1000),
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    record_llm_call(
        stage="visual_table_recovery",
        model=payload.get("model"),
        duration_ms=int((time.monotonic() - started) * 1000),
        ok=True,
        response_payload=response_payload,
    )
    return response_payload


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
        if preferred_output_format != "html":
            guidance.append(
                "When emitting markdown, repeat each vertically merged value in every row it spans instead of leaving blank cells."
            )
    guidance.append(
        "If the cropped region actually contains multiple distinct tables, output them as separate tables with their own header rows instead of one fused table."
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

    markdown_lines = [line.rstrip() for line in markdown.splitlines() if line.strip()]
    if len(markdown_lines) < 2:
        return content

    lines = content.splitlines()
    label_index: int | None = None
    if label_number is not None:
        label_pattern = re.compile(rf"(?:{_TABLE_PREFIX}\s*)?{re.escape(label_number)}")
        for index, line in enumerate(lines):
            if label_pattern.search(line):
                label_index = index
                break
    else:
        # "사후환경영향조사계획 표"처럼 번호 없는 라벨은 라벨 텍스트
        # 자체를 후보 본문에서 찾는다.
        label_index = _find_text_label_line_index(lines, table_label)
    if label_index is None:
        if label_number is None:
            return content
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
            if not _recovered_table_passes_sanity(
                recovered_markdown=recovered_markdown,
                issue_types=task.issue_types,
            ):
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
        if _page_number_from_scoped_label(table_label) is None:
            # judge가 준 페이지 번호를 검증한다. 라벨이 그 페이지에 없으면
            # 재배치하고, 어디에도 없으면 포기한다 — 라벨 없는 페이지에서
            # 아무 표나 crop하면 엉뚱한 표가 라벨 자리에 들어간다.
            page_number = self._resolve_label_page(pdf_path, table_label, page_number)
            if page_number is None:
                return None
        image_url, crop, crop_text = self._render_table_region_data_url(pdf_path, page_number, table_label)
        normalized_issue_types = tuple(issue_types)
        continuation_urls, continuation_text = (
            self._render_continuation_data_urls(pdf_path, page_number)
            if "split_multipage_table" in normalized_issue_types
            else ([], "")
        )
        prompt = self._build_prompt(
            table_label,
            page_number,
            candidate_text,
            issue_types=normalized_issue_types,
            preferred_output_format=preferred_output_format,
            continuation_page_count=len(continuation_urls),
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
                        ]
                        + [
                            {"type": "input_image", "image_url": continuation_url, "detail": "high"}
                            for continuation_url in continuation_urls
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
            grounding=_recovery_grounding_ratio(f"{crop_text}\n{continuation_text}", markdown),
        )

    def _build_prompt(
        self,
        table_label: str,
        page_number: int,
        candidate_text: str,
        *,
        issue_types: Iterable[str] = (),
        preferred_output_format: str = "markdown",
        continuation_page_count: int = 0,
    ) -> str:
        excerpt = _candidate_excerpt(candidate_text, table_label)
        target_descriptor = table_label
        if _page_number_from_scoped_label(table_label) is not None:
            target_descriptor = f"primary broken table on page {page_number}"
        normalized_issue_types = tuple(issue_types)
        rendered_issue_types = ", ".join(normalized_issue_types) if normalized_issue_types else "none provided"
        issue_guidance = _issue_specific_prompt_guidance(normalized_issue_types, preferred_output_format)
        issue_guidance_block = "\n".join(f"- {line}" for line in issue_guidance) or "- No extra issue-specific guidance."
        continuation_note = ""
        if continuation_page_count:
            continuation_note = (
                f"The table continues onto the next page. {continuation_page_count} additional image(s) after "
                "the first show the top of the following page(s) — merge all parts into ONE table, do not repeat "
                "the header rows from continuation pages.\n"
            )
        return (
            f"Target table label: {target_descriptor}\n"
            f"Source page number: {page_number}\n"
            f"{continuation_note}\n"
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
        """라벨이 있는 페이지를 찾는다.

        같은 라벨이 목차와 본문에 모두 등장할 수 있다. 라벨 앵커 아래에
        실제 표가 감지되는 페이지를 우선하고, 없으면 마지막 등장 페이지를
        쓴다 (목차는 앞쪽에 있으므로).
        """
        label_number = _label_number(table_label)
        target_variants = {table_label, table_label.replace(" ", "")}
        if label_number is not None:
            target_variants.add(label_number)
        matches: list[int] = []
        with fitz.open(pdf_path) as document:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_text = page.get_text("text")
                if not any(variant in page_text for variant in target_variants):
                    continue
                matches.append(page_index + 1)
                anchor = self._find_label_anchor(page, table_label)
                if anchor is not None and self._detect_table_rect(page, anchor) is not None:
                    return page_index + 1
        return matches[-1] if matches else None

    def _resolve_label_page(self, pdf_path: Path, table_label: str, claimed_page: int) -> int | None:
        """task가 들고 온 페이지 번호를 검증하고, 틀렸으면 재배치한다.

        judge가 인쇄된 쪽번호를 PDF 페이지 번호로 착각해 보고하는 경우가
        실제로 있다 (표 4.2-2가 4페이지인데 인쇄 쪽번호 44를 반환).
        지정 페이지에 라벨 앵커가 없으면 문서 전체에서 다시 찾고,
        어디에도 없으면 None을 반환해 맹목적인 crop을 막는다.
        """
        with fitz.open(pdf_path) as document:
            if 1 <= claimed_page <= document.page_count:
                page = document.load_page(claimed_page - 1)
                if self._find_label_anchor(page, table_label) is not None:
                    return claimed_page
        return self._find_page_number(pdf_path, table_label)

    def _render_continuation_data_urls(self, pdf_path: Path, page_number: int, max_pages: int = 1) -> tuple[list[str], str]:
        """다중 페이지 표의 이어지는 페이지 상단(60%)을 렌더링한다.

        split_multipage_table 이슈에서 첫 페이지 crop만 보내면 vision 모델이
        표의 뒷부분을 아예 못 보고 재구성하게 된다. 이어지는 페이지의 표는
        관례상 페이지 상단에 위치하므로 상단 60%를 함께 보낸다.
        그라운딩 검증에 쓰도록 해당 영역의 텍스트도 함께 반환한다.
        """
        urls: list[str] = []
        texts: list[str] = []
        with fitz.open(pdf_path) as document:
            for offset in range(1, max_pages + 1):
                page_index = page_number - 1 + offset
                if page_index >= document.page_count:
                    break
                page = document.load_page(page_index)
                clip = fitz.Rect(0.0, 0.0, page.rect.width, page.rect.height * 0.6)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), clip=clip, alpha=False)
                encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
                urls.append(f"data:image/png;base64,{encoded}")
                texts.append(page.get_text("text", clip=clip))
        return urls, "\n".join(texts)

    def _render_table_region_data_url(self, pdf_path: Path, page_number: int, table_label: str) -> tuple[str, TableCrop, str]:
        with fitz.open(pdf_path) as document:
            page = document.load_page(page_number - 1)
            crop = self._build_table_crop(page, page_number, table_label)
            clip = crop.clip
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), clip=clip, alpha=False)
            crop_text = page.get_text("text", clip=clip)
        encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
        return f"data:image/png;base64,{encoded}", crop, crop_text

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
            # 표 감지가 실패한 경우 표의 실제 높이를 모른다. 고정 높이(420px)로
            # 자르면 긴 표의 아래쪽이 잘려나가므로, 라벨부터 페이지 끝까지 잡는다 —
            # 이미지가 커지는 비용이 표가 잘리는 비용보다 싸다.
            top = max(0.0, label_anchor.y0 - 24.0)
            clip = fitz.Rect(0.0, top, page.rect.width, page.rect.height)
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
