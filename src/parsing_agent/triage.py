from __future__ import annotations

from dataclasses import dataclass

import fitz

from parsing_agent.config import WorkflowConfig
from parsing_agent.filetype import is_pdf_source
from parsing_agent.models import DocumentSource


@dataclass(frozen=True, slots=True)
class TriageDecision:
    complexity: str
    selected_parsers: list[str]
    reasons: list[str]
    sampled_pages: int
    page_count: int | None
    table_hits: int = 0
    image_block_count: int = 0
    text_block_count: int = 0
    multi_column_signals: int = 0


def _configured_pdf_parsers(config: WorkflowConfig, *roles: str) -> list[str]:
    allowed_roles = set(roles)
    return [
        parser_name
        for parser_name in config.parser_names
        if config.pdf_parser_roles.get(parser_name) in allowed_roles
    ]


def _configured_primary_pdf_parsers(config: WorkflowConfig) -> list[str]:
    primary_only = _configured_pdf_parsers(config, "primary")
    if primary_only:
        return primary_only
    return _configured_pdf_parsers(config, "primary", "support") or list(config.parser_names)


def _configured_primary_and_support_pdf_parsers(config: WorkflowConfig) -> list[str]:
    primary_and_support = _configured_pdf_parsers(config, "primary", "support")
    if primary_and_support:
        return primary_and_support
    return _configured_primary_pdf_parsers(config)


def triage_document(source: DocumentSource, config: WorkflowConfig) -> TriageDecision:
    if not config.triage_enabled:
        return TriageDecision(
            complexity="disabled",
            selected_parsers=list(config.parser_names),
            reasons=["triage_disabled"],
            sampled_pages=0,
            page_count=source.page_count,
        )

    if source.media_type.startswith("text/"):
        return TriageDecision(
            complexity="text",
            selected_parsers=["text-fallback"],
            reasons=["text_like_source"],
            sampled_pages=0,
            page_count=source.page_count,
        )

    if not is_pdf_source(source):
        return TriageDecision(
            complexity="unknown",
            selected_parsers=list(config.parser_names),
            reasons=["unsupported_for_triage"],
            sampled_pages=0,
            page_count=source.page_count,
        )

    sample_pages = max(1, min(source.page_count or 1, config.triage_sample_pages))
    table_hits = 0
    image_block_count = 0
    text_block_count = 0
    multi_column_signals = 0
    reasons: list[str] = []

    with fitz.open(source.path) as document:
        for page_index in range(sample_pages):
            page = document.load_page(page_index)
            if hasattr(page, "find_tables"):
                try:
                    table_hits += len(getattr(page.find_tables(), "tables", []))
                except Exception:
                    pass

            blocks = page.get_text("blocks")
            x_positions: set[int] = set()
            for block in blocks:
                x0 = float(block[0])
                block_type = block[6] if len(block) > 6 else 0
                if block_type != 0:
                    image_block_count += 1
                    continue
                if str(block[4]).strip():
                    text_block_count += 1
                    x_positions.add(int(x0 // 100))
            if len(x_positions) >= 2:
                multi_column_signals += 1

    if table_hits:
        reasons.append("sample_pages_show_tables")
    if image_block_count:
        reasons.append("sample_pages_show_images")
    if multi_column_signals:
        reasons.append("sample_pages_show_multi_column_layout")
    if source.page_count and source.page_count >= 12:
        reasons.append("document_has_many_pages")
    if not reasons:
        reasons.append("sample_pages_plain_text_only")

    if "sample_pages_plain_text_only" in reasons and (source.page_count or 0) <= sample_pages:
        return TriageDecision(
            complexity="simple",
            selected_parsers=["source-text"],
            reasons=reasons,
            sampled_pages=sample_pages,
            page_count=source.page_count,
            table_hits=table_hits,
            image_block_count=image_block_count,
            text_block_count=text_block_count,
            multi_column_signals=multi_column_signals,
        )

    selected_pdf_parsers = _configured_primary_pdf_parsers(config)
    if table_hits or image_block_count:
        selected_pdf_parsers = _configured_primary_and_support_pdf_parsers(config)

    if table_hits or image_block_count or multi_column_signals or (source.page_count or 0) >= 12:
        return TriageDecision(
            complexity="complex",
            selected_parsers=selected_pdf_parsers,
            reasons=reasons,
            sampled_pages=sample_pages,
            page_count=source.page_count,
            table_hits=table_hits,
            image_block_count=image_block_count,
            text_block_count=text_block_count,
            multi_column_signals=multi_column_signals,
        )

    return TriageDecision(
        complexity="medium",
        selected_parsers=selected_pdf_parsers,
        reasons=reasons,
        sampled_pages=sample_pages,
        page_count=source.page_count,
        table_hits=table_hits,
        image_block_count=image_block_count,
        text_block_count=text_block_count,
        multi_column_signals=multi_column_signals,
    )
