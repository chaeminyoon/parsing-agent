from __future__ import annotations

import mimetypes
from pathlib import Path

from pypdf import PdfReader

from parsing_agent.config import WorkflowConfig
from parsing_agent.filetype import is_pdf, is_text_like
from parsing_agent.format_parsers import (
    extract_docx_text,
    extract_html_text,
    extract_odt_text,
    extract_pptx_text,
    extract_xlsx_text,
)
from parsing_agent.models import DocumentSource
from parsing_agent.ocr import run_ocr, should_run_ocr
from parsing_agent.textutil import read_text_with_fallback


def detect_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def extract_text_from_pdf(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip(), len(pages)


def extract_source_text(path: Path, media_type: str) -> tuple[str | None, int | None]:
    suffix = path.suffix.lower()
    # 바이너리 오피스 포맷: 평가/요약이 쓰는 extracted_text를 평문으로 채운다.
    if suffix == ".docx":
        return extract_docx_text(path), None
    if suffix == ".pptx":
        text, slide_count = extract_pptx_text(path)
        return text, slide_count
    if suffix == ".xlsx":
        return extract_xlsx_text(path), None
    if suffix == ".odt":
        return extract_odt_text(path), None
    # HTML은 마크업이 아니라 가시 텍스트가 평가 기준이 되어야 한다.
    if suffix in (".html", ".htm") or media_type == "text/html":
        return extract_html_text(path), None
    if is_text_like(media_type, path):
        return read_text_with_fallback(path), None
    if is_pdf(media_type, path):
        return extract_text_from_pdf(path)
    return None, None


def build_document_source(
    path: Path,
    run_id: str,
    config: WorkflowConfig | None = None,
    artifact_dir: Path | None = None,
) -> DocumentSource:
    resolved = path.resolve()
    media_type = detect_media_type(resolved)
    extracted_text, page_count = extract_source_text(resolved, media_type)
    workflow_config = config or WorkflowConfig()
    ocr_metadata = {
        "enabled": workflow_config.ocr_enabled,
        "applied": False,
        "provider": workflow_config.ocr_provider,
        "reason": "not_required",
        "input_text_characters": len(extracted_text or ""),
        "page_count": page_count,
    }
    ocr_artifacts: dict[str, str] = {}

    if should_run_ocr(
        media_type=media_type,
        path=resolved,
        extracted_text=extracted_text,
        config=workflow_config,
    ):
        ocr_result = run_ocr(
            input_path=resolved,
            output_dir=artifact_dir or resolved.parent / f"{resolved.stem}_ocr",
            config=workflow_config,
            page_count=page_count,
            extracted_text=extracted_text,
        )
        ocr_metadata = {
            "enabled": workflow_config.ocr_enabled,
            **ocr_result.metadata,
        }
        ocr_artifacts = {name: str(path) for name, path in ocr_result.artifacts.items()}
        if ocr_result.applied and ocr_result.text:
            extracted_text = ocr_result.text

    return DocumentSource(
        path=resolved,
        media_type=media_type,
        size_bytes=resolved.stat().st_size,
        run_id=run_id,
        extracted_text=extracted_text,
        page_count=page_count,
        ocr_metadata=ocr_metadata,
        ocr_artifacts=ocr_artifacts,
    )
