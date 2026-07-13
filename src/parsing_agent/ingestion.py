from __future__ import annotations

import mimetypes
from pathlib import Path

from pypdf import PdfReader

from parsing_agent.config import WorkflowConfig
from parsing_agent.filetype import is_pdf, is_text_like
from parsing_agent.format_parsers import (
    extract_data_text,
    extract_docx_text,
    extract_html_text,
    extract_odt_text,
    extract_pptx_text,
    extract_xlsx_text,
    extract_xml_text,
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
    # 데이터 포맷은 값 시퀀스가 평가 기준 — 키/태그 반복이 표 접기를 감점하지 않게.
    if suffix in (".json", ".yaml", ".yml"):
        return extract_data_text(path), None
    if suffix == ".xml":
        return extract_xml_text(path), None
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

    diagnostics = estimate_pdf_image_ratio(resolved) if is_pdf(media_type, resolved) else {}

    return DocumentSource(
        path=resolved,
        media_type=media_type,
        size_bytes=resolved.stat().st_size,
        run_id=run_id,
        extracted_text=extracted_text,
        page_count=page_count,
        ocr_metadata=ocr_metadata,
        ocr_artifacts=ocr_artifacts,
        diagnostics=diagnostics,
    )


# 이 비율 이상이 이미지 영역이면 coverage/추출 결과를 저신뢰로 고지한다.
# (coverage의 기준인 추출 텍스트에는 이미지 속 텍스트·그림표가 아예 없다 —
#  골든 파일럿에서 사람 라벨이 확정한 맹점.)
_IMAGE_DIAGNOSTIC_MAX_PAGES = 40


def estimate_pdf_image_ratio(path: Path, max_pages: int = _IMAGE_DIAGNOSTIC_MAX_PAGES) -> dict:
    """PDF의 이미지 블록 면적 비율과 개수를 표본 추정한다. 실패 시 빈 dict."""
    try:
        import fitz

        with fitz.open(path) as document:
            page_total = min(document.page_count, max_pages)
            total_area = 0.0
            image_area = 0.0
            image_blocks = 0
            for index in range(page_total):
                page = document.load_page(index)
                total_area += float(page.rect.get_area())
                try:
                    blocks = page.get_text("dict").get("blocks", [])
                except Exception:  # noqa: BLE001 - 페이지 하나의 실패는 건너뛴다
                    continue
                for block in blocks:
                    if block.get("type") == 1:  # image block
                        x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
                        image_area += max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
                        image_blocks += 1
            if total_area <= 0:
                return {}
            return {
                "image_area_ratio": round(min(image_area / total_area, 1.0), 4),
                "image_block_count": image_blocks,
                "sampled_pages": page_total,
            }
    except Exception:  # noqa: BLE001 - 진단 실패가 인제스천을 막으면 안 된다
        return {}
