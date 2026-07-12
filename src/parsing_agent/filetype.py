"""파일 타입 판별의 단일 소스.

media type과 suffix 기반 판별이 모듈마다 복붙되어 있던 것을 한곳으로 모은다.
새 포맷을 추가할 때 여기의 상수/판별자만 갱신하면 ingestion·parsers·ocr·
evaluation·judge·repair·workflow가 함께 따라온다.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parsing_agent.models import DocumentSource

# 원문을 그대로 텍스트로 읽을 수 있는 suffix (raw-text fallback 대상)
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".html", ".htm", ".xml"}

# OCR이 유일한 텍스트 경로인 이미지 포맷 (로드맵의 OCR 연동 포맷)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}


def is_pdf(media_type: str, path: Path) -> bool:
    return media_type == "application/pdf" or path.suffix.lower() == ".pdf"


def is_pdf_source(source: "DocumentSource") -> bool:
    return is_pdf(source.media_type, source.path)


def is_image(media_type: str, path: Path) -> bool:
    return media_type.startswith("image/") or path.suffix.lower() in IMAGE_SUFFIXES


def is_image_source(source: "DocumentSource") -> bool:
    return is_image(source.media_type, source.path)


def is_text_like(media_type: str, path: Path) -> bool:
    return media_type.startswith("text/") or path.suffix.lower() in TEXT_SUFFIXES


def is_text_like_source(source: "DocumentSource") -> bool:
    return is_text_like(source.media_type, source.path)
