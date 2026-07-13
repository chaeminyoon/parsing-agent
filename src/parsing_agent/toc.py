"""PDF 북마크(TOC) 기반 구조 복원 — 문서가 스스로 선언한 목차가 ground truth다.

정부·기관 보고서 PDF는 대부분 북마크를 갖고 있는데 기존 구조 평가는
장/절 번호 정규식 cue에만 의존했다 (영문·비정형 문서에서 실측 0.0).
TOC가 있으면:
- 평가: 후보에 살아남은 TOC 제목 비율이 구조 보존율의 하한이 된다.
- 수리: 후보에 있는 제목은 헤딩으로 승격, 사라진 제목은 페이지 마커
  위치에 복원한다. 재채점-롤백 루프가 안전망.
"""
from __future__ import annotations

import re
from pathlib import Path

import fitz

from parsing_agent.models import DocumentSource

_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_PAGE_MARKER_RE = re.compile(r"^<!-- page (\d+) -->$")


def read_pdf_toc(path: Path | str) -> list[tuple[int, str, int]]:
    """(level, title, page) 목록. 북마크가 없거나 파일이 깨졌으면 빈 목록."""
    try:
        with fitz.open(path) as document:
            entries = document.get_toc() or []
    except (OSError, RuntimeError, ValueError):
        return []
    toc: list[tuple[int, str, int]] = []
    for entry in entries:
        if len(entry) < 3:
            continue
        level, title, page = entry[0], str(entry[1] or "").strip(), entry[2]
        if title and isinstance(level, int) and level >= 1:
            toc.append((min(level, 6), title, int(page) if isinstance(page, (int, float)) else 0))
    return toc


def _normalize(text: str) -> str:
    return " ".join(_MD_HEADING_RE.sub("", text).split()).lower()


def toc_title_coverage(toc: list[tuple[int, str, int]], candidate_text: str) -> float | None:
    """후보 텍스트에 살아남은 TOC 제목의 비율. TOC가 없으면 None."""
    if not toc:
        return None
    haystack = _normalize(candidate_text)
    found = sum(1 for _, title, _ in toc if _normalize(title) and _normalize(title) in haystack)
    return found / len(toc)


def restore_headings_from_toc(source: DocumentSource, content: str) -> str:
    """TOC 제목을 헤딩으로 승격하고, 사라진 제목은 페이지 마커 뒤에 복원한다."""
    toc = read_pdf_toc(source.path)
    if not toc:
        return content

    lines = content.splitlines()
    line_index: dict[str, int] = {}
    page_marker_index: dict[int, int] = {}
    for i, line in enumerate(lines):
        normalized = _normalize(line)
        if normalized and normalized not in line_index:
            line_index[normalized] = i
        marker = _PAGE_MARKER_RE.match(line.strip())
        if marker:
            page_marker_index[int(marker.group(1))] = i

    promotions: dict[int, str] = {}  # 줄 인덱스 → 새 헤딩 줄
    insertions: dict[int, list[str]] = {}  # 줄 인덱스 뒤 → 삽입할 헤딩들
    for level, title, page in toc:
        key = _normalize(title)
        if not key:
            continue
        heading = f"{'#' * level} {' '.join(title.split())}"
        position = line_index.get(key)
        if position is not None:
            if not lines[position].lstrip().startswith("#"):
                promotions[position] = heading
            continue
        # 제목이 통째로 사라졌다 — 해당 페이지 마커가 있으면 그 뒤에 복원.
        marker_position = page_marker_index.get(page)
        if marker_position is not None:
            insertions.setdefault(marker_position, []).append(heading)

    if not promotions and not insertions:
        return content

    result: list[str] = []
    for i, line in enumerate(lines):
        result.append(promotions.get(i, line))
        for heading in insertions.get(i, []):
            result.extend(["", heading, ""])
    return "\n".join(result)
