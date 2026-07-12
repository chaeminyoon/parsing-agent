"""텍스트 IO·정규화·마크다운 표 렌더링 공용 유틸.

parsers.py 내부에 있던 것을 추출해 format_parsers/ingestion/models가
parsers(무거운 fitz/pypdf 의존)를 거치지 않고 쓰게 한다.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

# 한국어 문서 현실을 반영한 디코딩 순서: UTF-8 계열 → 레거시 완성형
TEXT_READ_ENCODINGS = ("utf-8", "utf-8-sig", "cp949", "euc-kr")


def normalize_markdown_text(text: str) -> str:
    normalized = text.replace("\ufeff", "").replace("\u00a0", " ")
    return unicodedata.normalize("NFC", normalized)


def read_text_with_fallback(path: Path) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in TEXT_READ_ENCODINGS:
        try:
            return normalize_markdown_text(path.read_text(encoding=encoding))
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return normalize_markdown_text(path.read_text(encoding="utf-8"))


def clean_table_cell(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split()).replace("|", "\\|")


def rows_to_markdown(rows: list[list[object]]) -> str:
    cleaned_rows = [
        [clean_table_cell(cell) for cell in row]
        for row in rows
        if any(clean_table_cell(cell) for cell in row)
    ]
    if not cleaned_rows:
        return ""

    max_columns = max(len(row) for row in cleaned_rows)
    normalized_rows = [row + [""] * (max_columns - len(row)) for row in cleaned_rows]
    header = normalized_rows[0]
    body = normalized_rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)
