from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from typing import Any

from parsing_agent.config import WorkflowConfig
from parsing_agent.filetype import is_image, is_pdf


@dataclass(slots=True)
class OcrResult:
    applied: bool
    provider: str
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)


class _TextHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"br", "p", "div", "tr", "li", "table", "thead", "tbody"}:
            self._parts.append("\n")
        if tag in {"td", "th"}:
            self._parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "tr", "li", "table"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def text(self) -> str:
        return _normalize_text("\n".join(self._parts))


class _TableMarkdownParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self._current_colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._current_row = []
            return
        if tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []
            self._current_colspan = _positive_int(_attr_value(attrs, "colspan"), default=1)
            return
        if self._current_cell is None:
            return
        if tag == "br":
            self._current_cell.append(" ")
        elif tag == "li":
            self._current_cell.append("- ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            cell_text = _normalize_inline_text(" ".join(self._current_cell))
            self._current_row.append(cell_text)
            for _ in range(max(0, self._current_colspan - 1)):
                self._current_row.append("")
            self._current_cell = None
            self._current_colspan = 1
            return
        if tag == "tr" and self._current_row is not None:
            if any(cell.strip() for cell in self._current_row):
                self._rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None and data.strip():
            self._current_cell.append(data.strip())

    def markdown(self) -> str:
        if not self._rows:
            return ""
        column_count = max(len(row) for row in self._rows)
        rows = [row + [""] * (column_count - len(row)) for row in self._rows]
        header = rows[0]
        body = rows[1:]
        markdown_rows = [
            _markdown_table_row(header),
            _markdown_table_row(["---"] * column_count),
        ]
        markdown_rows.extend(_markdown_table_row(row) for row in body)
        return "\n".join(markdown_rows)


def should_run_ocr(
    *,
    media_type: str,
    path: Path,
    extracted_text: str | None,
    config: WorkflowConfig,
) -> bool:
    if not config.ocr_enabled:
        return False
    if config.ocr_provider.strip().lower() != "surya":
        return False
    # 이미지 입력은 OCR이 유일한 텍스트 경로이므로 길이 기준 없이 항상 실행한다.
    # (surya CLI는 이미지 입력을 그대로 받는다.)
    if is_image(media_type, path):
        return True
    if not is_pdf(media_type, path):
        return False
    return len((extracted_text or "").strip()) < config.ocr_min_text_characters


def run_ocr(
    *,
    input_path: Path,
    output_dir: Path,
    config: WorkflowConfig,
    page_count: int | None = None,
    extracted_text: str | None = None,
) -> OcrResult:
    provider = config.ocr_provider.strip().lower()
    if provider != "surya":
        return OcrResult(
            applied=False,
            provider=provider,
            metadata={
                "applied": False,
                "provider": provider,
                "reason": "unsupported_provider",
            },
        )
    return run_surya_ocr(
        input_path=input_path,
        output_dir=output_dir,
        config=config,
        page_count=page_count,
        extracted_text=extracted_text,
    )


def run_surya_ocr(
    *,
    input_path: Path,
    output_dir: Path,
    config: WorkflowConfig,
    page_count: int | None = None,
    extracted_text: str | None = None,
) -> OcrResult:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = _surya_command(input_path=input_path, output_dir=output_dir, config=config)

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.ocr_timeout_seconds,
        )
        elapsed_ms = _elapsed_ms(started)
        if completed.returncode != 0:
            message = _preview(completed.stderr or completed.stdout or "Surya OCR failed.")
            if not config.ocr_fail_open:
                raise RuntimeError(message)
            return OcrResult(
                applied=False,
                provider="surya",
                metadata={
                    "applied": False,
                    "provider": "surya",
                    "error": message,
                    "elapsed_ms": elapsed_ms,
                    "input_text_characters": len(extracted_text or ""),
                    "page_count": page_count,
                },
            )

        raw_result_path = _find_surya_result_json(output_dir)
        if raw_result_path is None:
            message = "Surya OCR completed but no result JSON was found."
            if not config.ocr_fail_open:
                raise RuntimeError(message)
            return OcrResult(
                applied=False,
                provider="surya",
                metadata={
                    "applied": False,
                    "provider": "surya",
                    "error": message,
                    "elapsed_ms": elapsed_ms,
                    "input_text_characters": len(extracted_text or ""),
                    "page_count": page_count,
                },
            )

        result_json = json.loads(raw_result_path.read_text(encoding="utf-8"))
        text = extract_text_from_surya_result(result_json)
        stats = summarize_surya_result(result_json)

        raw_artifact_path = output_dir / "ocr_raw.json"
        if raw_result_path.resolve() != raw_artifact_path.resolve():
            shutil.copy2(raw_result_path, raw_artifact_path)
        text_artifact_path = output_dir / "ocr_text.md"
        summary_artifact_path = output_dir / "ocr_summary.json"

        metadata = {
            "applied": True,
            "provider": "surya",
            "elapsed_ms": elapsed_ms,
            "input_text_characters": len(extracted_text or ""),
            "output_text_characters": len(text),
            "page_count": page_count,
            **stats,
        }
        text_artifact_path.write_text(text, encoding="utf-8")
        summary_artifact_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return OcrResult(
            applied=True,
            provider="surya",
            text=text,
            metadata=metadata,
            artifacts={
                "ocr_raw": raw_artifact_path,
                "ocr_text": text_artifact_path,
                "ocr_summary": summary_artifact_path,
            },
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = _elapsed_ms(started)
        message = f"Surya OCR timed out after {config.ocr_timeout_seconds}s."
        if not config.ocr_fail_open:
            raise RuntimeError(message) from exc
        return OcrResult(
            applied=False,
            provider="surya",
            metadata={
                "applied": False,
                "provider": "surya",
                "error": message,
                "elapsed_ms": elapsed_ms,
                "input_text_characters": len(extracted_text or ""),
                "page_count": page_count,
            },
        )
    except Exception as exc:
        elapsed_ms = _elapsed_ms(started)
        if not config.ocr_fail_open:
            raise
        return OcrResult(
            applied=False,
            provider="surya",
            metadata={
                "applied": False,
                "provider": "surya",
                "error": _preview(str(exc)),
                "elapsed_ms": elapsed_ms,
                "input_text_characters": len(extracted_text or ""),
                "page_count": page_count,
            },
        )


def extract_text_from_surya_result(result_json: Any) -> str:
    pages = _page_items(result_json)
    page_texts: list[str] = []
    for index, page in enumerate(pages, start=1):
        lines = _extract_text_items(page)
        page_body = _normalize_text("\n".join(lines))
        if page_body:
            page_texts.append(f"<!-- page {index} -->\n\n{page_body}")
    if page_texts:
        return "\n\n".join(page_texts).strip()
    return _normalize_text("\n".join(_extract_text_items(result_json)))


def summarize_surya_result(result_json: Any) -> dict[str, Any]:
    page_count = len(_page_items(result_json))
    block_count = 0
    table_block_count = 0
    confidences: list[float] = []

    for item in _walk(result_json):
        if not isinstance(item, dict):
            continue
        if _looks_like_text_block(item):
            block_count += 1
            block_type = str(
                item.get("type")
                or item.get("label")
                or item.get("block_type")
                or item.get("layout_label")
                or ""
            ).lower()
            html = str(item.get("html") or "")
            if "table" in block_type or "<table" in html.lower():
                table_block_count += 1
        confidence = item.get("confidence") or item.get("score")
        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

    summary: dict[str, Any] = {
        "ocr_page_count": page_count or None,
        "ocr_block_count": block_count,
        "ocr_table_block_count": table_block_count,
    }
    if confidences:
        summary["ocr_mean_confidence"] = sum(confidences) / len(confidences)
    return summary


def _surya_command(*, input_path: Path, output_dir: Path, config: WorkflowConfig) -> list[str]:
    return [config.ocr_command, str(input_path), "--output_dir", str(output_dir)]


def _find_surya_result_json(output_dir: Path) -> Path | None:
    preferred = [
        output_dir / "ocr_raw.json",
        output_dir / "results.json",
        output_dir / "result.json",
        output_dir / output_dir.name / "results.json",
    ]
    for path in preferred:
        if path.is_file():
            return path
    candidates = sorted(output_dir.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _page_items(result_json: Any) -> list[Any]:
    if isinstance(result_json, dict):
        for key in ("pages", "page_results", "results"):
            value = result_json.get(key)
            if isinstance(value, list):
                return value
        if result_json and all(isinstance(value, dict) for value in result_json.values()):
            return list(result_json.values())
        if result_json and all(isinstance(value, list) for value in result_json.values()):
            return [item for value in result_json.values() for item in value]
    if isinstance(result_json, list):
        return result_json
    return []


def _extract_text_items(value: Any) -> list[str]:
    items: list[str] = []
    for item in _walk(value):
        if not isinstance(item, dict):
            continue
        html = item.get("html")
        if isinstance(html, str) and html.strip():
            text = _html_to_text(html)
            if text:
                items.append(text)
            continue
        for key in ("text", "content", "markdown"):
            text = item.get(key)
            if isinstance(text, str) and text.strip():
                items.append(_normalize_text(text))
                break
    return _dedupe_adjacent(items)


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _looks_like_text_block(item: dict[str, Any]) -> bool:
    return any(isinstance(item.get(key), str) and item.get(key, "").strip() for key in ("html", "text", "content"))


def _html_to_text(html: str) -> str:
    if "<table" in html.lower():
        table_markdown = _html_table_to_markdown(html)
        if table_markdown:
            return table_markdown
    parser = _TextHtmlParser()
    parser.feed(html)
    return parser.text()


def _html_table_to_markdown(html: str) -> str:
    parser = _TableMarkdownParser()
    parser.feed(html)
    return parser.markdown()


def _markdown_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(_escape_markdown_table_cell(cell) for cell in cells) + " |"


def _escape_markdown_table_cell(cell: str) -> str:
    return _normalize_inline_text(cell).replace("|", "\\|")


def _normalize_inline_text(text: str) -> str:
    return " ".join(text.split())


def _attr_value(attrs: list[tuple[str, str | None]], name: str) -> str | None:
    for key, value in attrs:
        if key.lower() == name:
            return value
    return None


def _positive_int(value: str | None, *, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _normalize_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    compacted: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line
        if blank and previous_blank:
            continue
        compacted.append(line)
        previous_blank = blank
    return "\n".join(compacted).strip()


def _dedupe_adjacent(items: list[str]) -> list[str]:
    deduped: list[str] = []
    previous: str | None = None
    for item in items:
        if item and item != previous:
            deduped.append(item)
        previous = item
    return deduped


def _elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


def _preview(text: str, limit: int = 500) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."
