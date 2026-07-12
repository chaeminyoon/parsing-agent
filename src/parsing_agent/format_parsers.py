"""Structured parser adapters for the non-PDF roadmap formats.

Roadmap coverage:
- text-based office formats: ``.docx`` / ``.pptx`` (stdlib ``zipfile`` + ``ElementTree``,
  no python-docx/python-pptx dependency),
- tabular text: ``.csv`` (rendered as a markdown table),
- web/data formats: ``.html`` / ``.htm`` (visible text as markdown), ``.json`` /
  ``.yaml`` / ``.yml`` (hierarchy rendered as nested markdown, homogeneous object
  arrays as tables).

Each adapter self-guards on suffix/media type like the PDF adapters and returns
``[]`` when it does not apply, so the workflow fallback chain stays intact. All
adapters emit a single markdown ``content`` string (the project-wide parse-output
contract) plus lightweight structure metadata. The ``extract_*_text`` helpers give
ingestion a plain-text rendering of the same walk so ``DocumentSource.extracted_text``
and evaluation stay meaningful for binary formats.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

try:  # optional; the data adapter degrades to the raw-text fallback without it
    import yaml
except ImportError:  # pragma: no cover - exercised via the None branch in tests
    yaml = None

from parsing_agent.config import WorkflowConfig
from parsing_agent.interfaces import ParserAdapter
from parsing_agent.models import DocumentSource, ParseCandidate
from parsing_agent.textutil import normalize_markdown_text, read_text_with_fallback, rows_to_markdown

# Single source of truth for suffix→base-parser routing (used by workflow).
STRUCTURED_SUFFIX_PARSERS = {
    ".docx": "docx-structured",
    ".pptx": "pptx-structured",
    ".csv": "csv-structured",
    ".html": "html-structured",
    ".htm": "html-structured",
    ".json": "data-structured",
    ".yaml": "data-structured",
    ".yml": "data-structured",
}

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"

_HEADING_STYLE_RE = re.compile(r"heading\s*([1-9])", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Shared block model: ("heading", level, text) | ("para", text) | ("list", text)
# | ("table", rows) | ("marker", label)
# ---------------------------------------------------------------------------


def _render_blocks_markdown(blocks: list[tuple]) -> str:
    parts: list[str] = []
    for block in blocks:
        kind = block[0]
        if kind == "heading":
            parts.append(f"{'#' * block[1]} {block[2]}")
        elif kind == "para":
            parts.append(block[1])
        elif kind == "list":
            parts.append(f"- {block[1]}")
        elif kind == "table":
            rendered = rows_to_markdown(block[1])
            if rendered:
                parts.append(rendered)
        elif kind == "marker":
            parts.append(f"<!-- {block[1]} -->")
    return normalize_markdown_text("\n\n".join(parts)).strip()


def _render_blocks_plain(blocks: list[tuple]) -> str:
    parts: list[str] = []
    for block in blocks:
        kind = block[0]
        if kind == "heading":
            parts.append(block[2])
        elif kind in ("para", "list"):
            parts.append(block[1])
        elif kind == "table":
            parts.extend(" ".join(str(cell) for cell in row if cell) for row in block[1])
    return normalize_markdown_text("\n".join(part for part in parts if part.strip())).strip()


def _clean_inline(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# DOCX (word/document.xml)
# ---------------------------------------------------------------------------


def _docx_paragraph_text(paragraph: ElementTree.Element) -> str:
    return _clean_inline("".join(node.text or "" for node in paragraph.iter(f"{_W}t")))


def _docx_heading_level(paragraph: ElementTree.Element) -> int | None:
    properties = paragraph.find(f"{_W}pPr")
    if properties is None:
        return None
    outline = properties.find(f"{_W}outlineLvl")
    if outline is not None:
        raw = outline.get(f"{_W}val")
        if raw is not None and raw.isdigit():
            return min(int(raw) + 1, 6)
    style = properties.find(f"{_W}pStyle")
    if style is not None:
        match = _HEADING_STYLE_RE.search(style.get(f"{_W}val") or "")
        if match:
            return min(int(match.group(1)), 6)
    return None


def _docx_is_list_item(paragraph: ElementTree.Element) -> bool:
    properties = paragraph.find(f"{_W}pPr")
    return properties is not None and properties.find(f"{_W}numPr") is not None


def _docx_table_rows(table: ElementTree.Element) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.findall(f"{_W}tr"):
        cells = [
            _clean_inline(" ".join(_docx_paragraph_text(p) for p in cell.iter(f"{_W}p")))
            for cell in row.findall(f"{_W}tc")
        ]
        rows.append(cells)
    return rows


def _docx_blocks(path: Path) -> tuple[list[tuple], dict[str, int]]:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    body = root.find(f"{_W}body")
    blocks: list[tuple] = []
    stats = {"paragraph_count": 0, "heading_count": 0, "table_count": 0}
    if body is None:
        return blocks, stats
    for child in body:
        if child.tag == f"{_W}p":
            text = _docx_paragraph_text(child)
            if not text:
                continue
            level = _docx_heading_level(child)
            if level is not None:
                blocks.append(("heading", level, text))
                stats["heading_count"] += 1
            elif _docx_is_list_item(child):
                blocks.append(("list", text))
                stats["paragraph_count"] += 1
            else:
                blocks.append(("para", text))
                stats["paragraph_count"] += 1
        elif child.tag == f"{_W}tbl":
            rows = _docx_table_rows(child)
            if rows:
                blocks.append(("table", rows))
                stats["table_count"] += 1
    return blocks, stats


def parse_docx(path: Path) -> tuple[str, str, dict[str, int]]:
    blocks, stats = _docx_blocks(path)
    return _render_blocks_markdown(blocks), _render_blocks_plain(blocks), stats


def extract_docx_text(path: Path) -> str:
    return parse_docx(path)[1]


# ---------------------------------------------------------------------------
# PPTX (ppt/slides/slideN.xml)
# ---------------------------------------------------------------------------

_SLIDE_NAME_RE = re.compile(r"^ppt/slides/slide(\d+)\.xml$")


def _pptx_shape_paragraphs(shape: ElementTree.Element) -> list[str]:
    paragraphs: list[str] = []
    for paragraph in shape.iter(f"{_A}p"):
        text = _clean_inline("".join(node.text or "" for node in paragraph.iter(f"{_A}t")))
        if text:
            paragraphs.append(text)
    return paragraphs


def _pptx_is_title_shape(shape: ElementTree.Element) -> bool:
    placeholder = shape.find(f"{_P}nvSpPr/{_P}nvPr/{_P}ph")
    return placeholder is not None and placeholder.get("type") in ("title", "ctrTitle")


def _pptx_table_rows(table: ElementTree.Element) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.findall(f"{_A}tr"):
        cells = [
            _clean_inline(" ".join(_pptx_shape_paragraphs(cell)))
            for cell in row.findall(f"{_A}tc")
        ]
        rows.append(cells)
    return rows


def _pptx_blocks(path: Path) -> tuple[list[tuple], dict[str, int]]:
    with zipfile.ZipFile(path) as archive:
        slide_names = sorted(
            (name for name in archive.namelist() if _SLIDE_NAME_RE.match(name)),
            key=lambda name: int(_SLIDE_NAME_RE.match(name).group(1)),
        )
        blocks: list[tuple] = []
        stats = {"slide_count": len(slide_names), "table_count": 0}
        for index, name in enumerate(slide_names, start=1):
            root = ElementTree.fromstring(archive.read(name))
            tree = root.find(f"{_P}cSld/{_P}spTree")
            if tree is None:
                continue
            blocks.append(("marker", f"slide {index}"))
            for child in tree:
                if child.tag == f"{_P}sp":
                    paragraphs = _pptx_shape_paragraphs(child)
                    if not paragraphs:
                        continue
                    if _pptx_is_title_shape(child):
                        blocks.append(("heading", 2, paragraphs[0]))
                        blocks.extend(("para", text) for text in paragraphs[1:])
                    else:
                        blocks.extend(("list", text) for text in paragraphs)
                elif child.tag == f"{_P}graphicFrame":
                    for table in child.iter(f"{_A}tbl"):
                        rows = _pptx_table_rows(table)
                        if rows:
                            blocks.append(("table", rows))
                            stats["table_count"] += 1
    return blocks, stats


def parse_pptx(path: Path) -> tuple[str, str, dict[str, int]]:
    blocks, stats = _pptx_blocks(path)
    return _render_blocks_markdown(blocks), _render_blocks_plain(blocks), stats


def extract_pptx_text(path: Path) -> tuple[str, int]:
    _, plain, stats = parse_pptx(path)
    return plain, stats["slide_count"]


# ---------------------------------------------------------------------------
# HTML (visible text → markdown; stdlib html.parser)
# ---------------------------------------------------------------------------


class _HtmlToBlocks(HTMLParser):
    _SKIP_TAGS = {"script", "style", "head", "noscript", "template"}
    _HEADING_LEVELS = {f"h{n}": n for n in range(1, 7)}
    _FLUSH_TAGS = {"p", "div", "section", "article", "blockquote", "pre",
                   "header", "footer", "main", "aside", "figure", "figcaption"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple] = []
        self._buffer: list[str] = []
        self._pending: tuple | None = None  # ("heading", level) | ("list",)
        self._skip_depth = 0
        self._table_rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self._HEADING_LEVELS:
            self._flush()
            self._pending = ("heading", self._HEADING_LEVELS[tag])
        elif tag == "li":
            self._flush()
            self._pending = ("list",)
        elif tag in self._FLUSH_TAGS:
            self._flush()
        elif tag == "br":
            self._buffer.append(" ")
        elif tag == "table":
            self._flush()
            self._table_rows = []
        elif tag == "tr" and self._table_rows is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(_clean_inline("".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._table_rows is not None and self._row is not None:
            self._table_rows.append(self._row)
            self._row = None
        elif tag == "table" and self._table_rows is not None:
            if any(any(cell for cell in row) for row in self._table_rows):
                self.blocks.append(("table", self._table_rows))
            self._table_rows = None
        elif tag in self._HEADING_LEVELS or tag in self._FLUSH_TAGS or tag == "li":
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._cell is not None:
            self._cell.append(data)
        else:
            self._buffer.append(data)

    def _flush(self) -> None:
        text = _clean_inline("".join(self._buffer))
        self._buffer = []
        pending, self._pending = self._pending, None
        if not text:
            return
        if pending and pending[0] == "heading":
            self.blocks.append(("heading", pending[1], text))
        elif pending and pending[0] == "list":
            self.blocks.append(("list", text))
        else:
            self.blocks.append(("para", text))

    def close(self) -> None:  # noqa: D102 - flush trailing text on close
        super().close()
        self._flush()


def _html_blocks(path: Path) -> list[tuple]:
    parser = _HtmlToBlocks()
    parser.feed(read_text_with_fallback(path))
    parser.close()
    return parser.blocks


def extract_html_text(path: Path) -> str:
    return _render_blocks_plain(_html_blocks(path))


# ---------------------------------------------------------------------------
# JSON / YAML (hierarchy → nested markdown; object arrays → tables)
# ---------------------------------------------------------------------------


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _is_table_like(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 1
        and all(isinstance(item, dict) and item for item in value)
        and all(all(_is_scalar(cell) for cell in item.values()) for item in value)
    )


def _data_table_rows(items: list[dict]) -> list[list[str]]:
    headers: list[str] = []
    for item in items:
        for key in item:
            if key not in headers:
                headers.append(key)
    rows = [headers]
    rows.extend([("" if item.get(key) is None else str(item.get(key, ""))) for key in headers] for item in items)
    return rows


def _render_data(value: object, depth: int = 0) -> list[str]:
    indent = "  " * depth
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_scalar(item):
                rendered = "" if item is None else str(item)
                lines.append(f"{indent}- **{key}:** {rendered}".rstrip())
            elif _is_table_like(item):
                lines.append(f"{indent}- **{key}:**")
                lines.append("")
                lines.append(rows_to_markdown(_data_table_rows(item)))
                lines.append("")
            else:
                lines.append(f"{indent}- **{key}:**")
                lines.extend(_render_data(item, depth + 1))
    elif isinstance(value, list):
        if _is_table_like(value):
            lines.append(rows_to_markdown(_data_table_rows(value)))
        else:
            for item in value:
                if _is_scalar(item):
                    rendered = "" if item is None else str(item)
                    lines.append(f"{indent}- {rendered}".rstrip())
                else:
                    lines.append(f"{indent}-")
                    lines.extend(_render_data(item, depth + 1))
    else:
        rendered = "" if value is None else str(value)
        if rendered:
            lines.append(f"{indent}{rendered}")
    return lines


def render_data_markdown(value: object) -> str:
    return normalize_markdown_text("\n".join(_render_data(value))).strip()


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class DocxParserAdapter(ParserAdapter):
    name = "docx-structured"

    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        if source.path.suffix.lower() != ".docx":
            return []
        markdown, _, stats = parse_docx(source.path)
        if not markdown.strip():
            return []
        return [
            ParseCandidate(
                parser_name=self.name,
                content=markdown,
                format_name=config.output_format,
                metadata={"media_type": source.media_type, **stats},
                source_path=source.path,
            )
        ]


class PptxParserAdapter(ParserAdapter):
    name = "pptx-structured"

    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        if source.path.suffix.lower() != ".pptx":
            return []
        markdown, _, stats = parse_pptx(source.path)
        if not markdown.strip():
            return []
        return [
            ParseCandidate(
                parser_name=self.name,
                content=markdown,
                format_name=config.output_format,
                metadata={"media_type": source.media_type, "page_count": stats["slide_count"], **stats},
                source_path=source.path,
            )
        ]


class CsvParserAdapter(ParserAdapter):
    name = "csv-structured"

    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        if source.path.suffix.lower() != ".csv" and source.media_type != "text/csv":
            return []
        text = read_text_with_fallback(source.path)
        if not text.strip():
            return []
        try:
            dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = [row for row in csv.reader(io.StringIO(text), dialect) if any(cell.strip() for cell in row)]
        markdown = rows_to_markdown(rows)
        if not markdown:
            return []
        return [
            ParseCandidate(
                parser_name=self.name,
                content=markdown,
                format_name=config.output_format,
                metadata={
                    "media_type": source.media_type,
                    "row_count": len(rows),
                    "column_count": max(len(row) for row in rows),
                    "delimiter": dialect.delimiter,
                },
                source_path=source.path,
            )
        ]


class HtmlParserAdapter(ParserAdapter):
    name = "html-structured"

    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        if source.path.suffix.lower() not in (".html", ".htm") and source.media_type != "text/html":
            return []
        blocks = _html_blocks(source.path)
        markdown = _render_blocks_markdown(blocks)
        if not markdown.strip():
            return []
        table_count = sum(1 for block in blocks if block[0] == "table")
        heading_count = sum(1 for block in blocks if block[0] == "heading")
        return [
            ParseCandidate(
                parser_name=self.name,
                content=markdown,
                format_name=config.output_format,
                metadata={
                    "media_type": source.media_type,
                    "table_count": table_count,
                    "heading_count": heading_count,
                },
                source_path=source.path,
            )
        ]


class DataParserAdapter(ParserAdapter):
    """Structured rendering for JSON/YAML; invalid documents fall back to raw text."""

    name = "data-structured"

    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        suffix = source.path.suffix.lower()
        if suffix == ".json" or source.media_type == "application/json":
            data_format = "json"
        elif suffix in (".yaml", ".yml"):
            if yaml is None:
                return []
            data_format = "yaml"
        else:
            return []
        text = read_text_with_fallback(source.path)
        try:
            value = json.loads(text) if data_format == "json" else yaml.safe_load(text)
        except Exception:  # noqa: BLE001 - malformed documents use the raw-text fallback
            return []
        markdown = render_data_markdown(value)
        if not markdown.strip():
            return []
        return [
            ParseCandidate(
                parser_name=self.name,
                content=markdown,
                format_name=config.output_format,
                metadata={
                    "media_type": source.media_type,
                    "data_format": data_format,
                    "top_level_type": type(value).__name__,
                },
                source_path=source.path,
            )
        ]
