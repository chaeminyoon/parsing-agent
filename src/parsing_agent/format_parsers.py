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
    ".xlsx": "xlsx-structured",
    ".odt": "odt-structured",
    ".csv": "csv-structured",
    ".html": "html-structured",
    ".htm": "html-structured",
    ".json": "data-structured",
    ".yaml": "data-structured",
    ".yml": "data-structured",
    ".xml": "xml-structured",
}

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_ODF_TEXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
_ODF_TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"

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


# ---------------------------------------------------------------------------
# XLSX (xl/worksheets/sheetN.xml — SpreadsheetML)
# ---------------------------------------------------------------------------

_R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_CELL_REF_RE = re.compile(r"^([A-Z]+)(\d+)$")


def _excel_column_index(ref: str) -> int | None:
    match = _CELL_REF_RE.match(ref or "")
    if match is None:
        return None
    index = 0
    for ch in match.group(1):
        index = index * 26 + (ord(ch) - ord("A") + 1)
    return index - 1


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        _clean_inline("".join(t.text or "" for t in si.iter(f"{_S}t")))
        for si in root.findall(f"{_S}si")
    ]


def _xlsx_cell_value(cell: ElementTree.Element, shared: list[str]) -> str:
    cell_type = cell.get("t", "n")
    if cell_type == "inlineStr":
        return _clean_inline("".join(t.text or "" for t in cell.iter(f"{_S}t")))
    value = cell.findtext(f"{_S}v") or ""
    if cell_type == "s":
        try:
            return shared[int(value)]
        except (ValueError, IndexError):
            return ""
    if cell_type == "b":
        return "TRUE" if value == "1" else "FALSE"
    return _clean_inline(value)


def _xlsx_sheet_rows(root: ElementTree.Element, shared: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in root.iter(f"{_S}row"):
        cells: list[str] = []
        next_index = 0
        for cell in row.findall(f"{_S}c"):
            index = _excel_column_index(cell.get("r", ""))
            if index is None:
                index = next_index
            while len(cells) < index:
                cells.append("")
            cells.append(_xlsx_cell_value(cell, shared))
            next_index = index + 1
        if any(cell for cell in cells):
            rows.append(cells)
    return rows


def _xlsx_blocks(path: Path) -> tuple[list[tuple], dict[str, int]]:
    blocks: list[tuple] = []
    stats = {"sheet_count": 0, "table_count": 0}
    with zipfile.ZipFile(path) as archive:
        shared = _xlsx_shared_strings(archive)
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
        targets = {rel.get("Id"): rel.get("Target", "") for rel in rels_root.iter(f"{rel_ns}Relationship")}
        for sheet in workbook.iter(f"{_S}sheet"):
            target = targets.get(sheet.get(f"{_R_NS}id"), "")
            if not target:
                continue
            member = target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"
            if member not in archive.namelist():
                continue
            rows = _xlsx_sheet_rows(ElementTree.fromstring(archive.read(member)), shared)
            stats["sheet_count"] += 1
            blocks.append(("heading", 2, sheet.get("name") or f"Sheet{stats['sheet_count']}"))
            if rows:
                blocks.append(("table", rows))
                stats["table_count"] += 1
    return blocks, stats


def parse_xlsx(path: Path) -> tuple[str, str, dict[str, int]]:
    blocks, stats = _xlsx_blocks(path)
    return _render_blocks_markdown(blocks), _render_blocks_plain(blocks), stats


def extract_xlsx_text(path: Path) -> str:
    return parse_xlsx(path)[1]


# ---------------------------------------------------------------------------
# ODT (content.xml — OpenDocument Text)
# ---------------------------------------------------------------------------


def _odt_text(element: ElementTree.Element) -> str:
    return _clean_inline("".join(element.itertext()))


def _odt_table_rows(table: ElementTree.Element) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.findall(f"{_ODF_TABLE}table-row"):
        cells: list[str] = []
        for cell in row.findall(f"{_ODF_TABLE}table-cell"):
            repeat = int(cell.get(f"{_ODF_TABLE}number-columns-repeated", "1") or "1")
            cells.extend([_odt_text(cell)] * min(repeat, 64))
        rows.append(cells)
    return rows


def _odt_blocks(path: Path) -> tuple[list[tuple], dict[str, int]]:
    office = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("content.xml"))
    body = root.find(f"{office}body/{office}text")
    blocks: list[tuple] = []
    stats = {"paragraph_count": 0, "heading_count": 0, "table_count": 0}
    if body is None:
        return blocks, stats
    for child in body:
        if child.tag == f"{_ODF_TEXT}h":
            text = _odt_text(child)
            if text:
                level = int(child.get(f"{_ODF_TEXT}outline-level", "1") or "1")
                blocks.append(("heading", min(level, 6), text))
                stats["heading_count"] += 1
        elif child.tag == f"{_ODF_TEXT}p":
            text = _odt_text(child)
            if text:
                blocks.append(("para", text))
                stats["paragraph_count"] += 1
        elif child.tag == f"{_ODF_TEXT}list":
            for item in child.iter(f"{_ODF_TEXT}list-item"):
                text = _odt_text(item)
                if text:
                    blocks.append(("list", text))
                    stats["paragraph_count"] += 1
        elif child.tag == f"{_ODF_TABLE}table":
            rows = _odt_table_rows(child)
            if rows:
                blocks.append(("table", rows))
                stats["table_count"] += 1
    return blocks, stats


def parse_odt(path: Path) -> tuple[str, str, dict[str, int]]:
    blocks, stats = _odt_blocks(path)
    return _render_blocks_markdown(blocks), _render_blocks_plain(blocks), stats


def extract_odt_text(path: Path) -> str:
    return parse_odt(path)[1]


# ---------------------------------------------------------------------------
# XML (범용 — 계층은 리스트로, 동질 반복 요소는 표로)
# ---------------------------------------------------------------------------


def _xml_local_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _xml_is_flat(element: ElementTree.Element) -> bool:
    return len(element) == 0


def _xml_flat_children_table(children: list[ElementTree.Element]) -> list[list[str]] | None:
    """같은 태그의 평평한 형제가 2개 이상이면 attribute+text를 표로 편다."""
    if len(children) < 2 or not all(_xml_is_flat(child) for child in children):
        return None
    if len({_xml_local_name(child.tag) for child in children}) != 1:
        return None
    headers: list[str] = []
    for child in children:
        for key in child.attrib:
            name = _xml_local_name(key)
            if name not in headers:
                headers.append(name)
    has_text = any(_clean_inline(child.text or "") for child in children)
    columns = headers + (["text"] if has_text else [])
    if not columns:
        return None
    rows = [columns]
    for child in children:
        attrs = {_xml_local_name(k): v for k, v in child.attrib.items()}
        row = [attrs.get(name, "") for name in headers]
        if has_text:
            row.append(_clean_inline(child.text or ""))
        rows.append(row)
    return rows


def _render_xml_element(element: ElementTree.Element, depth: int = 0) -> list[str]:
    indent = "  " * depth
    name = _xml_local_name(element.tag)
    attrs = " ".join(f'{_xml_local_name(k)}="{v}"' for k, v in element.attrib.items())
    text = _clean_inline(element.text or "")
    label = f"{name} ({attrs})" if attrs else name
    lines = [f"{indent}- **{label}:** {text}".rstrip()]

    children = list(element)
    index = 0
    while index < len(children):
        child = children[index]
        # 같은 태그의 연속 형제 묶음을 찾아 표 후보로 검사한다.
        group = [child]
        while (
            index + len(group) < len(children)
            and children[index + len(group)].tag == child.tag
        ):
            group.append(children[index + len(group)])
        table_rows = _xml_flat_children_table(group)
        if table_rows is not None:
            lines.append("")
            lines.append(rows_to_markdown(table_rows))
            lines.append("")
        else:
            for member in group:
                lines.extend(_render_xml_element(member, depth + 1))
        index += len(group)
    return lines


def render_xml_markdown(root: ElementTree.Element) -> str:
    return normalize_markdown_text("\n".join(_render_xml_element(root))).strip()


# ---------------------------------------------------------------------------
# 추가 어댑터 (xlsx / odt / xml)
# ---------------------------------------------------------------------------


class XlsxParserAdapter(ParserAdapter):
    name = "xlsx-structured"

    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        if source.path.suffix.lower() != ".xlsx":
            return []
        markdown, _, stats = parse_xlsx(source.path)
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


class OdtParserAdapter(ParserAdapter):
    name = "odt-structured"

    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        if source.path.suffix.lower() != ".odt":
            return []
        markdown, _, stats = parse_odt(source.path)
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


class XmlParserAdapter(ParserAdapter):
    """범용 XML의 구조 보존 렌더. 파싱 실패 시 raw-text 폴백에 맡긴다."""

    name = "xml-structured"

    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        if source.path.suffix.lower() != ".xml":
            return []
        text = read_text_with_fallback(source.path)
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError:
            return []
        markdown = render_xml_markdown(root)
        if not markdown.strip():
            return []
        return [
            ParseCandidate(
                parser_name=self.name,
                content=markdown,
                format_name=config.output_format,
                metadata={"media_type": source.media_type, "root_tag": _xml_local_name(root.tag)},
                source_path=source.path,
            )
        ]
