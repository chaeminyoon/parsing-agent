from pathlib import Path

import fitz

from parsing_agent.config import WorkflowConfig
from parsing_agent.models import DocumentSource
from parsing_agent.parsers import (
    LayoutFirstPdfParserAdapter,
    _TABLE_LABEL_LINE_RE,
    _table_to_html,
    build_default_parser_registry,
)
from parsing_agent.textutil import read_text_with_fallback, rows_to_markdown


class _SimpleTable:
    bbox = (70, 200, 520, 320)
    row_count = 3
    col_count = 2
    cells = [
        (70, 200, 295, 240),
        (295, 200, 520, 240),
        (70, 240, 295, 280),
        (295, 240, 520, 280),
        (70, 280, 295, 320),
        (295, 280, 520, 320),
    ]

    def extract(self):
        return [
            ["name", "value"],
            ["alpha", "10"],
            ["beta", "20"],
        ]


class _MergedTable:
    bbox = (70, 200, 520, 320)
    row_count = 3
    col_count = 2
    cells = [
        (70, 200, 520, 240),
        (70, 200, 520, 240),
        (70, 240, 295, 280),
        (295, 240, 520, 280),
        (70, 280, 295, 320),
        (295, 280, 520, 320),
    ]

    def extract(self):
        return [
            ["merged header", ""],
            ["alpha", "10"],
            ["beta", "20"],
        ]


class _ContinuationTable:
    bbox = (70, 40, 520, 140)
    row_count = 2
    col_count = 2
    cells = [
        (70, 40, 295, 90),
        (295, 40, 520, 90),
        (70, 90, 295, 140),
        (295, 90, 520, 140),
    ]

    def extract(self):
        return [
            ["gamma", "30"],
            ["delta", "40"],
        ]


class _Finder:
    def __init__(self, tables):
        self.tables = tables


class _SimplePage:
    rect = fitz.Rect(0, 0, 600, 800)

    def find_tables(self):
        return _Finder([_SimpleTable()])

    def get_text(self, mode: str):
        if mode == "blocks":
            return [
                (70, 80, 520, 120, "Heading text\n", 0, 0),
                (75, 220, 510, 300, "duplicate table text\n", 1, 0),
                (70, 360, 520, 390, "After table\n", 2, 0),
            ]
        raise AssertionError(mode)


def test_read_text_with_fallback_reads_cp949_korean(tmp_path) -> None:
    sample_path = tmp_path / "sample.md"
    sample_path.write_bytes("제4장 지역개황\n표 4.2-2 토지이용 현황\n".encode("cp949"))

    text = read_text_with_fallback(sample_path)

    assert "제4장 지역개황" in text
    assert "표 4.2-2" in text


def test_read_text_with_fallback_normalizes_nbsp(tmp_path) -> None:
    sample_path = tmp_path / "sample.md"
    sample_path.write_text("제4장\u00a0지역개황", encoding="utf-8")

    text = read_text_with_fallback(sample_path)

    assert text == "제4장 지역개황"


class _MergedTablePage:
    rect = fitz.Rect(0, 0, 600, 800)

    def find_tables(self):
        return _Finder([_MergedTable()])

    def get_text(self, mode: str):
        if mode == "blocks":
            return [
                (70, 180, 520, 220, "mergedwordswithoutspaces\n", 0, 0),
            ]
        raise AssertionError(mode)


class _MixedFallbackPage:
    rect = fitz.Rect(0, 0, 600, 800)

    def find_tables(self):
        return _Finder([_SimpleTable()])

    def get_text(self, mode: str):
        if mode == "blocks":
            return [
                (70, 80, 520, 120, "Heading text\n", 0, 0),
                (70, 140, 520, 180, "bodytextwithoutspaces\n", 1, 0),
            ]
        raise AssertionError(mode)


class _BottomSimpleTablePage:
    rect = fitz.Rect(0, 0, 600, 800)

    def find_tables(self):
        table = _SimpleTable()
        table.bbox = (70, 640, 520, 780)
        return _Finder([table])

    def get_text(self, mode: str):
        if mode == "blocks":
            return []
        raise AssertionError(mode)


class _ReferenceOnlyTablePage:
    rect = fitz.Rect(0, 0, 600, 800)

    def find_tables(self):
        return _Finder([_MergedTable()])

    def get_text(self, mode: str):
        if mode == "blocks":
            return []
        raise AssertionError(mode)


class _TopContinuationPage:
    rect = fitz.Rect(0, 0, 600, 800)

    def find_tables(self):
        return _Finder([_ContinuationTable()])

    def get_text(self, mode: str):
        if mode == "blocks":
            return []
        raise AssertionError(mode)


class _Document:
    def __init__(self, pages):
        self._pages = pages
        self.page_count = len(pages)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self._pages)


def _build_pdf_source(tmp_path: Path, *, extracted_text: str = "", page_count: int = 1) -> DocumentSource:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    return DocumentSource(
        path=pdf_path,
        media_type="application/pdf",
        size_bytes=0,
        run_id="run-layout",
        extracted_text=extracted_text,
        page_count=page_count,
    )


def test_rows_to_markdown_normalizes_cells() -> None:
    markdown = rows_to_markdown([["a|b", None], ["line\nbreak", 3]])

    assert markdown == "| a\\|b |  |\n| --- | --- |\n| line break | 3 |"


def test_table_to_html_preserves_merged_cell_span() -> None:
    html = _table_to_html(_MergedTable())

    assert '<th colspan="2">merged header</th>' in html
    assert html.count("merged header") == 1
    assert "<td>alpha</td>" in html
    assert "<td>10</td>" in html


def test_table_label_line_regex_matches_expected_label_forms() -> None:
    assert _TABLE_LABEL_LINE_RE.search("Table 1")
    assert _TABLE_LABEL_LINE_RE.search("table <1.2-3>")
    assert _TABLE_LABEL_LINE_RE.search("\uD45C 1")
    assert _TABLE_LABEL_LINE_RE.search("\uD45C <1.2>")
    assert not _TABLE_LABEL_LINE_RE.search("Heading 1")
    assert not _TABLE_LABEL_LINE_RE.search("Table of contents")


def test_layout_first_parser_keeps_simple_tables_usable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("parsing_agent.parsers.fitz.open", lambda path: _Document([_SimplePage()]))
    monkeypatch.setattr("parsing_agent.parsers._extract_pdf_page_texts", lambda source: [])
    source = _build_pdf_source(tmp_path)

    candidates = LayoutFirstPdfParserAdapter().parse(
        source,
        WorkflowConfig(layout_first_image_captioning_enabled=False),
    )

    candidate = candidates[0]
    assert "| name | value |" in candidate.content
    assert "duplicate table text" not in candidate.content
    assert candidate.content.index("Heading text") < candidate.content.index("| name | value |")
    assert candidate.metadata["table_regions"] == [
        {
            "table_id": "p1-t1",
            "page": 1,
            "bbox": [70.0, 200.0, 520.0, 320.0],
            "row_count": 3,
            "col_count": 2,
            "extraction_mode": "structured",
        }
    ]


def test_layout_first_parser_falls_back_to_source_text_for_complex_table_pages(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("parsing_agent.parsers.fitz.open", lambda path: _Document([_MergedTablePage()]))
    monkeypatch.setattr(
        "parsing_agent.parsers._extract_pdf_page_texts",
        lambda source: ["merged words with spaces\nTable 1 merged header alpha 10 beta 20"],
    )
    source = _build_pdf_source(tmp_path, extracted_text="merged words with spaces")

    candidates = LayoutFirstPdfParserAdapter().parse(
        source,
        WorkflowConfig(layout_first_image_captioning_enabled=False),
    )

    candidate = candidates[0]
    assert "merged words with spaces" in candidate.content
    assert "mergedwordswithoutspaces" not in candidate.content
    assert "| merged header |" not in candidate.content
    assert candidate.metadata["source_text_fallback_page_count"] == 1
    assert candidate.metadata["table_regions"] == [
        {
            "table_id": "p1-t1",
            "page": 1,
            "bbox": [70.0, 200.0, 520.0, 320.0],
            "row_count": 3,
            "col_count": 2,
            "extraction_mode": "reference",
        }
    ]


def test_layout_first_parser_uses_source_fallback_for_body_while_preserving_simple_table(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("parsing_agent.parsers.fitz.open", lambda path: _Document([_MixedFallbackPage()]))
    monkeypatch.setattr(
        "parsing_agent.parsers._extract_pdf_page_texts",
        lambda source: ["Heading text\nTable 1\nRecovered body text with spaces"],
    )
    source = _build_pdf_source(tmp_path, extracted_text="Heading text\nRecovered body text with spaces")

    candidates = LayoutFirstPdfParserAdapter().parse(
        source,
        WorkflowConfig(layout_first_image_captioning_enabled=False),
    )

    candidate = candidates[0]
    assert "Recovered body text with spaces" in candidate.content
    assert "bodytextwithoutspaces" not in candidate.content
    assert "| name | value |" in candidate.content
    assert candidate.content.index("Table 1") < candidate.content.index("| name | value |")
    assert candidate.content.index("| name | value |") < candidate.content.index("Recovered body text with spaces")
    assert candidate.metadata["source_text_fallback_page_count"] == 1
    assert candidate.metadata["table_regions"] == [
        {
            "table_id": "p1-t1",
            "page": 1,
            "bbox": [70.0, 200.0, 520.0, 320.0],
            "row_count": 3,
            "col_count": 2,
            "extraction_mode": "structured",
        }
    ]


def test_layout_first_parser_preserves_reference_marker_when_source_text_is_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("parsing_agent.parsers.fitz.open", lambda path: _Document([_ReferenceOnlyTablePage()]))
    monkeypatch.setattr("parsing_agent.parsers._extract_pdf_page_texts", lambda source: [])
    source = _build_pdf_source(tmp_path)

    candidates = LayoutFirstPdfParserAdapter().parse(
        source,
        WorkflowConfig(layout_first_image_captioning_enabled=False),
    )

    candidate = candidates[0]
    assert "[Table reference: id=p1-t1 page=1 bbox=70.0,200.0,520.0,320.0 rows=3 cols=2]" in candidate.content
    assert candidate.metadata["table_regions"] == [
        {
            "table_id": "p1-t1",
            "page": 1,
            "bbox": [70.0, 200.0, 520.0, 320.0],
            "row_count": 3,
            "col_count": 2,
            "extraction_mode": "reference",
        }
    ]


def test_layout_first_parser_does_not_complete_multipage_tables(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "parsing_agent.parsers.fitz.open",
        lambda path: _Document([_BottomSimpleTablePage(), _TopContinuationPage()]),
    )
    monkeypatch.setattr(
        "parsing_agent.parsers._extract_pdf_page_texts",
        lambda source: ["", "gamma 30\ndelta 40"],
    )
    source = _build_pdf_source(tmp_path, page_count=2)

    candidates = LayoutFirstPdfParserAdapter().parse(
        source,
        WorkflowConfig(layout_first_image_captioning_enabled=False, layout_first_table_format="html"),
    )

    candidate = candidates[0]
    assert "<table data-continuation" not in candidate.content
    assert candidate.content.count("<table>") == 1
    assert "gamma 30" in candidate.content
    assert "<th>name</th>" in candidate.content
    assert candidate.metadata["table_regions"] == [
        {
            "table_id": "p1-t1",
            "page": 1,
            "bbox": [70.0, 640.0, 520.0, 780.0],
            "row_count": 3,
            "col_count": 2,
            "extraction_mode": "structured",
        },
        {
            "table_id": "p2-t1",
            "page": 2,
            "bbox": [70.0, 40.0, 520.0, 140.0],
            "row_count": 2,
            "col_count": 2,
            "extraction_mode": "reference",
            "continued_from_page": 1,
        },
    ]


def test_default_registry_includes_layout_first_pdf() -> None:
    registry = build_default_parser_registry()

    assert registry.has("layout-first-pdf")
