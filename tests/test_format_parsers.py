"""로드맵 포맷(docx/pptx/csv/html/json/yaml) 구조화 파서의 회귀 테스트.

docx/pptx는 라이브러리 없이 실제 zip+XML 파일을 만들어 파싱 경로 전체를 검증한다.
"""

import zipfile
from pathlib import Path

from parsing_agent.config import WorkflowConfig
from parsing_agent.format_parsers import (
    CsvParserAdapter,
    DataParserAdapter,
    DocxParserAdapter,
    HtmlParserAdapter,
    PptxParserAdapter,
    extract_docx_text,
    extract_html_text,
    extract_pptx_text,
)
from parsing_agent.ingestion import extract_source_text
from parsing_agent.models import DocumentSource
from parsing_agent.workflow import WorkflowRunner

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def _source(path: Path, media_type: str) -> DocumentSource:
    return DocumentSource(
        path=path,
        media_type=media_type,
        size_bytes=path.stat().st_size if path.exists() else 0,
        run_id="format-test",
    )


def _write_docx(path: Path) -> None:
    document = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="{_W_NS}">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
      <w:r><w:t>사업 개요</w:t></w:r>
    </w:p>
    <w:p><w:r><w:t>본문 첫 문단이다.</w:t></w:r></w:p>
    <w:p>
      <w:pPr><w:numPr><w:ilvl w:val="0"/></w:numPr></w:pPr>
      <w:r><w:t>첫 번째 항목</w:t></w:r>
    </w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>지표</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>값</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>최대 파고</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>8.18</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)


def _write_pptx(path: Path) -> None:
    slide1 = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="{_P_NS}" xmlns:a="{_A_NS}">
  <p:cSld><p:spTree>
    <p:sp>
      <p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
      <p:txBody><a:p><a:r><a:t>발표 제목</a:t></a:r></a:p></p:txBody>
    </p:sp>
    <p:sp>
      <p:txBody><a:p><a:r><a:t>첫 번째 불릿</a:t></a:r></a:p></p:txBody>
    </p:sp>
  </p:spTree></p:cSld>
</p:sld>"""
    slide2 = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="{_P_NS}" xmlns:a="{_A_NS}">
  <p:cSld><p:spTree>
    <p:graphicFrame>
      <a:graphic><a:graphicData>
        <a:tbl>
          <a:tr>
            <a:tc><a:txBody><a:p><a:r><a:t>연도</a:t></a:r></a:p></a:txBody></a:tc>
            <a:tc><a:txBody><a:p><a:r><a:t>건수</a:t></a:r></a:p></a:txBody></a:tc>
          </a:tr>
          <a:tr>
            <a:tc><a:txBody><a:p><a:r><a:t>2023</a:t></a:r></a:p></a:txBody></a:tc>
            <a:tc><a:txBody><a:p><a:r><a:t>139</a:t></a:r></a:p></a:txBody></a:tc>
          </a:tr>
        </a:tbl>
      </a:graphicData></a:graphic>
    </p:graphicFrame>
  </p:spTree></p:cSld>
</p:sld>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide1)
        archive.writestr("ppt/slides/slide2.xml", slide2)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def test_docx_adapter_emits_headings_lists_and_tables(tmp_path: Path) -> None:
    docx_path = tmp_path / "report.docx"
    _write_docx(docx_path)

    candidates = DocxParserAdapter().parse(_source(docx_path, "application/octet-stream"), WorkflowConfig())

    assert len(candidates) == 1
    content = candidates[0].content
    assert "# 사업 개요" in content
    assert "본문 첫 문단이다." in content
    assert "- 첫 번째 항목" in content
    assert "| 지표 | 값 |" in content
    assert "| 최대 파고 | 8.18 |" in content
    assert candidates[0].metadata["table_count"] == 1
    assert candidates[0].metadata["heading_count"] == 1


def test_extract_docx_text_is_plain(tmp_path: Path) -> None:
    docx_path = tmp_path / "report.docx"
    _write_docx(docx_path)

    plain = extract_docx_text(docx_path)

    assert "사업 개요" in plain
    assert "최대 파고" in plain
    assert "#" not in plain
    assert "|" not in plain


def test_docx_adapter_skips_other_suffixes(tmp_path: Path) -> None:
    other = tmp_path / "report.txt"
    other.write_text("plain", encoding="utf-8")

    assert DocxParserAdapter().parse(_source(other, "text/plain"), WorkflowConfig()) == []


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------


def test_pptx_adapter_emits_slide_markers_titles_and_tables(tmp_path: Path) -> None:
    pptx_path = tmp_path / "deck.pptx"
    _write_pptx(pptx_path)

    candidates = PptxParserAdapter().parse(_source(pptx_path, "application/octet-stream"), WorkflowConfig())

    assert len(candidates) == 1
    content = candidates[0].content
    assert "<!-- slide 1 -->" in content
    assert "<!-- slide 2 -->" in content
    assert "## 발표 제목" in content
    assert "- 첫 번째 불릿" in content
    assert "| 연도 | 건수 |" in content
    assert "| 2023 | 139 |" in content
    assert candidates[0].metadata["slide_count"] == 2
    assert candidates[0].metadata["page_count"] == 2


def test_extract_pptx_text_returns_slide_count(tmp_path: Path) -> None:
    pptx_path = tmp_path / "deck.pptx"
    _write_pptx(pptx_path)

    plain, slide_count = extract_pptx_text(pptx_path)

    assert slide_count == 2
    assert "발표 제목" in plain
    assert "2023" in plain


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_csv_adapter_renders_markdown_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("이름,나이,도시\n김철수,34,부산\n이영희,29,서울\n", encoding="utf-8")

    candidates = CsvParserAdapter().parse(_source(csv_path, "text/csv"), WorkflowConfig())

    assert len(candidates) == 1
    content = candidates[0].content
    assert "| 이름 | 나이 | 도시 |" in content
    assert "| --- | --- | --- |" in content
    assert "| 김철수 | 34 | 부산 |" in content
    assert candidates[0].metadata["row_count"] == 3
    assert candidates[0].metadata["column_count"] == 3
    assert candidates[0].metadata["delimiter"] == ","


def test_csv_adapter_reads_cp949(tmp_path: Path) -> None:
    csv_path = tmp_path / "legacy.csv"
    csv_path.write_bytes("항목,값\n최대파고,8.18\n".encode("cp949"))

    candidates = CsvParserAdapter().parse(_source(csv_path, "text/csv"), WorkflowConfig())

    assert len(candidates) == 1
    assert "| 항목 | 값 |" in candidates[0].content
    assert "| 최대파고 | 8.18 |" in candidates[0].content


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def test_html_adapter_converts_structure_and_skips_scripts(tmp_path: Path) -> None:
    html_path = tmp_path / "page.html"
    html_path.write_text(
        """<html><head><script>var hidden = "secret";</script></head>
        <body>
          <h1>보고서 요약</h1>
          <p>첫 번째 문단.</p>
          <ul><li>항목 하나</li><li>항목 둘</li></ul>
          <table>
            <tr><th>구분</th><th>수치</th></tr>
            <tr><td>충돌</td><td>52</td></tr>
          </table>
        </body></html>""",
        encoding="utf-8",
    )

    candidates = HtmlParserAdapter().parse(_source(html_path, "text/html"), WorkflowConfig())

    assert len(candidates) == 1
    content = candidates[0].content
    assert "# 보고서 요약" in content
    assert "첫 번째 문단." in content
    assert "- 항목 하나" in content
    assert "| 구분 | 수치 |" in content
    assert "| 충돌 | 52 |" in content
    assert "secret" not in content
    assert candidates[0].metadata["table_count"] == 1


def test_extract_html_text_returns_visible_text_only(tmp_path: Path) -> None:
    html_path = tmp_path / "page.htm"
    html_path.write_text(
        "<html><body><h2>제목</h2><p>본문 <b>강조</b> 텍스트</p></body></html>",
        encoding="utf-8",
    )

    plain = extract_html_text(html_path)

    assert "제목" in plain
    assert "본문 강조 텍스트" in plain
    assert "<" not in plain
    assert "#" not in plain


# ---------------------------------------------------------------------------
# JSON / YAML
# ---------------------------------------------------------------------------


def test_data_adapter_renders_object_array_as_table(tmp_path: Path) -> None:
    json_path = tmp_path / "records.json"
    json_path.write_text(
        '[{"name": "collision", "months": 52}, {"name": "casualty", "months": 47}]',
        encoding="utf-8",
    )

    candidates = DataParserAdapter().parse(_source(json_path, "application/json"), WorkflowConfig())

    assert len(candidates) == 1
    content = candidates[0].content
    assert "| name | months |" in content
    assert "| collision | 52 |" in content
    assert candidates[0].metadata["data_format"] == "json"
    assert candidates[0].metadata["top_level_type"] == "list"


def test_data_adapter_renders_nested_mapping(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "project: parse-everything\nlimits:\n  timeout: 900\n  retries: 2\n",
        encoding="utf-8",
    )

    candidates = DataParserAdapter().parse(_source(yaml_path, "application/octet-stream"), WorkflowConfig())

    assert len(candidates) == 1
    content = candidates[0].content
    assert "- **project:** parse-everything" in content
    assert "- **limits:**" in content
    assert "- **timeout:** 900" in content
    assert candidates[0].metadata["data_format"] == "yaml"


def test_data_adapter_falls_back_on_invalid_json(tmp_path: Path) -> None:
    json_path = tmp_path / "broken.json"
    json_path.write_text('{"unclosed": ', encoding="utf-8")

    assert DataParserAdapter().parse(_source(json_path, "application/json"), WorkflowConfig()) == []


def test_text_fallback_reads_cp949_txt(tmp_path: Path) -> None:
    """레거시 완성형 .txt가 text-fallback에서 죽어 전체 파서 실패로 이어지던 버그의 회귀 테스트."""
    from parsing_agent.parsers import LocalTextParserAdapter

    txt_path = tmp_path / "legacy.txt"
    txt_path.write_bytes("제4장 지역개황\n".encode("cp949"))

    candidates = LocalTextParserAdapter().parse(_source(txt_path, "text/plain"), WorkflowConfig())

    assert len(candidates) == 1
    assert "제4장 지역개황" in candidates[0].content


# ---------------------------------------------------------------------------
# 라우팅 & 인제스천 연결
# ---------------------------------------------------------------------------


def test_base_parser_routing_prefers_structured_adapters(tmp_path: Path) -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))

    cases = {
        "report.docx": ("application/octet-stream", "docx-structured"),
        "deck.pptx": ("application/octet-stream", "pptx-structured"),
        "data.csv": ("text/csv", "csv-structured"),
        "page.html": ("text/html", "html-structured"),
        "records.json": ("application/json", "data-structured"),
        "config.yaml": ("application/octet-stream", "data-structured"),
        "scan.png": ("image/png", "source-text"),
        "photo.jpeg": ("image/jpeg", "source-text"),
        "notes.txt": ("text/plain", "text-fallback"),
    }
    for file_name, (media_type, expected) in cases.items():
        source = DocumentSource(
            path=tmp_path / file_name,
            media_type=media_type,
            size_bytes=0,
            run_id="routing-test",
        )
        assert runner._base_parser_name_for_source(source) == expected, file_name


def test_extract_source_text_populates_docx_and_pptx(tmp_path: Path) -> None:
    docx_path = tmp_path / "report.docx"
    _write_docx(docx_path)
    pptx_path = tmp_path / "deck.pptx"
    _write_pptx(pptx_path)

    docx_text, docx_pages = extract_source_text(docx_path, "application/octet-stream")
    pptx_text, pptx_pages = extract_source_text(pptx_path, "application/octet-stream")

    assert docx_text is not None and "사업 개요" in docx_text
    assert docx_pages is None
    assert pptx_text is not None and "발표 제목" in pptx_text
    assert pptx_pages == 2


def test_extract_source_text_uses_visible_html_text(tmp_path: Path) -> None:
    html_path = tmp_path / "page.html"
    html_path.write_text("<html><body><p>보이는 본문</p></body></html>", encoding="utf-8")

    text, page_count = extract_source_text(html_path, "text/html")

    assert text == "보이는 본문"
    assert page_count is None
