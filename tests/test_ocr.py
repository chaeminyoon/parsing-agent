from pathlib import Path

from parsing_agent.config import WorkflowConfig
from parsing_agent.ingestion import build_document_source
from parsing_agent.ocr import OcrResult, extract_text_from_surya_result, should_run_ocr


def test_should_run_ocr_for_pdf_with_too_little_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    config = WorkflowConfig(ocr_enabled=True, ocr_min_text_characters=50)

    assert should_run_ocr(
        media_type="application/pdf",
        path=pdf_path,
        extracted_text="",
        config=config,
    )


def test_should_not_run_ocr_when_disabled(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    config = WorkflowConfig(ocr_enabled=False, ocr_min_text_characters=50)

    assert not should_run_ocr(
        media_type="application/pdf",
        path=pdf_path,
        extracted_text="",
        config=config,
    )


def test_extract_text_from_surya_result_handles_html_blocks() -> None:
    result = {
        "pages": [
            {
                "text_lines": [
                    {"html": "<h1>제1장 요약문</h1>"},
                    {"html": "<table><tr><th>항목</th><th>값</th></tr><tr><td>대기질</td><td>양호</td></tr></table>"},
                ]
            }
        ]
    }

    text = extract_text_from_surya_result(result)

    assert "<!-- page 1 -->" in text
    assert "제1장 요약문" in text
    assert "대기질" in text
    assert "| 항목 | 값 |" in text
    assert "| --- | --- |" in text
    assert "| 대기질 | 양호 |" in text


def test_build_document_source_applies_ocr_when_pdf_text_is_empty(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    ocr_dir = tmp_path / "ocr"

    monkeypatch.setattr(
        "parsing_agent.ingestion.extract_source_text",
        lambda path, media_type: ("", 3),
    )
    monkeypatch.setattr(
        "parsing_agent.ingestion.run_ocr",
        lambda **kwargs: OcrResult(
            applied=True,
            provider="surya",
            text="OCR body",
            metadata={
                "applied": True,
                "provider": "surya",
                "ocr_page_count": 3,
                "ocr_block_count": 12,
            },
            artifacts={"ocr_text": ocr_dir / "ocr_text.md"},
        ),
    )

    source = build_document_source(
        pdf_path,
        run_id="ocr-test",
        config=WorkflowConfig(ocr_enabled=True, ocr_min_text_characters=50),
        artifact_dir=ocr_dir,
    )

    assert source.extracted_text == "OCR body"
    assert source.ocr_metadata["applied"] is True
    assert source.ocr_metadata["provider"] == "surya"
    assert source.ocr_metadata["ocr_block_count"] == 12
    assert source.ocr_artifacts["ocr_text"].endswith("ocr_text.md")
