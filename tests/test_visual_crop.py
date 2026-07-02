"""비전 crop 전략 검증: 라벨 윈도우 확장, 다중 페이지 이어짐 이미지."""

import json

import fitz

from parsing_agent.visual_repair import OpenAIVisualTableRecoverer


def _make_pdf(tmp_path, pages: int = 1, label: str = "표 9.1-1"):
    path = tmp_path / "crop-test.pdf"
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=595, height=842)
        if index == 0:
            page.insert_text((72, 120), label)
            page.insert_text((72, 150), "table body line")
        else:
            page.insert_text((72, 80), "continuation body line")
    document.save(path)
    document.close()
    return path


def _recoverer() -> OpenAIVisualTableRecoverer:
    return OpenAIVisualTableRecoverer(model="gpt-test", api_key="test-key")


def test_label_window_crop_extends_to_page_bottom(tmp_path) -> None:
    pdf = _make_pdf(tmp_path)
    with fitz.open(pdf) as document:
        page = document.load_page(0)
        crop = _recoverer()._build_table_crop(page, 1, "표 9.1-1")

        assert crop.method == "label-window"
        # 고정 높이(420px)가 아니라 페이지 끝까지 잡아야 긴 표가 잘리지 않는다.
        assert crop.clip.y1 == page.rect.height


def _fake_response(label: str) -> dict:
    body = json.dumps(
        {
            "table_label": label,
            "page_number": 1,
            "confidence": 0.9,
            "markdown": "| a | b |\n| --- | --- |\n| 1 | 2 |",
        },
        ensure_ascii=False,
    )
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": body}]}]}


def test_split_multipage_issue_sends_continuation_image(tmp_path, monkeypatch) -> None:
    pdf = _make_pdf(tmp_path, pages=2)
    captured = {}

    def fake_post(*, url, api_key, payload, timeout_seconds):
        captured["payload"] = payload
        return _fake_response("표 9.1-1")

    monkeypatch.setattr("parsing_agent.visual_repair._post_response", fake_post)

    recovery = _recoverer()._recover_single_table(
        pdf, "", "표 9.1-1", page_number=1, issue_types=("split_multipage_table",)
    )

    assert recovery is not None
    content = captured["payload"]["input"][0]["content"]
    images = [item for item in content if item["type"] == "input_image"]
    assert len(images) == 2
    prompt = next(item["text"] for item in content if item["type"] == "input_text")
    assert "continues onto the next page" in prompt


def test_non_split_issue_sends_single_image(tmp_path, monkeypatch) -> None:
    pdf = _make_pdf(tmp_path, pages=2)
    captured = {}

    def fake_post(*, url, api_key, payload, timeout_seconds):
        captured["payload"] = payload
        return _fake_response("표 9.1-1")

    monkeypatch.setattr("parsing_agent.visual_repair._post_response", fake_post)

    recovery = _recoverer()._recover_single_table(
        pdf, "", "표 9.1-1", page_number=1, issue_types=("missing_header",)
    )

    assert recovery is not None
    images = [item for item in captured["payload"]["input"][0]["content"] if item["type"] == "input_image"]
    assert len(images) == 1


def test_split_issue_on_last_page_does_not_crash(tmp_path, monkeypatch) -> None:
    pdf = _make_pdf(tmp_path, pages=1)

    monkeypatch.setattr(
        "parsing_agent.visual_repair._post_response",
        lambda *, url, api_key, payload, timeout_seconds: _fake_response("표 9.1-1"),
    )

    recovery = _recoverer()._recover_single_table(
        pdf, "", "표 9.1-1", page_number=1, issue_types=("split_multipage_table",)
    )

    assert recovery is not None
