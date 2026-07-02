import json
from pathlib import Path

from parsing_agent.config import WorkflowConfig
from parsing_agent.judge import OpenAICompatibleJudge, build_default_judge
from parsing_agent.models import DocumentSource, EvaluationMetrics, ParseCandidate


def test_multimodal_judge_sends_pdf_page_images(monkeypatch) -> None:
    captured_payload = {}

    def _fake_post_response(*, url, api_key, payload, timeout_seconds):
        del url, api_key, timeout_seconds
        captured_payload.update(payload)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"overall_score": 0.8, "coverage_score": 0.7, "structure_score": 0.8, "table_score": 0.9, "hallucination_risk": 0.1, "editorial_readiness": 0.75, "notes": ["grounded"], "issues": []}',
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr("parsing_agent.judge._post_response", _fake_post_response)
    monkeypatch.setattr(
        "parsing_agent.judge._render_pdf_page_data_urls",
        lambda source, max_pages: [(1, "data:image/png;base64,AAA"), (2, "data:image/png;base64,BBB")],
    )

    judge = OpenAICompatibleJudge(model="gpt-4.1-mini", api_key="test-key", enable_multimodal_grounding=True)
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="judge-test",
        extracted_text="source text",
        page_count=3,
    )
    candidate = ParseCandidate(parser_name="layout-first-pdf", content="candidate markdown", format_name="md")
    metrics = EvaluationMetrics(
        text_coverage=0.7,
        normalized_similarity=0.7,
        structure_retention=0.7,
        table_preservation=0.7,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
    )

    result = judge.judge(source, candidate, metrics)

    content = captured_payload["input"][1]["content"]
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert content[2]["type"] == "input_image"
    assert result.metadata["grounding_pages"] == [1, 2]
    assert result.notes == ["grounded"]


def test_multimodal_judge_preserves_structured_table_findings(monkeypatch) -> None:
    def _fake_post_response(*, url, api_key, payload, timeout_seconds):
        del url, api_key, payload, timeout_seconds
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                '{"overall_score": 0.8, "coverage_score": 0.7, "structure_score": 0.8, '
                                '"table_score": 0.9, "hallucination_risk": 0.1, "editorial_readiness": 0.75, '
                                '"notes": ["grounded"], "issues": [], '
                                '"table_findings": [{"issue_type": "missing_header", "table_label": "표 4.2-2", "page_number": 4}]}'
                            ),
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr("parsing_agent.judge._post_response", _fake_post_response)
    monkeypatch.setattr(
        "parsing_agent.judge._render_pdf_page_data_urls",
        lambda source, max_pages: [(1, "data:image/png;base64,AAA")],
    )

    judge = OpenAICompatibleJudge(model="gpt-4.1-mini", api_key="test-key", enable_multimodal_grounding=True)
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="judge-structured-findings-test",
        extracted_text="source text",
        page_count=3,
    )
    candidate = ParseCandidate(parser_name="layout-first-pdf", content="candidate markdown", format_name="md")
    metrics = EvaluationMetrics(
        text_coverage=0.7,
        normalized_similarity=0.7,
        structure_retention=0.7,
        table_preservation=0.7,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
    )

    result = judge.judge(source, candidate, metrics)

    assert result.table_findings == [
        {"issue_type": "missing_header", "table_label": "표 4.2-2", "page_number": 4}
    ]


def test_judge_prompt_uses_feedback_log_hints(tmp_path, monkeypatch) -> None:
    captured_payload = {}
    feedback_log = tmp_path / "judge_feedback.jsonl"
    feedback_log.write_text(
        "\n".join(
            [
                json.dumps({"issues": ["표 4.2-2 merged cell issue"]}, ensure_ascii=False),
                json.dumps({"issues": ["table row omission detected"]}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    def _fake_post_chat_completion(*, url, api_key, payload, timeout_seconds):
        del url, api_key, timeout_seconds
        captured_payload.update(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"overall_score": 0.8, "coverage_score": 0.8, "structure_score": 0.8, "table_score": 0.8, "hallucination_risk": 0.1, "editorial_readiness": 0.8, "notes": [], "issues": []}'
                    }
                }
            ]
        }

    monkeypatch.setattr("parsing_agent.judge._post_chat_completion", _fake_post_chat_completion)

    config = WorkflowConfig(
        judge_api_key="test-key",
        judge_base_url="https://api.openai.com/v1",
        judge_model="gpt-4.1-mini",
        judge_multimodal_grounding_enabled=False,
        judge_feedback_log_path=str(feedback_log),
    )
    judge = build_default_judge(config)
    assert judge is not None

    source = DocumentSource(
        path=Path("sample.txt"),
        media_type="text/plain",
        size_bytes=0,
        run_id="judge-feedback-test",
        extracted_text="source text",
    )
    candidate = ParseCandidate(parser_name="text-fallback", content="candidate text", format_name="md")
    metrics = EvaluationMetrics(
        text_coverage=0.7,
        normalized_similarity=0.7,
        structure_retention=0.7,
        table_preservation=0.7,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
    )

    judge.judge(source, candidate, metrics)

    prompt = captured_payload["messages"][1]["content"]
    assert "Extra review instructions from prior failures" in prompt
    assert "merged cells" in prompt


def test_judge_prompt_explicitly_requires_table_findings_when_table_cues_exist(monkeypatch) -> None:
    captured_payload = {}

    def _fake_post_chat_completion(*, url, api_key, payload, timeout_seconds):
        del url, api_key, timeout_seconds
        captured_payload.update(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"overall_score": 0.8, "coverage_score": 0.8, "structure_score": 0.8, "table_score": 0.8, "hallucination_risk": 0.1, "editorial_readiness": 0.8, "notes": [], "issues": [], "table_findings": []}'
                    }
                }
            ]
        }

    monkeypatch.setattr("parsing_agent.judge._post_chat_completion", _fake_post_chat_completion)

    judge = OpenAICompatibleJudge(model="gpt-4.1-mini", api_key="test-key", enable_multimodal_grounding=False)
    source = DocumentSource(
        path=Path("sample.txt"),
        media_type="text/plain",
        size_bytes=0,
        run_id="judge-table-cue-test",
        extracted_text="표 4.2-2 원문\n항목 값",
    )
    candidate = ParseCandidate(parser_name="text-fallback", content="표 4.2-2\n| 항목 | 값 |", format_name="md")
    metrics = EvaluationMetrics(
        text_coverage=0.8,
        normalized_similarity=0.8,
        structure_retention=0.8,
        table_preservation=0.5,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
    )

    judge.judge(source, candidate, metrics)

    prompt = captured_payload["messages"][1]["content"]
    assert "If the document contains table cues" in prompt
    assert "table_findings" in prompt
    assert "Do not leave table_findings empty" in prompt


def test_judge_uses_stratified_evidence_for_long_documents(monkeypatch) -> None:
    captured_payload = {}

    def _fake_post_chat_completion(*, url, api_key, payload, timeout_seconds):
        del url, api_key, timeout_seconds
        captured_payload.update(payload)
        return {
            "choices": [
                {"message": {"content": '{"overall_score": 0.8, "notes": [], "issues": []}'}}
            ]
        }

    monkeypatch.setattr("parsing_agent.judge._post_chat_completion", _fake_post_chat_completion)
    judge = OpenAICompatibleJudge(
        model="gpt-4.1-mini",
        api_key="test-key",
        enable_multimodal_grounding=False,
        max_source_characters=1_200,
        max_candidate_characters=1_200,
        evidence_segments=4,
        table_evidence_limit=2,
    )
    source = DocumentSource(
        path=Path("sample.txt"),
        media_type="text/plain",
        size_bytes=0,
        run_id="judge-evidence-test",
        extracted_text=("intro\n" + "x" * 3_000 + "\n표 4-2\nsource table\n" + "y" * 3_000),
    )
    candidate = ParseCandidate(
        parser_name="text-fallback",
        content=("intro\n" + "a" * 3_000 + "\n표 4-2\n| candidate table |\n" + "b" * 3_000),
        format_name="md",
    )
    metrics = EvaluationMetrics(0.8, 0.8, 0.8, 0.8, 0.0, 0.0, 0.0)

    result = judge.judge(source, candidate, metrics)

    prompt = captured_payload["messages"][1]["content"]
    assert len(prompt) < 4_000
    assert "[Table evidence: 4-2]" in prompt
    assert result.metadata["evidence"]["mode"] == "stratified_evidence"
    assert result.metadata["evidence"]["source_evidence_character_count"] <= 1_200
