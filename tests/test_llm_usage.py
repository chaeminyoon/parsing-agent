"""LLM 호출 사용량 계측 검증."""

from urllib import error as urllib_error

import pytest

from parsing_agent.judge import JudgeUnavailableError, _call_with_retry
from parsing_agent.llm_usage import (
    extract_token_usage,
    llm_usage_summary,
    record_llm_call,
    reset_llm_usage,
)


@pytest.fixture(autouse=True)
def _clean_usage():
    reset_llm_usage()
    yield
    reset_llm_usage()


def test_summary_aggregates_by_stage() -> None:
    record_llm_call(
        stage="judge",
        model="gpt-test",
        duration_ms=120,
        ok=True,
        response_payload={"usage": {"prompt_tokens": 100, "completion_tokens": 20}},
    )
    record_llm_call(stage="judge", model="gpt-test", duration_ms=80, ok=False, error="TimeoutError: x")
    record_llm_call(
        stage="visual_table_recovery",
        model="gpt-vision",
        duration_ms=300,
        ok=True,
        response_payload={"usage": {"input_tokens": 500, "output_tokens": 50}},
    )

    summary = llm_usage_summary()

    assert summary["total_calls"] == 3
    assert summary["total_errors"] == 1
    judge_stage = summary["by_stage"]["judge"]
    assert judge_stage["calls"] == 2
    assert judge_stage["errors"] == 1
    assert judge_stage["prompt_tokens"] == 100
    assert judge_stage["completion_tokens"] == 20
    vision_stage = summary["by_stage"]["visual_table_recovery"]
    assert vision_stage["prompt_tokens"] == 500
    assert vision_stage["models"] == ["gpt-vision"]


def test_extract_token_usage_supports_both_api_shapes() -> None:
    assert extract_token_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 5}}) == (10, 5)
    assert extract_token_usage({"usage": {"input_tokens": 7, "output_tokens": 3}}) == (7, 3)
    assert extract_token_usage({"no_usage": True}) == (None, None)
    assert extract_token_usage(None) == (None, None)


def test_call_with_retry_records_success_with_attempts() -> None:
    calls = {"count": 0}

    def flaky(**kwargs):
        calls["count"] += 1
        if calls["count"] < 2:
            raise urllib_error.URLError("reset")
        return {"ok": True, "usage": {"prompt_tokens": 11, "completion_tokens": 2}}

    _call_with_retry(
        flaky,
        max_retries=2,
        backoff_seconds=0,
        usage_stage="judge",
        payload={"model": "gpt-test"},
    )

    summary = llm_usage_summary()
    assert summary["total_calls"] == 1
    record = summary["calls"][0]
    assert record["ok"] is True
    assert record["attempts"] == 2
    assert record["model"] == "gpt-test"
    assert record["prompt_tokens"] == 11


def test_call_with_retry_records_failure() -> None:
    def dead(**kwargs):
        raise TimeoutError("down")

    with pytest.raises(JudgeUnavailableError):
        _call_with_retry(dead, max_retries=0, backoff_seconds=0, usage_stage="llm_text_repair")

    summary = llm_usage_summary()
    assert summary["total_errors"] == 1
    assert summary["by_stage"]["llm_text_repair"]["errors"] == 1
    assert "TimeoutError" in summary["calls"][0]["error"]
