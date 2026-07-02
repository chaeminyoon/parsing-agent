"""P0 안정성 수정 검증: judge 재시도/폴백, 파서 폴백 체인, 청크 격리, 수리 롤백."""

from pathlib import Path
from types import SimpleNamespace
from urllib import error as urllib_error

import pytest

from parsing_agent.config import WorkflowConfig
from parsing_agent.evaluation import DeterministicEvaluator
from parsing_agent.judge import JudgeUnavailableError, _call_with_retry, _parse_judge_verdict
from parsing_agent.models import DocumentSource, EvaluationMetrics, ParseCandidate
from parsing_agent.repair import HeuristicRepairer
from parsing_agent.workflow import WorkflowRunner


def _make_source(run_id: str, media_type: str = "text/plain", suffix: str = ".txt") -> DocumentSource:
    return DocumentSource(
        path=Path(f"sample{suffix}"),
        media_type=media_type,
        size_bytes=0,
        run_id=run_id,
        extracted_text="제1장 사업개요\n표 4.2-2 환경영향 조사항목\n본문 내용입니다.",
    )


def _make_candidate(content: str = "제1장 사업개요\n표 4.2-2 환경영향 조사항목\n본문 내용입니다.") -> ParseCandidate:
    return ParseCandidate(parser_name="text-fallback", content=content, format_name="md")


def _make_metrics(total_score: float) -> EvaluationMetrics:
    return EvaluationMetrics(
        text_coverage=total_score,
        normalized_similarity=total_score,
        structure_retention=total_score,
        table_preservation=total_score,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=total_score,
    )


# --- judge JSON 폴백 ---------------------------------------------------------


def test_parse_judge_verdict_accepts_strict_json() -> None:
    assert _parse_judge_verdict('{"overall_score": 0.8}') == {"overall_score": 0.8}


def test_parse_judge_verdict_accepts_fenced_json() -> None:
    raw = '```json\n{"overall_score": 0.7, "notes": ["ok"]}\n```'
    assert _parse_judge_verdict(raw)["overall_score"] == 0.7


def test_parse_judge_verdict_accepts_json_embedded_in_prose() -> None:
    raw = 'Here is my judgement:\n{"overall_score": 0.65}\nHope this helps.'
    assert _parse_judge_verdict(raw)["overall_score"] == 0.65


def test_parse_judge_verdict_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        _parse_judge_verdict("I cannot produce JSON right now.")


# --- judge HTTP 재시도 -------------------------------------------------------


def test_call_with_retry_recovers_from_transient_error() -> None:
    calls = {"count": 0}

    def flaky(**kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise urllib_error.URLError("connection reset")
        return {"ok": True}

    result = _call_with_retry(flaky, max_retries=2, backoff_seconds=0)

    assert result == {"ok": True}
    assert calls["count"] == 3


def test_call_with_retry_does_not_retry_non_retryable_http_error() -> None:
    calls = {"count": 0}

    def unauthorized(**kwargs):
        calls["count"] += 1
        raise urllib_error.HTTPError("https://api", 401, "unauthorized", None, None)

    with pytest.raises(JudgeUnavailableError):
        _call_with_retry(unauthorized, max_retries=3, backoff_seconds=0)
    assert calls["count"] == 1


def test_call_with_retry_raises_after_exhausting_retries() -> None:
    def always_timeout(**kwargs):
        raise TimeoutError("timed out")

    with pytest.raises(JudgeUnavailableError):
        _call_with_retry(always_timeout, max_retries=1, backoff_seconds=0)


# --- judge fail-open ---------------------------------------------------------


class _RaisingJudge:
    def judge(self, source, candidate, metrics):
        raise JudgeUnavailableError("judge api down")


def test_evaluator_falls_back_to_deterministic_metrics_when_judge_fails() -> None:
    evaluator = DeterministicEvaluator(WorkflowConfig(judge_weight=0.25), judge=_RaisingJudge())

    metrics = evaluator.evaluate(_make_source("judge-fail-open"), _make_candidate())

    assert metrics.judge_result is None
    assert metrics.llm_judge_score is None
    assert any("Judge unavailable" in note for note in metrics.notes)
    assert metrics.total_score > 0


def test_evaluator_raises_when_judge_fail_open_disabled() -> None:
    evaluator = DeterministicEvaluator(
        WorkflowConfig(judge_weight=0.25, judge_fail_open=False),
        judge=_RaisingJudge(),
    )

    with pytest.raises(JudgeUnavailableError):
        evaluator.evaluate(_make_source("judge-fail-closed"), _make_candidate())


# --- parse 노드 파서 폴백 체인 -----------------------------------------------


class _FailingParser:
    def parse(self, source, config):
        raise RuntimeError("parser exploded")


class _BackupParser:
    def parse(self, source, config):
        return [ParseCandidate(parser_name="backup", content="복구된 본문 내용", format_name="md")]


class _TwoParserRegistry:
    def __init__(self):
        self._adapters = {"primary": _FailingParser(), "backup": _BackupParser()}

    def get(self, name):
        return self._adapters[name]

    def has(self, name):
        return name in self._adapters


def test_parse_node_falls_back_to_next_parser_when_primary_crashes() -> None:
    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False, parser_names=["primary", "backup"]),
        parser_registry=_TwoParserRegistry(),
    )
    source = _make_source("parse-fallback", media_type="application/pdf", suffix=".pdf")

    result = runner._parse_document_node({"source": source})

    assert result["candidate"].parser_name == "backup"
    assert result["parse_errors"][0]["parser"] == "primary"
    assert "RuntimeError" in result["parse_errors"][0]["error"]


def test_parse_node_raises_with_error_summary_when_all_parsers_fail() -> None:
    class _AllFailRegistry:
        def get(self, name):
            return _FailingParser()

        def has(self, name):
            return name == "primary"

    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False, parser_names=["primary"]),
        parser_registry=_AllFailRegistry(),
    )
    source = _make_source("parse-all-fail", media_type="application/pdf", suffix=".pdf")

    with pytest.raises(ValueError, match="All parsers failed"):
        runner._parse_document_node({"source": source})


# --- visual repair 청크 예외 격리 ---------------------------------------------


class _ExplodingChunkRepairer(HeuristicRepairer):
    def apply_chunk_repair(self, source, candidate, task):
        raise RuntimeError("vision api down")


def test_repair_chunk_node_isolates_chunk_exception() -> None:
    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False),
        repairer=_ExplodingChunkRepairer(),
    )

    result = runner._repair_chunk_node(
        {
            "source": _make_source("chunk-isolation"),
            "task": SimpleNamespace(task_id="task-1"),
            "candidate": _make_candidate(),
        }
    )

    chunk_results = result["repair_task_results"]
    assert len(chunk_results) == 1
    assert chunk_results[0].task_id == "task-1"
    assert chunk_results[0].candidate is None


# --- 수리 악화 시 롤백 --------------------------------------------------------


class _FixedScoreEvaluator:
    def __init__(self, total_score: float):
        self._total_score = total_score

    def evaluate(self, source, candidate):
        return _make_metrics(self._total_score)


def test_evaluate_node_rolls_back_candidate_when_repair_regresses_score() -> None:
    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False),
        evaluator=_FixedScoreEvaluator(0.55),
    )
    best_candidate = _make_candidate("원본 최고 후보")
    best_metrics = _make_metrics(0.8)
    regressed_candidate = _make_candidate("수리로 악화된 후보")

    result = runner._evaluate_candidate_node(
        {
            "source": _make_source("rollback"),
            "candidate": regressed_candidate,
            "iteration_count": 1,
            "accuracy_snapshots": [{"metrics": {"total_score": 0.8}}],
            "best_candidate": best_candidate,
            "best_metrics": best_metrics,
        }
    )

    assert result["candidate"] is best_candidate
    assert result["metrics"] is best_metrics
    assert result["rollback_events"][0]["regressed_score"] == 0.55
    assert result["rollback_events"][0]["restored_score"] == 0.8
    assert result["accuracy_snapshots"][-1]["rolled_back"] is True


def test_evaluate_node_updates_best_candidate_when_score_improves() -> None:
    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False),
        evaluator=_FixedScoreEvaluator(0.9),
    )
    improved_candidate = _make_candidate("수리로 개선된 후보")

    result = runner._evaluate_candidate_node(
        {
            "source": _make_source("best-update"),
            "candidate": improved_candidate,
            "iteration_count": 1,
            "accuracy_snapshots": [{"metrics": {"total_score": 0.8}}],
            "best_candidate": _make_candidate("이전 최고 후보"),
            "best_metrics": _make_metrics(0.8),
        }
    )

    assert result["best_candidate"] is improved_candidate
    assert result["best_metrics"].total_score == 0.9
    assert "rollback_events" not in result
