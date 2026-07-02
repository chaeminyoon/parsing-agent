"""이슈 단위 LLM 텍스트 수리(llm_text_repair 전략) 검증."""

import json
from pathlib import Path

from parsing_agent.config import WorkflowConfig
from parsing_agent.llm_repair import (
    OpenAITargetedTextRepairer,
    TargetedRepairOutcome,
    locate_issue_window,
)
from parsing_agent.models import DocumentSource, EvaluationMetrics, ParseCandidate, RepairAction
from parsing_agent.repair import HeuristicRepairer, RepairTarget
from parsing_agent.workflow import WorkflowRunner


def _make_source(run_id: str) -> DocumentSource:
    return DocumentSource(
        path=Path("sample.txt"),
        media_type="text/plain",
        size_bytes=0,
        run_id=run_id,
        extracted_text="제1장 사업개요\n사업의 목적과 범위를 기술한다.",
    )


def _make_target(
    issue_type: str = "repetition_noise",
    route_name: str = "remove_repeated_lines",
    **kwargs,
) -> RepairTarget:
    return RepairTarget(
        target_kind="text",
        issue_type=issue_type,
        route_name=route_name,
        description="repeated lines detected",
        **kwargs,
    )


# --- 이슈 윈도우 탐색 ----------------------------------------------------------


def test_locate_issue_window_finds_duplicate_lines() -> None:
    content = "\n".join(
        [
            "서론입니다.",
            "사업의 목적과 범위를 기술한다.",
            "본문 중간입니다.",
            "사업의 목적과 범위를 기술한다.",
            "결론입니다.",
        ]
    )
    window = locate_issue_window(content, _make_target(), window_lines=4)

    assert window is not None
    start, end = window
    assert start <= 3 < end


def test_locate_issue_window_returns_none_without_anchor() -> None:
    content = "정상적인 본문 한 줄."
    target = _make_target(issue_type="unknown_issue", route_name="unknown_route")

    assert locate_issue_window(content, target, window_lines=10) is None


# --- OpenAITargetedTextRepairer 가드레일 ---------------------------------------


def _make_repairer(**kwargs) -> OpenAITargetedTextRepairer:
    kwargs.setdefault("max_retries", 0)
    return OpenAITargetedTextRepairer(
        model="gpt-test",
        api_key="test-key",
        min_confidence=0.6,
        window_lines=10,
        **kwargs,
    )


def _fake_completion(payload: dict):
    def fake(**kwargs):
        return {"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]}

    return fake


_DUPLICATE_CONTENT = "\n".join(
    [
        "사업의 목적과 범위를 기술한다.",
        "본문 중간입니다.",
        "사업의 목적과 범위를 기술한다.",
    ]
)


def test_repair_target_applies_confident_fix(monkeypatch) -> None:
    fixed_passage = "사업의 목적과 범위를 기술한다.\n본문 중간입니다."
    monkeypatch.setattr(
        "parsing_agent.llm_repair._post_chat_completion",
        _fake_completion({"fixed_text": fixed_passage, "confidence": 0.9, "changed": True}),
    )
    repairer = _make_repairer()

    outcome = repairer.repair_target(_make_source("llm-fix"), _DUPLICATE_CONTENT, _make_target())

    assert outcome is not None
    assert outcome.content == fixed_passage
    assert outcome.action.action_name == "llm_targeted_text_repair"
    assert outcome.action.route_name == "llm:remove_repeated_lines"


def test_repair_target_rejects_low_confidence(monkeypatch) -> None:
    monkeypatch.setattr(
        "parsing_agent.llm_repair._post_chat_completion",
        _fake_completion({"fixed_text": "수정본", "confidence": 0.3, "changed": True}),
    )

    assert _make_repairer().repair_target(_make_source("llm-lowconf"), _DUPLICATE_CONTENT, _make_target()) is None


def test_repair_target_rejects_unchanged_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "parsing_agent.llm_repair._post_chat_completion",
        _fake_completion({"fixed_text": _DUPLICATE_CONTENT, "confidence": 0.9, "changed": False}),
    )

    assert _make_repairer().repair_target(_make_source("llm-unchanged"), _DUPLICATE_CONTENT, _make_target()) is None


def test_repair_target_rejects_runaway_length(monkeypatch) -> None:
    monkeypatch.setattr(
        "parsing_agent.llm_repair._post_chat_completion",
        _fake_completion({"fixed_text": "긴 내용 " * 500, "confidence": 0.9, "changed": True}),
    )

    assert _make_repairer().repair_target(_make_source("llm-runaway"), _DUPLICATE_CONTENT, _make_target()) is None


def test_repair_target_survives_api_failure(monkeypatch) -> None:
    def exploding(**kwargs):
        raise TimeoutError("api down")

    monkeypatch.setattr("parsing_agent.llm_repair._post_chat_completion", exploding)

    assert _make_repairer().repair_target(_make_source("llm-apidown"), _DUPLICATE_CONTENT, _make_target()) is None


# --- route 에스컬레이션 ---------------------------------------------------------


class _FakeTextRepairer:
    def __init__(self):
        self.calls: list[RepairTarget] = []

    def repair_target(self, source, content, target):
        self.calls.append(target)
        return TargetedRepairOutcome(
            content=content.replace("중복 줄", "정리된 줄", 1),
            action=RepairAction(
                action_name="llm_targeted_text_repair",
                description="fixed",
                before_excerpt="중복 줄",
                after_excerpt="정리된 줄",
                issue_type=target.issue_type,
                route_name=f"llm:{target.route_name}",
            ),
        )


def _stalled_route_state() -> dict:
    return {
        "iteration_count": 1,
        "accuracy_snapshots": [
            {"metrics": {"total_score": 0.50}},
            {"metrics": {"total_score": 0.505}},
        ],
        "repairs": [
            RepairAction(
                action_name="merge_wrapped_lines",
                description="Merge lines",
                before_excerpt="a",
                after_excerpt="b",
                issue_type="wrapped_line_noise",
                route_name="merge_wrapped_lines",
            )
        ],
        "repair_targets": [
            RepairTarget(
                target_kind="text",
                issue_type="wrapped_line_noise",
                route_name="merge_wrapped_lines",
                description="merge wrapped lines",
            ),
        ],
    }


def test_route_escalates_stalled_heuristic_to_llm_repair() -> None:
    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False, llm_text_repair_enabled=True),
        repairer=HeuristicRepairer(text_repairer=_FakeTextRepairer()),
    )

    result = runner._route_repair_strategy_node(_stalled_route_state())

    plan = result["repair_plan"]
    assert len(plan) == 1
    assert plan[0].strategy == "llm_text_repair"
    assert plan[0].targets[0].route_name == "merge_wrapped_lines"


def test_route_skips_target_after_llm_escalation_also_attempted() -> None:
    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False, llm_text_repair_enabled=True),
        repairer=HeuristicRepairer(text_repairer=_FakeTextRepairer()),
    )
    state = _stalled_route_state()
    state["repairs"].append(
        RepairAction(
            action_name="llm_targeted_text_repair",
            description="LLM fix",
            before_excerpt="a",
            after_excerpt="b",
            issue_type="wrapped_line_noise",
            route_name="llm:merge_wrapped_lines",
        )
    )

    result = runner._route_repair_strategy_node(state)

    assert result["repair_plan"] == []
    skipped = result["repair_plan_history"][0]["steps"][0]
    assert skipped["skip_reason"] == "route_already_attempted_after_stalled_score"


# --- repair 노드에서 LLM 스텝 실행 ----------------------------------------------


def test_repair_node_executes_llm_text_repair_step() -> None:
    fake_repairer = _FakeTextRepairer()
    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False, llm_text_repair_enabled=True),
        repairer=HeuristicRepairer(text_repairer=fake_repairer),
    )
    state = _stalled_route_state()
    routed = runner._route_repair_strategy_node(state)
    candidate = ParseCandidate(
        parser_name="text-fallback",
        content="본문입니다.\n중복 줄\n중복 줄",
        format_name="md",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.5,
        normalized_similarity=0.5,
        structure_retention=0.5,
        table_preservation=0.5,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.5,
    )

    result = runner._repair_candidate_node(
        {
            "source": _make_source("llm-repair-node"),
            "candidate": candidate,
            "metrics": metrics,
            "repair_plan": routed["repair_plan"],
            "iteration_count": 1,
        }
    )

    assert len(fake_repairer.calls) == 1
    assert "정리된 줄" in result["candidate"].content
    assert any(action.route_name == "llm:merge_wrapped_lines" for action in result["repairs"])
