"""워크플로 상태/계획 모델 — 노드 사이를 오가는 구조화된 계약.

workflow.py 분할: 상태 dataclass와 WorkflowState TypedDict만 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


from parsing_agent.models import (
    DocumentSource,
    EvaluationMetrics,
    ParseCandidate,
    RepairAction,
    WorkflowResult,
)
from parsing_agent.repair import (
    RepairTarget,
)

@dataclass(frozen=True, slots=True)
class RepairChunkTask:
    task_id: str
    table_label: str
    page_number: int
    issue_types: tuple[str, ...] = ()
    preferred_output_format: str = "markdown"


@dataclass(slots=True)
class RepairChunkResult:
    """visual repair 청크 하나의 결과. candidate가 None이면 거부/실패."""
    task_id: str
    candidate: ParseCandidate | None = None
    action: RepairAction | None = None
    rejections: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RepairPlanStep:
    strategy: str
    route_name: str
    targets: tuple[RepairTarget, ...]
    priority: int = 50
    expected_gain: float = 0.0
    estimated_cost: float = 0.0
    risk_level: str = "low"
    max_attempts: int = 1
    verification_rule: str = "score_delta_non_negative"
    skip_reason: str | None = None


@dataclass(slots=True)
class RepairOutcome:
    action_name: str
    issue_type: str | None
    route_name: str | None
    verification_rule: str
    before_score: float
    before_metrics: dict[str, float]
    after_score: float | None = None
    after_metrics: dict[str, float] | None = None
    score_delta: float | None = None
    changed_metrics: dict[str, float] = field(default_factory=dict)
    verification_passed: bool | None = None
    failure_reason: str | None = None


class WorkflowState(TypedDict, total=False):
    source: DocumentSource
    candidate: ParseCandidate
    metrics: EvaluationMetrics
    repairs: list[RepairAction]
    accuracy_snapshots: list[dict[str, object]]
    iteration_count: int
    repair_targets: list[RepairTarget]
    repair_plan: list[RepairPlanStep]
    repair_plan_history: list[dict[str, object]]
    repair_outcomes: list[RepairOutcome]
    diagnosed_issues_history: list[dict[str, object]]
    failed_visual_task_keys: list[str]
    attempted_repair_routes: list[str]
    visual_repair_rejections: list[dict[str, object]]
    parse_errors: list[dict[str, str]]
    best_candidate: ParseCandidate
    best_metrics: EvaluationMetrics
    rollback_events: list[dict[str, object]]
    result: WorkflowResult
