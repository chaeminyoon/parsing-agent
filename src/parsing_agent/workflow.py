from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import TypedDict
from uuid import uuid4

from langsmith import Client, tracing_context
from langgraph.graph import END, START, StateGraph

from parsing_agent.config import WorkflowConfig
from parsing_agent.evaluation import DeterministicEvaluator
from parsing_agent.filetype import is_image_source, is_pdf_source
from parsing_agent.format_parsers import STRUCTURED_SUFFIX_PARSERS
from parsing_agent.ingestion import build_document_source
from parsing_agent.judge import build_default_judge
from parsing_agent.interfaces import CandidateEvaluator, CandidateRepairer
from parsing_agent.llm_repair import build_default_targeted_text_repairer
from parsing_agent.llm_usage import llm_usage_summary, reset_llm_usage
from parsing_agent.monitoring import append_judge_feedback_record
from parsing_agent.models import (
    DocumentSource,
    DocumentSummary,
    EvaluationMetrics,
    ParseCandidate,
    RepairAction,
    WorkflowResult,
    load_document_source_text,
)
from parsing_agent.parsers import ParserRegistry, build_default_parser_registry
from parsing_agent.repair import (
    HeuristicRepairer,
    RepairTarget,
    apply_table_normalizations,
    identify_repair_targets,
)
from parsing_agent.reporting import write_workflow_artifacts
from parsing_agent.visual_repair import (
    OpenAIVisualTableRecoverer,
    _page_table_selector_from_label,
    replace_page_table_block,
    replace_table_block,
)
from parsing_agent.visual_repair import build_default_visual_table_recoverer

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


_SUPPORT_PAGE_MARKER_RE = re.compile(r"^<!-- page (\d+) -->$")
_TABLE_REFERENCE_LINE_RE = re.compile(r"^\[Table reference:.*\]$")
_IMAGE_OMITTED_LINE_RE = re.compile(r"^\[Image block omitted:.*\]$")


def _summarize_ocr_metadata(metadata: dict[str, object] | None) -> dict[str, object]:
    if not metadata:
        return {}
    allowed_keys = {
        "enabled",
        "applied",
        "provider",
        "reason",
        "elapsed_ms",
        "input_text_characters",
        "output_text_characters",
        "page_count",
        "ocr_page_count",
        "ocr_block_count",
        "ocr_table_block_count",
        "ocr_mean_confidence",
    }
    return {key: value for key, value in metadata.items() if key in allowed_keys}


_TRACE_COLLECTION_ITEM_LIMIT = 20


def _summarize_trace_collection(value) -> dict[str, object]:
    """분기 판단에 쓰이는 컬렉션을 문장 없이 구조화된 필드로 요약한다.

    description/notes 같은 자유 문장은 제외하고, 라우팅이 실제로 참조하는
    enum(issue_type, route_name, strategy)과 수치(confidence, score_delta)만
    트레이스에 내보낸다.
    """
    items = list(value)[:_TRACE_COLLECTION_ITEM_LIMIT]
    if items and all(isinstance(item, RepairTarget) for item in items):
        return {
            "type": "repair_targets",
            "count": len(value),
            "items": [
                {
                    "issue_type": target.issue_type,
                    "route_name": target.route_name,
                    "severity": target.severity,
                    "confidence": target.confidence,
                    "table_label": target.table_label,
                    "page_number": target.page_number,
                    "repairability": target.repairability,
                    "expected_gain": target.expected_gain,
                    "risk_level": target.risk_level,
                }
                for target in items
            ],
        }
    if items and all(isinstance(item, RepairPlanStep) for item in items):
        return {
            "type": "repair_plan",
            "count": len(value),
            "items": [
                {
                    "strategy": step.strategy,
                    "route_name": step.route_name,
                    "priority": step.priority,
                    "expected_gain": step.expected_gain,
                    "estimated_cost": step.estimated_cost,
                    "risk_level": step.risk_level,
                    "skip_reason": step.skip_reason,
                    "issue_types": sorted({target.issue_type for target in step.targets}),
                }
                for step in items
            ],
        }
    if items and all(isinstance(item, RepairAction) for item in items):
        return {
            "type": "repair_actions",
            "count": len(value),
            "items": [
                {
                    "action_name": action.action_name,
                    "issue_type": action.issue_type,
                    "route_name": action.route_name,
                }
                for action in items
            ],
        }
    if items and all(isinstance(item, RepairOutcome) for item in items):
        return {
            "type": "repair_outcomes",
            "count": len(value),
            "items": [
                {
                    "action_name": outcome.action_name,
                    "issue_type": outcome.issue_type,
                    "route_name": outcome.route_name,
                    "score_delta": outcome.score_delta,
                    "verification_passed": outcome.verification_passed,
                    "failure_reason": outcome.failure_reason,
                }
                for outcome in items
            ],
        }
    return {"type": "collection", "count": len(value)}


def _summarize_langsmith_payload(payload: dict) -> dict[str, object]:
    """Replace graph state with a trace-safe operational summary."""
    fields: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, DocumentSource):
            fields[str(key)] = {
                "type": "DocumentSource",
                "filename": value.path.name,
                "size_bytes": value.size_bytes,
                "media_type": value.media_type,
                "page_count": value.page_count,
                "has_extracted_text": value.extracted_text is not None,
                "extracted_text_character_count": len(value.extracted_text or ""),
                "ocr": _summarize_ocr_metadata(value.ocr_metadata),
            }
            continue
        if isinstance(value, ParseCandidate):
            image_data = value.metadata.get("embedded_image_data_urls")
            fields[str(key)] = {
                "type": "ParseCandidate",
                "parser_name": value.parser_name,
                "format_name": value.format_name,
                "content_character_count": len(value.content)
                or value.metadata.get("content_character_count", 0),
                "metadata_keys": sorted(str(item) for item in value.metadata)[:30],
                "embedded_image_count": len(image_data) if isinstance(image_data, dict) else 0,
            }
            continue
        if isinstance(value, EvaluationMetrics):
            fields[str(key)] = {
                "type": "EvaluationMetrics",
                "total_score": value.total_score,
                "text_coverage": value.text_coverage,
                "normalized_similarity": value.normalized_similarity,
                "structure_retention": value.structure_retention,
                "table_preservation": value.table_preservation,
                "llm_judge_score": value.llm_judge_score,
                "table_cell_similarity": value.table_cell_similarity,
                "table_issues": list(value.table_issues),
                "issue_types": sorted({issue.issue_type for issue in value.issues}),
                "judge_table_finding_count": 0
                if value.judge_result is None
                else len(value.judge_result.table_findings),
            }
            continue
        if isinstance(value, WorkflowResult):
            fields[str(key)] = {
                "type": "WorkflowResult",
                "parser_name": value.best_candidate.parser_name,
                "parsed_text_character_count": len(value.best_candidate.content),
                "quality_score": value.metrics.total_score,
                "quality_gate_passed": bool(
                    value.report.get("quality_gate", {}).get("passed", False)
                ),
                "repair_count": len(value.repairs),
            }
            continue
        if isinstance(value, dict):
            fields[str(key)] = {
                "type": "mapping",
                "key_count": len(value),
                "keys": sorted(str(item) for item in value)[:30],
            }
            continue
        if isinstance(value, (list, tuple, set)):
            fields[str(key)] = _summarize_trace_collection(value)
            continue
        if isinstance(value, str):
            fields[str(key)] = {"type": "text", "character_count": len(value)}
            continue
        fields[str(key)] = {"type": type(value).__name__}

    return {"trace_payload_policy": "summary_only", "fields": fields}


class WorkflowRunner:
    def __init__(
        self,
        config: WorkflowConfig | None = None,
        parser_registry: ParserRegistry | None = None,
        evaluator: CandidateEvaluator | None = None,
        repairer: CandidateRepairer | None = None,
    ) -> None:
        self._config = config or WorkflowConfig()
        self._parser_registry = parser_registry or build_default_parser_registry()
        self._validate_configured_parsers()
        self._evaluator = evaluator or DeterministicEvaluator(self._config, judge=build_default_judge(self._config))
        self._repairer = repairer or HeuristicRepairer(
            visual_table_recoverer=build_default_visual_table_recoverer(self._config),
            text_repairer=build_default_targeted_text_repairer(self._config),
        )
        self._source_text_cache: dict[str, str] = {}
        self._candidate_content_cache: dict[str, str] = {}
        self._graph = self._build_graph()

    def run(
        self,
        input_path: Path,
        output_dir: Path,
        run_id: str | None = None,
    ) -> tuple[WorkflowResult, dict[str, Path]]:
        self._source_text_cache.clear()
        self._candidate_content_cache.clear()
        reset_llm_usage()
        resolved_output_dir = Path(output_dir)
        source = self._externalize_source_text(
            build_document_source(
                Path(input_path),
                run_id=run_id or uuid4().hex,
                config=self._config,
                artifact_dir=resolved_output_dir / "ocr",
            )
        )
        trace_metadata = self._langsmith_metadata(source)
        invoke_config = {
            "run_name": "parsing-agent-workflow",
            "tags": ["parsing-agent-system", "langgraph"],
            "metadata": trace_metadata,
        }
        with tracing_context(
            enabled=self._config.langsmith_tracing,
            project_name=self._config.langsmith_project,
            tags=["parsing-agent-system", "langgraph"],
            metadata=trace_metadata,
            client=self._build_langsmith_client(),
        ):
            graph_state = self._graph.invoke({"source": source}, config=invoke_config)
        result = graph_state["result"]
        expected_artifacts = self._artifact_paths(output_dir, source.path)
        result.artifacts = {name: str(path) for name, path in expected_artifacts.items()}
        feedback_log_path = append_judge_feedback_record(self._config, result)
        result.report.setdefault("monitoring", {})
        result.report["monitoring"]["judge_feedback_log_path"] = str(feedback_log_path)
        result.report["monitoring"]["llm_usage"] = llm_usage_summary()
        written_artifacts = write_workflow_artifacts(result, output_dir)
        artifacts = self._finalize_artifacts(written_artifacts, expected_artifacts)
        artifacts.update({name: Path(path) for name, path in result.source.ocr_artifacts.items()})
        result.artifacts = {name: str(path) for name, path in artifacts.items()}
        return result, artifacts

    def get_graph(self):
        return self._graph.get_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("parse", self._parse_document_node)
        graph.add_node("evaluate", self._evaluate_candidate_node)
        graph.add_node("inspect", self._inspect_quality_issues_node)
        graph.add_node("route", self._route_repair_strategy_node)
        graph.add_node("repair", self._repair_candidate_node)
        graph.add_node("finalize", self._finalize_output_node)
        graph.add_edge(START, "parse")
        graph.add_edge("parse", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {
                "inspect": "inspect",
                "finalize": "finalize",
            },
        )
        graph.add_conditional_edges(
            "inspect",
            self._route_after_quality_inspection,
            {
                "route": "route",
            },
        )
        graph.add_conditional_edges(
            "route",
            self._route_after_repair_strategy,
            {
                "repair": "repair",
                "finalize": "finalize",
            },
        )
        graph.add_edge("repair", "evaluate")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _parse_document_node(self, state: WorkflowState) -> WorkflowState:
        source = self._materialize_source_text(state["source"])
        base_parser_name = self._base_parser_name_for_source(source)
        parse_errors: list[dict[str, str]] = []
        candidate: ParseCandidate | None = None
        for parser_name in self._parser_fallback_chain(base_parser_name):
            try:
                parsed_candidates = self._run_parsers(source, [parser_name])
            except Exception as exc:  # noqa: BLE001 - 개별 파서 실패는 다음 파서로 폴백한다
                parse_errors.append(
                    {"parser": parser_name, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            if not parsed_candidates:
                parse_errors.append({"parser": parser_name, "error": "no_candidates_produced"})
                continue
            candidate = self._select_base_candidate(source, parsed_candidates)
            break
        if candidate is None:
            error_summary = "; ".join(f"{item['parser']}: {item['error']}" for item in parse_errors)
            raise ValueError(f"All parsers failed for {source.path} ({error_summary})")
        if self._config.langsmith_tracing:
            candidate = self._externalize_candidate_content(candidate)
        return {
            "candidate": candidate,
            "repairs": [],
            "accuracy_snapshots": [],
            "iteration_count": 0,
            "repair_plan_history": [],
            "repair_outcomes": [],
            "diagnosed_issues_history": [],
            "failed_visual_task_keys": [],
            "attempted_repair_routes": [],
            "visual_repair_rejections": [],
            "parse_errors": parse_errors,
        }

    def _parser_fallback_chain(self, base_parser_name: str) -> list[str]:
        chain = [base_parser_name]
        for name in [*self._config.parser_names, "text-fallback", "source-text"]:
            if name not in chain and self._parser_registry.has(name):
                chain.append(name)
        return chain

    def _evaluate_candidate_node(self, state: WorkflowState) -> WorkflowState:
        source = self._materialize_source_text(state["source"])
        current_candidate = self._materialize_candidate_content(state["candidate"])
        snapshots = list(state.get("accuracy_snapshots") or [])
        metrics = self._evaluator.evaluate(source, current_candidate)
        stage = "initial_evaluation" if int(state.get("iteration_count", 0)) == 0 else "post_repair_evaluation"
        snapshots.append(
            self._build_accuracy_snapshot(
                stage=stage,
                iteration=int(state.get("iteration_count", 0)),
                candidate=current_candidate,
                metrics=metrics,
                repair_targets=state.get("repair_targets") or [],
                repair_actions=[],
            )
        )
        repair_outcomes = self._verify_repair_outcomes(
            outcomes=list(state.get("repair_outcomes") or []),
            metrics=metrics,
        )
        updates: WorkflowState = {
            "metrics": metrics,
            "accuracy_snapshots": snapshots,
            "repair_outcomes": repair_outcomes,
        }
        best_metrics = state.get("best_metrics")
        best_candidate = state.get("best_candidate")
        if best_metrics is None or best_candidate is None or metrics.total_score >= best_metrics.total_score:
            updates["best_candidate"] = state["candidate"]
            updates["best_metrics"] = metrics
        else:
            # 수리가 점수를 악화시킨 경우: 최고 성적 후보로 되돌려 다음
            # 라운드와 finalize가 악화된 결과 위에서 진행되지 않게 한다.
            rollback_events = list(state.get("rollback_events") or [])
            rollback_events.append(
                {
                    "iteration": int(state.get("iteration_count", 0)),
                    "stage": stage,
                    "regressed_score": float(metrics.total_score),
                    "restored_score": float(best_metrics.total_score),
                }
            )
            snapshots[-1] = {**snapshots[-1], "rolled_back": True}
            updates["candidate"] = best_candidate
            updates["metrics"] = best_metrics
            updates["rollback_events"] = rollback_events
        return updates

    def _route_after_evaluation(self, state: WorkflowState) -> str:
        metrics = state.get("metrics")
        iteration_count = int(state.get("iteration_count", 0))
        if metrics is not None and self._needs_candidate_repair(metrics, iteration_count):
            return "inspect"
        return "finalize"

    def _inspect_quality_issues_node(self, state: WorkflowState) -> WorkflowState:
        metrics = state.get("metrics")
        candidate = state.get("candidate")
        if metrics is None or candidate is None:
            return {"repair_targets": []}
        targets = identify_repair_targets(
            self._materialize_source_text(state["source"]),
            self._materialize_candidate_content(candidate),
            metrics,
        )
        history = list(state.get("diagnosed_issues_history") or [])
        history.append(
            {
                "iteration": int(state.get("iteration_count", 0)),
                "issues": [self._target_summary(target) for target in targets],
            }
        )
        return {"repair_targets": targets, "diagnosed_issues_history": history}

    def _route_after_quality_inspection(self, state: WorkflowState) -> str:
        return "route"

    def _route_repair_strategy_node(self, state: WorkflowState) -> WorkflowState:
        repair_targets = list(state.get("repair_targets") or [])
        iteration_count = int(state.get("iteration_count", 0))
        score_delta = self._latest_snapshot_score_delta(state)
        # 액션이 하나도 안 나온 no-op 스텝도 '시도됨'으로 집계해야
        # 같은 route를 헛돌지 않고 LLM 승격/스킵 판단이 가능하다.
        attempted_routes = set(state.get("attempted_repair_routes") or [])
        attempted_routes.update(
            action.route_name
            for action in state.get("repairs") or []
            if isinstance(action, RepairAction) and action.route_name
        )
        stalled = iteration_count >= 1 and score_delta is not None and score_delta < 0.01
        skipped_steps: list[RepairPlanStep] = []
        filtered_targets: list[RepairTarget] = []
        for target in repair_targets:
            if self._should_skip_repair_target(target, iteration_count, score_delta, attempted_routes):
                skipped_steps.append(
                    self._build_repair_plan_step(
                        strategy=self._repair_strategy_for_target(
                            target, stalled=stalled, attempted_routes=attempted_routes
                        ),
                        route_name=target.route_name,
                        targets=[target],
                        iteration_count=iteration_count,
                        score_delta=score_delta,
                        skip_reason="route_already_attempted_after_stalled_score",
                    )
                )
                continue
            filtered_targets.append(target)
        grouped_targets: dict[tuple[str, str], list[RepairTarget]] = {}
        for target in filtered_targets:
            strategy = self._repair_strategy_for_target(
                target, stalled=stalled, attempted_routes=attempted_routes
            )
            grouped_targets.setdefault((strategy, target.route_name), []).append(target)
        plan: list[RepairPlanStep] = []
        for strategy, route_name in sorted(
            grouped_targets,
            key=lambda item: (
                self._repair_strategy_priority(item[0], iteration_count, score_delta),
                item[1],
            ),
        ):
            step = self._build_repair_plan_step(
                strategy=strategy,
                route_name=route_name,
                targets=grouped_targets[(strategy, route_name)],
                iteration_count=iteration_count,
                score_delta=score_delta,
            )
            if step.skip_reason is None:
                plan.append(step)
            else:
                skipped_steps.append(step)
        plan_history = list(state.get("repair_plan_history") or [])
        plan_history.append(
            {
                "iteration": iteration_count,
                "score_delta": score_delta,
                "steps": [self._plan_step_summary(step) for step in [*plan, *skipped_steps]],
            }
        )
        return {"repair_plan": plan, "repair_plan_history": plan_history}

    def _route_after_repair_strategy(self, state: WorkflowState) -> str:
        if not state.get("repair_plan"):
            return "finalize"
        return "repair"

    def _repair_candidate_node(self, state: WorkflowState) -> WorkflowState:
        source = self._materialize_source_text(state["source"])
        metrics = state.get("metrics")
        candidate = state.get("candidate")
        if metrics is None or candidate is None:
            return {}
        materialized_candidate = self._materialize_candidate_content(candidate)
        repair_plan = list(state.get("repair_plan") or [])
        current_repairs = list(state.get("repairs") or [])
        current_outcomes = list(state.get("repair_outcomes") or [])
        failed_visual_task_keys = list(state.get("failed_visual_task_keys") or [])
        repaired_candidate = materialized_candidate
        actions: list[RepairAction] = []
        visual_targets: list[RepairTarget] = []
        attempted_repair_routes = list(state.get("attempted_repair_routes") or [])
        visual_repair_rejections = list(state.get("visual_repair_rejections") or [])

        def record_attempt(route_name: str) -> None:
            if route_name not in attempted_repair_routes:
                attempted_repair_routes.append(route_name)

        for step in repair_plan:
            if step.strategy == "visual_table_repair":
                visual_targets.extend(step.targets)
                record_attempt(step.route_name)
                continue
            if step.strategy == "llm_text_repair" and isinstance(self._repairer, HeuristicRepairer):
                record_attempt(f"llm:{step.route_name}")
                repaired_candidate, llm_actions = self._repairer.repair_llm_targets(
                    source,
                    repaired_candidate,
                    metrics,
                    list(step.targets),
                    max_targets=self._config.llm_text_repair_max_targets,
                )
                actions.extend(llm_actions)
                continue
            step_targets = list(step.targets)
            record_attempt(step.route_name)
            if isinstance(self._repairer, HeuristicRepairer):
                try:
                    repaired_candidate, heuristic_actions = self._repairer.repair_heuristics(
                        source,
                        repaired_candidate,
                        metrics,
                        targets=step_targets,
                    )
                except TypeError:
                    repaired_candidate, heuristic_actions = self._repairer.repair_heuristics(
                        source,
                        repaired_candidate,
                        metrics,
                    )
            else:
                repaired_candidate, heuristic_actions = self._repairer.repair(
                    source,
                    repaired_candidate,
                    metrics,
                )
            actions.extend(heuristic_actions)

        if visual_targets and isinstance(self._repairer, HeuristicRepairer):
            if (
                self._config.repair_fanout_enabled
                and self._should_plan_chunk_repairs(metrics)
            ):
                try:
                    chunk_tasks = self._repairer.plan_chunk_repairs(
                        source,
                        repaired_candidate,
                        metrics,
                        self._config.repair_fanout_max_tasks,
                        targets=visual_targets,
                    )
                except TypeError:
                    chunk_tasks = self._repairer.plan_chunk_repairs(
                        source,
                        repaired_candidate,
                        metrics,
                        self._config.repair_fanout_max_tasks,
                    )
                chunk_tasks = [
                    task
                    for task in chunk_tasks
                    if self._visual_task_key(
                        task.table_label,
                        task.page_number,
                        task.issue_types,
                    )
                    not in failed_visual_task_keys
                ]
                if chunk_tasks:
                    repair_task_results = [
                        result
                        for task in chunk_tasks
                        for result in self._repair_chunk_node(
                            {
                                "source": source,
                                "task": task,
                                "candidate": repaired_candidate,
                            }
                        )["repair_task_results"]
                    ]
                    for chunk_result in repair_task_results:
                        visual_repair_rejections.extend(chunk_result.rejections)
                    merged = self._merge_repair_chunks_node(
                        {
                            "candidate": repaired_candidate,
                            "pending_candidate": repaired_candidate,
                            "pending_actions": [],
                            "repair_tasks": [
                                RepairChunkTask(
                                    task_id=task.task_id,
                                    table_label=task.table_label,
                                    page_number=task.page_number,
                                    issue_types=task.issue_types,
                                    preferred_output_format=task.preferred_output_format,
                                )
                                for task in chunk_tasks
                            ],
                            "repair_task_results": repair_task_results,
                        }
                    )
                    failed_by_task_id = {
                        result.task_id
                        for result in repair_task_results
                        if result.candidate is None
                    }
                    for task in chunk_tasks:
                        if task.task_id not in failed_by_task_id:
                            continue
                        task_key = self._visual_task_key(
                            task.table_label,
                            task.page_number,
                            task.issue_types,
                        )
                        if task_key not in failed_visual_task_keys:
                            failed_visual_task_keys.append(task_key)
                    if merged:
                        repaired_candidate = self._materialize_candidate_content(merged["pending_candidate"])
                        actions.extend(list(merged.get("pending_actions") or []))
        elif visual_targets:
            repaired_candidate, visual_actions = self._repairer.repair(source, repaired_candidate, metrics)
            actions.extend(visual_actions)

        if self._config.langsmith_tracing:
            repaired_candidate = self._externalize_candidate_content(repaired_candidate)
        pending_outcomes = self._build_pending_repair_outcomes(
            actions=actions,
            metrics=metrics,
            repair_plan=repair_plan,
        )
        return {
            "candidate": repaired_candidate,
            "repairs": current_repairs + actions,
            "repair_outcomes": current_outcomes + pending_outcomes,
            "iteration_count": int(state.get("iteration_count", 0)) + 1,
            "repair_targets": [],
            "repair_plan": [],
            "failed_visual_task_keys": failed_visual_task_keys,
            "attempted_repair_routes": attempted_repair_routes,
            "visual_repair_rejections": visual_repair_rejections,
        }

    def _repair_chunk_node(self, state) -> WorkflowState:
        if not isinstance(self._repairer, HeuristicRepairer):
            return {
                "repair_task_results": [
                    RepairChunkResult(task_id=state["task"].task_id)
                ]
            }
        task = state["task"]
        rejections: list[dict[str, object]] = []
        try:
            try:
                result = self._repairer.apply_chunk_repair(
                    state["source"], state["candidate"], task, rejection_sink=rejections
                )
            except TypeError:
                result = self._repairer.apply_chunk_repair(state["source"], state["candidate"], task)
        except Exception as exc:  # noqa: BLE001 - 청크 하나의 실패가 나머지 수리를 막으면 안 된다
            rejections.append(
                {
                    "task_id": task.task_id,
                    "table_label": getattr(task, "table_label", None),
                    "page_number": getattr(task, "page_number", None),
                    "reason": "chunk_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            result = None
        if result is None:
            return {
                "repair_task_results": [
                    RepairChunkResult(task_id=task.task_id, rejections=rejections)
                ]
            }
        candidate, action = result
        patch_label = candidate.metadata.get("repair_chunk_table_label")
        patch_markdown = candidate.metadata.get("repair_chunk_markdown")
        if isinstance(patch_label, str) and isinstance(patch_markdown, str):
            candidate = replace(candidate, content="")
        return {
            "repair_task_results": [
                RepairChunkResult(
                    task_id=task.task_id,
                    candidate=candidate,
                    action=action,
                )
            ]
        }

    def _merge_repair_chunks_node(self, state: WorkflowState) -> WorkflowState:
        tasks = {task.task_id: task for task in state.get("repair_tasks") or []}
        if not tasks:
            return {}
        current_candidate = self._materialize_candidate_content(
            state.get("pending_candidate") or state["candidate"]
        )
        pending_actions = list(state.get("pending_actions") or [])
        for result in sorted(
            (result for result in state.get("repair_task_results") or [] if result.task_id in tasks),
            key=lambda item: tasks[item.task_id].page_number,
        ):
            if result.candidate is None or result.action is None:
                continue
            patch_label = result.candidate.metadata.get("repair_chunk_table_label")
            patch_markdown = result.candidate.metadata.get("repair_chunk_markdown")
            if isinstance(patch_label, str) and isinstance(patch_markdown, str):
                if patch_label.startswith("__page_table__:"):
                    page_number, table_index = _page_table_selector_from_label(patch_label)
                    transformed_content = replace_table_block(
                        current_candidate.content,
                        patch_label,
                        patch_markdown,
                        candidate_metadata=current_candidate.metadata,
                    )
                    if transformed_content == current_candidate.content:
                        transformed_content = (
                            replace_page_table_block(
                                current_candidate.content,
                                page_number,
                                patch_markdown,
                                table_index=table_index,
                            )
                            if page_number is not None
                            else current_candidate.content
                        )
                else:
                    transformed_content = replace_table_block(
                        current_candidate.content,
                        patch_label,
                        patch_markdown,
                        candidate_metadata=current_candidate.metadata,
                    )
                current_candidate = replace(
                    current_candidate,
                    content=transformed_content,
                    repaired_from=current_candidate.repaired_from or current_candidate.parser_name,
                )
            else:
                current_candidate = result.candidate
            pending_actions.append(result.action)
        current_candidate = replace(
            current_candidate,
            metadata={
                **current_candidate.metadata,
                "repair_actions": [action.action_name for action in pending_actions],
                "repair_issue_types": [action.issue_type for action in pending_actions if action.issue_type is not None],
                "repair_routes": [action.route_name for action in pending_actions if action.route_name is not None],
            },
        )
        return {
            "pending_candidate": self._externalize_candidate_content(current_candidate)
            if self._config.langsmith_tracing
            else current_candidate,
            "pending_actions": pending_actions,
        }

    def _finalize_output_node(self, state: WorkflowState) -> WorkflowState:
        candidate = self._materialize_candidate_content(state["candidate"])
        metrics = state["metrics"]
        repairs = list(state.get("repairs") or [])
        post_loop_normalizations: list[str] = []
        if self._config.post_loop_normalization_enabled:
            # 채점 루프가 끝난 뒤의 무손실 표 정규화. 현재 결정적 채점기가
            # 이 정규화들을 감점하기 때문에(라벨 기반 표 메트릭의 오판)
            # 루프 안이 아니라 여기서 적용한다 — apply_table_normalizations 참고.
            normalized_content, post_loop_normalizations = apply_table_normalizations(candidate.content)
            if post_loop_normalizations:
                candidate = replace(candidate, content=normalized_content)
                metrics.notes.append(
                    "Post-loop table normalizations applied after scoring "
                    f"(metrics reflect pre-normalization content): {', '.join(post_loop_normalizations)}"
                )
        summary = self._build_document_summary(state["source"], candidate.content)
        result = WorkflowResult(
            source=state["source"],
            best_candidate=candidate,
            metrics=metrics,
            document_summary=summary,
            repairs=repairs,
            report={
                "quality_gate": {
                    "passed": not self._quality_gate_failures(metrics),
                    "selected_candidate_passed": not self._quality_gate_failures(metrics),
                    "selected_candidate_failed_checks": self._quality_gate_failures(metrics),
                },
                "monitoring": {
                    "judge_grounding_pages": []
                    if metrics.judge_result is None
                    else metrics.judge_result.metadata.get("grounding_pages", []),
                    "used_chunk_repairs": any(
                        action.route_name == "recover_tables_from_pdf_image" for action in repairs
                    ),
                    "image_caption_enrichment": {
                        "count": 0,
                        "paths": [],
                    },
                    "failed_visual_task_keys": list(state.get("failed_visual_task_keys") or []),
                    "post_loop_normalizations": post_loop_normalizations,
                    "visual_repair_rejections": list(state.get("visual_repair_rejections") or []),
                    "parse_errors": list(state.get("parse_errors") or []),
                    "rollback_events": list(state.get("rollback_events") or []),
                },
                "diagnosed_issues": list(state.get("diagnosed_issues_history") or []),
                "repair_plan": list(state.get("repair_plan_history") or []),
                "repair_outcomes": [
                    self._repair_outcome_summary(outcome)
                    for outcome in list(state.get("repair_outcomes") or [])
                ],
                "skipped_repairs": self._skipped_repair_summaries(
                    list(state.get("repair_plan_history") or [])
                ),
                "accuracy_snapshots": list(state.get("accuracy_snapshots") or []),
            },
        )
        return {"result": result}

    def _target_summary(self, target: RepairTarget) -> dict[str, object]:
        return {
            "target_kind": target.target_kind,
            "issue_type": target.issue_type,
            "route_name": target.route_name,
            "description": target.description,
            "table_label": target.table_label,
            "page_number": target.page_number,
            "source_name": target.source_name,
            "severity": target.severity,
            "confidence": target.confidence,
            "source_excerpt": target.source_excerpt,
            "candidate_excerpt": target.candidate_excerpt,
            "bbox": target.bbox,
            "expected_gain": target.expected_gain,
            "estimated_cost": target.estimated_cost,
            "risk_level": target.risk_level,
            "repairability": target.repairability,
        }

    def _build_repair_plan_step(
        self,
        *,
        strategy: str,
        route_name: str,
        targets: list[RepairTarget],
        iteration_count: int,
        score_delta: float | None,
        skip_reason: str | None = None,
    ) -> RepairPlanStep:
        gain_fallback = {"visual_table_repair": 0.08, "llm_text_repair": 0.05}.get(strategy, 0.03)
        cost_fallback = {"visual_table_repair": 1.0, "llm_text_repair": 0.5}.get(strategy, 0.0)
        expected_gain = round(
            sum(
                target.expected_gain if target.expected_gain > 0 else gain_fallback
                for target in targets
            ),
            4,
        )
        estimated_cost = round(
            sum(
                target.estimated_cost if target.estimated_cost > 0 else cost_fallback
                for target in targets
            ),
            4,
        )
        risk_level = self._plan_risk_level(targets)
        priority = self._repair_strategy_priority(strategy, iteration_count, score_delta)
        confidence = max((target.confidence for target in targets), default=0.0)
        effective_skip_reason = skip_reason
        # 비용 게이트는 재수리 라운드에만 적용한다. 첫 라운드는 PRD의
        # inspect → route → repair 강제 경로를 보장해야 하므로 스킵하지 않는다.
        if (
            effective_skip_reason is None
            and iteration_count >= 1
            and estimated_cost > 0
            and expected_gain < 0.03
        ):
            effective_skip_reason = "expected_gain_below_cost_gate"
        if effective_skip_reason is None and risk_level == "high" and confidence < 0.4:
            effective_skip_reason = "low_confidence_high_risk_repair"
        return RepairPlanStep(
            strategy=strategy,
            route_name=route_name,
            targets=tuple(targets),
            priority=priority,
            expected_gain=expected_gain,
            estimated_cost=estimated_cost,
            risk_level=risk_level,
            max_attempts=1 if strategy == "heuristic" else self._config.repair_fanout_max_tasks,
            verification_rule=self._verification_rule_for_strategy(strategy),
            skip_reason=effective_skip_reason,
        )

    def _plan_risk_level(self, targets: list[RepairTarget]) -> str:
        levels = {target.risk_level for target in targets}
        if "high" in levels:
            return "high"
        if "medium" in levels:
            return "medium"
        return "low"

    def _verification_rule_for_strategy(self, strategy: str) -> str:
        if strategy == "visual_table_repair":
            return "table_preservation_or_score_improves"
        return "score_delta_non_negative"

    def _plan_step_summary(self, step: RepairPlanStep) -> dict[str, object]:
        return {
            "strategy": step.strategy,
            "route_name": step.route_name,
            "priority": step.priority,
            "expected_gain": step.expected_gain,
            "estimated_cost": step.estimated_cost,
            "risk_level": step.risk_level,
            "max_attempts": step.max_attempts,
            "verification_rule": step.verification_rule,
            "skip_reason": step.skip_reason,
            "targets": [self._target_summary(target) for target in step.targets],
        }

    def _metrics_summary(self, metrics: EvaluationMetrics) -> dict[str, float]:
        return {
            "text_coverage": float(metrics.text_coverage),
            "normalized_similarity": float(metrics.normalized_similarity),
            "structure_retention": float(metrics.structure_retention),
            "table_preservation": float(metrics.table_preservation),
            "empty_block_penalty": float(metrics.empty_block_penalty),
            "repetition_penalty": float(metrics.repetition_penalty),
            "total_score": float(metrics.total_score),
        }

    def _build_pending_repair_outcomes(
        self,
        *,
        actions: list[RepairAction],
        metrics: EvaluationMetrics,
        repair_plan: list[RepairPlanStep],
    ) -> list[RepairOutcome]:
        if not actions:
            return []
        verification_by_route = {
            step.route_name: step.verification_rule
            for step in repair_plan
        }
        before_metrics = self._metrics_summary(metrics)
        return [
            RepairOutcome(
                action_name=action.action_name,
                issue_type=action.issue_type,
                route_name=action.route_name,
                verification_rule=verification_by_route.get(
                    action.route_name or "",
                    "score_delta_non_negative",
                ),
                before_score=metrics.total_score,
                before_metrics=before_metrics,
            )
            for action in actions
        ]

    def _verify_repair_outcomes(
        self,
        *,
        outcomes: list[RepairOutcome],
        metrics: EvaluationMetrics,
    ) -> list[RepairOutcome]:
        after_metrics = self._metrics_summary(metrics)
        for outcome in outcomes:
            if outcome.verification_passed is not None:
                continue
            outcome.after_score = metrics.total_score
            outcome.after_metrics = after_metrics
            outcome.score_delta = round(metrics.total_score - outcome.before_score, 6)
            outcome.changed_metrics = {
                key: round(after_metrics[key] - outcome.before_metrics.get(key, 0.0), 6)
                for key in after_metrics
            }
            outcome.verification_passed = self._outcome_passed(outcome)
            if not outcome.verification_passed:
                outcome.failure_reason = "repair_did_not_improve_required_metric"
        return outcomes

    def _outcome_passed(self, outcome: RepairOutcome) -> bool:
        score_delta = outcome.score_delta if outcome.score_delta is not None else 0.0
        if outcome.verification_rule == "table_preservation_or_score_improves":
            table_delta = outcome.changed_metrics.get("table_preservation", 0.0)
            return table_delta > 0 or score_delta >= 0
        return score_delta >= 0

    def _repair_outcome_summary(self, outcome: RepairOutcome) -> dict[str, object]:
        return {
            "action_name": outcome.action_name,
            "issue_type": outcome.issue_type,
            "route_name": outcome.route_name,
            "verification_rule": outcome.verification_rule,
            "before_score": outcome.before_score,
            "after_score": outcome.after_score,
            "score_delta": outcome.score_delta,
            "changed_metrics": outcome.changed_metrics,
            "verification_passed": outcome.verification_passed,
            "failure_reason": outcome.failure_reason,
        }

    def _skipped_repair_summaries(
        self,
        repair_plan_history: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        skipped: list[dict[str, object]] = []
        for entry in repair_plan_history:
            steps = entry.get("steps")
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict) or not step.get("skip_reason"):
                    continue
                skipped.append(
                    {
                        "iteration": entry.get("iteration"),
                        "route_name": step.get("route_name"),
                        "strategy": step.get("strategy"),
                        "skip_reason": step.get("skip_reason"),
                        "targets": step.get("targets"),
                    }
                )
        return skipped

    def _run_parsers(self, source: DocumentSource, parser_names: list[str]) -> list[ParseCandidate]:
        candidates: list[ParseCandidate] = []
        for name in parser_names:
            candidates.extend(self._parser_registry.get(name).parse(source, self._config))
        return candidates

    def _base_parser_name_for_source(self, source: DocumentSource) -> str:
        suffix = source.path.suffix.lower()
        # Structured adapters win over the raw-text fallback (text/csv, text/html 등).
        structured_name = STRUCTURED_SUFFIX_PARSERS.get(suffix)
        if structured_name and self._parser_registry.has(structured_name):
            return structured_name
        # 이미지 입력은 인제스천의 OCR이 extracted_text를 채우고 source-text가 소비한다.
        if is_image_source(source) and self._parser_registry.has("source-text"):
            return "source-text"
        if source.media_type.startswith("text/") and self._parser_registry.has("text-fallback"):
            return "text-fallback"
        if suffix in {".md", ".markdown", ".txt"} and self._parser_registry.has("text-fallback"):
            return "text-fallback"
        return self._config.parser_names[0]

    def _build_accuracy_snapshot(
        self,
        *,
        stage: str,
        iteration: int,
        candidate: ParseCandidate,
        metrics: EvaluationMetrics,
        repair_targets: list[object],
        repair_actions: list[RepairAction],
    ) -> dict[str, object]:
        normalized_targets: list[dict[str, object]] = []
        for raw_target in repair_targets:
            target = raw_target
            if not isinstance(target, RepairTarget):
                continue
            normalized_targets.append(
                {
                    "target_kind": target.target_kind,
                    "issue_type": target.issue_type,
                    "route_name": target.route_name,
                    "description": target.description,
                    "table_label": target.table_label,
                    "page_number": target.page_number,
                    "source_name": target.source_name,
                    "severity": target.severity,
                    "confidence": target.confidence,
                    "expected_gain": target.expected_gain,
                    "estimated_cost": target.estimated_cost,
                    "risk_level": target.risk_level,
                    "repairability": target.repairability,
                    "source_excerpt": target.source_excerpt,
                    "candidate_excerpt": target.candidate_excerpt,
                    "bbox": target.bbox,
                }
            )
        return {
            "stage": stage,
            "iteration": iteration,
            "parser_name": candidate.parser_name,
            "format_name": candidate.format_name,
            "content": candidate.content,
            "metrics": {
                "text_coverage": metrics.text_coverage,
                "normalized_similarity": metrics.normalized_similarity,
                "structure_retention": metrics.structure_retention,
                "table_preservation": metrics.table_preservation,
                "empty_block_penalty": metrics.empty_block_penalty,
                "repetition_penalty": metrics.repetition_penalty,
                "llm_judge_score": metrics.llm_judge_score,
                "total_score": metrics.total_score,
                "table_issues": list(metrics.table_issues),
                "notes": list(metrics.notes),
                "issues": [
                    {
                        "issue_type": issue.issue_type,
                        "metric_name": issue.metric_name,
                        "severity": issue.severity,
                        "confidence": issue.confidence,
                        "description": issue.description,
                        "page_number": issue.page_number,
                        "table_label": issue.table_label,
                        "repairability": issue.repairability,
                    }
                    for issue in metrics.issues
                ],
            },
            "repair_targets": normalized_targets,
            "repair_actions": [
                {
                    "action_name": action.action_name,
                    "issue_type": action.issue_type,
                    "route_name": action.route_name,
                    "description": action.description,
                }
                for action in repair_actions
            ],
        }

    def _select_base_candidate(
        self,
        source: DocumentSource,
        candidates: list[ParseCandidate],
    ) -> ParseCandidate:
        for candidate in candidates:
            if not self._candidate_verification_failures(source, candidate):
                return candidate
        if candidates:
            return candidates[0]
        raise ValueError(f"No parse candidates produced for {source.path}")

    def _externalize_source_text(self, source: DocumentSource) -> DocumentSource:
        if source.extracted_text is None:
            return source
        self._source_text_cache[source.run_id] = source.extracted_text
        return replace(source, extracted_text=None)

    def _externalize_candidate_content(self, candidate: ParseCandidate) -> ParseCandidate:
        if not self._config.langsmith_tracing or not candidate.content:
            return candidate
        cache_key = str(candidate.metadata.get("content_cache_key") or uuid4().hex)
        self._candidate_content_cache[cache_key] = candidate.content
        return replace(
            candidate,
            content="",
            metadata={
                **candidate.metadata,
                "content_cache_key": cache_key,
                "content_character_count": len(self._candidate_content_cache[cache_key]),
            },
        )

    def _materialize_candidate_content(self, candidate: ParseCandidate) -> ParseCandidate:
        if candidate.content or not self._config.langsmith_tracing:
            return candidate
        cache_key = candidate.metadata.get("content_cache_key")
        if not isinstance(cache_key, str):
            return candidate
        cached_content = self._candidate_content_cache.get(cache_key)
        if cached_content is None:
            return candidate
        return replace(candidate, content=cached_content)

    def _materialize_source_text(self, source: DocumentSource) -> DocumentSource:
        if source.extracted_text is not None:
            return source
        extracted_text = self._source_text_cache.get(source.run_id)
        if extracted_text is None:
            return source
        return replace(source, extracted_text=extracted_text)

    def _llm_text_repair_available(self) -> bool:
        return (
            self._config.llm_text_repair_enabled
            and isinstance(self._repairer, HeuristicRepairer)
            and getattr(self._repairer, "_text_repairer", None) is not None
        )

    def _repair_strategy_for_target(
        self,
        target: RepairTarget,
        *,
        stalled: bool = False,
        attempted_routes: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        if target.route_name == "recover_tables_from_pdf_image":
            return "visual_table_repair"
        if self._llm_text_repair_available():
            if target.repairability == "llm":
                return "llm_text_repair"
            # heuristic으로 이미 시도했는데 점수가 정체된 이슈는 LLM 수리로 승격한다.
            if (
                stalled
                and target.route_name in attempted_routes
                and f"llm:{target.route_name}" not in attempted_routes
            ):
                return "llm_text_repair"
        return "heuristic"

    def _repair_strategy_priority(
        self,
        strategy: str,
        iteration_count: int,
        score_delta: float | None,
    ) -> int:
        stalled = iteration_count >= 1 and score_delta is not None and score_delta < 0.01
        if stalled:
            if strategy == "visual_table_repair":
                return 0
            if strategy == "heuristic":
                return 1
            return 2
        if strategy == "heuristic":
            return 0
        return 1

    def _should_skip_repair_target(
        self,
        target: RepairTarget,
        iteration_count: int,
        score_delta: float | None,
        attempted_routes: set[str],
    ) -> bool:
        if target.route_name == "recover_tables_from_pdf_image":
            return False
        if iteration_count < 1 or score_delta is None or score_delta >= 0.01:
            return False
        if target.route_name not in attempted_routes:
            return False
        # heuristic이 정체됐어도 LLM 승격이 남아 있으면 스킵하지 않는다.
        if self._llm_text_repair_available() and f"llm:{target.route_name}" not in attempted_routes:
            return False
        return True

    def _latest_snapshot_score_delta(self, state: WorkflowState) -> float | None:
        raw_snapshots = state.get("accuracy_snapshots") or []
        if len(raw_snapshots) < 2:
            return None
        totals: list[float] = []
        for snapshot in raw_snapshots[-2:]:
            if not isinstance(snapshot, dict):
                return None
            metrics = snapshot.get("metrics")
            if not isinstance(metrics, dict):
                return None
            total_score = metrics.get("total_score")
            if not isinstance(total_score, (int, float)):
                return None
            totals.append(float(total_score))
        if len(totals) != 2:
            return None
        return totals[1] - totals[0]

    def _visual_task_key(
        self,
        table_label: str,
        page_number: int,
        issue_types: tuple[str, ...] | list[str] | None,
    ) -> str:
        normalized_issue_types = tuple(sorted(str(issue_type) for issue_type in (issue_types or ())))
        if normalized_issue_types:
            return f"{table_label}|{page_number}|{','.join(normalized_issue_types)}"
        return f"{table_label}|{page_number}|"

    def _needs_candidate_repair(self, metrics: EvaluationMetrics, iteration_count: int) -> bool:
        if self._config.max_repair_rounds <= 0 or iteration_count >= self._config.max_repair_rounds:
            return False
        return bool(self._quality_gate_failures(metrics)) or self._has_repairable_table_issues(metrics)

    def _has_repairable_table_issues(self, metrics: EvaluationMetrics) -> bool:
        return bool(metrics.table_issues)

    def _should_plan_chunk_repairs(self, metrics: EvaluationMetrics) -> bool:
        if not isinstance(self._repairer, HeuristicRepairer):
            return True
        if not isinstance(getattr(self._repairer, "_visual_table_recoverer", None), OpenAIVisualTableRecoverer):
            return True
        return metrics.text_coverage >= self._config.min_text_coverage

    def _candidate_verification_failures(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
    ) -> list[str]:
        if not self._is_usable_candidate(candidate):
            return ["empty_content"]
        if not is_pdf_source(source):
            return []
        if self._has_meaningful_pdf_content(candidate.content):
            return []
        return ["placeholder_only_content"]

    def _validate_configured_parsers(self) -> None:
        for name in self._config.parser_names:
            self._parser_registry.get(name)

    def _artifact_paths(self, output_dir: Path, source_path: Path) -> dict[str, Path]:
        resolved_output_dir = Path(output_dir)
        artifact_stem = source_path.stem
        output_suffix = self._normalized_output_suffix()
        return {
            "parsed_output": resolved_output_dir / f"{artifact_stem}.{output_suffix}",
            "json_report": resolved_output_dir / f"{artifact_stem}.json",
        }

    def _build_langsmith_client(self) -> Client | None:
        if not self._config.langsmith_tracing:
            return None
        if not any(
            (
                self._config.langsmith_api_key,
                self._config.langsmith_endpoint,
                self._config.langsmith_workspace_id,
            )
        ):
            return None
        return Client(
            api_key=self._config.langsmith_api_key,
            api_url=self._config.langsmith_endpoint,
            workspace_id=self._config.langsmith_workspace_id,
            # LangGraph state can contain parsed text and image data URLs. Replace
            # it with an operational summary before it leaves the process.
            hide_inputs=(
                _summarize_langsmith_payload
                if self._config.langsmith_hide_inputs
                else False
            ),
            hide_outputs=(
                _summarize_langsmith_payload
                if self._config.langsmith_hide_outputs
                else False
            ),
        )

    def _langsmith_metadata(self, source: DocumentSource) -> dict[str, object]:
        ocr_metadata = source.ocr_metadata or {}
        return {
            "source_filename": source.path.name,
            "source_size_bytes": source.size_bytes,
            "source_text_character_count": len(self._source_text_cache.get(source.run_id, "")),
            "media_type": source.media_type,
            "run_id": source.run_id,
            "page_count": source.page_count,
            "ocr_enabled": self._config.ocr_enabled,
            "ocr_provider": ocr_metadata.get("provider", self._config.ocr_provider),
            "ocr_applied": ocr_metadata.get("applied", False),
            "ocr_elapsed_ms": ocr_metadata.get("elapsed_ms"),
            "ocr_page_count": ocr_metadata.get("ocr_page_count") or ocr_metadata.get("page_count"),
            "ocr_block_count": ocr_metadata.get("ocr_block_count"),
            "ocr_table_block_count": ocr_metadata.get("ocr_table_block_count"),
            "ocr_mean_confidence": ocr_metadata.get("ocr_mean_confidence"),
            "ocr_output_text_characters": ocr_metadata.get("output_text_characters"),
            "trace_payload_policy": "summary_only",
            "parser_names": list(self._config.parser_names),
            "triage_enabled": self._config.triage_enabled,
            "triage_sample_pages": self._config.triage_sample_pages,
            "max_repair_rounds": self._config.max_repair_rounds,
            "repair_fanout_enabled": self._config.repair_fanout_enabled,
            "judge_multimodal_grounding_enabled": self._config.judge_multimodal_grounding_enabled,
            "min_total_score": self._config.min_total_score,
            "min_text_coverage": self._config.min_text_coverage,
        }

    def _normalized_output_suffix(self) -> str:
        suffix = self._config.output_format.strip().lstrip(".")
        return suffix or "txt"

    def _finalize_artifacts(
        self,
        written_artifacts: dict[str, Path],
        expected_artifacts: dict[str, Path],
    ) -> dict[str, Path]:
        finalized = dict(written_artifacts)
        actual_output_path = written_artifacts["parsed_output"]
        expected_output_path = expected_artifacts["parsed_output"]
        if actual_output_path != expected_output_path:
            expected_output_path.parent.mkdir(parents=True, exist_ok=True)
            if expected_output_path.exists():
                expected_output_path.unlink()
            actual_output_path.replace(expected_output_path)
            finalized["parsed_output"] = expected_output_path
        finalized["json_report"] = expected_artifacts["json_report"]
        return finalized

    def _quality_gate_failures(self, metrics: EvaluationMetrics) -> list[str]:
        failed_checks: list[str] = []
        if metrics.total_score < self._config.min_total_score:
            failed_checks.append("total_score")
        if metrics.text_coverage < self._config.min_text_coverage:
            failed_checks.append("text_coverage")
        hallucination_threshold = self._config.max_hallucination_risk
        hallucination_risk = None if metrics.judge_result is None else metrics.judge_result.hallucination_risk
        if (
            hallucination_threshold is not None
            and hallucination_risk is not None
            and hallucination_risk > hallucination_threshold
        ):
            failed_checks.append("hallucination_risk")
        return failed_checks

    def _content_preview(self, content: str, limit: int = 40) -> str:
        preview = " ".join(content.split())
        if len(preview) <= limit:
            return preview
        return preview[: limit - 3] + "..."

    def _has_meaningful_pdf_content(self, content: str) -> bool:
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _SUPPORT_PAGE_MARKER_RE.match(stripped):
                continue
            if _TABLE_REFERENCE_LINE_RE.match(stripped):
                continue
            if _IMAGE_OMITTED_LINE_RE.match(stripped):
                continue
            return True
        return False

    def _build_document_summary(self, source: DocumentSource, selected_content: str) -> DocumentSummary:
        summary_text = self._summary_text(self._materialize_source_text(source), selected_content)
        return DocumentSummary(
            file_name=source.path.name,
            media_type=source.media_type,
            page_count=source.page_count,
            overview=self._content_preview(summary_text, limit=240),
            stats=self._content_stats(selected_content),
        )

    def _summary_text(self, source: DocumentSource, selected_content: str) -> str:
        try:
            return load_document_source_text(source)
        except (OSError, ValueError):
            return selected_content

    def _content_stats(self, content: str) -> dict[str, int]:
        lines = content.splitlines()
        return {
            "character_count": len(content),
            "word_count": len(content.split()),
            "line_count": len(lines),
            "heading_count": sum(1 for line in lines if line.lstrip().startswith("#")),
            "table_count": (sum(1 for line in lines if "|" in line) // 2) + content.lower().count("<table"),
        }

    def _is_usable_candidate(self, candidate: ParseCandidate) -> bool:
        return bool(candidate.content.strip())
