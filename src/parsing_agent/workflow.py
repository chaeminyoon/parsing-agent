from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import TypedDict
from uuid import uuid4

from langsmith import Client, tracing_context
from langgraph.graph import END, START, StateGraph

from parsing_agent.config import WorkflowConfig
from parsing_agent.evaluation import DeterministicEvaluator
from parsing_agent.ingestion import build_document_source
from parsing_agent.judge import build_default_judge
from parsing_agent.interfaces import CandidateEvaluator, CandidateRepairer
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
from parsing_agent.repair import HeuristicRepairer, RepairTarget, identify_repair_targets
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
    task_id: str
    candidate: ParseCandidate | None = None
    action: RepairAction | None = None


@dataclass(frozen=True, slots=True)
class RepairPlanStep:
    strategy: str
    route_name: str
    targets: tuple[RepairTarget, ...]


class WorkflowState(TypedDict, total=False):
    source: DocumentSource
    candidate: ParseCandidate
    metrics: EvaluationMetrics
    repairs: list[RepairAction]
    accuracy_snapshots: list[dict[str, object]]
    iteration_count: int
    repair_targets: list[RepairTarget]
    repair_plan: list[RepairPlanStep]
    failed_visual_task_keys: list[str]
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
                "table_issue_count": len(value.table_issues),
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
            fields[str(key)] = {"type": "collection", "count": len(value)}
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
            visual_table_recoverer=build_default_visual_table_recoverer(self._config)
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
        parsed_candidates = self._run_parsers(source, [base_parser_name])
        candidate = self._select_base_candidate(source, parsed_candidates)
        if self._config.langsmith_tracing:
            candidate = self._externalize_candidate_content(candidate)
        return {
            "candidate": candidate,
            "repairs": [],
            "accuracy_snapshots": [],
            "iteration_count": 0,
            "failed_visual_task_keys": [],
        }

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
        return {"metrics": metrics, "accuracy_snapshots": snapshots}

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
            state["source"],
            self._materialize_candidate_content(candidate),
            metrics,
        )
        return {"repair_targets": targets}

    def _route_after_quality_inspection(self, state: WorkflowState) -> str:
        return "route"

    def _route_repair_strategy_node(self, state: WorkflowState) -> WorkflowState:
        repair_targets = list(state.get("repair_targets") or [])
        iteration_count = int(state.get("iteration_count", 0))
        score_delta = self._latest_snapshot_score_delta(state)
        attempted_routes = {
            action.route_name
            for action in state.get("repairs") or []
            if isinstance(action, RepairAction) and action.route_name
        }
        filtered_targets: list[RepairTarget] = []
        for target in repair_targets:
            if self._should_skip_repair_target(target, iteration_count, score_delta, attempted_routes):
                continue
            filtered_targets.append(target)
        grouped_targets: dict[tuple[str, str], list[RepairTarget]] = {}
        for target in filtered_targets:
            strategy = self._repair_strategy_for_target(target)
            grouped_targets.setdefault((strategy, target.route_name), []).append(target)
        plan: list[RepairPlanStep] = []
        for strategy, route_name in sorted(
            grouped_targets,
            key=lambda item: (
                self._repair_strategy_priority(item[0], iteration_count, score_delta),
                item[1],
            ),
        ):
            plan.append(
                RepairPlanStep(
                    strategy=strategy,
                    route_name=route_name,
                    targets=tuple(grouped_targets[(strategy, route_name)]),
                )
            )
        return {"repair_plan": plan}

    def _route_after_repair_strategy(self, state: WorkflowState) -> str:
        if not state.get("repair_plan"):
            return "finalize"
        return "repair"

    def _repair_candidate_node(self, state: WorkflowState) -> WorkflowState:
        source = state["source"]
        metrics = state.get("metrics")
        candidate = state.get("candidate")
        if metrics is None or candidate is None:
            return {}
        materialized_candidate = self._materialize_candidate_content(candidate)
        repair_plan = list(state.get("repair_plan") or [])
        current_repairs = list(state.get("repairs") or [])
        failed_visual_task_keys = list(state.get("failed_visual_task_keys") or [])
        repaired_candidate = materialized_candidate
        actions: list[RepairAction] = []
        visual_targets: list[RepairTarget] = []
        for step in repair_plan:
            if step.strategy == "visual_table_repair":
                visual_targets.extend(step.targets)
                continue
            step_targets = list(step.targets)
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
        return {
            "candidate": repaired_candidate,
            "repairs": current_repairs + actions,
            "iteration_count": int(state.get("iteration_count", 0)) + 1,
            "repair_targets": [],
            "repair_plan": [],
            "failed_visual_task_keys": failed_visual_task_keys,
        }

    def _repair_chunk_node(self, state) -> WorkflowState:
        if not isinstance(self._repairer, HeuristicRepairer):
            return {
                "repair_task_results": [
                    RepairChunkResult(task_id=state["task"].task_id)
                ]
            }
        task = state["task"]
        result = self._repairer.apply_chunk_repair(state["source"], state["candidate"], task)
        if result is None:
            return {"repair_task_results": [RepairChunkResult(task_id=task.task_id)]}
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
                },
                "accuracy_snapshots": list(state.get("accuracy_snapshots") or []),
            },
        )
        return {"result": result}

    def _run_parsers(self, source: DocumentSource, parser_names: list[str]) -> list[ParseCandidate]:
        candidates: list[ParseCandidate] = []
        for name in parser_names:
            candidates.extend(self._parser_registry.get(name).parse(source, self._config))
        return candidates

    def _base_parser_name_for_source(self, source: DocumentSource) -> str:
        if source.media_type.startswith("text/") and self._parser_registry.has("text-fallback"):
            return "text-fallback"
        if source.path.suffix.lower() in {".md", ".markdown", ".txt"} and self._parser_registry.has("text-fallback"):
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

    def _repair_strategy_for_target(self, target: RepairTarget) -> str:
        if target.route_name == "recover_tables_from_pdf_image":
            return "visual_table_repair"
        return "heuristic"

    def _repair_strategy_priority(
        self,
        strategy: str,
        iteration_count: int,
        score_delta: float | None,
    ) -> int:
        stalled = iteration_count >= 1 and score_delta is not None and score_delta < 0.01
        if strategy == "visual_table_repair" and stalled:
            return 0
        if strategy == "heuristic" and stalled:
            return 1
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
        return target.route_name in attempted_routes

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
        if source.media_type != "application/pdf" and source.path.suffix.lower() != ".pdf":
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
