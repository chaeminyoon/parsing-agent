from dataclasses import replace
from pathlib import Path

from parsing_agent.config import WorkflowConfig
from parsing_agent.evaluation import TABLE_ISSUE_MERGED_CELL_LOSS, TABLE_ISSUE_NUMERIC_TOKEN_BREAK
from parsing_agent.models import DocumentSource, EvaluationMetrics, ParseCandidate, RepairAction
from parsing_agent.repair import HeuristicRepairer
from parsing_agent.visual_repair import VisualRepairTask
from parsing_agent.repair import RepairTarget
from parsing_agent.workflow import RepairChunkResult, RepairChunkTask, RepairPlanStep, WorkflowRunner


class _FanoutRepairer(HeuristicRepairer):
    def __init__(self) -> None:
        super().__init__(visual_table_recoverer=None)

    def repair_heuristics(self, source, candidate, metrics):
        del source, metrics
        return candidate, []

    def plan_chunk_repairs(self, source, candidate, metrics, max_tasks):
        del source, candidate, metrics
        return [
            VisualRepairTask(
                task_id="task-1",
                table_label="table 4.2-2",
                page_number=4,
                issue_types=(TABLE_ISSUE_MERGED_CELL_LOSS,),
                preferred_output_format="html",
            ),
            VisualRepairTask(
                task_id="task-2",
                table_label="table 4.2-3",
                page_number=5,
                issue_types=(TABLE_ISSUE_NUMERIC_TOKEN_BREAK,),
                preferred_output_format="markdown",
            ),
        ][:max_tasks]

    def apply_chunk_repair(self, source, candidate, task):
        del source
        patched_markdown = f"| item | value |\n| --- | --- |\n| {task.table_label} | patched |"
        updated = replace(
            candidate,
            metadata={
                **candidate.metadata,
                "repair_chunk_table_label": task.table_label,
                "repair_chunk_markdown": patched_markdown,
            },
        )
        action = RepairAction(
            action_name="recover_table_from_pdf_image",
            description=f"Recovered {task.table_label}",
            before_excerpt=candidate.content,
            after_excerpt=patched_markdown,
            issue_type="table_visual_recovery",
            route_name="recover_tables_from_pdf_image",
        )
        return updated, action


class _RaisingVisualRecoverer:
    def recover_task(self, source, candidate_content, task):
        del source, candidate_content, task
        raise ValueError("bad visual repair payload")


class _TrackingVisualRecoverer:
    def __init__(self) -> None:
        self.called = False

    def recover_task(self, source, candidate_content, task):
        del source, candidate_content, task
        self.called = True
        return None


def test_repair_fanout_routes_chunk_tasks_and_merges_results() -> None:
    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, repair_fanout_max_tasks=4),
        repairer=_FanoutRepairer(),
    )
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="fanout-test",
        extracted_text="source",
        page_count=10,
    )
    metrics = EvaluationMetrics(
        text_coverage=0.4,
        normalized_similarity=0.4,
        structure_retention=0.4,
        table_preservation=0.2,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.3,
    )
    candidate = ParseCandidate(
        parser_name="layout-first-pdf",
        content=(
            "table 4.2-2\n"
            "| old | first |\n"
            "| --- | --- |\n"
            "| 1 | 2 |\n\n"
            "table 4.2-3\n"
            "| old | second |\n"
            "| --- | --- |\n"
            "| 3 | 4 |\n"
        ),
        format_name="md",
    )
    prepared = runner._repair_candidate_node(
        {
            "source": source,
            "candidate": candidate,
            "metrics": metrics,
            "iteration_count": 0,
            "repairs": [],
            "repair_plan": [
                RepairPlanStep(
                    strategy="visual_table_repair",
                    route_name="recover_tables_from_pdf_image",
                    targets=(
                        RepairTarget(
                            target_kind="table",
                            issue_type="table_missing_header",
                            route_name="recover_tables_from_pdf_image",
                            description="recover tables",
                        ),
                    ),
                )
            ],
        }
    )

    repaired_candidate = prepared["candidate"]
    assert "| table 4.2-2 | patched |" in repaired_candidate.content
    assert "| table 4.2-3 | patched |" in repaired_candidate.content
    assert len(prepared["repairs"]) == 2
    assert prepared["iteration_count"] == 1


def test_merge_repair_chunks_targets_indexed_page_scoped_table() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, repair_fanout_max_tasks=4))
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content=(
            "<!-- page 6 -->\n"
            "| old | first |\n"
            "| --- | --- |\n"
            "| 1 | 2 |\n\n"
            "| old | second |\n"
            "| --- | --- |\n"
            "| 3 | 4 |\n"
        ),
        format_name="md",
    )
    merged = runner._merge_repair_chunks_node(
        {
            "candidate": candidate,
            "pending_candidate": candidate,
            "pending_actions": [],
            "repair_tasks": [
                RepairChunkTask(
                    task_id="page-task-2",
                    table_label="__page_table__:6:2",
                    page_number=6,
                    issue_types=(TABLE_ISSUE_NUMERIC_TOKEN_BREAK,),
                    preferred_output_format="markdown",
                )
            ],
            "repair_task_results": [
                RepairChunkResult(
                    task_id="page-task-2",
                    candidate=ParseCandidate(
                        parser_name="opendataloader-pdf",
                        content=candidate.content,
                        format_name="md",
                        repaired_from="opendataloader-pdf",
                        metadata={
                            "repair_chunk_table_label": "__page_table__:6:2",
                            "repair_chunk_markdown": "| new | second |\n| --- | --- |\n| 30 | 40 |",
                        },
                    ),
                    action=RepairAction(
                        action_name="recover_table_from_pdf_image",
                        description="Recovered second table",
                        before_excerpt="old second",
                        after_excerpt="new second",
                        issue_type="table_visual_recovery",
                        route_name="recover_tables_from_pdf_image",
                    ),
                )
            ],
        }
    )

    pending_candidate = merged["pending_candidate"]
    assert pending_candidate is not None
    assert "| old | first |" in pending_candidate.content
    assert "| new | second |" in pending_candidate.content
    assert "| old | second |" not in pending_candidate.content


def test_apply_chunk_repair_returns_none_when_visual_recoverer_raises() -> None:
    repairer = HeuristicRepairer(visual_table_recoverer=_RaisingVisualRecoverer())
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="visual-repair-exception",
        extracted_text="source",
        page_count=2,
    )
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="broken table",
        format_name="md",
    )
    task = VisualRepairTask(
        task_id="task-1",
        table_label="table 4.2-2",
        page_number=1,
        issue_types=(TABLE_ISSUE_MERGED_CELL_LOSS,),
    )

    result = repairer.apply_chunk_repair(source, candidate, task)

    assert result is None


def test_apply_chunk_repair_skips_out_of_range_page_without_calling_recoverer() -> None:
    recoverer = _TrackingVisualRecoverer()
    repairer = HeuristicRepairer(visual_table_recoverer=recoverer)
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="visual-repair-page-range",
        extracted_text="source",
        page_count=1,
    )
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="broken table",
        format_name="md",
    )
    task = VisualRepairTask(
        task_id="task-2",
        table_label="__page_table__:2",
        page_number=2,
        issue_types=(TABLE_ISSUE_NUMERIC_TOKEN_BREAK,),
    )

    result = repairer.apply_chunk_repair(source, candidate, task)

    assert result is None
    assert recoverer.called is False


class _RetryAwareRepairer(HeuristicRepairer):
    def __init__(self) -> None:
        super().__init__(visual_table_recoverer=None)
        self.applied_labels: list[str] = []

    def repair_heuristics(self, source, candidate, metrics, targets=None):
        del source, metrics, targets
        return candidate, []

    def plan_chunk_repairs(self, source, candidate, metrics, max_tasks, targets=None):
        del source, candidate, metrics, max_tasks, targets
        return [
            VisualRepairTask(
                task_id="task-repeat",
                table_label="표 4.2-2",
                page_number=4,
                issue_types=(TABLE_ISSUE_MERGED_CELL_LOSS,),
                preferred_output_format="html",
            )
        ]

    def apply_chunk_repair(self, source, candidate, task):
        del source, candidate
        self.applied_labels.append(task.table_label)
        return None


def test_repair_candidate_skips_previously_failed_visual_task() -> None:
    repairer = _RetryAwareRepairer()
    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, repair_fanout_max_tasks=4),
        repairer=repairer,
    )
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="visual-retry-skip",
        extracted_text="source",
        page_count=10,
    )
    metrics = EvaluationMetrics(
        text_coverage=0.8,
        normalized_similarity=0.8,
        structure_retention=0.8,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.4,
        table_issues=[TABLE_ISSUE_MERGED_CELL_LOSS],
    )
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="broken table",
        format_name="md",
    )

    prepared = runner._repair_candidate_node(
        {
            "source": source,
            "candidate": candidate,
            "metrics": metrics,
            "iteration_count": 1,
            "repairs": [],
                "failed_visual_task_keys": ["표 4.2-2|4|merged_cell_loss"],
            "repair_plan": [
                RepairPlanStep(
                    strategy="visual_table_repair",
                    route_name="recover_tables_from_pdf_image",
                    targets=(
                        RepairTarget(
                            target_kind="table",
                            issue_type="table_merged_cell_loss",
                            route_name="recover_tables_from_pdf_image",
                            description="recover tables",
                            table_label="표 4.2-2",
                            page_number=4,
                            source_name="judge_table_finding",
                        ),
                    ),
                )
            ],
        }
    )

    assert repairer.applied_labels == []
    assert prepared["failed_visual_task_keys"] == ["표 4.2-2|4|merged_cell_loss"]
