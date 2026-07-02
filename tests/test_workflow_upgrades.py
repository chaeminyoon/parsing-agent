import base64
from pathlib import Path

from parsing_agent.cli import build_parser
from parsing_agent.config import WorkflowConfig
from parsing_agent.evaluation import TABLE_ISSUE_MERGED_CELL_LOSS, TABLE_ISSUE_MISSING_HEADER
from parsing_agent.interfaces import CandidateEvaluator, CandidateRepairer
from parsing_agent.models import (
    DocumentSource,
    EvaluationIssue,
    EvaluationMetrics,
    JudgeResult,
    ParseCandidate,
    RepairAction,
    WorkflowResult,
)
from parsing_agent.parsers import OpenDataLoaderPdfParserAdapter, build_default_parser_registry
from parsing_agent.repair import HeuristicRepairer, identify_repair_targets
from parsing_agent.visual_repair import OpenAIVisualTableRecoverer, VisualRepairTask
from parsing_agent.repair import RepairTarget
from parsing_agent.workflow import RepairOutcome, WorkflowRunner, WorkflowState

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WnR6zsAAAAASUVORK5CYII="
)


class _StubEvaluator(CandidateEvaluator):
    def evaluate(self, source: DocumentSource, candidate: ParseCandidate) -> EvaluationMetrics:
        del source, candidate
        return EvaluationMetrics(
            text_coverage=0.95,
            normalized_similarity=0.9,
            structure_retention=0.85,
            table_preservation=0.8,
            empty_block_penalty=0.0,
            repetition_penalty=0.0,
            total_score=0.91,
        )


class _ImageSensitiveEvaluator(CandidateEvaluator):
    def evaluate(self, source: DocumentSource, candidate: ParseCandidate) -> EvaluationMetrics:
        del source
        if "![image chart]" in candidate.content:
            return EvaluationMetrics(
                text_coverage=0.4,
                normalized_similarity=0.4,
                structure_retention=0.4,
                table_preservation=0.4,
                empty_block_penalty=0.0,
                repetition_penalty=0.0,
                total_score=0.4,
            )
        return EvaluationMetrics(
            text_coverage=0.92,
            normalized_similarity=0.92,
            structure_retention=0.92,
            table_preservation=0.92,
            empty_block_penalty=0.0,
            repetition_penalty=0.0,
            total_score=0.92,
        )


class _StubRepairer(CandidateRepairer):
    def repair(self, source: DocumentSource, candidate: ParseCandidate, metrics: EvaluationMetrics):
        del source, metrics
        return candidate, []


class _LoopingEvaluator(CandidateEvaluator):
    def evaluate(self, source: DocumentSource, candidate: ParseCandidate) -> EvaluationMetrics:
        del source
        if "good" in candidate.content:
            return EvaluationMetrics(
                text_coverage=0.96,
                normalized_similarity=0.96,
                structure_retention=0.96,
                table_preservation=0.96,
                empty_block_penalty=0.0,
                repetition_penalty=0.0,
                total_score=0.96,
            )
        return EvaluationMetrics(
            text_coverage=0.4,
            normalized_similarity=0.4,
            structure_retention=0.4,
            table_preservation=0.4,
            empty_block_penalty=0.0,
            repetition_penalty=0.0,
            total_score=0.4,
            notes=["initial quality is low"],
        )


class _LoopingRepairer(CandidateRepairer):
    def repair(self, source: DocumentSource, candidate: ParseCandidate, metrics: EvaluationMetrics):
        del source, metrics
        repaired = ParseCandidate(
            parser_name=candidate.parser_name,
            content=candidate.content.replace("bad", "good"),
            format_name=candidate.format_name,
            metadata=dict(candidate.metadata),
            source_path=candidate.source_path,
            repaired_from=candidate.repaired_from or candidate.parser_name,
        )
        return repaired, [
            RepairAction(
                action_name="rewrite_low_quality_text",
                description="Rewrite the low-quality text block.",
                before_excerpt="bad",
                after_excerpt="good",
                issue_type="wrapped_line_noise",
                route_name="rewrite_text_block",
            )
        ]


class _ChunkPlanningRecoverer(OpenAIVisualTableRecoverer):
    def __init__(self) -> None:
        super().__init__(model="test", api_key="test", max_tables_per_round=2)
        self.plan_calls = 0

    def plan_tasks(
        self,
        source: DocumentSource,
        content: str,
        metrics: EvaluationMetrics,
        *,
        candidate_metadata: dict[str, object] | None = None,
        max_tasks: int,
    ) -> list[VisualRepairTask]:
        del source, content, metrics, candidate_metadata
        self.plan_calls += 1
        return [
            VisualRepairTask(
                task_id="task-1",
                table_label="Table 1",
                page_number=1,
                issue_types=(TABLE_ISSUE_MERGED_CELL_LOSS,),
            )
        ][:max_tasks]


def test_workflow_runner_report_includes_quality_gate_and_monitoring(tmp_path, monkeypatch) -> None:
    source = DocumentSource(
        path=tmp_path / "sample.txt",
        media_type="text/plain",
        size_bytes=0,
        run_id="workflow-upgrade-test",
        extracted_text="source text",
        page_count=None,
    )
    source.path.write_text("source text", encoding="utf-8")
    monkeypatch.setattr("parsing_agent.workflow.build_document_source", lambda path, run_id, **kwargs: source)

    runner = WorkflowRunner(
        config=WorkflowConfig(
            judge_weight=0,
            langsmith_tracing=False,
            judge_feedback_log_path=str(tmp_path / "judge_feedback.jsonl"),
        ),
        parser_registry=build_default_parser_registry(),
        evaluator=_StubEvaluator(),
        repairer=_StubRepairer(),
    )

    result, artifacts = runner.run(source.path, output_dir=tmp_path / "outputs")

    assert isinstance(result, WorkflowResult)
    assert "triage" not in result.report
    assert "branch" not in result.report
    assert "artifacts" not in result.report
    assert "document_summary" not in result.report
    assert "candidate_mapping" not in result.report["monitoring"]
    assert result.report["monitoring"]["judge_grounding_pages"] == []
    assert result.report["monitoring"]["used_chunk_repairs"] is False
    assert result.report["monitoring"]["image_caption_enrichment"]["count"] == 0
    assert Path(result.report["monitoring"]["judge_feedback_log_path"]).exists()
    assert result.report["quality_gate"] == {
        "passed": True,
        "selected_candidate_passed": True,
        "selected_candidate_failed_checks": [],
    }
    assert result.artifacts["parsed_output"] == str(artifacts["parsed_output"])


def test_cli_parser_does_not_expose_fallback_parser_option() -> None:
    parser = build_parser()

    option_strings = {option for action in parser._actions for option in action.option_strings}

    assert "--fallback-parser" not in option_strings


def test_cli_parser_uses_workflow_config_default_for_max_repair_rounds() -> None:
    parser = build_parser()

    args = parser.parse_args(["sample.txt"])

    assert args.max_repair_rounds == WorkflowConfig().max_repair_rounds


def test_workflow_result_does_not_expose_candidates_field() -> None:
    assert "candidates" not in WorkflowResult.__dataclass_fields__


def test_workflow_runner_does_not_enrich_candidate_during_finalize(tmp_path, monkeypatch) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    image_path = assets_dir / "chart.png"
    image_path.write_bytes(_PNG_BYTES)
    source = DocumentSource(
        path=tmp_path / "sample.md",
        media_type="text/markdown",
        size_bytes=0,
        run_id="workflow-image-enrichment-test",
        extracted_text="Before\n\n![image chart](assets/chart.png)\n\nAfter",
        page_count=None,
    )
    source.path.write_text(source.extracted_text, encoding="utf-8")
    monkeypatch.setattr("parsing_agent.workflow.build_document_source", lambda path, run_id, **kwargs: source)
    monkeypatch.setattr(
        "parsing_agent.enrichment._post_response",
        lambda payload, config, timeout_seconds: {"output_text": "Site layout overview"},
    )

    runner = WorkflowRunner(
        config=WorkflowConfig(
            judge_weight=0,
            langsmith_tracing=False,
            max_repair_rounds=0,
            judge_feedback_log_path=str(tmp_path / "judge_feedback.jsonl"),
            judge_api_key="test-key",
            post_selection_image_captioning_enabled=True,
            post_selection_image_caption_model="gpt-test",
        ),
        parser_registry=build_default_parser_registry(),
        evaluator=_StubEvaluator(),
        repairer=_StubRepairer(),
    )

    result, artifacts = runner.run(source.path, output_dir=tmp_path / "outputs")

    assert "![image chart]" in result.best_candidate.content
    assert "Image: Site layout overview" not in result.best_candidate.content
    assert result.report["monitoring"]["image_caption_enrichment"]["count"] == 0
    assert result.report["monitoring"]["image_caption_enrichment"]["paths"] == []
    assert result.artifacts["parsed_output"] == str(artifacts["parsed_output"])


def test_workflow_runner_does_not_re_evaluate_metrics_during_finalize(tmp_path, monkeypatch) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    image_path = assets_dir / "chart.png"
    image_path.write_bytes(_PNG_BYTES)
    source = DocumentSource(
        path=tmp_path / "sample.md",
        media_type="text/markdown",
        size_bytes=0,
        run_id="workflow-image-reevaluate-test",
        extracted_text="Before\n\n![image chart](assets/chart.png)\n\nAfter",
        page_count=None,
    )
    source.path.write_text(source.extracted_text, encoding="utf-8")
    monkeypatch.setattr("parsing_agent.workflow.build_document_source", lambda path, run_id, **kwargs: source)
    monkeypatch.setattr(
        "parsing_agent.enrichment._post_response",
        lambda payload, config, timeout_seconds: {"output_text": "Site layout overview"},
    )

    runner = WorkflowRunner(
        config=WorkflowConfig(
            judge_weight=0,
            langsmith_tracing=False,
            judge_feedback_log_path=str(tmp_path / "judge_feedback.jsonl"),
            judge_api_key="test-key",
            post_selection_image_captioning_enabled=True,
            post_selection_image_caption_model="gpt-test",
        ),
        parser_registry=build_default_parser_registry(),
        evaluator=_ImageSensitiveEvaluator(),
        repairer=_StubRepairer(),
    )

    result, _artifacts = runner.run(source.path, output_dir=tmp_path / "outputs")

    assert result.metrics.total_score == 0.4
    assert result.metrics.text_coverage == 0.4
    assert "![image chart]" in result.best_candidate.content


def test_workflow_runner_writes_accuracy_snapshots_for_initial_and_repaired_iterations(tmp_path, monkeypatch) -> None:
    source = DocumentSource(
        path=tmp_path / "sample.txt",
        media_type="text/plain",
        size_bytes=0,
        run_id="workflow-accuracy-snapshot-test",
        extracted_text="bad content",
        page_count=None,
    )
    source.path.write_text("bad content", encoding="utf-8")
    monkeypatch.setattr("parsing_agent.workflow.build_document_source", lambda path, run_id, **kwargs: source)

    runner = WorkflowRunner(
        config=WorkflowConfig(
            judge_weight=0,
            langsmith_tracing=False,
            max_repair_rounds=2,
            judge_feedback_log_path=str(tmp_path / "judge_feedback.jsonl"),
        ),
        parser_registry=build_default_parser_registry(),
        evaluator=_LoopingEvaluator(),
        repairer=_LoopingRepairer(),
    )

    result, artifacts = runner.run(source.path, output_dir=tmp_path / "outputs")

    snapshot_dir = Path(artifacts["accuracy_snapshot_dir"])
    assert snapshot_dir.exists()
    assert (snapshot_dir / "01_iter_00_initial_evaluation.md").read_text(encoding="utf-8") == "bad content"
    assert (snapshot_dir / "02_iter_01_post_repair_evaluation.md").read_text(encoding="utf-8") == "good content"
    snapshot_manifest = result.report["accuracy_snapshots"]
    assert len(snapshot_manifest) == 2
    assert snapshot_manifest[0]["stage"] == "initial_evaluation"
    assert snapshot_manifest[1]["stage"] == "post_repair_evaluation"
    assert snapshot_manifest[1]["metrics"]["total_score"] == 0.96


def test_workflow_considers_table_judge_issues_repairable_even_when_score_passes() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))
    metrics = EvaluationMetrics(
        text_coverage=0.95,
        normalized_similarity=0.9,
        structure_retention=0.85,
        table_preservation=0.5,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.91,
        table_issues=[TABLE_ISSUE_MISSING_HEADER],
        judge_result=JudgeResult(overall_score=0.8, issues=["Table 4.2-2 missing header row."]),
    )

    assert runner._has_repairable_table_issues(metrics) is True
    assert runner._needs_candidate_repair(metrics, 0) is True


def test_repair_candidate_skips_chunk_repairs_when_text_coverage_is_below_gate() -> None:
    recoverer = _ChunkPlanningRecoverer()
    runner = WorkflowRunner(
        config=WorkflowConfig(
            judge_weight=0,
            langsmith_tracing=False,
            min_text_coverage=0.7,
        ),
        repairer=HeuristicRepairer(visual_table_recoverer=recoverer),
    )
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="repair-low-coverage-test",
        extracted_text="source",
        page_count=1,
    )
    metrics = EvaluationMetrics(
        text_coverage=0.6,
        normalized_similarity=0.85,
        structure_retention=0.85,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.65,
        table_issues=[TABLE_ISSUE_MERGED_CELL_LOSS],
    )
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="Table 1\nbroken table",
        format_name="md",
        metadata={
            "support_parser_metadata": {
                "layout-first-pdf": {
                    "table_regions": [{"table_id": "p1-t1", "page": 1, "extraction_mode": "reference"}]
                }
            }
        },
        source_path=source.path,
    )

    result = runner._repair_candidate_node(
        {
            "source": source,
            "candidate": candidate,
            "metrics": metrics,
            "iteration_count": 0,
            "repairs": [],
            "repair_plan": [],
        }
    )

    assert result["candidate"].content == candidate.content
    assert result["iteration_count"] == 1
    assert result["repairs"] == []
    assert recoverer.plan_calls == 0


def test_default_parser_registry_does_not_include_mock_parser() -> None:
    registry = build_default_parser_registry()

    assert registry.has("mock") is False


def test_verify_candidate_reports_placeholder_only_candidates() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="verify-placeholder-test",
        extracted_text="source",
        page_count=2,
    )
    placeholder_candidate = ParseCandidate(
        parser_name="layout-first-pdf",
        content="<!-- page 1 -->\n[Table reference: id=p1-t1 page=1 bbox=1,2,3,4 rows=2 cols=2]\n",
        format_name="md",
        source_path=source.path,
    )
    usable_candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="<!-- page 1 -->\nTable 1\n| a | b |\n| --- | --- |\n| 1 | 2 |\n",
        format_name="md",
        source_path=source.path,
    )

    assert runner._candidate_verification_failures(source, placeholder_candidate) == ["placeholder_only_content"]
    assert runner._candidate_verification_failures(source, usable_candidate) == []


def test_inspect_repair_targets_routes_text_and_table_issues() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="inspect-routing-test",
        extracted_text="source",
        page_count=2,
    )
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="# Heading\n# Heading\nwrapped line\ncontinues here\n",
        format_name="md",
        source_path=source.path,
    )
    metrics = EvaluationMetrics(
        text_coverage=0.92,
        normalized_similarity=0.6,
        structure_retention=0.5,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.5,
        table_issues=[TABLE_ISSUE_MISSING_HEADER],
    )

    result = runner._inspect_quality_issues_node(
        {
            "source": source,
            "candidate": candidate,
            "metrics": metrics,
        }
    )

    routed_pairs = {(target.issue_type, target.route_name) for target in result["repair_targets"]}
    assert ("structure_heading_noise", "deduplicate_headings") in routed_pairs
    assert ("wrapped_line_noise", "merge_wrapped_lines") in routed_pairs
    assert (TABLE_ISSUE_MISSING_HEADER, "recover_tables_from_pdf_image") in routed_pairs


def test_route_without_repair_plan_finalizes() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))

    assert runner._route_after_quality_inspection({"repair_targets": []}) == "route"
    assert runner._route_after_repair_strategy({"repair_plan": []}) == "finalize"


def test_workflow_state_excludes_legacy_graph_fields() -> None:
    annotations = WorkflowState.__annotations__

    assert "triage_decision" not in annotations
    assert "selected_parser_names" not in annotations
    assert "repair_tasks" not in annotations
    assert "repair_task_results" not in annotations


def test_route_repair_strategy_builds_explicit_strategy_steps() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))

    result = runner._route_repair_strategy_node(
        {
            "repair_targets": [
                RepairTarget(
                    target_kind="text",
                    issue_type="wrapped_line_noise",
                    route_name="merge_wrapped_lines",
                    description="merge wrapped lines",
                ),
                RepairTarget(
                    target_kind="table",
                    issue_type=TABLE_ISSUE_MISSING_HEADER,
                    route_name="recover_tables_from_pdf_image",
                    description="recover table",
                ),
            ]
        }
    )

    plan = result["repair_plan"]
    assert [step.strategy for step in plan] == ["heuristic", "visual_table_repair"]
    assert [step.route_name for step in plan] == ["merge_wrapped_lines", "recover_tables_from_pdf_image"]
    assert [target.issue_type for target in plan[0].targets] == ["wrapped_line_noise"]
    assert [target.issue_type for target in plan[1].targets] == [TABLE_ISSUE_MISSING_HEADER]


def test_inspect_preserves_structured_table_finding_payload_in_repair_targets() -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="inspect-structured-targets",
        extracted_text="source text",
        page_count=12,
    )
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="표 4.2-2\nbroken table",
        format_name="md",
        source_path=source.path,
    )
    metrics = EvaluationMetrics(
        text_coverage=0.8,
        normalized_similarity=0.8,
        structure_retention=0.8,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MISSING_HEADER],
        judge_result=JudgeResult(
            overall_score=0.8,
            table_findings=[
                {"issue_type": TABLE_ISSUE_MISSING_HEADER, "table_label": "표 4.2-2", "page_number": 4}
            ],
        ),
    )

    targets = identify_repair_targets(source, candidate, metrics)

    table_targets = [target for target in targets if target.issue_type == TABLE_ISSUE_MISSING_HEADER]
    assert len(table_targets) == 1
    assert table_targets[0].table_label == "표 4.2-2"
    assert table_targets[0].page_number == 4
    assert table_targets[0].source_name == "judge_table_finding"


def test_inspect_targets_include_metric_evidence_and_repair_estimates() -> None:
    source = DocumentSource(
        path=Path("sample.md"),
        media_type="text/markdown",
        size_bytes=0,
        run_id="inspect-evidence-targets",
        extracted_text="# Source title\nbody",
    )
    candidate = ParseCandidate(
        parser_name="text-fallback",
        content="# Broken\n# Broken\nbody",
        format_name="md",
        source_path=source.path,
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.8,
        structure_retention=0.4,
        table_preservation=0.9,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        issues=[
            EvaluationIssue(
                issue_type="structure_heading_noise",
                metric_name="structure_retention",
                severity="high",
                confidence=0.82,
                description="heading duplicated",
                source_excerpt="# Source title",
                candidate_excerpt="# Broken # Broken",
                repairability="heuristic",
            )
        ],
    )

    targets = identify_repair_targets(source, candidate, metrics)

    structure_target = next(target for target in targets if target.issue_type == "structure_heading_noise")
    assert structure_target.severity == "high"
    assert structure_target.confidence == 0.82
    assert structure_target.source_excerpt == "# Source title"
    assert structure_target.candidate_excerpt == "# Broken # Broken"
    assert structure_target.repairability == "heuristic"
    assert structure_target.expected_gain > 0
    assert structure_target.estimated_cost == 0.0


def test_inspect_synthesizes_table_targets_from_parser_metadata_when_judge_findings_are_missing() -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="inspect-synthetic-table-targets",
        extracted_text="source text",
        page_count=12,
    )
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="broken table",
        format_name="md",
        source_path=source.path,
        metadata={
            "table_regions": [
                {"page_number": 4, "label": "??4.2-2", "table_id": "page-4-table-1"},
            ]
        },
    )
    metrics = EvaluationMetrics(
        text_coverage=0.8,
        normalized_similarity=0.8,
        structure_retention=0.8,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MISSING_HEADER],
    )

    targets = identify_repair_targets(source, candidate, metrics)

    table_targets = [target for target in targets if target.issue_type == TABLE_ISSUE_MISSING_HEADER]
    assert any(target.source_name == "parser_table_region" for target in table_targets)
    assert any(target.table_label == "??4.2-2" and target.page_number == 4 for target in table_targets)


def test_route_repair_strategy_preserves_structured_visual_target_payload() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))

    result = runner._route_repair_strategy_node(
        {
            "repair_targets": [
                RepairTarget(
                    target_kind="table",
                    issue_type=TABLE_ISSUE_MISSING_HEADER,
                    route_name="recover_tables_from_pdf_image",
                    description="recover table",
                    table_label="표 4.2-2",
                    page_number=4,
                    source_name="judge_table_finding",
                ),
            ]
        }
    )

    plan = result["repair_plan"]
    assert len(plan) == 1
    assert plan[0].strategy == "visual_table_repair"
    assert plan[0].targets[0].table_label == "표 4.2-2"
    assert plan[0].targets[0].page_number == 4
    assert plan[0].targets[0].source_name == "judge_table_finding"
    assert plan[0].expected_gain > 0
    assert plan[0].estimated_cost > 0
    assert plan[0].verification_rule == "table_preservation_or_score_improves"
    assert result["repair_plan_history"][0]["steps"][0]["targets"][0]["table_label"] == "표 4.2-2"


def test_route_repair_strategy_plans_low_value_visual_repairs_on_first_round() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))

    result = runner._route_repair_strategy_node(
        {
            "repair_targets": [
                RepairTarget(
                    target_kind="table",
                    issue_type=TABLE_ISSUE_MISSING_HEADER,
                    route_name="recover_tables_from_pdf_image",
                    description="recover table",
                    expected_gain=0.01,
                    estimated_cost=1.0,
                    risk_level="medium",
                ),
            ]
        }
    )

    assert len(result["repair_plan"]) == 1
    assert result["repair_plan"][0].skip_reason is None


def test_route_repair_strategy_records_skipped_low_value_visual_repairs_after_first_round() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))

    result = runner._route_repair_strategy_node(
        {
            "iteration_count": 1,
            "repair_targets": [
                RepairTarget(
                    target_kind="table",
                    issue_type=TABLE_ISSUE_MISSING_HEADER,
                    route_name="recover_tables_from_pdf_image",
                    description="recover table",
                    expected_gain=0.01,
                    estimated_cost=1.0,
                    risk_level="medium",
                ),
            ],
        }
    )

    assert result["repair_plan"] == []
    skipped_step = result["repair_plan_history"][0]["steps"][0]
    assert skipped_step["skip_reason"] == "expected_gain_below_cost_gate"
    assert skipped_step["estimated_cost"] == 1.0


def test_repair_outcomes_are_verified_against_post_repair_metrics() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))
    outcome = RepairOutcome(
        action_name="recover_table_from_pdf_image",
        issue_type=TABLE_ISSUE_MISSING_HEADER,
        route_name="recover_tables_from_pdf_image",
        verification_rule="table_preservation_or_score_improves",
        before_score=0.4,
        before_metrics={
            "text_coverage": 0.8,
            "normalized_similarity": 0.8,
            "structure_retention": 0.8,
            "table_preservation": 0.3,
            "empty_block_penalty": 0.0,
            "repetition_penalty": 0.0,
            "total_score": 0.4,
        },
    )
    metrics = EvaluationMetrics(
        text_coverage=0.8,
        normalized_similarity=0.8,
        structure_retention=0.8,
        table_preservation=0.55,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.52,
    )

    verified = runner._verify_repair_outcomes(outcomes=[outcome], metrics=metrics)

    assert verified[0].verification_passed is True
    assert verified[0].score_delta == 0.12
    assert verified[0].changed_metrics["table_preservation"] == 0.25


def test_route_prioritizes_visual_repair_after_stalled_snapshot_delta() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))

    result = runner._route_repair_strategy_node(
        {
            "iteration_count": 1,
            "accuracy_snapshots": [
                {"metrics": {"total_score": 0.50}},
                {"metrics": {"total_score": 0.505}},
            ],
            "repair_targets": [
                RepairTarget(
                    target_kind="text",
                    issue_type="wrapped_line_noise",
                    route_name="merge_wrapped_lines",
                    description="merge wrapped lines",
                ),
                RepairTarget(
                    target_kind="table",
                    issue_type=TABLE_ISSUE_MISSING_HEADER,
                    route_name="recover_tables_from_pdf_image",
                    description="recover table",
                ),
            ],
        }
    )

    plan = result["repair_plan"]
    assert [step.strategy for step in plan] == ["visual_table_repair", "heuristic"]


def test_route_skips_repeated_heuristic_after_stalled_snapshot_delta() -> None:
    # LLM 승격이 불가능한 구성에서는 정체된 heuristic 재시도를 스킵해야 한다.
    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False, llm_text_repair_enabled=False)
    )

    result = runner._route_repair_strategy_node(
        {
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
                RepairTarget(
                    target_kind="table",
                    issue_type=TABLE_ISSUE_MISSING_HEADER,
                    route_name="recover_tables_from_pdf_image",
                    description="recover table",
                ),
            ],
        }
    )

    plan = result["repair_plan"]
    assert len(plan) == 1
    assert plan[0].strategy == "visual_table_repair"


def test_parse_document_preserves_embedded_image_data_urls() -> None:
    class _ImageParser:
        def parse(self, source, config):
            del source, config
            return [
                ParseCandidate(
                    parser_name="text-fallback",
                    content="![image](assets/chart.png)",
                    format_name="md",
                    metadata={"embedded_image_data_urls": {"assets/chart.png": "data:image/png;base64,AAAA"}},
                )
            ]

    class _Registry:
        def get(self, name):
            assert name == "text-fallback"
            return _ImageParser()

        def has(self, name):
            return name == "text-fallback"

    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False, parser_names=["text-fallback"]),
        parser_registry=_Registry(),
    )
    source = DocumentSource(
        path=Path("sample.md"),
        media_type="text/markdown",
        size_bytes=0,
        run_id="parse-preserve-images",
        extracted_text="source",
    )

    result = runner._parse_document_node({"source": source})

    assert result["candidate"].metadata["embedded_image_data_urls"] == {"assets/chart.png": "data:image/png;base64,AAAA"}


def test_repair_candidate_noop_still_advances_loop_iteration() -> None:
    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False),
        repairer=_StubRepairer(),
    )
    source = DocumentSource(
        path=Path("sample.txt"),
        media_type="text/plain",
        size_bytes=0,
        run_id="repair-noop-loop",
        extracted_text="source",
    )
    candidate = ParseCandidate(
        parser_name="text-fallback",
        content="unchanged",
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
            "source": source,
            "candidate": candidate,
            "metrics": metrics,
            "repairs": [],
            "iteration_count": 0,
            "repair_plan": [],
        }
    )

    assert result["candidate"].content == "unchanged"
    assert result["iteration_count"] == 1
    assert result["repairs"] == []


def test_chunk_repair_planning_uses_support_metadata_for_html_preference(monkeypatch) -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="support-repair-context-test",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MERGED_CELL_LOSS],
        judge_result=JudgeResult(overall_score=0.8, issues=["Table 4.2-2 on p.4 needs visual repair."]),
    )
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="표 4.2-2\nbroken table\n",
        format_name="md",
        metadata={
            "support_parser_metadata": {
                "layout-first-pdf": {
                    "table_format": "html",
                    "table_regions": [{"table_id": "p1-t1", "page": 1, "extraction_mode": "reference"}],
                }
            }
        },
    )
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test", max_tables_per_round=2)
    monkeypatch.setattr(recoverer, "_find_page_number", lambda path, table_label: 4)

    tasks = HeuristicRepairer(visual_table_recoverer=recoverer).plan_chunk_repairs(
        source,
        candidate,
        metrics,
        max_tasks=1,
    )

    assert len(tasks) == 1
    assert tasks[0].preferred_output_format == "html"


def test_chunk_repair_planning_accepts_pdf_suffix_without_exact_pdf_media_type(monkeypatch) -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/octet-stream",
        size_bytes=0,
        run_id="pdf-suffix-repair-test",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MERGED_CELL_LOSS],
        judge_result=JudgeResult(overall_score=0.8, issues=["Table 4.2-2 on p.4 needs visual repair."]),
    )
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="표 4.2-2\nbroken table\n",
        format_name="md",
        metadata={
            "support_parser_metadata": {
                "layout-first-pdf": {
                    "table_format": "html",
                    "table_regions": [{"table_id": "p1-t1", "page": 1, "extraction_mode": "reference"}],
                }
            }
        },
    )
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test", max_tables_per_round=2)
    monkeypatch.setattr(recoverer, "_find_page_number", lambda path, table_label: 4)

    tasks = HeuristicRepairer(visual_table_recoverer=recoverer).plan_chunk_repairs(
        source,
        candidate,
        metrics,
        max_tasks=1,
    )

    assert len(tasks) == 1
    assert tasks[0].preferred_output_format == "html"


def test_chunk_repair_planning_skips_ambiguous_page_scoped_fallback_for_multi_table_page() -> None:
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="page-scoped-ambiguity-test",
        extracted_text="source text",
    )
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.4,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.0,
        table_issues=[TABLE_ISSUE_MERGED_CELL_LOSS],
        judge_result=JudgeResult(overall_score=0.8, issues=["p.6 table needs visual repair."]),
    )
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="broken table content",
        format_name="md",
        metadata={
            "support_parser_metadata": {
                "layout-first-pdf": {
                    "table_regions": [
                        {"table_id": "p6-t1", "page": 6, "extraction_mode": "reference"},
                        {"table_id": "p6-t2", "page": 6, "extraction_mode": "reference"},
                    ]
                }
            }
        },
    )
    recoverer = OpenAIVisualTableRecoverer(model="test", api_key="test", max_tables_per_round=2)

    tasks = HeuristicRepairer(visual_table_recoverer=recoverer).plan_chunk_repairs(
        source,
        candidate,
        metrics,
        max_tasks=2,
    )

    assert [task.table_label for task in tasks] == ["__page_table__:6:1", "__page_table__:6:2"]


def test_finalize_output_does_not_mutate_candidate_content() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="Before\n![image chart](assets/chart.png)\nAfter\n",
        format_name="md",
    )
    state = runner._finalize_output_node(
        {
            "source": DocumentSource(
                path=Path("sample.md"),
                media_type="text/markdown",
                size_bytes=0,
                run_id="finalize-no-mutate",
                extracted_text="source",
            ),
            "candidate": candidate,
            "metrics": EvaluationMetrics(
                text_coverage=0.9,
                normalized_similarity=0.9,
                structure_retention=0.9,
                table_preservation=0.9,
                empty_block_penalty=0.0,
                repetition_penalty=0.0,
                total_score=0.8,
            ),
            "repairs": [],
            "accuracy_snapshots": [],
        }
    )
    assert state["result"].best_candidate.content == candidate.content
    assert state["result"].report["diagnosed_issues"] == []
    assert state["result"].report["repair_plan"] == []
    assert state["result"].report["repair_outcomes"] == []
    assert state["result"].report["skipped_repairs"] == []


def test_externalize_source_text_keeps_it_out_of_graph_state() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="source-text-cache-test",
        extracted_text="source body",
        page_count=1,
    )

    compact_source = runner._externalize_source_text(source)
    restored_source = runner._materialize_source_text(compact_source)

    assert compact_source.extracted_text is None
    assert runner._source_text_cache[source.run_id] == "source body"
    assert restored_source.extracted_text == "source body"


def test_langsmith_client_hides_graph_payloads_and_keeps_source_summary(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Client:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("parsing_agent.workflow.Client", _Client)
    runner = WorkflowRunner(
        config=WorkflowConfig(
            langsmith_tracing=True,
            langsmith_api_key="test-key",
            langsmith_hide_inputs=True,
            langsmith_hide_outputs=True,
        )
    )
    source = DocumentSource(
        path=Path("large-report.pdf"),
        media_type="application/pdf",
        size_bytes=42_000_000,
        run_id="run-1",
        extracted_text="document body",
        page_count=120,
    )

    compact_source = runner._externalize_source_text(source)
    metadata = runner._langsmith_metadata(compact_source)
    runner._build_langsmith_client()

    assert callable(captured["hide_inputs"])
    assert callable(captured["hide_outputs"])
    assert metadata["source_filename"] == "large-report.pdf"
    assert metadata["source_size_bytes"] == 42_000_000
    assert metadata["source_text_character_count"] == len("document body")
    assert metadata["trace_payload_policy"] == "summary_only"
    assert "source_path" not in metadata

    payload_summary = captured["hide_outputs"](
        {"candidate": ParseCandidate("parser", "very large body", "md")}
    )
    candidate_summary = payload_summary["fields"]["candidate"]
    assert candidate_summary["content_character_count"] == len("very large body")
    assert "very large body" not in str(payload_summary)


def test_opendataloader_parser_runs_converter_in_quiet_mode(tmp_path: Path) -> None:
    converter_calls: list[dict[str, object]] = []

    def fake_converter(**kwargs):
        converter_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sample.md").write_text("# sample\n", encoding="utf-8")

    parser = OpenDataLoaderPdfParserAdapter(converter=fake_converter)
    source = DocumentSource(
        path=tmp_path / "sample.pdf",
        media_type="application/pdf",
        size_bytes=0,
        run_id="quiet-opendataloader",
        extracted_text="source",
    )
    source.path.write_bytes(b"%PDF-1.4")

    parser.parse(source, WorkflowConfig(judge_weight=0, langsmith_tracing=False))

    assert converter_calls
    assert converter_calls[0]["quiet"] is True


def test_opendataloader_parser_extracts_table_grounding_metadata(tmp_path: Path) -> None:
    def fake_converter(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sample.md").write_text(
            "\n".join(
                [
                    "<!-- page 3 -->",
                    "표 4.2-2",
                    "",
                    "| 항목 | 값 |",
                    "| --- | --- |",
                    "| A | 1 |",
                    "",
                    "<!-- page 4 -->",
                    "표 4.2-3",
                    "",
                    "| 항목 | 값 |",
                    "| --- | --- |",
                    "| B | 2 |",
                ]
            ),
            encoding="utf-8",
        )

    parser = OpenDataLoaderPdfParserAdapter(converter=fake_converter)
    source = DocumentSource(
        path=tmp_path / "sample.pdf",
        media_type="application/pdf",
        size_bytes=0,
        run_id="metadata-opendataloader",
        extracted_text="source",
        page_count=4,
    )
    source.path.write_bytes(b"%PDF-1.4")

    candidates = parser.parse(source, WorkflowConfig(judge_weight=0, langsmith_tracing=False))

    candidate = candidates[0]
    assert candidate.metadata["table_format"] == "markdown"
    assert candidate.metadata["table_label_pages"]["표 4.2-2"] == 3
    assert candidate.metadata["table_label_pages"]["4.2-2"] == 3
    assert candidate.metadata["table_label_pages"]["표 4.2-3"] == 4
    assert candidate.metadata["table_label_positions"]["표 4.2-2"]["page"] == 3
    assert candidate.metadata["table_label_positions"]["표 4.2-3"]["page"] == 4
    assert candidate.metadata["table_regions"] == [
        {
            "table_id": "p3-t1",
            "page": 3,
            "row_count": 2,
            "col_count": 2,
            "label": "표 4.2-2",
            "extraction_mode": "markdown",
        },
        {
            "table_id": "p4-t1",
            "page": 4,
            "row_count": 2,
            "col_count": 2,
            "label": "표 4.2-3",
            "extraction_mode": "markdown",
        },
    ]


