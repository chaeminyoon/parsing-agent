from __future__ import annotations

import json
from pathlib import Path

from parsing_agent.models import JudgeResult, WorkflowResult


def _build_judge_payload(judge_result: JudgeResult | None, llm_judge_score: float | None) -> dict[str, object] | None:
    if judge_result is None and llm_judge_score is None:
        return None
    if judge_result is None:
        return {
            "overall_score": llm_judge_score,
            "coverage_score": None,
            "structure_score": None,
            "table_score": None,
            "hallucination_risk": None,
            "editorial_readiness": None,
            "notes": [],
            "issues": [],
            "table_findings": [],
        }
    return {
        "overall_score": judge_result.overall_score,
        "coverage_score": judge_result.coverage_score,
        "structure_score": judge_result.structure_score,
        "table_score": judge_result.table_score,
        "hallucination_risk": judge_result.hallucination_risk,
        "editorial_readiness": judge_result.editorial_readiness,
        "notes": judge_result.notes,
        "issues": judge_result.issues,
        "table_findings": judge_result.table_findings,
    }


def build_report_payload(result: WorkflowResult) -> dict[str, object]:
    return {
        "run_id": result.source.run_id,
        "source": {
            "path": str(result.source.path),
            "media_type": result.source.media_type,
            "size_bytes": result.source.size_bytes,
            "page_count": result.source.page_count,
            "ocr": result.source.ocr_metadata,
            "ocr_artifacts": result.source.ocr_artifacts,
        },
        "best_candidate": {
            "parser_name": result.best_candidate.parser_name,
            "format_name": result.best_candidate.format_name,
            "repaired_from": result.best_candidate.repaired_from,
        },
        "metrics": {
            "text_coverage": result.metrics.text_coverage,
            "normalized_similarity": result.metrics.normalized_similarity,
            "structure_retention": result.metrics.structure_retention,
            "table_preservation": result.metrics.table_preservation,
            "empty_block_penalty": result.metrics.empty_block_penalty,
            "repetition_penalty": result.metrics.repetition_penalty,
            "llm_judge_score": result.metrics.llm_judge_score,
            "judge": _build_judge_payload(result.metrics.judge_result, result.metrics.llm_judge_score),
            "total_score": result.metrics.total_score,
            "notes": result.metrics.notes,
        },
        "repairs": [
            {
                "action_name": action.action_name,
                "description": action.description,
                "before_excerpt": action.before_excerpt,
                "after_excerpt": action.after_excerpt,
                "issue_type": action.issue_type,
                "route_name": action.route_name,
            }
            for action in result.repairs
        ],
        "artifacts": dict(result.artifacts),
        "document_summary": None
        if result.document_summary is None
        else {
            "file_name": result.document_summary.file_name,
            "media_type": result.document_summary.media_type,
            "page_count": result.document_summary.page_count,
            "overview": result.document_summary.overview,
            "stats": result.document_summary.stats,
        },
        "report": result.report,
    }


def write_workflow_artifacts(
    result: WorkflowResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = result.source.path.stem
    format_name = result.best_candidate.format_name.strip().lstrip(".") or "txt"
    parsed_output = output_dir / f"{artifact_stem}.{format_name}"
    json_report = output_dir / f"{artifact_stem}.json"
    snapshot_dir = output_dir / f"{artifact_stem}_accuracy_snapshots"

    parsed_output.write_text(result.best_candidate.content, encoding="utf-8")
    snapshot_manifest = _write_accuracy_snapshot_artifacts(result, snapshot_dir)
    if snapshot_manifest:
        result.report["accuracy_snapshots"] = snapshot_manifest
    json_report.write_text(
        json.dumps(build_report_payload(result), indent=2),
        encoding="utf-8",
    )
    artifacts = {"parsed_output": parsed_output, "json_report": json_report}
    if snapshot_manifest:
        artifacts["accuracy_snapshot_dir"] = snapshot_dir
    return artifacts


def _write_accuracy_snapshot_artifacts(result: WorkflowResult, snapshot_dir: Path) -> list[dict[str, object]]:
    raw_snapshots = result.report.get("accuracy_snapshots")
    if not isinstance(raw_snapshots, list) or not raw_snapshots:
        return []
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for index, snapshot in enumerate(raw_snapshots, start=1):
        if not isinstance(snapshot, dict):
            continue
        stage = str(snapshot.get("stage") or f"snapshot_{index}")
        iteration = int(snapshot.get("iteration") or 0)
        snapshot_stem = f"{index:02d}_iter_{iteration:02d}_{stage}"
        markdown_path = snapshot_dir / f"{snapshot_stem}.md"
        metrics_path = snapshot_dir / f"{snapshot_stem}.json"
        markdown_path.write_text(str(snapshot.get("content") or ""), encoding="utf-8")
        metrics_payload = {
            "stage": stage,
            "iteration": iteration,
            "parser_name": snapshot.get("parser_name"),
            "format_name": snapshot.get("format_name"),
            "metrics": snapshot.get("metrics"),
            "repair_targets": snapshot.get("repair_targets"),
            "repair_actions": snapshot.get("repair_actions"),
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        manifest.append(
            {
                "stage": stage,
                "iteration": iteration,
                "markdown_path": str(markdown_path),
                "metrics_path": str(metrics_path),
                "metrics": snapshot.get("metrics"),
                "repair_targets": snapshot.get("repair_targets"),
                "repair_actions": snapshot.get("repair_actions"),
            }
        )
    return manifest
