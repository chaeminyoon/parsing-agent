"""LangSmith 트레이스 요약 — 자유 문장 대신 구조화 필드만 내보낸다.

workflow.py 분할: 트레이스 페이로드 요약 로직만 둔다.
"""
from __future__ import annotations



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

from parsing_agent.workflow_state import RepairOutcome, RepairPlanStep  # noqa: F401


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
