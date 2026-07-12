from __future__ import annotations

import json
from pathlib import Path

from parsing_agent.config import WorkflowConfig
from parsing_agent.models import WorkflowResult


def load_judge_prompt_hints(config: WorkflowConfig) -> list[str]:
    if not config.judge_prompt_tuning_enabled:
        return []
    log_path = Path(config.judge_feedback_log_path)
    if not log_path.exists():
        return []

    records = _read_recent_feedback_records(log_path, config.judge_feedback_log_max_records)
    table_issue_count = sum(_contains_any(record.get("issues", []), ("표", "table")) for record in records)
    image_issue_count = sum(_contains_any(record.get("issues", []), ("그림", "image", "chart")) for record in records)
    structure_issue_count = sum(_contains_any(record.get("issues", []), ("구조", "heading", "section")) for record in records)

    hints: list[str] = []
    if table_issue_count >= 2:
        hints.append("Pay extra attention to missing table rows, merged cells, and numeric omissions.")
    if image_issue_count >= 2:
        hints.append("Check whether important figures, maps, or charts were omitted from the candidate output.")
    if structure_issue_count >= 2:
        hints.append("Check heading hierarchy, section continuity, and repeated structural markers carefully.")
    return hints


def append_judge_feedback_record(config: WorkflowConfig, result: WorkflowResult) -> Path:
    log_path = Path(config.judge_feedback_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": result.source.run_id,
        "source_path": str(result.source.path),
        "parser_name": result.best_candidate.parser_name,
        "repaired_from": result.best_candidate.repaired_from,
        "total_score": result.metrics.total_score,
        "text_coverage": result.metrics.text_coverage,
        "table_preservation": result.metrics.table_preservation,
        "llm_judge_score": result.metrics.llm_judge_score,
        "issues": [] if result.metrics.judge_result is None else list(result.metrics.judge_result.issues),
        "notes": list(result.metrics.notes),
        "repair_routes": [action.route_name for action in result.repairs if action.route_name is not None],
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return log_path


def _read_recent_feedback_records(log_path: Path, max_records: int) -> list[dict[str, object]]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, object]] = []
    for line in lines[-max(max_records, 1) :]:
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _contains_any(values, needles: tuple[str, ...]) -> bool:
    for value in values:
        text = str(value).lower()
        if any(needle.lower() in text for needle in needles):
            return True
    return False
