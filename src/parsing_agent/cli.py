from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import find_dotenv, load_dotenv

from parsing_agent.config import WorkflowConfig
from parsing_agent.workflow import WorkflowRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the parsing agent workflow.")
    parser.add_argument("input_path", help="Path to the source document.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for workflow artifacts.",
    )
    parser.add_argument(
        "--parser",
        action="append",
        dest="parsers",
        help="Parser adapter name to run. Repeat to run multiple parsers.",
    )
    parser.add_argument(
        "--judge-model",
        help="Optional LLM judge model name. Requires OPENAI_API_KEY or a compatible API key in the environment.",
    )
    parser.add_argument(
        "--judge-base-url",
        help="Optional OpenAI-compatible base URL for the LLM judge.",
    )
    parser.add_argument(
        "--judge-weight",
        type=float,
        default=WorkflowConfig().judge_weight,
        help="Blend weight for the LLM judge score between 0 and 1.",
    )
    parser.add_argument(
        "--min-total-score",
        type=float,
        default=WorkflowConfig().min_total_score,
        help="Minimum total score required to pass the quality gate.",
    )
    parser.add_argument(
        "--min-text-coverage",
        type=float,
        default=WorkflowConfig().min_text_coverage,
        help="Minimum text coverage required to pass the quality gate.",
    )
    parser.add_argument(
        "--max-hallucination-risk",
        type=float,
        default=WorkflowConfig().max_hallucination_risk,
        help="Optional maximum hallucination risk allowed when judge data is available.",
    )
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=WorkflowConfig().max_repair_rounds,
        help="Maximum number of repair and re-evaluation rounds per candidate.",
    )
    parser.add_argument(
        "--langsmith-project",
        help="Optional LangSmith project name for this run.",
    )
    parser.add_argument(
        "--langsmith-tracing",
        action="store_true",
        help="Force-enable LangSmith tracing for this run.",
    )
    return parser


def _load_project_dotenv() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)


def _resolve_output_dir(input_path: Path, output_dir: Path) -> Path:
    for ancestor in input_path.parents:
        if ancestor.name.lower() != "data":
            continue
        relative_parent = input_path.parent.relative_to(ancestor)
        if not relative_parent.parts:
            return output_dir / input_path.stem
        return output_dir.joinpath(*relative_parent.parts, input_path.stem)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    _load_project_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = Path(args.input_path)
    resolved_output_dir = _resolve_output_dir(input_path, Path(args.output_dir))
    base_config = WorkflowConfig()
    config = WorkflowConfig(
        parser_names=args.parsers or base_config.parser_names,
        judge_model=args.judge_model if args.judge_model is not None else base_config.judge_model,
        judge_base_url=args.judge_base_url or base_config.judge_base_url,
        judge_weight=args.judge_weight,
        min_total_score=args.min_total_score,
        min_text_coverage=args.min_text_coverage,
        max_hallucination_risk=args.max_hallucination_risk,
        max_repair_rounds=args.max_repair_rounds,
        langsmith_project=args.langsmith_project or base_config.langsmith_project,
        langsmith_tracing=args.langsmith_tracing or base_config.langsmith_tracing,
    )
    runner = WorkflowRunner(config=config)
    result, artifacts = runner.run(input_path, output_dir=resolved_output_dir)
    print(f"Best score: {result.metrics.total_score:.3f}")
    if result.document_summary is not None:
        stats = result.document_summary.stats
        print(f"Document: {result.document_summary.file_name}")
        print(
            "Stats: "
            f"{stats.get('character_count', 0)} chars, "
            f"{stats.get('word_count', 0)} words, "
            f"{stats.get('line_count', 0)} lines"
        )
    if config.langsmith_tracing:
        project_name = config.langsmith_project or "default"
        print(f"LangSmith: enabled (project={project_name})")
    print(f"Output: {artifacts['parsed_output']}")
    print(f"Report: {artifacts['json_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
