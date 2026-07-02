"""외부 파서와의 head-to-head 벤치마크.

사용:
    uv sync --extra bench
    uv run python benchmarks/run_head_to_head.py data/*.pdf
    uv run python benchmarks/run_head_to_head.py --no-agent data/문서.pdf   # API 호출 없이

엔진:
- opendataloader     : 이 파이프라인의 기본 파서 단발 출력 (베이스라인)
- pymupdf4llm        : PyMuPDF 기반 마크다운 변환
- markitdown         : Microsoft MarkItDown (pdfminer 기반)
- docling            : IBM Docling (--extra bench-docling 필요, 첫 실행 시 모델 다운로드)
- parsing-agent      : 풀 루프 (judge + 수리, OPENAI_API_KEY 필요)

채점: 모든 엔진을 **동일한 결정적 채점기**(judge 없음)로 잰다.
주의 — parsing-agent는 이 채점기를 내부에서 직접 최적화하므로 구조적으로
유리하다. 이 표는 "같은 잣대로 쟀을 때의 방향"이지 중립 벤치마크가 아니다.
중립 검증은 golden/ 의 사람 라벨 상관 분석으로 한다.

marker 제외 사유: GPL-3.0 (라이선스 비호환 우려). 부분 실행은
--engines 로 선택하며, 결과는 기존 results.json에 병합된다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parsing_agent.config import WorkflowConfig  # noqa: E402
from parsing_agent.evaluation import DeterministicEvaluator  # noqa: E402
from parsing_agent.ingestion import build_document_source  # noqa: E402
from parsing_agent.models import ParseCandidate  # noqa: E402
from parsing_agent.parsers import OpenDataLoaderPdfParserAdapter  # noqa: E402

METRIC_COLUMNS = ["total_score", "text_coverage", "normalized_similarity", "structure_retention", "table_preservation"]


def _scoring_config() -> WorkflowConfig:
    return WorkflowConfig(judge_weight=0, langsmith_tracing=False, ocr_enabled=False)


def run_opendataloader(pdf: Path, workdir: Path) -> str:
    config = _scoring_config()
    source = build_document_source(pdf, run_id=f"bench-odl-{pdf.stem}", config=config, artifact_dir=workdir / "ocr")
    candidates = OpenDataLoaderPdfParserAdapter().parse(source, config)
    if not candidates:
        raise RuntimeError("opendataloader produced no candidates")
    return candidates[0].content


def run_pymupdf4llm(pdf: Path, workdir: Path) -> str:
    del workdir
    import pymupdf4llm

    return pymupdf4llm.to_markdown(str(pdf), show_progress=False)


def run_markitdown(pdf: Path, workdir: Path) -> str:
    del workdir
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(pdf)).text_content


def run_docling(pdf: Path, workdir: Path) -> str:
    del workdir
    from docling.document_converter import DocumentConverter

    return DocumentConverter().convert(str(pdf)).document.export_to_markdown()


def run_parsing_agent(pdf: Path, workdir: Path) -> str:
    from parsing_agent.workflow import WorkflowRunner

    runner = WorkflowRunner(config=WorkflowConfig(langsmith_tracing=False, ocr_enabled=False))
    result, _ = runner.run(pdf, workdir / f"agent-{pdf.stem}")
    return result.best_candidate.content


ENGINES = {
    "opendataloader": run_opendataloader,
    "pymupdf4llm": run_pymupdf4llm,
    "markitdown": run_markitdown,
    "docling": run_docling,
    "parsing-agent": run_parsing_agent,
}


def score(pdf: Path, content: str, workdir: Path, engine: str) -> dict[str, float]:
    config = _scoring_config()
    source = build_document_source(pdf, run_id=f"bench-score-{pdf.stem}", config=config, artifact_dir=workdir / "ocr")
    candidate = ParseCandidate(parser_name=engine, content=content, format_name="md")
    metrics = DeterministicEvaluator(config).evaluate(source, candidate)
    return {column: round(float(getattr(metrics, column)), 4) for column in METRIC_COLUMNS}


def render_markdown(results: dict) -> str:
    lines = [
        "# Head-to-head 벤치마크 결과",
        "",
        "채점기: parsing-agent의 결정적 메트릭 (judge 제외). "
        "**parsing-agent는 이 채점기를 내부에서 직접 최적화하므로 구조적으로 유리하다** — "
        "같은 잣대로 쟀을 때의 방향으로만 해석할 것.",
        "",
    ]
    for doc, engines in results.items():
        lines.append(f"## {doc}")
        lines.append("")
        lines.append("| 엔진 | total | coverage | similarity | structure | table | 시간(s) |")
        lines.append("|---|---|---|---|---|---|---|")
        best = max(
            (data["metrics"]["total_score"] for data in engines.values() if "metrics" in data),
            default=None,
        )
        for engine, data in engines.items():
            if "error" in data:
                lines.append(f"| {engine} | 실패: {data['error']} | | | | | {data['seconds']} |")
                continue
            metrics = data["metrics"]
            total = f"**{metrics['total_score']}**" if metrics["total_score"] == best else str(metrics["total_score"])
            lines.append(
                f"| {engine} | {total} | {metrics['text_coverage']} | {metrics['normalized_similarity']} | "
                f"{metrics['structure_retention']} | {metrics['table_preservation']} | {data['seconds']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--no-agent", action="store_true", help="parsing-agent 풀 루프(API 호출) 생략")
    parser.add_argument("--engines", nargs="*", choices=sorted(ENGINES), help="실행할 엔진 부분 선택")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    load_dotenv(find_dotenv())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workdir = args.output_dir / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    if args.engines:
        engines = {name: ENGINES[name] for name in args.engines}
    else:
        engines = dict(ENGINES)
        if args.no_agent:
            engines.pop("parsing-agent")

    # 부분 실행 결과가 기존 결과를 지우지 않도록 병합한다
    results_path = args.output_dir / "results.json"
    results: dict[str, dict] = {}
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    for pdf in args.pdfs:
        results.setdefault(pdf.name, {})
        for engine, runner in engines.items():
            started = time.monotonic()
            try:
                content = runner(pdf, workdir)
                elapsed = round(time.monotonic() - started, 1)
                results[pdf.name][engine] = {
                    "seconds": elapsed,
                    "characters": len(content),
                    "metrics": score(pdf, content, workdir, engine),
                }
                (workdir / f"{engine}-{pdf.stem}.md").write_text(content, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 - 엔진 하나의 실패가 벤치마크를 멈추면 안 된다
                results[pdf.name][engine] = {
                    "seconds": round(time.monotonic() - started, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            print(f"{pdf.name} / {engine}: {results[pdf.name][engine]}")

    (args.output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "RESULTS.md").write_text(render_markdown(results), encoding="utf-8")
    print(f"\n결과 저장: {args.output_dir}/results.json, {args.output_dir}/RESULTS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
