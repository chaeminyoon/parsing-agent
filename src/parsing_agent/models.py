from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parsing_agent.textutil import read_text_with_fallback


def load_document_source_text(source: "DocumentSource") -> str:
    if source.extracted_text is not None:
        return source.extracted_text
    if source.media_type.startswith("text/"):
        try:
            return read_text_with_fallback(source.path)
        except UnicodeDecodeError:
            return source.path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(
        f"DocumentSource.extracted_text is required for non-text media type {source.media_type!r}."
    )


@dataclass(slots=True)
class DocumentSource:
    path: Path
    media_type: str
    size_bytes: int
    run_id: str
    extracted_text: str | None = None
    page_count: int | None = None
    ocr_metadata: dict[str, Any] = field(default_factory=dict)
    ocr_artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ParseCandidate:
    parser_name: str
    content: str
    format_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    repaired_from: str | None = None


@dataclass(slots=True)
class EvaluationIssue:
    issue_type: str
    metric_name: str
    severity: str
    confidence: float
    description: str
    source_excerpt: str | None = None
    candidate_excerpt: str | None = None
    page_number: int | None = None
    table_label: str | None = None
    bbox: list[float] | None = None
    repairability: str | None = None


@dataclass(slots=True)
class EvaluationMetrics:
    text_coverage: float
    normalized_similarity: float
    structure_retention: float
    table_preservation: float
    empty_block_penalty: float
    repetition_penalty: float
    total_score: float
    table_issues: list[str] = field(default_factory=list)
    llm_judge_score: float | None = None
    judge_result: JudgeResult | None = None
    # PDF 괘선 표 그리드 대비 셀 단위 유사도(TEDS-lite). 기준 표가 없으면 None.
    # 진단용 — total_score에는 섞지 않는다 (골든셋 상관 확인 전까지).
    table_cell_similarity: float | None = None
    notes: list[str] = field(default_factory=list)
    issues: list[EvaluationIssue] = field(default_factory=list)


@dataclass(slots=True)
class JudgeResult:
    overall_score: float
    coverage_score: float | None = None
    structure_score: float | None = None
    table_score: float | None = None
    hallucination_risk: float | None = None
    editorial_readiness: float | None = None
    notes: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    table_findings: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.overall_score


@dataclass(slots=True)
class RepairAction:
    action_name: str
    description: str
    before_excerpt: str
    after_excerpt: str
    issue_type: str | None = None
    route_name: str | None = None


@dataclass(slots=True)
class DocumentSummary:
    file_name: str
    media_type: str
    page_count: int | None
    overview: str
    stats: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowResult:
    source: DocumentSource
    best_candidate: ParseCandidate
    metrics: EvaluationMetrics
    document_summary: DocumentSummary | None = None
    repairs: list[RepairAction] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
