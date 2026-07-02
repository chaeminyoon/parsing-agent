from __future__ import annotations

import base64
import json
import re
import socket
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request

import fitz
from langsmith import tracing_context

from parsing_agent.config import WorkflowConfig
from parsing_agent.interfaces import CandidateJudge
from parsing_agent.llm_usage import record_llm_call
from parsing_agent.monitoring import load_judge_prompt_hints
from parsing_agent.models import DocumentSource, EvaluationMetrics, JudgeResult, ParseCandidate, load_document_source_text

_SYSTEM_PROMPT = """You are judging the quality of a parsed document against its source.
Return strict JSON with this schema:
{
  "overall_score": number between 0 and 1,
  "coverage_score": number between 0 and 1,
  "structure_score": number between 0 and 1,
  "table_score": number between 0 and 1,
  "hallucination_risk": number between 0 and 1,
  "editorial_readiness": number between 0 and 1,
  "notes": ["short note", "..."],
  "issues": ["specific issue", "..."],
  "table_findings": [
    {
      "issue_type": "missing_header | split_multipage_table | merged_cell_loss | numeric_token_break | table_text_duplication",
      "table_label": "표 4.2-2",
      "page_number": 12
    }
  ]
}
`hallucination_risk` is penalty-compatible: 0 means low risk, 1 means high risk.
`issues` may be omitted or an empty list.
`table_findings` may be omitted or an empty list.
`overall_score` should reflect content fidelity, structural preservation, formatting usefulness, and penalize hallucination risk.
`issues` and `notes` are human-readable commentary ONLY and are never machine-parsed.
Any problem that should trigger a repair MUST appear in `table_findings` with an exact taxonomy `issue_type` value; a problem mentioned only in `issues` will not be acted on.
Use `page_number` as the grounded PDF page number you actually inspected, not a printed page label from the document body.
Do not include any prose outside the JSON object."""
_TABLE_LABEL_RE = re.compile(r"(?:table|\uD45C)\s*(?:<\s*)?(\d+(?:\.\d+)*(?:-\d+)?)", re.IGNORECASE)
def _extract_message_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        raise ValueError("LLM judge response did not include choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        joined_text = "\n".join(part for part in text_parts if part)
        if not joined_text:
            raise ValueError("LLM judge response content list included no text blocks.")
        return joined_text
    raise ValueError("LLM judge response content format was not recognized.")


def _post_chat_completion(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with tracing_context(enabled=False):
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _post_response(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with tracing_context(enabled=False):
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


class JudgeUnavailableError(RuntimeError):
    """LLM judge 호출이 재시도 후에도 실패했을 때 발생한다."""


_RETRYABLE_HTTP_STATUSES = {408, 409, 429, 500, 502, 503, 504}
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, urllib_error.HTTPError):
        return exc.code in _RETRYABLE_HTTP_STATUSES
    return isinstance(exc, (urllib_error.URLError, TimeoutError, socket.timeout, ConnectionError))


def _call_with_retry(
    post_fn,
    *,
    max_retries: int,
    backoff_seconds: float = 1.0,
    usage_stage: str = "judge",
    **kwargs,
) -> dict[str, Any]:
    """LLM HTTP 호출을 일시 오류에 한해 지수 백오프로 재시도한다.

    비-일시 오류(4xx 등)와 재시도 소진은 JudgeUnavailableError로 감싸
    호출자가 judge 실패를 한 가지 타입으로 처리할 수 있게 한다.
    성공/실패와 무관하게 호출 횟수·소요시간·토큰을 llm_usage에 남긴다.
    """
    model = None
    payload = kwargs.get("payload")
    if isinstance(payload, dict):
        model = payload.get("model")
    started = time.monotonic()
    last_error: Exception | None = None
    attempts = 0
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        try:
            response_payload = post_fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 - 모든 실패를 JudgeUnavailableError로 정규화
            last_error = exc
            if attempt >= max_retries or not _is_retryable_error(exc):
                break
            time.sleep(backoff_seconds * (2**attempt))
            continue
        record_llm_call(
            stage=usage_stage,
            model=model,
            duration_ms=int((time.monotonic() - started) * 1000),
            ok=True,
            attempts=attempts,
            response_payload=response_payload,
        )
        return response_payload
    record_llm_call(
        stage=usage_stage,
        model=model,
        duration_ms=int((time.monotonic() - started) * 1000),
        ok=False,
        attempts=attempts,
        error=f"{type(last_error).__name__}: {last_error}",
    )
    raise JudgeUnavailableError(
        f"LLM judge request failed after {max_retries + 1} attempt(s): {type(last_error).__name__}: {last_error}"
    ) from last_error


def _parse_judge_verdict(raw_text: str) -> dict[str, Any]:
    """judge 응답 텍스트에서 JSON verdict를 최대한 복원한다.

    strict JSON을 우선 시도하고, 실패하면 코드펜스 내부 → 첫 번째
    중괄호 블록 순서로 폴백한다. 전부 실패하면 ValueError.
    """
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if 0 <= brace_start < brace_end:
        try:
            parsed = json.loads(text[brace_start : brace_end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ValueError("LLM judge response did not contain a parseable JSON object.")


def _extract_response_text(response_payload: dict[str, Any]) -> str:
    output_items = response_payload.get("output") or []
    text_parts: list[str] = []
    for item in output_items:
        if item.get("type") != "message":
            continue
        for content_item in item.get("content") or []:
            if content_item.get("type") == "output_text":
                text = content_item.get("text")
                if text:
                    text_parts.append(str(text))
    if text_parts:
        return "\n".join(text_parts)
    raise ValueError("LLM judge multimodal response did not include output_text content.")


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def _coerce_optional_score(payload: dict[str, Any], field_name: str) -> float | None:
    value = payload.get(field_name)
    if value is None:
        return None
    try:
        return _clamp(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_table_findings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    findings: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        finding: dict[str, Any] = {}
        issue_type = item.get("issue_type")
        if issue_type is not None:
            finding["issue_type"] = str(issue_type)
        table_label = item.get("table_label")
        if table_label is not None:
            finding["table_label"] = str(table_label)
        page_number = item.get("page_number")
        if page_number is not None:
            try:
                finding["page_number"] = int(page_number)
            except (TypeError, ValueError):
                pass
        if finding:
            findings.append(finding)
    return findings


def _is_pdf_source(source: DocumentSource) -> bool:
    return source.media_type == "application/pdf" or source.path.suffix.lower() == ".pdf"


def _extract_table_labels(text: str, limit: int = 5) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for match in _TABLE_LABEL_RE.finditer(text):
        label = f"표 {match.group(1)}"
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def _has_table_cues(text: str) -> bool:
    if _TABLE_LABEL_RE.search(text):
        return True
    stripped = text.strip()
    if "<table" in stripped.lower():
        return True
    markdown_table_lines = 0
    for line in text.splitlines():
        raw = line.strip()
        if raw.count("|") >= 2 and (raw.startswith("|") or raw.endswith("|")):
            markdown_table_lines += 1
            if markdown_table_lines >= 2:
                return True
    return False


def _build_table_review_guidance(
    source: DocumentSource,
    source_text: str,
    candidate_text: str,
    metrics: EvaluationMetrics,
) -> str:
    source_labels = _extract_table_labels(source_text)
    candidate_labels = _extract_table_labels(candidate_text)
    table_cues_present = (
        _has_table_cues(source_text)
        or _has_table_cues(candidate_text)
        or metrics.table_preservation < 0.85
    )
    if not table_cues_present:
        return ""
    guidance_lines = [
        "If the document contains table cues, inspect table fidelity explicitly.",
        "Do not leave table_findings empty when you can identify a broken, partial, duplicated, or structurally damaged table.",
        "Each table_findings entry should use one taxonomy issue_type and include table_label when visible.",
    ]
    if source.page_count is not None:
        guidance_lines.append(
            f"Use grounded PDF page numbers only. Valid page_number range is 1..{source.page_count}."
        )
    if source_labels:
        guidance_lines.append(f"Source table cues: {', '.join(source_labels)}")
    if candidate_labels:
        guidance_lines.append(f"Candidate table cues: {', '.join(candidate_labels)}")
    return "\n".join(guidance_lines)


def _render_pdf_page_data_urls(source: DocumentSource, max_pages: int) -> list[tuple[int, str]]:
    if not _is_pdf_source(source):
        return []
    page_limit = max(0, min(source.page_count or max_pages, max_pages))
    if page_limit <= 0:
        return []
    images: list[tuple[int, str]] = []
    with fitz.open(source.path) as document:
        for page_index in range(min(page_limit, document.page_count)):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
            encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
            images.append((page_index + 1, f"data:image/png;base64,{encoded}"))
    return images


class OpenAICompatibleJudge(CandidateJudge):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
        enable_multimodal_grounding: bool = True,
        grounding_max_pages: int = 2,
        prompt_hints: list[str] | None = None,
        system_prompt: str | None = None,
        max_source_characters: int = 12_000,
        max_candidate_characters: int = 12_000,
        evidence_segments: int = 4,
        table_evidence_limit: int = 3,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._enable_multimodal_grounding = enable_multimodal_grounding
        self._grounding_max_pages = grounding_max_pages
        self._prompt_hints = list(prompt_hints or [])
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._max_source_characters = max_source_characters
        self._max_candidate_characters = max_candidate_characters
        self._evidence_segments = max(2, evidence_segments)
        self._table_evidence_limit = max(0, table_evidence_limit)

    def judge(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
        metrics: EvaluationMetrics,
    ) -> JudgeResult:
        source_text = load_document_source_text(source)
        evidence = _build_judge_evidence(
            source_text=source_text,
            candidate_text=candidate.content,
            max_source_characters=self._max_source_characters,
            max_candidate_characters=self._max_candidate_characters,
            segment_count=self._evidence_segments,
            table_evidence_limit=self._table_evidence_limit,
        )
        prompt = self._build_prompt(
            source,
            evidence["source_text"],
            evidence["candidate_text"],
            metrics,
        )
        grounding_pages = _render_pdf_page_data_urls(source, self._grounding_max_pages) if self._enable_multimodal_grounding else []
        if grounding_pages:
            response_payload = _call_with_retry(
                _post_response,
                max_retries=self._max_retries,
                url=f"{self._base_url}/responses",
                api_key=self._api_key,
                payload={
                    "model": self._model,
                    "input": [
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": self._system_prompt}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": prompt}]
                            + [
                                {"type": "input_image", "image_url": image_url, "detail": "high"}
                                for _, image_url in grounding_pages
                            ],
                        },
                    ],
                },
                timeout_seconds=self._timeout_seconds,
            )
            verdict = _parse_judge_verdict(_extract_response_text(response_payload))
        else:
            response_payload = _call_with_retry(
                _post_chat_completion,
                max_retries=self._max_retries,
                url=f"{self._base_url}/chat/completions",
                api_key=self._api_key,
                payload={
                    "model": self._model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout_seconds=self._timeout_seconds,
            )
            verdict = _parse_judge_verdict(_extract_message_content(response_payload))
        overall_score = _coerce_optional_score(verdict, "overall_score")
        if overall_score is None:
            overall_score = _coerce_optional_score(verdict, "score")
        if overall_score is None:
            raise ValueError("LLM judge verdict did not include a usable overall_score.")
        return JudgeResult(
            overall_score=overall_score,
            coverage_score=_coerce_optional_score(verdict, "coverage_score"),
            structure_score=_coerce_optional_score(verdict, "structure_score"),
            table_score=_coerce_optional_score(verdict, "table_score"),
            hallucination_risk=_coerce_optional_score(verdict, "hallucination_risk"),
            editorial_readiness=_coerce_optional_score(verdict, "editorial_readiness"),
            notes=_coerce_string_list(verdict.get("notes")),
            issues=_coerce_string_list(verdict.get("issues")),
            table_findings=_coerce_table_findings(verdict.get("table_findings")),
            metadata={
                "transport": "responses" if grounding_pages else "chat_completions",
                "grounding_enabled": bool(grounding_pages),
                "grounding_pages": [page_number for page_number, _ in grounding_pages],
                "evidence": evidence["metadata"],
            },
        )

    def _build_prompt(
        self,
        source: DocumentSource,
        source_text: str,
        candidate_text: str,
        metrics: EvaluationMetrics,
    ) -> str:
        tuning_text = ""
        if self._prompt_hints:
            tuning_text = "\n".join(f"- {hint}" for hint in self._prompt_hints)
            tuning_text = f"\n\nExtra review instructions from prior failures:\n{tuning_text}"
        table_guidance = _build_table_review_guidance(source, source_text, candidate_text, metrics)
        if table_guidance:
            table_guidance = f"\n\nTable review requirements:\n{table_guidance}"
        return (
            "Judge the parser output against the source.\n\n"
            f"Deterministic metrics: coverage={metrics.text_coverage:.3f}, "
            f"similarity={metrics.normalized_similarity:.3f}, "
            f"structure={metrics.structure_retention:.3f}, "
            f"table={metrics.table_preservation:.3f}, "
            f"empty_penalty={metrics.empty_block_penalty:.3f}, "
            f"repeat_penalty={metrics.repetition_penalty:.3f}"
            f"{tuning_text}{table_guidance}\n\n"
            "The source and candidate below are representative evidence, not full documents. "
            "Use the deterministic metrics for document-wide coverage; use the evidence to verify "
            "semantic fidelity, structural quality, and the listed table areas.\n\n"
            f"Source evidence:\n{source_text}\n\n"
            f"Candidate evidence:\n{candidate_text}"
        )


def build_default_judge(config: WorkflowConfig) -> CandidateJudge | None:
    if config.judge_weight <= 0:
        return None
    if not config.judge_model or not config.judge_api_key:
        return None
    return OpenAICompatibleJudge(
        model=config.judge_model,
        api_key=config.judge_api_key,
        base_url=config.judge_base_url,
        timeout_seconds=config.judge_timeout_seconds,
        enable_multimodal_grounding=config.judge_multimodal_grounding_enabled,
        grounding_max_pages=config.judge_grounding_max_pages,
        prompt_hints=load_judge_prompt_hints(config),
        system_prompt=config.judge_system_prompt,
        max_source_characters=config.judge_max_source_characters,
        max_candidate_characters=config.judge_max_candidate_characters,
        evidence_segments=config.judge_evidence_segments,
        table_evidence_limit=config.judge_table_evidence_limit,
        max_retries=config.judge_max_retries,
    )


def _build_judge_evidence(
    *,
    source_text: str,
    candidate_text: str,
    max_source_characters: int,
    max_candidate_characters: int,
    segment_count: int,
    table_evidence_limit: int,
) -> dict[str, object]:
    labels = _select_table_labels(source_text, candidate_text, table_evidence_limit)
    source_evidence = _build_evidence_excerpt(
        source_text,
        max_characters=max_source_characters,
        segment_count=segment_count,
        table_labels=labels,
    )
    candidate_evidence = _build_evidence_excerpt(
        candidate_text,
        max_characters=max_candidate_characters,
        segment_count=segment_count,
        table_labels=labels,
    )
    return {
        "source_text": source_evidence,
        "candidate_text": candidate_evidence,
        "metadata": {
            "mode": "stratified_evidence",
            "source_full_character_count": len(source_text),
            "candidate_full_character_count": len(candidate_text),
            "source_evidence_character_count": len(source_evidence),
            "candidate_evidence_character_count": len(candidate_evidence),
            "segment_count": segment_count,
            "table_labels": labels,
        },
    }


def _select_table_labels(source_text: str, candidate_text: str, limit: int) -> list[str]:
    if limit <= 0:
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for text in (source_text, candidate_text):
        for match in _TABLE_LABEL_RE.finditer(text):
            label = match.group(1)
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
    if len(labels) <= limit:
        return labels
    indexes = [round(index * (len(labels) - 1) / (limit - 1)) for index in range(limit)] if limit > 1 else [0]
    return [labels[index] for index in indexes]


def _build_evidence_excerpt(
    text: str,
    *,
    max_characters: int,
    segment_count: int,
    table_labels: list[str],
) -> str:
    if max_characters <= 0 or len(text) <= max_characters:
        return text

    pieces: list[str] = []
    table_budget = min(max_characters // 2, len(table_labels) * 1_000)
    if table_labels and table_budget:
        per_table_budget = max(300, table_budget // len(table_labels))
        for label in table_labels:
            pattern = re.compile(rf"(?:table|\uD45C)\s*(?:<\s*)?{re.escape(label)}", re.IGNORECASE)
            match = pattern.search(text)
            if match is None:
                continue
            start = max(0, match.start() - per_table_budget // 3)
            end = min(len(text), start + per_table_budget)
            pieces.append(f"[Table evidence: {label}]\n{text[start:end]}")

    used = sum(len(piece) for piece in pieces)
    remaining = max(1_000, max_characters - used)
    segment_budget = max(300, remaining // segment_count)
    max_start = max(0, len(text) - segment_budget)
    for index in range(segment_count):
        start = round(index * max_start / max(1, segment_count - 1))
        end = min(len(text), start + segment_budget)
        pieces.append(f"[Document segment {index + 1}/{segment_count}]\n{text[start:end]}")

    return "\n\n---\n\n".join(pieces)[:max_characters]
