"""이슈 단위 LLM 텍스트 수리.

heuristic 수리가 해결하지 못한(점수가 정체된) 이슈를 대상으로, 문제가
나타나는 후보 구간을 라인 윈도우로 잘라 원문 근거와 함께 LLM에 보내고
고친 텍스트로 해당 구간만 교체한다. 표 이미지 복구(visual_repair)와
달리 텍스트 이슈 전반(반복, 줄바꿈 노이즈, 누락 구조 등)을 다룬다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from parsing_agent.config import WorkflowConfig
from parsing_agent.judge import _call_with_retry, _parse_judge_verdict, _post_chat_completion
from parsing_agent.models import DocumentSource, RepairAction, load_document_source_text
from parsing_agent.repair import (
    RepairTarget,
    _looks_like_corrupted_table_line,
    _normalize_compare_line,
    _should_merge_lines,
)

_WORD_RE = re.compile(r"\w+")

_SYSTEM_PROMPT = """You are a meticulous editor fixing one specific defect in a parsed Korean document passage.
Rules:
- Fix ONLY the described defect inside the given passage.
- Ground every change in the provided source evidence. NEVER invent content absent from the source.
- Preserve markdown structure (headings, lists, tables) unless the defect is about that structure.
- Keep all Korean text exactly as written except where the defect requires changes.
Return strict JSON: {"fixed_text": "<corrected passage>", "confidence": <0..1>, "changed": <true|false>}
Set "changed" to false and return the passage unchanged if you cannot fix the defect confidently.
Do not include any prose outside the JSON object."""

_MIN_LENGTH_RATIO = 0.3
_MAX_LENGTH_RATIO = 3.5


@dataclass(slots=True)
class TargetedRepairOutcome:
    content: str
    action: RepairAction


def _first_duplicate_line_index(lines: list[str]) -> int | None:
    seen: set[str] = set()
    for index, line in enumerate(lines):
        normalized = _normalize_compare_line(line)
        if not normalized or len(normalized) <= 5:
            continue
        if normalized in seen:
            return index
        seen.add(normalized)
    return None


def _first_wrapped_line_index(lines: list[str]) -> int | None:
    for index in range(len(lines) - 1):
        if _should_merge_lines(lines[index], lines[index + 1]):
            return index
    return None


def _first_corrupted_table_line_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if _looks_like_corrupted_table_line(line):
            return index
    return None


def _first_blank_run_index(lines: list[str], run_length: int = 3) -> int | None:
    blank_run = 0
    for index, line in enumerate(lines):
        if line.strip():
            blank_run = 0
            continue
        blank_run += 1
        if blank_run >= run_length:
            return index - run_length + 1
    return None


def _best_overlap_line_index(lines: list[str], excerpt: str) -> int | None:
    excerpt_tokens = set(_WORD_RE.findall(excerpt.lower()))
    if len(excerpt_tokens) < 3:
        return None
    best_index: int | None = None
    best_score = 0.0
    for index, line in enumerate(lines):
        line_tokens = set(_WORD_RE.findall(line.lower()))
        if not line_tokens:
            continue
        score = len(excerpt_tokens & line_tokens) / len(line_tokens)
        if score > best_score:
            best_score = score
            best_index = index
    if best_score < 0.5:
        return None
    return best_index


def _anchor_line_index(lines: list[str], target: RepairTarget) -> int | None:
    """이슈가 실제로 나타나는 후보 라인을 이슈 타입 기반으로 찾는다."""
    issue_type = target.issue_type.lower()
    if "coverage" in issue_type or "missing" in issue_type:
        # 누락 내용은 잘린 꼬리인 경우가 많다. 문서 끝을 윈도우로 잡아
        # LLM이 원문 근거를 보고 이어붙일 수 있게 한다.
        return len(lines) - 1
    if "repetition" in issue_type or "repeated" in issue_type or "duplication" in issue_type:
        return _first_duplicate_line_index(lines)
    if "wrapped" in issue_type:
        return _first_wrapped_line_index(lines)
    if "table" in issue_type:
        return _first_corrupted_table_line_index(lines)
    if "blank" in issue_type or "empty" in issue_type:
        return _first_blank_run_index(lines)
    if target.candidate_excerpt:
        anchor = _best_overlap_line_index(lines, target.candidate_excerpt)
        if anchor is not None:
            return anchor
    if target.source_excerpt:
        return _best_overlap_line_index(lines, target.source_excerpt)
    return None


def locate_issue_window(content: str, target: RepairTarget, window_lines: int) -> tuple[int, int] | None:
    """이슈 주변 라인 스팬 (start, end)를 반환한다. 못 찾으면 None."""
    lines = content.splitlines()
    if not lines:
        return None
    anchor = _anchor_line_index(lines, target)
    if anchor is None:
        return None
    half = max(1, window_lines // 2)
    start = max(0, anchor - half)
    end = min(len(lines), start + window_lines)
    return start, end


def _source_evidence(source: DocumentSource, target: RepairTarget, max_characters: int) -> str:
    if target.source_excerpt:
        return target.source_excerpt[:max_characters]
    try:
        return load_document_source_text(source)[:max_characters]
    except (OSError, ValueError):
        return ""


def _excerpt(text: str, limit: int = 160) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


class OpenAITargetedTextRepairer:
    """이슈 하나를 라인 윈도우 단위로 LLM에 보내 고치는 수리기."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        min_confidence: float = 0.6,
        window_lines: int = 60,
        max_source_evidence_characters: int = 4_000,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._min_confidence = min_confidence
        self._window_lines = max(10, window_lines)
        self._max_source_evidence_characters = max_source_evidence_characters

    def repair_target(
        self,
        source: DocumentSource,
        content: str,
        target: RepairTarget,
    ) -> TargetedRepairOutcome | None:
        window = locate_issue_window(content, target, self._window_lines)
        if window is None:
            return None
        start, end = window
        lines = content.splitlines()
        passage = "\n".join(lines[start:end])
        if not passage.strip():
            return None
        evidence = _source_evidence(source, target, self._max_source_evidence_characters)
        verdict = self._request_fix(passage, evidence, target)
        if verdict is None:
            return None
        fixed_text = verdict.get("fixed_text")
        confidence = verdict.get("confidence")
        changed = verdict.get("changed")
        if not isinstance(fixed_text, str) or not fixed_text.strip():
            return None
        if changed is False or fixed_text == passage:
            return None
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            return None
        if confidence_value < self._min_confidence:
            return None
        length_ratio = len(fixed_text) / max(1, len(passage))
        issue_type = target.issue_type.lower()
        # 누락 내용 복원은 결과가 윈도우보다 길어지는 게 정상이므로 상한을 완화한다.
        max_ratio = 10.0 if ("coverage" in issue_type or "missing" in issue_type) else _MAX_LENGTH_RATIO
        if not (_MIN_LENGTH_RATIO <= length_ratio <= max_ratio):
            return None
        new_lines = [*lines[:start], *fixed_text.splitlines(), *lines[end:]]
        new_content = "\n".join(new_lines)
        if content.endswith("\n"):
            new_content += "\n"
        action = RepairAction(
            action_name="llm_targeted_text_repair",
            description=f"LLM fixed '{target.issue_type}' in lines {start + 1}-{end}.",
            before_excerpt=_excerpt(passage),
            after_excerpt=_excerpt(fixed_text),
            issue_type=target.issue_type,
            route_name=f"llm:{target.route_name}",
        )
        return TargetedRepairOutcome(content=new_content, action=action)

    def _request_fix(self, passage: str, evidence: str, target: RepairTarget) -> dict[str, Any] | None:
        prompt = (
            f"Defect type: {target.issue_type}\n"
            f"Defect description: {target.description}\n\n"
            f"Source evidence (ground truth, may be partial):\n{evidence or '(unavailable)'}\n\n"
            f"Passage to fix:\n{passage}"
        )
        try:
            response_payload = _call_with_retry(
                _post_chat_completion,
                max_retries=self._max_retries,
                usage_stage="llm_text_repair",
                url=f"{self._base_url}/chat/completions",
                api_key=self._api_key,
                payload={
                    "model": self._model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout_seconds=self._timeout_seconds,
            )
            choices = response_payload.get("choices") or []
            if not choices:
                return None
            message_content = (choices[0].get("message") or {}).get("content")
            if not isinstance(message_content, str):
                return None
            return _parse_judge_verdict(message_content)
        except Exception:  # noqa: BLE001 - 수리 실패는 해당 이슈 skip으로 처리한다
            return None


def build_default_targeted_text_repairer(config: WorkflowConfig) -> OpenAITargetedTextRepairer | None:
    if not config.llm_text_repair_enabled:
        return None
    if not config.llm_text_repair_model or not config.judge_api_key:
        return None
    return OpenAITargetedTextRepairer(
        model=config.llm_text_repair_model,
        api_key=config.judge_api_key,
        base_url=config.judge_base_url,
        timeout_seconds=config.llm_text_repair_timeout_seconds,
        max_retries=config.judge_max_retries,
        min_confidence=config.llm_text_repair_min_confidence,
        window_lines=config.llm_text_repair_window_lines,
    )
