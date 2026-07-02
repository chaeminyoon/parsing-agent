"""워크플로우 실행 중 발생한 LLM 호출의 횟수·소요시간·토큰을 집계한다.

judge, LLM 텍스트 수리, 비전 표 복구가 각각 얼마나 호출됐고 얼마나
걸렸는지를 리포트의 `monitoring.llm_usage`로 남기기 위한 모듈이다.
수집기는 프로세스 전역이며 run 시작 시 reset된다 — 한 프로세스에서
동시에 여러 run을 돌리면 집계가 섞이므로, 그 경우 run 단위 격리가
필요하다 (현재 배포 형태는 celery task당 단일 run이라 문제 없음).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

_MAX_DETAILED_RECORDS = 50


@dataclass(slots=True)
class LLMCallRecord:
    stage: str
    model: str | None
    duration_ms: int
    ok: bool
    attempts: int = 1
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


class _UsageCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[LLMCallRecord] = []

    def reset(self) -> None:
        with self._lock:
            self._records = []

    def record(self, record: LLMCallRecord) -> None:
        with self._lock:
            self._records.append(record)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
        by_stage: dict[str, dict[str, Any]] = {}
        for record in records:
            stage = by_stage.setdefault(
                record.stage,
                {
                    "calls": 0,
                    "errors": 0,
                    "total_attempts": 0,
                    "total_duration_ms": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "models": [],
                },
            )
            stage["calls"] += 1
            stage["total_attempts"] += record.attempts
            stage["total_duration_ms"] += record.duration_ms
            if not record.ok:
                stage["errors"] += 1
            if record.prompt_tokens is not None:
                stage["prompt_tokens"] += record.prompt_tokens
            if record.completion_tokens is not None:
                stage["completion_tokens"] += record.completion_tokens
            if record.model and record.model not in stage["models"]:
                stage["models"].append(record.model)
        return {
            "total_calls": len(records),
            "total_errors": sum(1 for record in records if not record.ok),
            "total_duration_ms": sum(record.duration_ms for record in records),
            "by_stage": by_stage,
            "calls": [
                {
                    "stage": record.stage,
                    "model": record.model,
                    "duration_ms": record.duration_ms,
                    "ok": record.ok,
                    "attempts": record.attempts,
                    "prompt_tokens": record.prompt_tokens,
                    "completion_tokens": record.completion_tokens,
                    "error": record.error,
                }
                for record in records[:_MAX_DETAILED_RECORDS]
            ],
        }


_collector = _UsageCollector()


def reset_llm_usage() -> None:
    _collector.reset()


def record_llm_call(
    *,
    stage: str,
    model: str | None,
    duration_ms: int,
    ok: bool,
    attempts: int = 1,
    response_payload: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    prompt_tokens, completion_tokens = extract_token_usage(response_payload)
    _collector.record(
        LLMCallRecord(
            stage=stage,
            model=model,
            duration_ms=duration_ms,
            ok=ok,
            attempts=attempts,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=error,
        )
    )


def llm_usage_summary() -> dict[str, Any]:
    return _collector.summary()


def extract_token_usage(response_payload: dict[str, Any] | None) -> tuple[int | None, int | None]:
    """chat completions(prompt/completion_tokens)와 responses API
    (input/output_tokens) 양쪽의 usage 필드를 지원한다."""
    if not isinstance(response_payload, dict):
        return None, None
    usage = response_payload.get("usage")
    if not isinstance(usage, dict):
        return None, None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    try:
        prompt_value = int(prompt) if prompt is not None else None
    except (TypeError, ValueError):
        prompt_value = None
    try:
        completion_value = int(completion) if completion is not None else None
    except (TypeError, ValueError):
        completion_value = None
    return prompt_value, completion_value
