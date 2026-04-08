"""Deterministic failure-tag extraction heuristics for Phase 4.

This module converts raw execution signals into a small fixed set of failure
tags. It does not perform any file I/O or metric-heavy parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final

SUPPORTED_FAILURE_TAGS: Final[tuple[str, ...]] = (
    "nan_loss",
    "inf_loss",
    "divergence",
    "no_improvement",
    "timeout",
    "oom",
    "runtime_error",
    "syntax_error",
)

_NAN_RE: Final[re.Pattern[str]] = re.compile(r"\bnan\b", re.IGNORECASE)
_INF_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:inf|infinity)\b", re.IGNORECASE)
_OOM_PATTERNS: Final[tuple[str, ...]] = (
    "cuda out of memory",
    "out of memory",
)
_DIVERGENCE_PATTERNS: Final[tuple[str, ...]] = (
    "diverged",
    "divergence",
    "exploding loss",
    "loss exploded",
)
_NO_IMPROVEMENT_PATTERNS: Final[tuple[str, ...]] = (
    "no improvement",
    "did not improve",
    "stagnant",
)
_TIMEOUT_PATTERNS: Final[tuple[str, ...]] = (
    "timeout",
    "timed out",
    "time limit exceeded",
)
_SYNTAX_ERROR_PATTERNS: Final[tuple[str, ...]] = (
    "syntaxerror",
    "syntax error",
)
_RUNTIME_ERROR_PATTERNS: Final[tuple[str, ...]] = (
    "runtimeerror",
    "runtime error",
)
_TIMEOUT_EXIT_CODES: Final[frozenset[int]] = frozenset({124})


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Raw execution signals for deterministic failure-tag extraction.

    Input:
    - raw process and exception fields from one execution attempt

    Output:
    - immutable execution-result container

    Failure behavior:
    - no validation beyond ordinary dataclass construction
    """

    exit_code: int | None
    stdout: str | None
    stderr: str | None
    exception: str | None
    duration_seconds: float | None


def extract_failure_tags(execution_result: ExecutionResult) -> list[str]:
    """Extract deterministic failure tags from raw execution signals.

    Input:
    - `execution_result`: raw stdout, stderr, exception, exit code, and duration

    Output:
    - an ordered deduplicated list of supported failure tags

    Failure behavior:
    - never raises for missing fields or empty text
    - returns an empty list when no supported signals are detected
    """

    text = _combined_text(execution_result)
    tags: list[str] = []

    if _contains_regex(text, _NAN_RE):
        tags.append("nan_loss")
    if _contains_regex(text, _INF_RE):
        tags.append("inf_loss")
    if _contains_any(text, _DIVERGENCE_PATTERNS):
        tags.append("divergence")
    if _contains_any(text, _NO_IMPROVEMENT_PATTERNS):
        tags.append("no_improvement")
    if _is_timeout(execution_result, text):
        tags.append("timeout")
    if _contains_any(text, _OOM_PATTERNS):
        tags.append("oom")

    has_syntax_error = _contains_any(text, _SYNTAX_ERROR_PATTERNS)
    if has_syntax_error:
        tags.append("syntax_error")

    if _should_tag_runtime_error(execution_result, text, tags, has_syntax_error):
        tags.append("runtime_error")

    return _dedupe_preserve_order(tags)


def _combined_text(execution_result: ExecutionResult) -> str:
    """Combine text fields into one normalized lowercase search string."""

    parts = [
        execution_result.stdout or "",
        execution_result.stderr or "",
        execution_result.exception or "",
    ]
    return "\n".join(parts).lower()


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether any literal pattern exists in normalized text."""

    return any(pattern in text for pattern in patterns)


def _contains_regex(text: str, pattern: re.Pattern[str]) -> bool:
    """Return whether a regex pattern matches normalized text."""

    return pattern.search(text) is not None


def _is_timeout(execution_result: ExecutionResult, text: str) -> bool:
    """Return whether the signals indicate a timeout."""

    if execution_result.exit_code in _TIMEOUT_EXIT_CODES:
        return True
    return _contains_any(text, _TIMEOUT_PATTERNS)


def _should_tag_runtime_error(
    execution_result: ExecutionResult,
    text: str,
    tags: list[str],
    has_syntax_error: bool,
) -> bool:
    """Return whether generic runtime_error should be emitted."""

    if has_syntax_error:
        return False
    if tags:
        return False
    if _contains_any(text, _RUNTIME_ERROR_PATTERNS):
        return True
    if execution_result.exception:
        return True
    return False


def _dedupe_preserve_order(tags: list[str]) -> list[str]:
    """Deduplicate tags while preserving first-seen order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        ordered.append(tag)
    return ordered


__all__ = [
    "ExecutionResult",
    "SUPPORTED_FAILURE_TAGS",
    "extract_failure_tags",
]
