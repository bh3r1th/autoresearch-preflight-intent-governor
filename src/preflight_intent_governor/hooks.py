"""Thin deterministic orchestration hooks for Phase 5.

This module wires together normalization, guard evaluation, heuristics, and
history persistence without duplicating business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .guard import GuardDecision, evaluate_preflight
from .heuristics import ExecutionResult, extract_failure_tags
from .history import AttemptRecord, append_attempt_record, make_attempt_id
from .normalize import (
    extract_function_signature_info,
    extract_supported_params,
    make_block_id,
)


@dataclass(frozen=True, slots=True)
class PostExecutionSummary:
    """Summary of deterministic post-execution recording.

    Input:
    - final recording details for one proposed code execution

    Output:
    - immutable summary containing block id, extracted param count, and records

    Failure behavior:
    - no validation beyond ordinary dataclass construction
    """

    block_id: str
    recorded_count: int
    records: list[AttemptRecord]


def run_preflight_check(
    *,
    file_path: str,
    function_source: str,
    proposed_code: str,
    attempt_records: list[AttemptRecord],
    start_line: int | None = None,
) -> GuardDecision:
    """Run deterministic preflight evaluation for one proposed code change.

    Input:
    - `file_path`: source file path for the target function
    - `function_source`: source text for the target function
    - `proposed_code`: proposed code snippet containing candidate params
    - `attempt_records`: already-loaded history records
    - `start_line`: optional approximate function start line for block-id fallback

    Output:
    - a `GuardDecision` from the Phase 3 guard module

    Failure behavior:
    - never performs file I/O
    - returns the guard decision even when no params are extracted
    """

    block_id = _compute_block_id(
        file_path=file_path,
        function_source=function_source,
        start_line=start_line,
    )
    proposed_params = extract_supported_params(proposed_code)
    return evaluate_preflight(
        block_id=block_id,
        proposed_params=list(proposed_params),
        attempt_records=attempt_records,
    )


def build_attempt_records(
    *,
    file_path: str,
    function_source: str,
    proposed_code: str,
    execution_result: ExecutionResult,
    outcome: str,
    run_id: str,
    timestamp: str | None = None,
    start_line: int | None = None,
) -> list[AttemptRecord]:
    """Build one attempt record per extracted param without writing to disk.

    Input:
    - `file_path`: source file path for the target function
    - `function_source`: source text for the target function
    - `proposed_code`: proposed code snippet containing candidate params
    - `execution_result`: raw execution signals
    - `outcome`: one of `success`, `fail`, or `unknown`
    - `run_id`: caller-provided run identifier
    - `timestamp`: optional explicit timestamp; generated in UTC when omitted
    - `start_line`: optional approximate function start line for block-id fallback

    Output:
    - a list of validated `AttemptRecord` objects in extracted param order

    Failure behavior:
    - returns an empty list when no supported params are extracted
    - propagates validation errors from history record construction
    """

    block_id = _compute_block_id(
        file_path=file_path,
        function_source=function_source,
        start_line=start_line,
    )
    function_name = _extract_function_name(function_source)
    proposed_params = extract_supported_params(proposed_code)
    if not proposed_params:
        return []

    failure_tags = extract_failure_tags(execution_result)
    record_timestamp = timestamp if timestamp is not None else _utc_timestamp()

    records: list[AttemptRecord] = []
    for param in proposed_params:
        attempt_id = make_attempt_id(
            run_id=run_id,
            file_path=file_path,
            function_name=function_name,
            block_id=block_id,
            param_name=param.canonical_param,
            param_value_normalized=param.normalized_value,
            outcome=outcome,
            line_number=param.line_number,
            failure_tags=failure_tags,
        )
        records.append(
            AttemptRecord(
                attempt_id=attempt_id,
                timestamp=record_timestamp,
                run_id=run_id,
                file_path=file_path,
                function_name=function_name,
                block_id=block_id,
                param_name=param.canonical_param,
                param_value_raw=param.raw_value,
                param_value_normalized=param.normalized_value,
                outcome=outcome,
                failure_tags=list(failure_tags),
                line_number=param.line_number,
            )
        )

    return records


def record_execution_result(
    *,
    file_path: str,
    function_source: str,
    proposed_code: str,
    execution_result: ExecutionResult,
    outcome: str,
    run_id: str,
    history_file_path: str | Path,
    timestamp: str | None = None,
    start_line: int | None = None,
) -> PostExecutionSummary:
    """Build and append attempt records for one execution result.

    Input:
    - `file_path`: source file path for the target function
    - `function_source`: source text for the target function
    - `proposed_code`: proposed code snippet containing candidate params
    - `execution_result`: raw execution signals
    - `outcome`: one of `success`, `fail`, or `unknown`
    - `run_id`: caller-provided run identifier
    - `history_file_path`: JSONL history file path
    - `timestamp`: optional explicit timestamp; generated in UTC when omitted
    - `start_line`: optional approximate function start line for block-id fallback

    Output:
    - a `PostExecutionSummary` describing the appended records

    Failure behavior:
    - appends nothing when no supported params are extracted
    - performs file I/O only through `history.append_attempt_record`
    - propagates validation and filesystem errors from lower layers
    """

    block_id = _compute_block_id(
        file_path=file_path,
        function_source=function_source,
        start_line=start_line,
    )
    records = build_attempt_records(
        file_path=file_path,
        function_source=function_source,
        proposed_code=proposed_code,
        execution_result=execution_result,
        outcome=outcome,
        run_id=run_id,
        timestamp=timestamp,
        start_line=start_line,
    )

    for record in records:
        append_attempt_record(history_file_path, record)

    return PostExecutionSummary(
        block_id=block_id,
        recorded_count=len(records),
        records=records,
    )


def _compute_block_id(
    *,
    file_path: str,
    function_source: str,
    start_line: int | None,
) -> str:
    """Compute a stable block id from file path and function source."""

    return make_block_id(
        file_path=file_path,
        function_source=function_source,
        start_line=start_line,
    )


def _extract_function_name(function_source: str) -> str:
    """Extract the normalized function name input for history records."""

    signature = extract_function_signature_info(function_source)
    return signature.function_name if signature.parsed else ""


def _utc_timestamp() -> str:
    """Return a deterministic UTC timestamp string format for new records."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "PostExecutionSummary",
    "build_attempt_records",
    "record_execution_result",
    "run_preflight_check",
]
