"""Deterministic preflight guard evaluation for Phase 3.

This module consumes already-normalized proposed params and already-loaded
attempt history records. It does not perform any file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _HasAttemptRecordFields(Protocol):
    """Protocol for the subset of history record fields used by guard logic."""

    attempt_id: str
    block_id: str
    param_name: str
    param_value_normalized: str
    outcome: str


class _HasProposedParamFields(Protocol):
    """Protocol for the subset of normalized proposal fields used by guard logic."""

    canonical_param: str
    normalized_value: str


@dataclass(frozen=True, slots=True)
class BlockedParam:
    """Blocked proposal detail for deterministic debugging output.

    Input:
    - one proposed canonical param/value pair and matching failed attempt ids

    Output:
    - immutable blocked-param detail

    Failure behavior:
    - no validation beyond ordinary dataclass construction
    """

    param_name: str
    proposed_normalized_value: str
    matched_attempt_ids: list[str]


@dataclass(frozen=True, slots=True)
class GuardDecision:
    """Explicit preflight decision result.

    Input:
    - final deterministic decision fields

    Output:
    - immutable decision record with allow/block result and debug details

    Failure behavior:
    - no validation beyond ordinary dataclass construction
    """

    allowed: bool
    reason_code: str
    blocked_params: list[BlockedParam]
    matched_attempt_ids: list[str]
    notes: list[str]


def get_failed_records_for_block(
    attempt_records: list[_HasAttemptRecordFields],
    block_id: str,
) -> list[_HasAttemptRecordFields]:
    """Return failed records for one block in existing order.

    Input:
    - `attempt_records`: already-loaded history records
    - `block_id`: target block id

    Output:
    - records for the block whose outcome is exactly `"fail"`

    Failure behavior:
    - never raises for empty inputs or no matches
    """

    return [
        record
        for record in attempt_records
        if record.block_id == block_id and record.outcome == "fail"
    ]


def build_failed_value_index(
    attempt_records: list[_HasAttemptRecordFields],
    block_id: str,
) -> dict[str, dict[str, list[str]]]:
    """Build a deterministic failed-value index for one block.

    Input:
    - `attempt_records`: already-loaded history records
    - `block_id`: target block id

    Output:
    - a nested mapping of:
      `param_name -> normalized_value -> [attempt_id, ...]`
    - inner attempt-id lists preserve input record order

    Failure behavior:
    - never raises for empty inputs or no matches
    """

    index: dict[str, dict[str, list[str]]] = {}
    for record in get_failed_records_for_block(attempt_records, block_id):
        per_param = index.setdefault(record.param_name, {})
        per_value = per_param.setdefault(record.param_value_normalized, [])
        per_value.append(record.attempt_id)
    return index


def find_repeated_failed_params(
    proposed_params: list[object],
    failed_value_index: dict[str, dict[str, list[str]]],
) -> list[BlockedParam]:
    """Return proposed params that repeat previously failed normalized values.

    Input:
    - `proposed_params`: normalized proposal entries
    - `failed_value_index`: nested failed-value mapping from `build_failed_value_index`

    Output:
    - blocked-param details in proposal order

    Failure behavior:
    - ignores malformed or unsupported proposal entries instead of raising
    """

    blocked: list[BlockedParam] = []

    for proposal in proposed_params:
        candidate = _extract_proposed_param(proposal)
        if candidate is None:
            continue

        param_name, normalized_value = candidate
        attempt_ids = failed_value_index.get(param_name, {}).get(normalized_value)
        if not attempt_ids:
            continue

        blocked.append(
            BlockedParam(
                param_name=param_name,
                proposed_normalized_value=normalized_value,
                matched_attempt_ids=list(attempt_ids),
            )
        )

    return blocked


def has_meaningful_drift(
    proposed_params: list[object],
    failed_value_index: dict[str, dict[str, list[str]]],
) -> bool:
    """Return whether the proposal introduces any new normalized value.

    Input:
    - `proposed_params`: normalized proposal entries
    - `failed_value_index`: nested failed-value mapping from `build_failed_value_index`

    Output:
    - `True` when at least one supported proposed param has a normalized value
      that does not appear in failed history for that param
    - `False` otherwise

    Failure behavior:
    - ignores malformed or unsupported proposal entries instead of raising
    """

    for proposal in proposed_params:
        candidate = _extract_proposed_param(proposal)
        if candidate is None:
            continue

        param_name, normalized_value = candidate
        if normalized_value not in failed_value_index.get(param_name, {}):
            return True

    return False


def evaluate_preflight(
    block_id: str,
    proposed_params: list[object],
    attempt_records: list[_HasAttemptRecordFields],
) -> GuardDecision:
    """Evaluate a deterministic preflight decision for one block.

    Rule order:
    1. gather failed history for the target block
    2. identify repeated failed proposed params
    3. detect whether any proposal introduces a new normalized value
    4. allow when there is no failed history for the block
    5. block when repeated failed params exist and there is no meaningful drift
    6. allow when there is at least one new direction
    7. otherwise allow because there is no matched conflict

    Input:
    - `block_id`: target normalized block id
    - `proposed_params`: normalized proposal entries
    - `attempt_records`: already-loaded history records

    Output:
    - a deterministic `GuardDecision`

    Failure behavior:
    - ignores malformed or unsupported proposal entries instead of raising
    """

    failed_records = get_failed_records_for_block(attempt_records, block_id)
    if not failed_records:
        return GuardDecision(
            allowed=True,
            reason_code="allow_no_conflict",
            blocked_params=[],
            matched_attempt_ids=[],
            notes=["no failed history for block"],
        )

    failed_value_index = build_failed_value_index(attempt_records, block_id)
    repeated_failed = find_repeated_failed_params(proposed_params, failed_value_index)
    meaningful_drift = has_meaningful_drift(proposed_params, failed_value_index)

    if repeated_failed and not meaningful_drift:
        matched_attempt_ids = _flatten_matched_attempt_ids(repeated_failed)
        reason_code = (
            "block_repeated_failed_value"
            if proposed_params and _all_supported_proposals_are_repeated(proposed_params, failed_value_index)
            else "block_no_meaningful_drift"
        )
        return GuardDecision(
            allowed=False,
            reason_code=reason_code,
            blocked_params=repeated_failed,
            matched_attempt_ids=matched_attempt_ids,
            notes=["proposal introduces no new normalized value for block"],
        )

    if not meaningful_drift and not _has_supported_proposals(proposed_params):
        return GuardDecision(
            allowed=False,
            reason_code="block_no_meaningful_drift",
            blocked_params=[],
            matched_attempt_ids=[],
            notes=["proposal introduces no supported normalized value for block"],
        )

    if meaningful_drift:
        return GuardDecision(
            allowed=True,
            reason_code="allow_new_direction",
            blocked_params=[],
            matched_attempt_ids=_flatten_matched_attempt_ids(repeated_failed),
            notes=["proposal introduces at least one new normalized value"],
        )

    return GuardDecision(
        allowed=True,
        reason_code="allow_no_conflict",
        blocked_params=[],
        matched_attempt_ids=[],
        notes=["no repeated failed normalized value matched proposal"],
    )


def _extract_proposed_param(proposal: object) -> tuple[str, str] | None:
    """Extract canonical param name and normalized value from a proposal entry."""

    if hasattr(proposal, "canonical_param") and hasattr(proposal, "normalized_value"):
        param_name = getattr(proposal, "canonical_param")
        normalized_value = getattr(proposal, "normalized_value")
    elif isinstance(proposal, dict):
        param_name = proposal.get("canonical_param")
        normalized_value = proposal.get("normalized_value")
    else:
        return None

    if not isinstance(param_name, str) or not isinstance(normalized_value, str):
        return None

    param_name = param_name.strip().lower()
    normalized_value = normalized_value.strip()
    if not param_name or not normalized_value:
        return None

    return (param_name, normalized_value)


def _flatten_matched_attempt_ids(blocked_params: list[BlockedParam]) -> list[str]:
    """Flatten matched attempt ids from blocked params in deterministic order."""

    flattened: list[str] = []
    for blocked in blocked_params:
        flattened.extend(blocked.matched_attempt_ids)
    return flattened


def _all_supported_proposals_are_repeated(
    proposed_params: list[object],
    failed_value_index: dict[str, dict[str, list[str]]],
) -> bool:
    """Return whether every supported proposal entry matches a failed value."""

    found_supported = False

    for proposal in proposed_params:
        candidate = _extract_proposed_param(proposal)
        if candidate is None:
            continue

        found_supported = True
        param_name, normalized_value = candidate
        if normalized_value not in failed_value_index.get(param_name, {}):
            return False

    return found_supported


def _has_supported_proposals(proposed_params: list[object]) -> bool:
    """Return whether at least one proposal entry is supported and well-formed."""

    for proposal in proposed_params:
        if _extract_proposed_param(proposal) is not None:
            return True
    return False


__all__ = [
    "BlockedParam",
    "GuardDecision",
    "build_failed_value_index",
    "evaluate_preflight",
    "find_repeated_failed_params",
    "get_failed_records_for_block",
    "has_meaningful_drift",
]
