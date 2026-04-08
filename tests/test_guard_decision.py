from dataclasses import dataclass

from preflight_intent_governor.guard import GuardDecision, evaluate_preflight
from preflight_intent_governor.history import AttemptRecord


@dataclass(frozen=True, slots=True)
class Proposal:
    canonical_param: str
    normalized_value: str


def _record(
    attempt_id: str,
    *,
    block_id: str,
    param_name: str,
    normalized_value: str,
    outcome: str,
) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=attempt_id,
        timestamp="2026-04-07T12:00:00Z",
        run_id="run-001",
        file_path="src/model.py",
        function_name="train_step",
        block_id=block_id,
        param_name=param_name,
        param_value_raw=normalized_value,
        param_value_normalized=normalized_value,
        outcome=outcome,
        failure_tags=[] if outcome != "fail" else ["oom"],
        line_number=42,
    )


def _history_records() -> list[AttemptRecord]:
    return [
        _record("att_1", block_id="blk_a", param_name="lr", normalized_value="0.0010000000", outcome="fail"),
        _record("att_2", block_id="blk_a", param_name="lr", normalized_value="0.0010000000", outcome="fail"),
        _record("att_3", block_id="blk_a", param_name="eps", normalized_value="0.0000010000", outcome="fail"),
        _record("att_4", block_id="blk_a", param_name="lr", normalized_value="0.0005000000", outcome="success"),
        _record("att_5", block_id="blk_b", param_name="lr", normalized_value="0.0010000000", outcome="fail"),
        _record("att_6", block_id="blk_a", param_name="weight_decay", normalized_value="0.0100000000", outcome="unknown"),
    ]


def test_evaluate_preflight_returns_structured_decision() -> None:
    decision = evaluate_preflight("blk_missing", [], _history_records())

    assert isinstance(decision, GuardDecision)
    assert decision.allowed is True
    assert decision.reason_code == "allow_no_conflict"
    assert decision.blocked_params == []
    assert decision.matched_attempt_ids == []
    assert isinstance(decision.notes, list)


def test_evaluate_preflight_blocks_when_only_previously_failed_values_are_proposed() -> None:
    decision = evaluate_preflight(
        "blk_a",
        [Proposal(canonical_param="lr", normalized_value="0.0010000000")],
        _history_records(),
    )

    assert decision.allowed is False
    assert decision.reason_code == "block_repeated_failed_value"
    assert [item.param_name for item in decision.blocked_params] == ["lr"]
    assert [item.proposed_normalized_value for item in decision.blocked_params] == ["0.0010000000"]
    assert decision.matched_attempt_ids == ["att_1", "att_2"]


def test_evaluate_preflight_blocks_no_meaningful_drift_for_empty_proposal_with_failed_history() -> None:
    decision = evaluate_preflight("blk_a", [], _history_records())

    assert decision.allowed is False
    assert decision.reason_code == "block_no_meaningful_drift"
    assert decision.blocked_params == []
    assert decision.matched_attempt_ids == []


def test_evaluate_preflight_allows_new_direction_for_mixed_repeated_and_new_values() -> None:
    decision = evaluate_preflight(
        "blk_a",
        [
            Proposal(canonical_param="lr", normalized_value="0.0010000000"),
            Proposal(canonical_param="eps", normalized_value="0.0000020000"),
        ],
        _history_records(),
    )

    assert decision.allowed is True
    assert decision.reason_code == "allow_new_direction"
    assert decision.blocked_params == []
    assert decision.matched_attempt_ids == ["att_1", "att_2"]


def test_evaluate_preflight_allows_new_value_for_previously_failed_same_param() -> None:
    decision = evaluate_preflight(
        "blk_a",
        [Proposal(canonical_param="lr", normalized_value="0.0005000000")],
        _history_records(),
    )

    assert decision.allowed is True
    assert decision.reason_code == "allow_new_direction"


def test_evaluate_preflight_allows_new_param_even_if_another_param_failed_before() -> None:
    decision = evaluate_preflight(
        "blk_a",
        [Proposal(canonical_param="weight_decay", normalized_value="0.0200000000")],
        _history_records(),
    )

    assert decision.allowed is True
    assert decision.reason_code == "allow_new_direction"


def test_evaluate_preflight_ignores_unknown_and_success_history_for_blocking() -> None:
    decision = evaluate_preflight(
        "blk_a",
        [
            Proposal(canonical_param="weight_decay", normalized_value="0.0100000000"),
            Proposal(canonical_param="lr", normalized_value="0.0005000000"),
        ],
        _history_records(),
    )

    assert decision.allowed is True
    assert decision.reason_code == "allow_new_direction"


def test_evaluate_preflight_duplicate_failed_history_records_do_not_change_outcome_semantics() -> None:
    records = _history_records() + [
        _record("att_dup", block_id="blk_a", param_name="lr", normalized_value="0.0010000000", outcome="fail")
    ]

    decision = evaluate_preflight(
        "blk_a",
        [Proposal(canonical_param="lr", normalized_value="0.0010000000")],
        records,
    )

    assert decision.allowed is False
    assert decision.reason_code == "block_repeated_failed_value"
    assert decision.matched_attempt_ids == ["att_1", "att_2", "att_dup"]


def test_evaluate_preflight_ignores_malformed_proposals_instead_of_crashing() -> None:
    decision = evaluate_preflight(
        "blk_a",
        [
            {"canonical_param": "lr", "normalized_value": "0.0010000000"},
            {"canonical_param": "eps"},
            object(),
        ],
        _history_records(),
    )

    assert decision.allowed is False
    assert decision.reason_code == "block_repeated_failed_value"
    assert decision.matched_attempt_ids == ["att_1", "att_2"]


def test_evaluate_preflight_defensively_normalizes_obvious_param_name_shape() -> None:
    decision = evaluate_preflight(
        "blk_a",
        [Proposal(canonical_param=" LR ", normalized_value="0.0010000000")],
        _history_records(),
    )

    assert decision.allowed is False
    assert decision.reason_code == "block_repeated_failed_value"
    assert [item.param_name for item in decision.blocked_params] == ["lr"]
    assert decision.matched_attempt_ids == ["att_1", "att_2"]
