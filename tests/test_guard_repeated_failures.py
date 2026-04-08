from dataclasses import dataclass

from preflight_intent_governor.guard import (
    build_failed_value_index,
    find_repeated_failed_params,
    get_failed_records_for_block,
)
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


def test_get_failed_records_for_block_filters_only_failures_for_target_block() -> None:
    records = _history_records()

    result = get_failed_records_for_block(records, "blk_a")

    assert [record.attempt_id for record in result] == ["att_1", "att_2", "att_3"]


def test_build_failed_value_index_has_deterministic_shape_and_order() -> None:
    records = _history_records()

    index = build_failed_value_index(records, "blk_a")

    assert list(index) == ["lr", "eps"]
    assert list(index["lr"]) == ["0.0010000000"]
    assert index["lr"]["0.0010000000"] == ["att_1", "att_2"]
    assert index["eps"]["0.0000010000"] == ["att_3"]


def test_find_repeated_failed_params_matches_same_block_same_param_same_value() -> None:
    index = build_failed_value_index(_history_records(), "blk_a")

    blocked = find_repeated_failed_params(
        [Proposal(canonical_param="lr", normalized_value="0.0010000000")],
        index,
    )

    assert len(blocked) == 1
    assert blocked[0].param_name == "lr"
    assert blocked[0].proposed_normalized_value == "0.0010000000"
    assert blocked[0].matched_attempt_ids == ["att_1", "att_2"]


def test_find_repeated_failed_params_ignores_unknown_success_unrelated_block_and_other_param() -> None:
    index = build_failed_value_index(_history_records(), "blk_a")

    blocked = find_repeated_failed_params(
        [
            Proposal(canonical_param="weight_decay", normalized_value="0.0100000000"),
            Proposal(canonical_param="lr", normalized_value="0.0005000000"),
            Proposal(canonical_param="lr", normalized_value="0.0020000000"),
            Proposal(canonical_param="weight_decay", normalized_value="0.0200000000"),
        ],
        index,
    )

    assert blocked == []


def test_find_repeated_failed_params_preserves_proposal_order() -> None:
    index = build_failed_value_index(_history_records(), "blk_a")

    blocked = find_repeated_failed_params(
        [
            Proposal(canonical_param="eps", normalized_value="0.0000010000"),
            Proposal(canonical_param="lr", normalized_value="0.0010000000"),
        ],
        index,
    )

    assert [item.param_name for item in blocked] == ["eps", "lr"]
    assert [item.matched_attempt_ids for item in blocked] == [["att_3"], ["att_1", "att_2"]]
