from dataclasses import dataclass

from preflight_intent_governor.guard import has_meaningful_drift
from preflight_intent_governor.guard import build_failed_value_index
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


def _failed_index() -> dict[str, dict[str, list[str]]]:
    records = [
        _record("att_1", block_id="blk_a", param_name="lr", normalized_value="0.0010000000", outcome="fail"),
        _record("att_2", block_id="blk_a", param_name="eps", normalized_value="0.0000010000", outcome="fail"),
    ]
    return build_failed_value_index(records, "blk_a")


def test_has_meaningful_drift_is_false_when_only_failed_values_are_repeated() -> None:
    index = _failed_index()

    result = has_meaningful_drift(
        [
            Proposal(canonical_param="lr", normalized_value="0.0010000000"),
            Proposal(canonical_param="eps", normalized_value="0.0000010000"),
        ],
        index,
    )

    assert result is False


def test_has_meaningful_drift_is_true_when_any_new_normalized_value_is_present() -> None:
    index = _failed_index()

    result = has_meaningful_drift(
        [
            Proposal(canonical_param="lr", normalized_value="0.0010000000"),
            Proposal(canonical_param="eps", normalized_value="0.0000020000"),
        ],
        index,
    )

    assert result is True


def test_has_meaningful_drift_is_true_for_new_value_on_same_param() -> None:
    index = _failed_index()

    result = has_meaningful_drift(
        [Proposal(canonical_param="lr", normalized_value="0.0005000000")],
        index,
    )

    assert result is True


def test_has_meaningful_drift_is_true_for_new_param_value_pair() -> None:
    index = _failed_index()

    result = has_meaningful_drift(
        [Proposal(canonical_param="weight_decay", normalized_value="0.0100000000")],
        index,
    )

    assert result is True


def test_has_meaningful_drift_is_false_for_empty_proposal() -> None:
    index = _failed_index()

    assert has_meaningful_drift([], index) is False


def test_has_meaningful_drift_ignores_malformed_entries() -> None:
    index = _failed_index()

    result = has_meaningful_drift(
        [
            {"canonical_param": "lr", "normalized_value": "0.0010000000"},
            {"canonical_param": "eps"},
            object(),
        ],
        index,
    )

    assert result is False
