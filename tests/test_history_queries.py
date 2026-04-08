import pytest

from preflight_intent_governor.history import (
    AttemptRecord,
    get_failed_records_for_block,
    get_recent_records_for_block,
    get_records_for_block,
    get_repeated_failed_values,
    group_records_by_param_name_for_block,
)


def _make_record(
    attempt_id: str,
    *,
    block_id: str,
    param_name: str,
    normalized: str,
    outcome: str,
    raw: str | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=attempt_id,
        timestamp=f"2026-04-07T12:00:{attempt_id[-1]}Z",
        run_id="run-001",
        file_path="src/model.py",
        function_name="train_step",
        block_id=block_id,
        param_name=param_name,
        param_value_raw=raw or normalized,
        param_value_normalized=normalized,
        outcome=outcome,
        failure_tags=[] if outcome != "fail" else ["oom"],
        line_number=42,
    )


def _records_fixture() -> list[AttemptRecord]:
    return [
        _make_record("att_1", block_id="blk_a", param_name="lr", normalized="0.0010000000", outcome="fail", raw="1e-3"),
        _make_record("att_2", block_id="blk_a", param_name="lr", normalized="0.0010000000", outcome="fail", raw="0.001"),
        _make_record("att_3", block_id="blk_a", param_name="lr", normalized="0.0100000000", outcome="fail", raw="0.01"),
        _make_record("att_4", block_id="blk_a", param_name="eps", normalized="0.0000010000", outcome="fail", raw="1e-6"),
        _make_record("att_5", block_id="blk_b", param_name="lr", normalized="0.0010000000", outcome="fail", raw="1e-3"),
        _make_record("att_6", block_id="blk_a", param_name="lr", normalized="0.0010000000", outcome="success", raw="1e-3"),
        _make_record("att_7", block_id="blk_a", param_name="lr", normalized="0.1000000000", outcome="unknown", raw="0.1"),
    ]


def test_get_records_for_block_returns_only_matching_block_in_existing_order() -> None:
    records = _records_fixture()

    result = get_records_for_block(records, "blk_a")

    assert [record.attempt_id for record in result] == ["att_1", "att_2", "att_3", "att_4", "att_6", "att_7"]


def test_get_failed_records_for_block_filters_to_failures_only() -> None:
    records = _records_fixture()

    result = get_failed_records_for_block(records, "blk_a")

    assert [record.attempt_id for record in result] == ["att_1", "att_2", "att_3", "att_4"]


def test_get_recent_records_for_block_returns_recent_slice_in_file_order() -> None:
    records = _records_fixture()

    result = get_recent_records_for_block(records, "blk_a", limit=3)

    assert [record.attempt_id for record in result] == ["att_4", "att_6", "att_7"]


def test_get_recent_records_for_block_limit_none_returns_all_matches() -> None:
    records = _records_fixture()

    result = get_recent_records_for_block(records, "blk_a", limit=None)

    assert [record.attempt_id for record in result] == ["att_1", "att_2", "att_3", "att_4", "att_6", "att_7"]


def test_get_recent_records_for_block_limit_zero_returns_empty_list() -> None:
    records = _records_fixture()

    assert get_recent_records_for_block(records, "blk_a", limit=0) == []


def test_get_recent_records_for_block_rejects_negative_limit() -> None:
    records = _records_fixture()

    with pytest.raises(ValueError):
        get_recent_records_for_block(records, "blk_a", limit=-1)


def test_get_repeated_failed_values_detects_second_failed_repeat_only() -> None:
    records = _records_fixture()

    result = get_repeated_failed_values(records, "blk_a", "lr")

    assert result == ["0.0010000000"]


def test_get_repeated_failed_values_ignores_successes_other_blocks_and_other_params() -> None:
    records = _records_fixture()

    assert get_repeated_failed_values(records, "blk_a", "eps") == []
    assert get_repeated_failed_values(records, "blk_b", "lr") == []


def test_group_records_by_param_name_for_block_preserves_order_within_each_group() -> None:
    records = _records_fixture()

    grouped = group_records_by_param_name_for_block(records, "blk_a")

    assert list(grouped) == ["lr", "eps"]
    assert [record.attempt_id for record in grouped["lr"]] == ["att_1", "att_2", "att_3", "att_6", "att_7"]
    assert [record.attempt_id for record in grouped["eps"]] == ["att_4"]


def test_query_helpers_do_not_reorder_duplicate_records() -> None:
    duplicate = _make_record(
        "att_dup",
        block_id="blk_a",
        param_name="lr",
        normalized="0.0010000000",
        outcome="fail",
        raw="1e-3",
    )
    records = [duplicate, duplicate]

    assert get_records_for_block(records, "blk_a") == [duplicate, duplicate]
    assert get_failed_records_for_block(records, "blk_a") == [duplicate, duplicate]
