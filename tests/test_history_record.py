import pytest

from preflight_intent_governor.history import AttemptRecord, make_attempt_id


def _make_record(**overrides: object) -> AttemptRecord:
    payload: dict[str, object] = {
        "attempt_id": "att_fixed",
        "timestamp": "2026-04-07T12:00:00Z",
        "run_id": "run-001",
        "file_path": "src/model.py",
        "function_name": "train_step",
        "block_id": "blk_abc123",
        "param_name": "lr",
        "param_value_raw": "1e-3",
        "param_value_normalized": "0.0010000000",
        "outcome": "fail",
        "failure_tags": ["oom"],
        "line_number": 42,
    }
    payload.update(overrides)
    return AttemptRecord(**payload)


def test_attempt_record_accepts_valid_values() -> None:
    record = _make_record()

    assert record.attempt_id == "att_fixed"
    assert record.outcome == "fail"
    assert record.failure_tags == ["oom"]
    assert record.line_number == 42


def test_attempt_record_rejects_invalid_outcome() -> None:
    with pytest.raises(ValueError):
        _make_record(outcome="partial")


@pytest.mark.parametrize(
    "failure_tags",
    [
        "oom",
        None,
        ["oom", 123],
    ],
)
def test_attempt_record_rejects_invalid_failure_tags(failure_tags: object) -> None:
    with pytest.raises(ValueError):
        _make_record(failure_tags=failure_tags)


def test_attempt_record_allows_null_line_number() -> None:
    record = _make_record(line_number=None)

    assert record.line_number is None


def test_attempt_record_allows_empty_failure_tags() -> None:
    record = _make_record(failure_tags=[])

    assert record.failure_tags == []


def test_attempt_record_copies_failure_tags_to_prevent_external_mutation() -> None:
    failure_tags = ["oom"]

    record = _make_record(failure_tags=failure_tags)
    failure_tags.append("timeout")

    assert record.failure_tags == ["oom"]


def test_make_attempt_id_is_deterministic_for_same_inputs() -> None:
    first = make_attempt_id(
        run_id="run-001",
        file_path="src/model.py",
        function_name="train_step",
        block_id="blk_abc123",
        param_name="lr",
        param_value_normalized="0.0010000000",
        outcome="fail",
        line_number=42,
        failure_tags=["oom"],
    )
    second = make_attempt_id(
        run_id="run-001",
        file_path="src/model.py",
        function_name="train_step",
        block_id="blk_abc123",
        param_name="lr",
        param_value_normalized="0.0010000000",
        outcome="fail",
        line_number=42,
        failure_tags=["oom"],
    )

    assert first == second
    assert first.startswith("att_")


def test_make_attempt_id_is_stable_when_failure_tag_order_differs() -> None:
    first = make_attempt_id(
        run_id="run-001",
        file_path="src/model.py",
        function_name="train_step",
        block_id="blk_abc123",
        param_name="lr",
        param_value_normalized="0.0010000000",
        outcome="fail",
        line_number=42,
        failure_tags=["oom", "timeout"],
    )
    second = make_attempt_id(
        run_id="run-001",
        file_path="src/model.py",
        function_name="train_step",
        block_id="blk_abc123",
        param_name="lr",
        param_value_normalized="0.0010000000",
        outcome="fail",
        line_number=42,
        failure_tags=["timeout", "oom"],
    )

    assert first == second


def test_make_attempt_id_changes_when_stable_inputs_change() -> None:
    base = make_attempt_id(
        run_id="run-001",
        file_path="src/model.py",
        function_name="train_step",
        block_id="blk_abc123",
        param_name="lr",
        param_value_normalized="0.0010000000",
        outcome="fail",
        line_number=42,
        failure_tags=["oom"],
    )
    changed = make_attempt_id(
        run_id="run-001",
        file_path="src/model.py",
        function_name="train_step",
        block_id="blk_abc123",
        param_name="lr",
        param_value_normalized="0.0100000000",
        outcome="fail",
        line_number=42,
        failure_tags=["oom"],
    )

    assert base != changed


def test_attempt_record_dict_roundtrip_preserves_all_fields() -> None:
    record = _make_record(line_number=None, failure_tags=[])

    restored = AttemptRecord.from_dict(record.to_dict())

    assert restored == record


def test_attempt_record_json_line_roundtrip_preserves_all_fields() -> None:
    record = _make_record()

    restored = AttemptRecord.from_json_line(record.to_json_line())

    assert restored == record


def test_attempt_record_json_line_is_deterministic_and_compact() -> None:
    record = _make_record()

    json_line = record.to_json_line()

    assert "\n" not in json_line
    assert '"attempt_id":"att_fixed"' in json_line
    assert '"outcome":"fail"' in json_line
    assert '"failure_tags":["oom"]' in json_line


def test_attempt_record_from_dict_rejects_missing_fields() -> None:
    data = _make_record().to_dict()
    del data["block_id"]

    with pytest.raises(ValueError):
        AttemptRecord.from_dict(data)


def test_attempt_record_from_json_line_rejects_non_object_payload() -> None:
    with pytest.raises(ValueError):
        AttemptRecord.from_json_line('["not", "an", "object"]')
