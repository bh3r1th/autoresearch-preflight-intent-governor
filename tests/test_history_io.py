from pathlib import Path

import pytest

from preflight_intent_governor.history import (
    AttemptRecord,
    append_attempt_record,
    load_attempt_records,
)


def _make_record(index: int, **overrides: object) -> AttemptRecord:
    payload: dict[str, object] = {
        "attempt_id": f"att_{index}",
        "timestamp": f"2026-04-07T12:00:{index:02d}Z",
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


def test_append_attempt_record_creates_file_if_missing(tmp_path: Path) -> None:
    path = tmp_path / "history" / "attempts.jsonl"

    append_attempt_record(path, _make_record(1))

    assert path.exists()
    loaded = load_attempt_records(path)
    assert loaded == [_make_record(1)]


def test_append_attempt_record_preserves_prior_records(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    first = _make_record(1)
    second = _make_record(2, attempt_id="att_2", param_value_raw="0.01", param_value_normalized="0.0100000000")

    append_attempt_record(path, first)
    append_attempt_record(path, second)

    assert load_attempt_records(path) == [first, second]


def test_load_attempt_records_preserves_file_order(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    records = [
        _make_record(1, param_name="lr"),
        _make_record(2, param_name="eps", param_value_raw="1e-6", param_value_normalized="0.0000010000"),
        _make_record(3, param_name="weight_decay", param_value_raw="0.01", param_value_normalized="0.0100000000"),
    ]

    for record in records:
        append_attempt_record(path, record)

    loaded = load_attempt_records(path)

    assert loaded == records


def test_load_attempt_records_missing_file_returns_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"

    assert load_attempt_records(path) == []


def test_load_attempt_records_empty_file_returns_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    path.write_text("", encoding="utf-8")

    assert load_attempt_records(path) == []


def test_load_attempt_records_skips_malformed_lines_when_not_strict(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    valid = _make_record(1)
    path.write_text(
        valid.to_json_line() + "\n" + "{bad json}\n" + valid.to_json_line() + "\n",
        encoding="utf-8",
    )

    loaded = load_attempt_records(path, strict=False)

    assert loaded == [valid, valid]


def test_load_attempt_records_raises_on_malformed_lines_when_strict(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid history line 1"):
        load_attempt_records(path, strict=True)


def test_repeated_append_and_load_cycles_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    first = _make_record(1)
    second = _make_record(2, attempt_id="att_2", outcome="success", failure_tags=[])

    append_attempt_record(path, first)
    after_first = load_attempt_records(path)
    append_attempt_record(path, second)
    after_second = load_attempt_records(path)

    assert after_first == [first]
    assert after_second == [first, second]


def test_duplicate_records_remain_visible_in_append_only_history(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    record = _make_record(1)

    append_attempt_record(path, record)
    append_attempt_record(path, record)

    assert load_attempt_records(path) == [record, record]
