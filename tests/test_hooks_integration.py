from pathlib import Path

from preflight_intent_governor.heuristics import ExecutionResult
from preflight_intent_governor.history import AttemptRecord, load_attempt_records
from preflight_intent_governor.hooks import (
    record_execution_result,
    run_preflight_check,
)
from preflight_intent_governor.normalize import make_block_id


def _function_source() -> str:
    return """
def train_step(batch):
    return batch
""".strip()


def _history_record(
    *,
    block_id: str,
    param_name: str,
    normalized_value: str,
    outcome: str,
    attempt_id: str = "att_seed",
) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=attempt_id,
        timestamp="2026-04-07T12:00:00Z",
        run_id="run-seed",
        file_path="src/model.py",
        function_name="train_step",
        block_id=block_id,
        param_name=param_name,
        param_value_raw=normalized_value,
        param_value_normalized=normalized_value,
        outcome=outcome,
        failure_tags=[] if outcome != "fail" else ["nan_loss"],
        line_number=1,
    )


def test_run_preflight_check_blocks_repeated_failed_value() -> None:
    function_source = _function_source()
    block_id = make_block_id("src/model.py", function_source, start_line=1)
    history = [
        _history_record(
            block_id=block_id,
            param_name="lr",
            normalized_value="0.0010000000",
            outcome="fail",
        )
    ]

    decision = run_preflight_check(
        file_path="src/model.py",
        function_source=function_source,
        proposed_code="optimizer = Adam(lr=1e-3)",
        attempt_records=history,
        start_line=1,
    )

    assert decision.allowed is False
    assert decision.reason_code == "block_repeated_failed_value"


def test_run_preflight_check_allows_new_value_direction() -> None:
    function_source = _function_source()
    block_id = make_block_id("src/model.py", function_source, start_line=1)
    history = [
        _history_record(
            block_id=block_id,
            param_name="lr",
            normalized_value="0.0010000000",
            outcome="fail",
        )
    ]

    decision = run_preflight_check(
        file_path="src/model.py",
        function_source=function_source,
        proposed_code="optimizer = Adam(lr=5e-4)",
        attempt_records=history,
        start_line=1,
    )

    assert decision.allowed is True
    assert decision.reason_code == "allow_new_direction"


def test_record_execution_result_writes_expected_failure_record(tmp_path: Path) -> None:
    history_path = tmp_path / "attempts.jsonl"
    function_source = _function_source()

    summary = record_execution_result(
        file_path="src/model.py",
        function_source=function_source,
        proposed_code="optimizer = Adam(lr=1e-3)",
        execution_result=ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="loss became nan",
            exception=None,
            duration_seconds=3.2,
        ),
        outcome="fail",
        run_id="run-001",
        history_file_path=history_path,
        timestamp="2026-04-07T12:00:00Z",
        start_line=1,
    )

    loaded = load_attempt_records(history_path)

    assert summary.recorded_count == 1
    assert len(loaded) == 1
    assert loaded[0].block_id == summary.block_id
    assert loaded[0].param_name == "lr"
    assert loaded[0].param_value_normalized == "0.0010000000"
    assert loaded[0].failure_tags == ["nan_loss"]


def test_record_execution_result_appends_one_record_per_param(tmp_path: Path) -> None:
    history_path = tmp_path / "attempts.jsonl"

    summary = record_execution_result(
        file_path="src/model.py",
        function_source=_function_source(),
        proposed_code="optimizer = Adam(lr=1e-3, eps=1e-6)",
        execution_result=ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="loss became nan",
            exception=None,
            duration_seconds=2.0,
        ),
        outcome="fail",
        run_id="run-002",
        history_file_path=history_path,
        timestamp="2026-04-07T12:00:01Z",
        start_line=1,
    )

    loaded = load_attempt_records(history_path)

    assert summary.recorded_count == 2
    assert [record.param_name for record in loaded] == ["lr", "eps"]


def test_full_flow_allows_then_records_then_blocks_repeat(tmp_path: Path) -> None:
    history_path = tmp_path / "attempts.jsonl"
    function_source = _function_source()
    proposed_code = "optimizer = Adam(lr=1e-3)"

    first_decision = run_preflight_check(
        file_path="src/model.py",
        function_source=function_source,
        proposed_code=proposed_code,
        attempt_records=[],
        start_line=1,
    )
    record_execution_result(
        file_path="src/model.py",
        function_source=function_source,
        proposed_code=proposed_code,
        execution_result=ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="loss became nan",
            exception=None,
            duration_seconds=4.0,
        ),
        outcome="fail",
        run_id="run-003",
        history_file_path=history_path,
        timestamp="2026-04-07T12:00:02Z",
        start_line=1,
    )

    second_decision = run_preflight_check(
        file_path="src/model.py",
        function_source=function_source,
        proposed_code=proposed_code,
        attempt_records=load_attempt_records(history_path),
        start_line=1,
    )

    assert first_decision.allowed is True
    assert first_decision.reason_code == "allow_no_conflict"
    assert second_decision.allowed is False
    assert second_decision.reason_code == "block_repeated_failed_value"


def test_no_param_case_allows_preflight_and_skips_recording(tmp_path: Path) -> None:
    history_path = tmp_path / "attempts.jsonl"
    function_source = _function_source()
    proposed_code = "optimizer = Adam(beta1=0.9)"

    decision = run_preflight_check(
        file_path="src/model.py",
        function_source=function_source,
        proposed_code=proposed_code,
        attempt_records=[],
        start_line=1,
    )
    summary = record_execution_result(
        file_path="src/model.py",
        function_source=function_source,
        proposed_code=proposed_code,
        execution_result=ExecutionResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            exception=None,
            duration_seconds=1.0,
        ),
        outcome="success",
        run_id="run-004",
        history_file_path=history_path,
        timestamp="2026-04-07T12:00:03Z",
        start_line=1,
    )

    assert decision.allowed is True
    assert decision.reason_code == "allow_no_conflict"
    assert summary.recorded_count == 0
    assert summary.records == []
    assert load_attempt_records(history_path) == []


def test_repeated_runs_produce_same_preflight_decision_and_stable_history_order(tmp_path: Path) -> None:
    history_path = tmp_path / "attempts.jsonl"
    function_source = _function_source()
    proposed_code = "optimizer = Adam(lr=1e-3, eps=1e-6)"

    first_summary = record_execution_result(
        file_path="src/model.py",
        function_source=function_source,
        proposed_code=proposed_code,
        execution_result=ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="loss became nan",
            exception=None,
            duration_seconds=1.0,
        ),
        outcome="fail",
        run_id="run-005",
        history_file_path=history_path,
        timestamp="2026-04-07T12:00:04Z",
        start_line=1,
    )
    second_summary = record_execution_result(
        file_path="src/model.py",
        function_source=function_source,
        proposed_code=proposed_code,
        execution_result=ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="loss became nan",
            exception=None,
            duration_seconds=1.0,
        ),
        outcome="fail",
        run_id="run-005",
        history_file_path=history_path,
        timestamp="2026-04-07T12:00:04Z",
        start_line=1,
    )

    loaded = load_attempt_records(history_path)
    decision_one = run_preflight_check(
        file_path="src/model.py",
        function_source=function_source,
        proposed_code="optimizer = Adam(lr=1e-3)",
        attempt_records=loaded,
        start_line=1,
    )
    decision_two = run_preflight_check(
        file_path="src/model.py",
        function_source=function_source,
        proposed_code="optimizer = Adam(lr=1e-3)",
        attempt_records=loaded,
        start_line=1,
    )

    assert first_summary.block_id == second_summary.block_id
    assert [record.param_name for record in loaded] == ["lr", "eps", "lr", "eps"]
    assert decision_one.allowed is False
    assert decision_one.reason_code == "block_repeated_failed_value"
    assert decision_two.allowed is False
    assert decision_two.reason_code == decision_one.reason_code
