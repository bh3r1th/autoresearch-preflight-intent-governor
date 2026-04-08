from __future__ import annotations

from pathlib import Path
import shutil

from preflight_intent_governor.heuristics import ExecutionResult
from preflight_intent_governor.history import load_attempt_records
from preflight_intent_governor.hooks import record_execution_result, run_preflight_check


def main() -> None:
    file_path = "demo/train.py"
    function_source = """
def train_step(batch):
    return batch
""".strip()
    proposed_code = "optimizer = Adam(lr=1e-3)"

    history_path = Path("demo_tmp") / "preflight_history.jsonl"
    _reset_demo_history(history_path)

    first_history = load_attempt_records(history_path)
    first_decision = run_preflight_check(
        file_path=file_path,
        function_source=function_source,
        proposed_code=proposed_code,
        attempt_records=first_history,
        start_line=1,
    )

    summary = record_execution_result(
        file_path=file_path,
        function_source=function_source,
        proposed_code=proposed_code,
        execution_result=ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="loss became nan",
            exception=None,
            duration_seconds=3.0,
        ),
        outcome="fail",
        run_id="demo-run-001",
        history_file_path=history_path,
        timestamp="2026-04-07T12:00:00Z",
        start_line=1,
    )

    second_history = load_attempt_records(history_path)
    second_decision = run_preflight_check(
        file_path=file_path,
        function_source=function_source,
        proposed_code=proposed_code,
        attempt_records=second_history,
        start_line=1,
    )

    print("First preflight decision:")
    print(f"  allowed={first_decision.allowed}")
    print(f"  reason_code={first_decision.reason_code}")
    print()
    print("Recorded execution:")
    print(f"  block_id={summary.block_id}")
    print(f"  history_write_count={summary.recorded_count}")
    print(f"  failure_tags={summary.records[0].failure_tags if summary.records else []}")
    print()
    print("Second preflight decision:")
    print(f"  allowed={second_decision.allowed}")
    print(f"  reason_code={second_decision.reason_code}")
    print(f"  matched_attempt_ids={second_decision.matched_attempt_ids}")


def _reset_demo_history(history_path: Path) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        history_path.unlink()


if __name__ == "__main__":
    main()
