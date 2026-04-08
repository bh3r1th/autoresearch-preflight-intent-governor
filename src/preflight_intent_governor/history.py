"""Deterministic JSONL attempt history storage and query helpers for Phase 2.

This module provides append-only persistence and stable in-memory queries for
parameter-attempt records. It does not make any guard decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

ALLOWED_OUTCOMES: Final[frozenset[str]] = frozenset({"success", "fail", "unknown"})


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """A single append-only parameter attempt record.

    Input:
    - field values for one deterministic attempt record

    Output:
    - an immutable validated record object

    Failure behavior:
    - raises `ValueError` for invalid `outcome`, invalid `failure_tags`,
      or invalid `line_number`
    """

    attempt_id: str
    timestamp: str
    run_id: str
    file_path: str
    function_name: str
    block_id: str
    param_name: str
    param_value_raw: str
    param_value_normalized: str
    outcome: str
    failure_tags: list[str]
    line_number: int | None

    def __post_init__(self) -> None:
        _validate_outcome(self.outcome)
        _validate_failure_tags(self.failure_tags)
        _validate_line_number(self.line_number)
        object.__setattr__(self, "failure_tags", list(self.failure_tags))

    def to_dict(self) -> dict[str, object]:
        """Convert the record to a deterministic flat dictionary.

        Input:
        - none

        Output:
        - a JSON-serializable dictionary in dataclass field order

        Failure behavior:
        - does not raise for already-valid records
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AttemptRecord:
        """Create a validated record from a flat dictionary.

        Input:
        - `data`: a dictionary containing all required record fields

        Output:
        - a validated `AttemptRecord`

        Failure behavior:
        - raises `ValueError` for missing fields or invalid values
        """

        required_fields = {
            "attempt_id",
            "timestamp",
            "run_id",
            "file_path",
            "function_name",
            "block_id",
            "param_name",
            "param_value_raw",
            "param_value_normalized",
            "outcome",
            "failure_tags",
            "line_number",
        }
        missing = sorted(required_fields.difference(data))
        if missing:
            raise ValueError(f"missing record fields: {missing}")

        return cls(
            attempt_id=_require_string(data["attempt_id"], field_name="attempt_id"),
            timestamp=_require_string(data["timestamp"], field_name="timestamp"),
            run_id=_require_string(data["run_id"], field_name="run_id"),
            file_path=_require_string(data["file_path"], field_name="file_path"),
            function_name=_require_string(data["function_name"], field_name="function_name"),
            block_id=_require_string(data["block_id"], field_name="block_id"),
            param_name=_require_string(data["param_name"], field_name="param_name"),
            param_value_raw=_require_string(data["param_value_raw"], field_name="param_value_raw"),
            param_value_normalized=_require_string(
                data["param_value_normalized"],
                field_name="param_value_normalized",
            ),
            outcome=_require_string(data["outcome"], field_name="outcome"),
            failure_tags=_coerce_failure_tags(data["failure_tags"]),
            line_number=_coerce_line_number(data["line_number"]),
        )

    def to_json_line(self) -> str:
        """Serialize the record as one deterministic JSONL line.

        Input:
        - none

        Output:
        - one compact JSON string without a trailing newline

        Failure behavior:
        - does not raise for already-valid records
        """

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json_line(cls, line: str) -> AttemptRecord:
        """Parse one JSONL line into a validated record.

        Input:
        - `line`: a single JSON object line

        Output:
        - a validated `AttemptRecord`

        Failure behavior:
        - raises `ValueError` for malformed JSON or invalid record data
        """

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON line") from exc

        if not isinstance(payload, dict):
            raise ValueError("history line must decode to an object")

        return cls.from_dict(payload)


def make_attempt_id(
    *,
    run_id: str,
    file_path: str,
    function_name: str,
    block_id: str,
    param_name: str,
    param_value_normalized: str,
    outcome: str,
    line_number: int | None,
    failure_tags: list[str],
) -> str:
    """Create a deterministic attempt id from stable record fields.

    Input:
    - stable attempt fields excluding `timestamp` and raw param text

    Output:
    - a short deterministic id string

    Failure behavior:
    - raises `ValueError` for invalid `outcome`, `failure_tags`, or `line_number`

    Notes:
    - `failure_tags` are sorted before hashing so id generation is stable even if
      the caller provides the same tags in a different order
    """

    _validate_outcome(outcome)
    _validate_failure_tags(failure_tags)
    _validate_line_number(line_number)

    payload = {
        "block_id": block_id,
        "failure_tags": sorted(failure_tags),
        "file_path": file_path,
        "function_name": function_name,
        "line_number": line_number,
        "outcome": outcome,
        "param_name": param_name,
        "param_value_normalized": param_value_normalized,
        "run_id": run_id,
    }
    digest = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        digest_size=12,
    ).hexdigest()
    return f"att_{digest}"


def append_attempt_record(path: str | Path, record: AttemptRecord) -> None:
    """Append one validated record to a JSONL file.

    Input:
    - `path`: target history file path
    - `record`: a validated attempt record

    Output:
    - appends exactly one newline-terminated JSON object

    Failure behavior:
    - creates parent directories as needed
    - propagates filesystem errors from directory creation or append
    """

    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record.to_json_line())
        handle.write("\n")


def load_attempt_records(path: str | Path, strict: bool = False) -> list[AttemptRecord]:
    """Load attempt records from a JSONL file in file order.

    Input:
    - `path`: history file path
    - `strict`: when true, raise on the first malformed line

    Output:
    - a list of valid records in append/file order

    Failure behavior:
    - returns an empty list for a missing file
    - skips blank lines
    - skips malformed lines when `strict` is false
    - raises `ValueError` with line context when `strict` is true
    """

    history_path = Path(path)
    if not history_path.exists():
        return []

    records: list[AttemptRecord] = []
    with history_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                records.append(AttemptRecord.from_json_line(line))
            except ValueError as exc:
                if strict:
                    raise ValueError(f"invalid history line {line_number}: {exc}") from exc

    return records


def get_records_for_block(records: list[AttemptRecord], block_id: str) -> list[AttemptRecord]:
    """Return all records for one block in existing order.

    Input:
    - `records`: loaded records in file order
    - `block_id`: target block id

    Output:
    - matching records in the same order they appeared in `records`

    Failure behavior:
    - never raises for an empty list or no matches
    """

    return [record for record in records if record.block_id == block_id]


def get_recent_records_for_block(
    records: list[AttemptRecord],
    block_id: str,
    limit: int | None = None,
) -> list[AttemptRecord]:
    """Return the most recent records for one block.

    Input:
    - `records`: loaded records in file order
    - `block_id`: target block id
    - `limit`: maximum number of recent records to return

    Output:
    - the last `limit` matching records, preserving file order within the slice
    - all matching records when `limit` is `None`
    - an empty list when `limit` is `0`

    Failure behavior:
    - raises `ValueError` for negative limits
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")

    matching = get_records_for_block(records, block_id)
    if limit is None:
        return matching
    if limit == 0:
        return []
    return matching[-limit:]


def get_failed_records_for_block(records: list[AttemptRecord], block_id: str) -> list[AttemptRecord]:
    """Return failed records for one block in existing order.

    Input:
    - `records`: loaded records in file order
    - `block_id`: target block id

    Output:
    - matching failed records in file order

    Failure behavior:
    - never raises for an empty list or no matches
    """

    return [
        record
        for record in records
        if record.block_id == block_id and record.outcome == "fail"
    ]


def get_repeated_failed_values(
    records: list[AttemptRecord],
    block_id: str,
    param_name: str,
) -> list[str]:
    """Return repeated failed normalized values for one block and param.

    Input:
    - `records`: loaded records in file order
    - `block_id`: target block id
    - `param_name`: target canonical parameter name

    Output:
    - normalized values that failed more than once, in first-seen order

    Failure behavior:
    - never raises for an empty list or no matches
    """

    counts: dict[str, int] = {}
    repeated: list[str] = []

    for record in records:
        if record.block_id != block_id:
            continue
        if record.param_name != param_name:
            continue
        if record.outcome != "fail":
            continue

        value = record.param_value_normalized
        counts[value] = counts.get(value, 0) + 1
        if counts[value] == 2:
            repeated.append(value)

    return repeated


def group_records_by_param_name_for_block(
    records: list[AttemptRecord],
    block_id: str,
) -> dict[str, list[AttemptRecord]]:
    """Group one block's records by param name in append order.

    Input:
    - `records`: loaded records in file order
    - `block_id`: target block id

    Output:
    - a dictionary keyed by param name with each value preserving file order

    Failure behavior:
    - never raises for an empty list or no matches
    """

    grouped: dict[str, list[AttemptRecord]] = {}
    for record in records:
        if record.block_id != block_id:
            continue
        grouped.setdefault(record.param_name, []).append(record)
    return grouped


def _require_string(value: object, *, field_name: str) -> str:
    """Validate that a field value is a string."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _coerce_failure_tags(value: object) -> list[str]:
    """Validate `failure_tags` as a list of strings."""

    if not isinstance(value, list):
        raise ValueError("failure_tags must be a list[str]")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("failure_tags must be a list[str]")
    return list(value)


def _validate_failure_tags(value: list[str]) -> None:
    """Validate `failure_tags`."""

    _coerce_failure_tags(value)


def _validate_outcome(value: str) -> None:
    """Validate `outcome` against locked allowed values."""

    if value not in ALLOWED_OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(ALLOWED_OUTCOMES)}")


def _coerce_line_number(value: object) -> int | None:
    """Validate `line_number` as an int or None."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("line_number must be an int or None")
    return value


def _validate_line_number(value: int | None) -> None:
    """Validate `line_number`."""

    _coerce_line_number(value)


__all__ = [
    "ALLOWED_OUTCOMES",
    "AttemptRecord",
    "append_attempt_record",
    "get_failed_records_for_block",
    "get_recent_records_for_block",
    "get_records_for_block",
    "get_repeated_failed_values",
    "group_records_by_param_name_for_block",
    "load_attempt_records",
    "make_attempt_id",
]
