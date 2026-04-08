# preflight-intent-governor

`preflight-intent-governor` is a minimal deterministic guard layer that prevents repeated failed
parameter configurations from executing.

It operates in two phases:
1) **Preflight check (before execution)**
2) **Post-execution recording (after execution)**

The system extracts normalized parameter values, records outcomes, and blocks future runs that
repeat previously failed values.

---

## Core Behavior

- Deterministic (no ML, no heuristics)
- Parameter-level tracking (not run-level)
- JSONL history (append-only)
- Pre-execution enforcement (fail fast, no wasted compute)

---

## What it does

On each run:
1. Extract parameters → normalize → compute `block_id`
2. Check history for conflicts
3. Decide:
   - `allow_no_conflict`
   - `block_repeated_failed_value`
   - `allow_new_direction`
4. After execution:
   - record outcome (`success` / `failure`)
   - attach failure tags

---

## Example (validated)
Run 1: eps = -1e-10 → FAIL → recorded
Run 2: eps = -1e-10 → BLOCKED (preflight)
Run 3: eps = 1e-10 → ALLOWED → SUCCESS


This demonstrates:
- memory of failure
- deterministic blocking
- safe exploration of new values

---

## Package Structure

- `normalize.py` → parameter normalization + block_id
- `history.py` → JSONL storage + retrieval
- `guard.py` → decision logic
- `heuristics.py` → failure tagging
- `hooks.py` → integration entry points

---

## Usage (integration)

```python
from preflight_intent_governor.hooks import run_preflight_check, record_execution_result

decision = run_preflight_check(...)
if not decision.allowed:
    exit(1)

# run workload

record_execution_result(...)
```

## Environment variables
AUTORESEARCH_GOVERNOR_ENABLED=1
AUTORESEARCH_GOVERNOR_HISTORY_PATH=./governor.jsonl
AUTORESEARCH_GOVERNOR_RUN_ID=<run_id>

## Verification
pytest -q

## Run normalization tests only
pytest -q tests/test_normalize_numeric.py tests/test_normalize_params.py tests/test_normalize_shape.py tests/test_normalize_block_id.py

## Scope
No retries
No auto-tuning
No LLM involvement
No probabilistic decisions

# This is a deterministic execution guard, not an evaluation or optimization system.
