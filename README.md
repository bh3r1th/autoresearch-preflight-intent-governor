# preflight-intent-governor

`preflight-intent-governor` is a minimal proof of concept for a deterministic Python guard layer.
The project is intended to normalize a small set of intent-adjacent tokens before any future guard
logic is evaluated. The design is deliberately simple, stdlib-only, and easy to inspect.

Phase 1 is limited to normalization utilities. It establishes the package layout, test harness, and
deterministic string normalization rules needed for numeric tokens, parameter names, shape names,
and block identifiers.

`normalize.py` owns the Phase 1 behavior. It provides the only implemented runtime logic in this
phase and is responsible for stable, explicit normalization transforms.

Out of scope for Phase 1 are guard decisions, history tracking, hooks, heuristics, fuzzy matching,
ML-assisted behavior, and any parser-heavy or AST-heavy analysis.

## Verification

```bash
pytest -q
```

Run only the Phase 1 normalization suite:

```bash
pytest -q tests/test_normalize_numeric.py tests/test_normalize_params.py tests/test_normalize_shape.py tests/test_normalize_block_id.py
```

If `make` is available locally, the same checks are available as:

```bash
make test
make test-phase1
```
