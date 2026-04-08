PYTEST ?= pytest
PHASE1_TESTS = tests/test_normalize_numeric.py tests/test_normalize_params.py tests/test_normalize_shape.py tests/test_normalize_block_id.py

.PHONY: test test-phase1

test:
	$(PYTEST) -q

test-phase1:
	$(PYTEST) -q $(PHASE1_TESTS)
