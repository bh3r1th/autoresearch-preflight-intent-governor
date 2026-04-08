from preflight_intent_governor.heuristics import ExecutionResult, extract_failure_tags


def _result(
    *,
    exit_code: int | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    exception: str | None = None,
    duration_seconds: float | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        exception=exception,
        duration_seconds=duration_seconds,
    )


def test_extract_failure_tags_detects_nan_loss() -> None:
    result = _result(stderr="loss became nan")

    assert extract_failure_tags(result) == ["nan_loss"]


def test_extract_failure_tags_detects_inf_loss() -> None:
    result = _result(stderr="loss = inf")

    assert extract_failure_tags(result) == ["inf_loss"]


def test_extract_failure_tags_detects_oom() -> None:
    result = _result(stderr="CUDA out of memory")

    assert extract_failure_tags(result) == ["oom"]


def test_extract_failure_tags_detects_timeout_from_exit_code() -> None:
    result = _result(exit_code=124)

    assert extract_failure_tags(result) == ["timeout"]


def test_extract_failure_tags_detects_syntax_error() -> None:
    result = _result(stderr="SyntaxError: invalid syntax")

    assert extract_failure_tags(result) == ["syntax_error"]


def test_extract_failure_tags_detects_generic_runtime_error_from_exception() -> None:
    result = _result(exception="RuntimeError: training loop failed")

    assert extract_failure_tags(result) == ["runtime_error"]


def test_extract_failure_tags_detects_divergence() -> None:
    result = _result(stdout="loss exploded after step 20")

    assert extract_failure_tags(result) == ["divergence"]


def test_extract_failure_tags_detects_no_improvement() -> None:
    result = _result(stdout="no improvement in validation loss")

    assert extract_failure_tags(result) == ["no_improvement"]


def test_extract_failure_tags_supports_multiple_specific_tags() -> None:
    result = _result(stderr="loss became nan and then CUDA out of memory")

    assert extract_failure_tags(result) == ["nan_loss", "oom"]


def test_extract_failure_tags_keeps_syntax_error_and_suppresses_runtime_error_fallback() -> None:
    result = _result(
        stderr="SyntaxError: invalid syntax",
        exception="RuntimeError: wrapper failed",
    )

    assert extract_failure_tags(result) == ["syntax_error"]


def test_extract_failure_tags_uses_runtime_error_only_when_no_specific_match_exists() -> None:
    result = _result(exception="generic failure in worker")

    assert extract_failure_tags(result) == ["runtime_error"]


def test_extract_failure_tags_does_not_add_runtime_error_when_specific_tag_exists() -> None:
    result = _result(
        stderr="RuntimeError: CUDA out of memory",
        exception="RuntimeError: CUDA out of memory",
    )

    assert extract_failure_tags(result) == ["oom"]


def test_extract_failure_tags_is_deterministic_and_deduplicated() -> None:
    result = _result(
        stdout="loss became nan then nan again and loss = inf",
        stderr="timeout timeout",
    )

    first = extract_failure_tags(result)
    second = extract_failure_tags(result)

    assert first == ["nan_loss", "inf_loss", "timeout"]
    assert second == first


def test_extract_failure_tags_returns_empty_list_when_no_supported_signal_exists() -> None:
    result = _result(stdout="training finished normally", stderr="all good")

    assert extract_failure_tags(result) == []
