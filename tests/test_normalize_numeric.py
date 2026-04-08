import pytest

from preflight_intent_governor.normalize import normalize_numeric_value


def test_normalize_numeric_value_equivalent_scientific_and_decimal_inputs_match() -> None:
    expected = "0.0010000000"

    assert normalize_numeric_value("1e-3") == expected
    assert normalize_numeric_value("1E-03") == expected
    assert normalize_numeric_value("0.001") == expected
    assert normalize_numeric_value("0.0010") == expected


def test_normalize_numeric_value_strips_outer_whitespace() -> None:
    assert normalize_numeric_value("  1e-3  ") == "0.0010000000"


def test_normalize_numeric_value_preserves_supported_signs() -> None:
    assert normalize_numeric_value("+1e-3") == "0.0010000000"
    assert normalize_numeric_value("-1e-3") == "-0.0010000000"


def test_normalize_numeric_value_zero_like_representations_collapse_to_canonical_zero() -> None:
    assert normalize_numeric_value("0") == "0.0000000000"
    assert normalize_numeric_value("-0") == "0.0000000000"
    assert normalize_numeric_value("0e0") == "0.0000000000"


def test_normalize_numeric_value_has_exact_canonical_width() -> None:
    result = normalize_numeric_value("2")

    assert result == "2.0000000000"
    assert "." in result
    whole, fraction = result.split(".")
    assert whole == "2"
    assert len(fraction) == 10


@pytest.mark.parametrize("raw", ["", "abc", "1.2.3", "1e", "--1"])
def test_normalize_numeric_value_rejects_invalid_numeric_strings(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_numeric_value(raw)
