from preflight_intent_governor.normalize import extract_supported_params


def test_extract_supported_params_from_direct_assignments_maps_to_canonical_names() -> None:
    code = """
lr = 1e-3
learning_rate = 0.0010
momentum = 0.9
""".strip()

    matches = extract_supported_params(code)

    assert [match.canonical_param for match in matches] == ["lr", "lr"]
    assert [match.raw_param for match in matches] == ["lr", "learning_rate"]
    assert [match.raw_value for match in matches] == ["1e-3", "0.0010"]
    assert [match.normalized_value for match in matches] == [
        "0.0010000000",
        "0.0010000000",
    ]


def test_extract_supported_params_supports_spaces_around_assignment() -> None:
    code = " learning_rate   =   1E-03 "

    matches = extract_supported_params(code)

    assert len(matches) == 1
    assert matches[0].canonical_param == "lr"
    assert matches[0].normalized_value == "0.0010000000"


def test_extract_supported_params_from_mixed_assignments_and_kwargs_preserves_source_order() -> None:
    code = """
lr = 1e-3
optimizer = Adam(params, lr=0.0010, eps=1e-6)
weight_decay = 0.01
""".strip()

    matches = extract_supported_params(code)

    assert [match.canonical_param for match in matches] == [
        "lr",
        "lr",
        "eps",
        "weight_decay",
    ]
    assert [match.raw_value for match in matches] == ["1e-3", "0.0010", "1e-6", "0.01"]


def test_extract_supported_params_supports_multiple_supported_params_on_one_line() -> None:
    code = "optimizer = AdamW(params, adamw_eps=1e-8, weight_decay=0.01, lr=1e-3)"

    matches = extract_supported_params(code)

    assert [match.canonical_param for match in matches] == ["eps", "weight_decay", "lr"]
    assert [match.normalized_value for match in matches] == [
        "0.0000000100",
        "0.0100000000",
        "0.0010000000",
    ]


def test_extract_supported_params_repeated_occurrences_preserve_source_order() -> None:
    code = """
lr = 1e-3
learning_rate = 0.0010
optimizer = Adam(params, lr=1E-03)
""".strip()

    matches = extract_supported_params(code)

    assert [match.canonical_param for match in matches] == ["lr", "lr", "lr"]
    assert [match.raw_param for match in matches] == ["lr", "learning_rate", "lr"]


def test_extract_supported_params_ignores_unknown_similar_names() -> None:
    code = """
lr_value = 1e-3
epsilon = 1e-6
my_weight_decay = 0.1
""".strip()

    assert extract_supported_params(code) == []


def test_extract_supported_params_ignores_comment_only_matches() -> None:
    code = """
# lr=1e-3
weight_decay = 0.01  # eps=1e-8 should not be extracted from the comment
""".strip()

    matches = extract_supported_params(code)

    assert len(matches) == 1
    assert matches[0].canonical_param == "weight_decay"
    assert matches[0].normalized_value == "0.0100000000"


def test_extract_supported_params_ignores_string_literal_matches() -> None:
    code = '''
message = "lr=1e-3 eps=1e-6"
weight_decay = 0.01
'''.strip()

    matches = extract_supported_params(code)

    assert len(matches) == 1
    assert matches[0].canonical_param == "weight_decay"
    assert matches[0].normalized_value == "0.0100000000"


def test_extract_supported_params_line_and_column_metadata_is_stable_for_debugging() -> None:
    code = """
alpha = 1
  lr = 1e-3
optimizer = Adam(params, eps=1e-6)
""".strip()

    matches = extract_supported_params(code)

    assert [match.line_number for match in matches] == [2, 3]
    assert [match.column_start for match in matches] == [3, 26]
    assert [match.column_end for match in matches] == [11, 33]
    assert [match.span_start for match in matches] == [12, 47]
    assert [match.span_end for match in matches] == [21, 55]
