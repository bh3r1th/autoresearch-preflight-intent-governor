from preflight_intent_governor.normalize import (
    extract_function_signature_info,
    make_block_id,
    make_fallback_block_id,
)


def test_make_block_id_is_stable_under_variable_rename() -> None:
    original = """
def train_step(batch, scale):
    if batch:
        value = scale + 1
    return batch
""".strip()
    renamed = """
def train_step(data, scale):
    if data:
        result = scale + 1
    return data
""".strip()

    assert make_block_id("src/model.py", original, start_line=10) == make_block_id(
        "src/model.py",
        renamed,
        start_line=10,
    )


def test_make_block_id_is_stable_under_comment_only_edits() -> None:
    base = """
def forward(x, y):
    if x:
        y = y + 1
    return y
""".strip()
    edited = """
def forward(x, y):
    # comment-only edit
    if x:
        y = y + 1
    return y
""".strip()

    assert make_block_id("pkg/module.py", base, start_line=20) == make_block_id(
        "pkg/module.py",
        edited,
        start_line=20,
    )


def test_make_block_id_is_stable_under_whitespace_only_edits() -> None:
    base = """
def forward(x, y):
    if x:
        y = y + 1
    return y
""".strip()
    edited = """
def forward(x, y):

    if x:
        y = y + 1

    return y
""".strip()

    assert make_block_id("pkg/module.py", base, start_line=20) == make_block_id(
        "pkg/module.py",
        edited,
        start_line=20,
    )


def test_make_block_id_changes_when_top_level_structure_changes() -> None:
    original = """
def forward(x):
    if x:
        x -= 1
    return x
""".strip()
    changed = """
def forward(x):
    if x:
        x -= 1
    while x:
        x -= 1
    return x
""".strip()

    assert make_block_id("pkg/module.py", original, start_line=5) != make_block_id(
        "pkg/module.py",
        changed,
        start_line=5,
    )


def test_make_block_id_changes_when_argument_count_changes() -> None:
    original = """
def forward(x):
    return x
""".strip()
    changed = """
def forward(x, y):
    return x
""".strip()

    assert make_block_id("pkg/module.py", original, start_line=5) != make_block_id(
        "pkg/module.py",
        changed,
        start_line=5,
    )


def test_extract_function_signature_info_counts_args_with_string_default_containing_comma() -> None:
    function_source = '''
def forward(self, label="x,y", scale=1):
    return scale
'''.strip()

    info = extract_function_signature_info(function_source)

    assert info.parsed is True
    assert info.function_name == "forward"
    assert info.argument_count == 3


def test_make_block_id_falls_back_when_function_signature_cannot_be_parsed() -> None:
    incomplete = "if x:\n    return x"

    block_id = make_block_id("pkg/module.py", incomplete, start_line=27)
    fallback = make_fallback_block_id("pkg/module.py", 27)

    assert block_id == fallback
    assert block_id.startswith("fbk_")


def test_make_fallback_block_id_is_deterministic_across_repeated_calls() -> None:
    first = make_fallback_block_id("pkg/module.py", 27)
    second = make_fallback_block_id("pkg/module.py", 27)

    assert first == second


def test_make_fallback_block_id_uses_deterministic_line_buckets() -> None:
    first = make_fallback_block_id("pkg/module.py", 21)
    same_bucket = make_fallback_block_id("pkg/module.py", 29)
    different_bucket = make_fallback_block_id("pkg/module.py", 31)

    assert first == same_bucket
    assert first != different_bucket
