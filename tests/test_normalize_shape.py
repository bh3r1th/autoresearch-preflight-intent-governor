from preflight_intent_governor.normalize import compute_top_level_control_flow_shape


def test_compute_top_level_control_flow_shape_counts_only_top_level_statements() -> None:
    function_source = """
def train_step(x, y):
    if x:
        if y:
            return y
    for item in y:
        while item:
            return item
    while x:
        x -= 1
    return x
""".strip()

    shape = compute_top_level_control_flow_shape(function_source)

    assert shape.if_count == 1
    assert shape.for_count == 1
    assert shape.while_count == 1
    assert shape.return_count == 1


def test_compute_top_level_control_flow_shape_ignores_decorators_comments_and_blank_lines() -> None:
    function_source = """
@decorator_one
@decorator_two(flag=True)
def run(value):
    # if this comment says return it should not count

    if value:
        value -= 1

    return value
""".strip()

    shape = compute_top_level_control_flow_shape(function_source)

    assert shape.if_count == 1
    assert shape.for_count == 0
    assert shape.while_count == 0
    assert shape.return_count == 1


def test_compute_top_level_control_flow_shape_ignores_nested_return_inside_top_level_if() -> None:
    function_source = """
def select_value(flag, value):
    if flag:
        return value
    return None
""".strip()

    shape = compute_top_level_control_flow_shape(function_source)

    assert shape.if_count == 1
    assert shape.for_count == 0
    assert shape.while_count == 0
    assert shape.return_count == 1


def test_compute_top_level_control_flow_shape_respects_method_indentation_and_nested_blocks() -> None:
    function_source = """
@classmethod
def build(self, items, enabled=True):
    if enabled:
        for item in items:
            if item:
                return item
    while False:
        return None
    return self
""".strip()

    shape = compute_top_level_control_flow_shape(function_source)

    assert shape.if_count == 1
    assert shape.for_count == 0
    assert shape.while_count == 1
    assert shape.return_count == 1


def test_compute_top_level_control_flow_shape_is_explicit_for_near_empty_functions() -> None:
    function_source = """
def noop():
    pass
""".strip()

    shape = compute_top_level_control_flow_shape(function_source)

    assert shape.if_count == 0
    assert shape.for_count == 0
    assert shape.while_count == 0
    assert shape.return_count == 0


def test_compute_top_level_control_flow_shape_ignores_docstring_only_line() -> None:
    function_source = '''
def describe():
    """return if for while"""
    return "done"
'''.strip()

    shape = compute_top_level_control_flow_shape(function_source)

    assert shape.if_count == 0
    assert shape.for_count == 0
    assert shape.while_count == 0
    assert shape.return_count == 1
