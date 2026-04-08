"""Deterministic normalization and extraction utilities for Phase 1.

This module owns:
- numeric canonicalization for a narrow set of parameter-like values
- regex-based extraction for explicitly supported parameters only
- lightweight function signature and control-flow shape extraction
- stable block identity generation with a deterministic fallback path
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import json
import re
from typing import Final

_CANONICAL_SCALE: Final[int] = 10
_CANONICAL_QUANTIZER: Final[Decimal] = Decimal("0.0000000001")
_LINE_ASSIGNMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?P<name>\b(?:learning_rate|adamw_eps|weight_decay|lr|eps)\b)
    \s*=\s*
    (?P<value>
        [+-]?
        (?:
            (?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?
            |
            \d+[eE][+-]?\d+
        )
    )
    """,
    re.VERBOSE,
)
_DEF_START_RE: Final[re.Pattern[str]] = re.compile(r"^(?P<indent>[ \t]*)def\s+(?P<name>[A-Za-z_]\w*)\s*\(")
_DEF_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*def\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<params>.*)\)\s*:\s*$",
    re.DOTALL,
)
_CONTROL_KEYWORDS: Final[tuple[str, ...]] = ("if", "for", "while", "return")
_PATH_SEP_RE: Final[re.Pattern[str]] = re.compile(r"[\\/]+")
_IDENTIFIER_CLEAN_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9_]+")
_SPLIT_NAME_RE: Final[re.Pattern[str]] = re.compile(r"[\s\-]+")
_SUPPORTED_PARAM_MAP: Final[dict[str, str]] = {
    "lr": "lr",
    "learning_rate": "lr",
    "eps": "eps",
    "adamw_eps": "eps",
    "weight_decay": "weight_decay",
}


@dataclass(frozen=True, slots=True)
class SupportedParamMatch:
    """A single supported parameter match extracted from source text.

    Input:
    - built internally from a regex match over source code

    Output:
    - an immutable record containing canonical name, raw match data, and location data

    Failure behavior:
    - instances are created only for recognized params with parseable numeric values
    """

    canonical_param: str
    raw_param: str
    raw_value: str
    normalized_value: str
    line_number: int
    column_start: int
    column_end: int
    span_start: int
    span_end: int

    def to_dict(self) -> dict[str, str | int]:
        """Return a deterministic dictionary representation of the match."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FunctionSignatureInfo:
    """Lightweight function signature metadata extracted from source text.

    Input:
    - a source string that may contain decorators and exactly one target function

    Output:
    - parsed name, argument count, line location, and parse-completeness fields

    Failure behavior:
    - does not raise for parse misses; instead returns `parsed=False` and default values
    """

    parsed: bool
    function_name: str
    normalized_function_name: str
    argument_count: int
    def_line_number: int | None
    body_indent: int | None
    header_start_index: int | None
    header_end_index: int | None


@dataclass(frozen=True, slots=True)
class ControlFlowShape:
    """Counts of top-level control-flow statements within a function body.

    Input:
    - a function source string

    Output:
    - deterministic counts for top-level `if`, `for`, `while`, and `return`

    Failure behavior:
    - returns zero counts when no parseable function body is found
    """

    if_count: int = 0
    for_count: int = 0
    while_count: int = 0
    return_count: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return a deterministic dictionary representation of the shape."""
        return asdict(self)


def normalize_numeric_value(raw: str) -> str:
    """Canonicalize a numeric string to a fixed-width decimal string.

    Input:
    - `raw`: a numeric string using plain decimal or scientific notation

    Output:
    - a decimal string with exactly 10 digits after the decimal point

    Failure behavior:
    - raises `ValueError` if `raw` is empty or cannot be parsed as a Decimal
    """

    text = raw.strip()
    if not text:
        raise ValueError("numeric value is empty")

    try:
        with localcontext() as ctx:
            ctx.prec = 50
            ctx.rounding = ROUND_HALF_EVEN
            value = Decimal(text)
            normalized = value.quantize(_CANONICAL_QUANTIZER)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid numeric value: {raw!r}") from exc

    if normalized == Decimal("-0.0000000000") or normalized == Decimal("0E-10"):
        normalized = Decimal("0").quantize(_CANONICAL_QUANTIZER)

    return format(normalized, "f")


def extract_supported_params(code: str) -> list[SupportedParamMatch]:
    """Extract supported parameter assignments from source code in source order.

    Input:
    - `code`: arbitrary source text that may contain direct assignments or call kwargs

    Output:
    - a list of immutable match records for supported params only

    Failure behavior:
    - never raises for unknown params or non-matching lines
    - skips matches whose numeric values cannot be normalized
    """

    matches: list[SupportedParamMatch] = []
    offset = 0

    for line_number, line in enumerate(code.splitlines(), start=1):
        search_text = _mask_strings_and_strip_comment(line)
        for match in _LINE_ASSIGNMENT_RE.finditer(search_text):
            raw_param = match.group("name")
            canonical_param = _SUPPORTED_PARAM_MAP.get(raw_param)
            if canonical_param is None:
                continue

            raw_value = match.group("value")
            try:
                normalized_value = normalize_numeric_value(raw_value)
            except ValueError:
                continue

            column_start = match.start() + 1
            column_end = match.end()
            matches.append(
                SupportedParamMatch(
                    canonical_param=canonical_param,
                    raw_param=raw_param,
                    raw_value=raw_value,
                    normalized_value=normalized_value,
                    line_number=line_number,
                    column_start=column_start,
                    column_end=column_end,
                    span_start=offset + match.start(),
                    span_end=offset + match.end(),
                )
            )

        offset += len(line) + 1

    return matches


def extract_function_signature_info(code: str) -> FunctionSignatureInfo:
    """Extract minimal function signature metadata from source text.

    Input:
    - `code`: source text expected to contain one target function, optionally with decorators

    Output:
    - parsed signature metadata including function name, normalized name, and argument count

    Failure behavior:
    - never raises for parse misses; returns `parsed=False` instead
    """

    lines = code.splitlines()
    header = _find_function_header(lines)
    if header is None:
        return FunctionSignatureInfo(
            parsed=False,
            function_name="",
            normalized_function_name="",
            argument_count=0,
            def_line_number=None,
            body_indent=None,
            header_start_index=None,
            header_end_index=None,
        )

    header_text, start_idx, end_idx, def_indent = header
    header_match = _DEF_HEADER_RE.match(header_text)
    if header_match is None:
        return FunctionSignatureInfo(
            parsed=False,
            function_name="",
            normalized_function_name="",
            argument_count=0,
            def_line_number=start_idx + 1,
            body_indent=_find_body_indent(lines, end_idx, def_indent),
            header_start_index=start_idx,
            header_end_index=end_idx,
        )

    function_name = header_match.group("name")
    params_text = header_match.group("params")
    return FunctionSignatureInfo(
        parsed=True,
        function_name=function_name,
        normalized_function_name=_normalize_identifier(function_name),
        argument_count=_count_signature_arguments(params_text),
        def_line_number=start_idx + 1,
        body_indent=_find_body_indent(lines, end_idx, def_indent),
        header_start_index=start_idx,
        header_end_index=end_idx,
    )


def compute_top_level_control_flow_shape(function_source: str) -> ControlFlowShape:
    """Count top-level control-flow statements relative to a function body.

    Input:
    - `function_source`: source text for a single function, optionally including decorators

    Output:
    - counts for top-level `if`, `for`, `while`, and `return` statements only

    Failure behavior:
    - never raises for parse misses; returns zero counts if the function body is unavailable
    """

    signature = extract_function_signature_info(function_source)
    if signature.def_line_number is None or signature.header_end_index is None:
        return ControlFlowShape()

    lines = function_source.splitlines()
    def_line_index = signature.header_end_index
    def_indent = _leading_indent(lines[signature.header_start_index or 0])
    body_indent = signature.body_indent

    if body_indent is None:
        return ControlFlowShape()

    counts = {keyword: 0 for keyword in _CONTROL_KEYWORDS}

    for raw_line in lines[def_line_index + 1 :]:
        if not raw_line.strip():
            continue

        line_without_comment = _strip_inline_comment(raw_line)
        if not line_without_comment.strip():
            continue

        current_indent = _leading_indent(raw_line)
        if current_indent <= def_indent:
            break
        if current_indent != body_indent:
            continue

        stripped = line_without_comment.lstrip()
        if _starts_with_keyword(stripped, "if"):
            counts["if"] += 1
        elif _starts_with_keyword(stripped, "for"):
            counts["for"] += 1
        elif _starts_with_keyword(stripped, "while"):
            counts["while"] += 1
        elif _starts_with_keyword(stripped, "return"):
            counts["return"] += 1

    return ControlFlowShape(
        if_count=counts["if"],
        for_count=counts["for"],
        while_count=counts["while"],
        return_count=counts["return"],
    )


def make_block_id(file_path: str, function_source: str, start_line: int | None = None) -> str:
    """Create a stable block id from function structure and location context.

    Input:
    - `file_path`: path to the source file containing the function
    - `function_source`: source text for the function, optionally with decorators
    - `start_line`: optional fallback line number when signature extraction fails

    Output:
    - a short deterministic block id string

    Failure behavior:
    - never raises for incomplete function parsing; falls back to location-based identity
    """

    signature = extract_function_signature_info(function_source)
    if not signature.parsed:
        line_number = start_line if start_line is not None else signature.def_line_number
        return make_fallback_block_id(file_path, line_number)

    shape = compute_top_level_control_flow_shape(function_source)
    payload = {
        "arg_count": signature.argument_count,
        "file_path": _normalize_file_path(file_path),
        "function_name": signature.normalized_function_name,
        "shape": shape.to_dict(),
    }
    digest = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        digest_size=10,
    ).hexdigest()
    return f"blk_{digest}"


def make_fallback_block_id(file_path: str, line_number: int | None) -> str:
    """Create a deterministic fallback block id from path and line bucket.

    Input:
    - `file_path`: path to the source file
    - `line_number`: approximate line number for the unresolved block

    Output:
    - a short deterministic fallback block id string

    Failure behavior:
    - never raises; missing line numbers are grouped into bucket `0`
    """

    bucket = 0 if line_number is None or line_number < 1 else (line_number - 1) // 10
    payload = {
        "bucket": bucket,
        "file_path": _normalize_file_path(file_path),
        "mode": "fallback",
    }
    digest = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        digest_size=10,
    ).hexdigest()
    return f"fbk_{digest}"


def normalize_numeric_token(value: str) -> str:
    """Compatibility wrapper for numeric normalization.

    Input:
    - `value`: a numeric string

    Output:
    - the fixed-width canonical numeric string from `normalize_numeric_value`

    Failure behavior:
    - raises `ValueError` under the same conditions as `normalize_numeric_value`
    """

    return normalize_numeric_value(value)


def normalize_param_name(value: str) -> str:
    """Compatibility helper for small deterministic identifier normalization.

    Input:
    - `value`: a parameter-like identifier

    Output:
    - a lowercased underscore-separated identifier

    Failure behavior:
    - never raises for ordinary strings
    """

    text = value.strip().lower()
    text = _SPLIT_NAME_RE.sub("_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_shape_name(value: str) -> str:
    """Compatibility helper for shape-like identifiers.

    Input:
    - `value`: a freeform shape-like name

    Output:
    - the same deterministic identifier normalization used for param-like names

    Failure behavior:
    - never raises for ordinary strings
    """

    return normalize_param_name(value)


def normalize_block_id(value: str) -> str:
    """Compatibility helper for normalizing freeform block-id text.

    Input:
    - `value`: a freeform identifier-like string

    Output:
    - a lowercased token that preserves only `a-z`, digits, `_`, `.`, `:`, and `-`

    Failure behavior:
    - never raises for ordinary strings
    """

    text = value.strip().lower().replace(" ", "-")
    text = re.sub(r"[^a-z0-9_.:-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-")


def _strip_inline_comment(line: str) -> str:
    """Remove trailing comments from a single line without parsing string literals."""

    return line.split("#", 1)[0]


def _mask_strings_and_strip_comment(line: str) -> str:
    """Mask string literal contents and strip comments while preserving positions."""

    chars = list(line)
    length = len(chars)
    index = 0
    quote_char: str | None = None
    triple_quoted = False

    while index < length:
        char = chars[index]

        if quote_char is None:
            if char == "#":
                for comment_index in range(index, length):
                    chars[comment_index] = " "
                break

            if char in {"'", '"'}:
                quote_char = char
                triple_quoted = line[index : index + 3] == char * 3
                if triple_quoted:
                    chars[index] = " "
                    if index + 1 < length:
                        chars[index + 1] = " "
                    if index + 2 < length:
                        chars[index + 2] = " "
                    index += 3
                    continue

                chars[index] = " "
                index += 1
                continue

            index += 1
            continue

        if triple_quoted:
            if line[index : index + 3] == quote_char * 3:
                chars[index] = " "
                if index + 1 < length:
                    chars[index + 1] = " "
                if index + 2 < length:
                    chars[index + 2] = " "
                quote_char = None
                triple_quoted = False
                index += 3
                continue

            chars[index] = " "
            index += 1
            continue

        if char == "\\" and index + 1 < length:
            chars[index] = " "
            chars[index + 1] = " "
            index += 2
            continue

        chars[index] = " "
        if char == quote_char:
            quote_char = None
        index += 1

    return "".join(chars)


def _find_function_header(
    lines: list[str],
) -> tuple[str, int, int, int] | None:
    """Collect the `def ...:` header for the first function found in source lines."""

    for start_idx, line in enumerate(lines):
        match = _DEF_START_RE.match(line)
        if match is None:
            continue

        parts = [line.strip()]
        paren_balance = line.count("(") - line.count(")")
        if paren_balance <= 0 and line.rstrip().endswith(":"):
            return (" ".join(parts), start_idx, start_idx, len(match.group("indent")))

        for end_idx in range(start_idx + 1, len(lines)):
            next_line = lines[end_idx].strip()
            parts.append(next_line)
            paren_balance += lines[end_idx].count("(") - lines[end_idx].count(")")
            if paren_balance <= 0 and lines[end_idx].rstrip().endswith(":"):
                return (" ".join(parts), start_idx, end_idx, len(match.group("indent")))

        return None

    return None


def _find_body_indent(lines: list[str], header_end_idx: int, def_indent: int) -> int | None:
    """Return the first real body indentation level for a parsed function header."""

    for line in lines[header_end_idx + 1 :]:
        if not line.strip():
            continue
        stripped = _strip_inline_comment(line)
        if not stripped.strip():
            continue
        current_indent = _leading_indent(line)
        if current_indent <= def_indent:
            return None
        return current_indent
    return None


def _count_signature_arguments(params_text: str) -> int:
    """Count explicit function parameters using a shallow comma split."""

    parts = _split_top_level_commas(params_text)
    count = 0
    for part in parts:
        token = part.strip()
        if not token or token in {"*", "/"}:
            continue
        count += 1
    return count


def _split_top_level_commas(text: str) -> list[str]:
    """Split a parameter list on commas not nested in brackets or braces."""

    if not text.strip():
        return []

    parts: list[str] = []
    start = 0
    depth = 0
    quote_char: str | None = None
    index = 0

    while index < len(text):
        char = text[index]

        if quote_char is not None:
            if char == "\\" and index + 1 < len(text):
                index += 2
                continue
            if char == quote_char:
                quote_char = None
            index += 1
            continue

        if char in {"'", '"'}:
            quote_char = char
            index += 1
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
        index += 1

    parts.append(text[start:])
    return parts


def _leading_indent(line: str) -> int:
    """Return the indentation width using raw leading spaces and tabs."""

    return len(line) - len(line.lstrip(" \t"))


def _starts_with_keyword(text: str, keyword: str) -> bool:
    """Return True when `text` starts with a whole Python keyword token."""

    if not text.startswith(keyword):
        return False
    if len(text) == len(keyword):
        return True
    return text[len(keyword)] in {" ", "\t", "(", ":"}


def _normalize_identifier(name: str) -> str:
    """Normalize a function or identifier name to a stable lowercase token."""

    text = _SPLIT_NAME_RE.sub("_", name.strip().lower())
    text = _IDENTIFIER_CLEAN_RE.sub("_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _normalize_file_path(file_path: str) -> str:
    """Normalize path separators and case for deterministic id serialization."""

    return _PATH_SEP_RE.sub("/", file_path.strip().lower())


__all__ = [
    "ControlFlowShape",
    "FunctionSignatureInfo",
    "SupportedParamMatch",
    "compute_top_level_control_flow_shape",
    "extract_function_signature_info",
    "extract_supported_params",
    "make_block_id",
    "make_fallback_block_id",
    "normalize_block_id",
    "normalize_numeric_token",
    "normalize_numeric_value",
    "normalize_param_name",
    "normalize_shape_name",
]
