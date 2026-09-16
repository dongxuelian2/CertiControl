"""Safe parsing for small numeric matrices."""

from __future__ import annotations

import re

import sympy as sp

_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_RATIONAL_RE = re.compile(r"^[+-]?\d+/\d+$")
_FLOAT_RE = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)$"
)


def _parse_scalar(token: str) -> sp.Expr:
    if _INTEGER_RE.fullmatch(token):
        return sp.Integer(token)
    if _RATIONAL_RE.fullmatch(token):
        numerator, denominator = token.split("/", 1)
        if int(denominator) == 0:
            raise ValueError(f"Invalid rational value {token!r}: denominator cannot be zero.")
        return sp.Rational(int(numerator), int(denominator))
    if _FLOAT_RE.fullmatch(token):
        return sp.Float(token)
    raise ValueError(
        f"Invalid matrix token {token!r}. Expected an integer, decimal/scientific float, or rational a/b."
    )


def parse_matrix(text: str) -> sp.ImmutableMatrix:
    """Parse a matrix without ``eval`` while preserving exact rationals.

    Rows may be separated by newlines or semicolons. Entries may be separated
    by whitespace, commas, or a mixture of both.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Matrix text must be a non-empty string.")

    raw_rows = [row.strip() for row in text.replace(";", "\n").splitlines() if row.strip()]
    rows: list[list[sp.Expr]] = []
    expected_columns: int | None = None
    for row_number, raw_row in enumerate(raw_rows, start=1):
        tokens = [token for token in re.split(r"[\s,]+", raw_row) if token]
        if not tokens:
            continue
        if expected_columns is None:
            expected_columns = len(tokens)
        elif len(tokens) != expected_columns:
            raise ValueError(
                "Matrix rows must all have the same length; "
                f"row 1 has {expected_columns} entries but row {row_number} has {len(tokens)}."
            )
        rows.append([_parse_scalar(token) for token in tokens])

    if not rows:
        raise ValueError("Matrix text did not contain any entries.")
    return sp.ImmutableMatrix(rows)
