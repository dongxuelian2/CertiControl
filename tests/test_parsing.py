import pytest
import sympy as sp

from certicontrol import parse_matrix


def test_parse_whitespace_integer_matrix():
    matrix = parse_matrix("0 1\n-2 -3")
    assert matrix == sp.ImmutableMatrix([[0, 1], [-2, -3]])


def test_parse_commas_and_rational_exactly():
    matrix = parse_matrix("0, 1\n-1/2, -3")
    assert matrix[1, 0] == sp.Rational(-1, 2)
    assert matrix[1, 0].is_Rational


def test_parse_float_and_scientific_notation():
    matrix = parse_matrix("1.25, -2e-3")
    assert isinstance(matrix[0, 0], sp.Float)
    assert isinstance(matrix[0, 1], sp.Float)


def test_parse_semicolon_rows():
    assert parse_matrix("1 0; 0 1") == sp.eye(2)


def test_reject_ragged_rows():
    with pytest.raises(ValueError, match="same length"):
        parse_matrix("1 2\n3")


def test_reject_arbitrary_expression():
    with pytest.raises(ValueError, match="Invalid matrix token"):
        parse_matrix("__import__('os').system('echo nope')")


def test_reject_zero_denominator():
    with pytest.raises(ValueError, match="denominator cannot be zero"):
        parse_matrix("1/0")
