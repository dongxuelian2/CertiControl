import numpy as np
import pytest
import sympy as sp

from certicontrol import LTISystem


def test_valid_siso_dimensions():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]])
    assert system.n == 2
    assert system.m == 1
    assert system.p is None
    assert system.has_exact_state_input_data


def test_valid_mimo_with_outputs():
    system = LTISystem(np.eye(3), np.ones((3, 2)), np.ones((2, 3)), np.zeros((2, 2)))
    assert system.n == 3
    assert system.m == 2
    assert system.p == 2


def test_a_must_be_square():
    with pytest.raises(ValueError, match="A must be square"):
        LTISystem([[1, 2, 3], [4, 5, 6]], [[1], [0]])


def test_b_rows_must_match_a():
    with pytest.raises(ValueError, match="B must have the same number of rows"):
        LTISystem(np.eye(2), np.ones((3, 1)))


def test_c_columns_must_match_state_dimension():
    with pytest.raises(ValueError, match="C must have 2 columns"):
        LTISystem(np.eye(2), np.ones((2, 1)), np.ones((1, 3)))


def test_d_input_dimension_must_match_b():
    with pytest.raises(ValueError, match="D must have the same number of columns"):
        LTISystem(np.eye(2), np.ones((2, 2)), D=np.ones((1, 1)))


def test_c_and_d_output_dimensions_must_match():
    with pytest.raises(ValueError, match="C and D must have the same number"):
        LTISystem(np.eye(2), np.ones((2, 1)), C=np.ones((2, 2)), D=np.ones((1, 1)))


def test_float_input_is_not_misrepresented_as_exact():
    system = LTISystem([[0.0, 1.0], [-2.0, -3.0]], [[0.0], [1.0]])
    assert not system.has_exact_state_input_data


def test_sympy_rational_input_is_retained_exactly():
    system = LTISystem(sp.Matrix([[0, sp.Rational(1, 2)], [0, 1]]), sp.eye(2))
    assert system.exact_A == sp.ImmutableMatrix([[0, sp.Rational(1, 2)], [0, 1]])
