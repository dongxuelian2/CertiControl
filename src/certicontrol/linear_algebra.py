"""Small shared linear-algebra helpers used by certificate modules."""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy import linalg

from .tolerance import NumericalRankResult


def exact_rank_and_pivots(matrix: sp.MatrixBase) -> tuple[int, tuple[int, ...]]:
    """Return exact rank and RREF pivot columns for a SymPy matrix."""
    _, pivots = matrix.rref()
    return len(pivots), tuple(int(index) for index in pivots)


def numerical_pivot_columns(matrix: np.ndarray, rank: int) -> tuple[int, ...]:
    """Return a rank-sized set of numerically independent columns via pivoted QR."""
    if rank == 0:
        return ()
    _, _, pivots = linalg.qr(matrix, mode="economic", pivoting=True)
    return tuple(int(index) for index in pivots[:rank])


def rank_sensitivity_warning(
    result: NumericalRankResult,
    *,
    matrix_name: str,
) -> str | None:
    """Describe rank decisions that are close to the active numerical threshold."""
    if result.singular_values.size == 0 or result.tolerance == 0:
        return None
    positive = result.singular_values[result.singular_values > 0]
    near = positive[(positive >= 0.1 * result.tolerance) & (positive <= 10.0 * result.tolerance)]
    if near.size:
        return (
            "Numerical rank is tolerance-sensitive: one or more singular values lie within "
            "a factor of 10 of the active threshold."
        )
    if (
        np.isfinite(result.condition_number)
        and result.condition_number > 1.0 / np.sqrt(np.finfo(float).eps)
    ):
        return (
            f"The {matrix_name} is severely ill-conditioned; floating-point rank may be fragile."
        )
    return None
