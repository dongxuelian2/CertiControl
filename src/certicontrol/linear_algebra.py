"""Small shared linear-algebra helpers used by certificate modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy import linalg

from .tolerance import NumericalRankResult, TolerancePolicy


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


def matrix_scale_tolerance(matrix: np.ndarray, policy: TolerancePolicy | None = None) -> float:
    """Return a scale-aware absolute tolerance derived from matrix singular values."""
    policy = policy or TolerancePolicy()
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"matrix must be two-dimensional; got shape {array.shape}.")
    singular_values = linalg.svdvals(array)
    return policy.threshold(singular_values, array.shape)


@dataclass(frozen=True)
class NumericalPositiveDefiniteResult:
    """Hermitian eigenvalue diagnostics for a positive-definiteness decision."""

    eigenvalues: np.ndarray
    lambda_min: float
    tolerance: float
    verdict: bool | None
    cholesky_success: bool


def numerical_positive_definite(
    matrix: np.ndarray,
    policy: TolerancePolicy | None = None,
) -> NumericalPositiveDefiniteResult:
    """Classify a Hermitian matrix as positive definite, not PD, or boundary-sensitive."""
    policy = policy or TolerancePolicy()
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"positive-definiteness input must be square; got shape {array.shape}.")
    eigenvalues = linalg.eigvalsh(array)
    lambda_min = float(np.real(eigenvalues[0])) if eigenvalues.size else float("nan")
    scales = np.sort(np.abs(eigenvalues))[::-1]
    tolerance = policy.threshold(scales, array.shape)
    if lambda_min > tolerance:
        verdict: bool | None = True
    elif lambda_min < -tolerance:
        verdict = False
    else:
        verdict = None
    try:
        linalg.cholesky(array, lower=True)
        cholesky_success = True
    except linalg.LinAlgError:
        cholesky_success = False
    return NumericalPositiveDefiniteResult(
        eigenvalues=eigenvalues,
        lambda_min=lambda_min,
        tolerance=tolerance,
        verdict=verdict,
        cholesky_success=cholesky_success,
    )



@dataclass(frozen=True)
class NumericalPositiveSemidefiniteResult:
    """Hermitian eigenvalue diagnostics for a positive-semidefiniteness decision."""

    eigenvalues: np.ndarray
    lambda_min: float
    tolerance: float
    verdict: bool | None
    classification: str


def numerical_positive_semidefinite(
    matrix: np.ndarray,
    policy: TolerancePolicy | None = None,
) -> NumericalPositiveSemidefiniteResult:
    """Classify a Hermitian matrix as PSD, indefinite, or tolerance-sensitive.

    A nonnegative eigenvalue that is smaller than the active tolerance is
    accepted as semidefinite rather than positive definite. A slightly
    negative eigenvalue within tolerance is reported as inconclusive instead
    of silently projecting the matrix onto the PSD cone.
    """
    policy = policy or TolerancePolicy()
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"positive-semidefiniteness input must be square; got shape {array.shape}.")
    eigenvalues = linalg.eigvalsh(array)
    lambda_min = float(np.real(eigenvalues[0])) if eigenvalues.size else float("nan")
    scales = np.sort(np.abs(eigenvalues))[::-1]
    tolerance = policy.threshold(scales, array.shape)
    if lambda_min > tolerance:
        verdict: bool | None = True
        classification = "POSITIVE_DEFINITE"
    elif lambda_min >= 0.0:
        verdict = True
        classification = "POSITIVE_SEMIDEFINITE"
    elif lambda_min >= -tolerance:
        verdict = None
        classification = "TOLERANCE_SENSITIVE"
    else:
        verdict = False
        classification = "INDEFINITE"
    return NumericalPositiveSemidefiniteResult(
        eigenvalues=eigenvalues,
        lambda_min=lambda_min,
        tolerance=tolerance,
        verdict=verdict,
        classification=classification,
    )


def exact_positive_semidefinite(
    matrix: sp.MatrixBase,
) -> tuple[bool | None, tuple[tuple[tuple[int, ...], sp.Expr], ...]]:
    """Check PSD exactly via all principal minors for exact Hermitian data.

    A Hermitian matrix is positive semidefinite iff every principal minor is
    nonnegative. This is deliberately different from the leading-principal-
    minor Sylvester criterion used for strict positive definiteness.
    """
    from itertools import combinations

    mat = sp.Matrix(matrix)
    if mat.rows != mat.cols:
        raise ValueError(f"positive-semidefiniteness input must be square; got shape {mat.shape}.")
    if mat != mat.conjugate().T:
        return False, ()

    evidence: list[tuple[tuple[int, ...], sp.Expr]] = []
    unknown = False
    for size in range(1, mat.rows + 1):
        for indices in combinations(range(mat.rows), size):
            minor = sp.simplify(mat.extract(indices, indices).det())
            evidence.append((tuple(int(index) for index in indices), minor))
            if minor.is_negative is True:
                return False, tuple(evidence)
            if minor.is_nonnegative is not True:
                unknown = True
    return (None if unknown else True), tuple(evidence)

def exact_positive_definite(matrix: sp.MatrixBase) -> tuple[bool, tuple[sp.Expr, ...]]:
    """Apply Sylvester's criterion and return all exact leading principal minors."""
    mat = sp.Matrix(matrix)
    if mat.rows != mat.cols:
        raise ValueError(f"positive-definiteness input must be square; got shape {mat.shape}.")
    if mat != mat.conjugate().T:
        return False, ()
    minors = tuple(sp.simplify(mat[:k, :k].det()) for k in range(1, mat.rows + 1))
    return all(minor.is_positive is True for minor in minors), minors