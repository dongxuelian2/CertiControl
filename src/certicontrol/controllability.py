"""Controllability certificates for continuous-time LTI systems."""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp
from scipy import linalg

from .certificates import Certificate
from .model import LTISystem, MatrixLike
from .tolerance import NumericalRankResult, TolerancePolicy, numerical_rank


def _validate_ab_shapes(A: Any, B: Any) -> tuple[int, int]:
    if len(A.shape) != 2 or len(B.shape) != 2:
        raise ValueError("A and B must both be two-dimensional matrices.")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; got shape {A.shape}.")
    if B.shape[0] != A.shape[0]:
        raise ValueError(
            f"B must have {A.shape[0]} rows to match A; got shape {B.shape}."
        )
    return int(A.shape[0]), int(B.shape[1])


def controllability_matrix(A: MatrixLike, B: MatrixLike) -> np.ndarray | sp.Matrix:
    """Return ``[B, AB, ..., A^(n-1)B]`` using a recurrence.

    SymPy input yields a SymPy matrix; otherwise a NumPy array is returned.
    """
    if isinstance(A, sp.MatrixBase) or isinstance(B, sp.MatrixBase):
        A_sp = sp.Matrix(A)
        B_sp = sp.Matrix(B)
        n, _ = _validate_ab_shapes(A_sp, B_sp)
        blocks: list[sp.Matrix] = [B_sp]
        current = B_sp
        for _ in range(1, n):
            current = A_sp * current
            blocks.append(current)
        return sp.Matrix.hstack(*blocks)

    A_np = np.asarray(A)
    B_np = np.asarray(B)
    n, _ = _validate_ab_shapes(A_np, B_np)
    blocks_np: list[np.ndarray] = [B_np]
    current_np = B_np
    for _ in range(1, n):
        current_np = A_np @ current_np
        blocks_np.append(current_np)
    return np.hstack(blocks_np)


def _exact_rank_and_pivots(matrix: sp.MatrixBase) -> tuple[int, tuple[int, ...]]:
    _, pivots = matrix.rref()
    return len(pivots), tuple(int(index) for index in pivots)


def _numerical_pivot_columns(matrix: np.ndarray, rank: int) -> tuple[int, ...]:
    if rank == 0:
        return ()
    _, _, pivots = linalg.qr(matrix, mode="economic", pivoting=True)
    return tuple(int(index) for index in pivots[:rank])


def _rank_sensitivity_warning(result: NumericalRankResult) -> str | None:
    if result.singular_values.size == 0 or result.tolerance == 0:
        return None
    positive = result.singular_values[result.singular_values > 0]
    near = positive[(positive >= 0.1 * result.tolerance) & (positive <= 10.0 * result.tolerance)]
    if near.size:
        return (
            "Numerical rank is tolerance-sensitive: one or more singular values lie within "
            "a factor of 10 of the active threshold."
        )
    if np.isfinite(result.condition_number) and result.condition_number > 1.0 / np.sqrt(np.finfo(float).eps):
        return "The controllability matrix is severely ill-conditioned; floating-point rank may be fragile."
    return None


def _exact_pbh(system: LTISystem) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate PBH exactly through left eigenspaces.

    For a left eigenspace basis ``Q`` at lambda, PBH fails exactly when
    ``B.T * Q`` has a non-trivial nullspace.  This avoids expensive symbolic
    rank simplification of matrices containing algebraic eigenvalues.
    """
    assert system.exact_A is not None and system.exact_B is not None
    A = sp.Matrix(system.exact_A)
    B = sp.Matrix(system.exact_B)
    modes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    eigen_data = A.T.eigenvects(simplify=False, error_when_incomplete=False)
    for eigenvalue, multiplicity, basis in eigen_data:
        if not basis:
            continue
        eigenspace = sp.Matrix.hstack(*basis)
        coupling = B.T * eigenspace
        coupling_nullspace = coupling.nullspace()
        failure_dimension = len(coupling_nullspace)
        rank = system.n - failure_dimension
        mode = {
            "eigenvalue": eigenvalue,
            "algebraic_multiplicity": int(multiplicity),
            "geometric_multiplicity": int(len(basis)),
            "rank": int(rank),
        }
        modes.append(mode)
        if coupling_nullspace:
            coefficients = coupling_nullspace[0]
            witness = eigenspace * coefficients
            failure = dict(mode)
            failure["witness"] = witness
            failure["eigenvector_residual"] = sp.Integer(0)
            failure["input_orthogonality_residual"] = sp.Integer(0)
            failures.append(failure)
    return not failures, modes, failures


def _numerical_pbh(
    system: LTISystem, policy: TolerancePolicy
) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    A = np.asarray(system.A)
    B = np.asarray(system.B)
    identity = np.eye(system.n, dtype=np.result_type(A.dtype, np.complex128))
    eigenvalues = linalg.eigvals(A)
    modes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for eigenvalue in eigenvalues:
        pbh_matrix = np.hstack((eigenvalue * identity - A, B))
        rank_result = numerical_rank(pbh_matrix, policy)
        mode = {
            "eigenvalue": complex(eigenvalue),
            "rank": rank_result.rank,
            "singular_values": rank_result.singular_values,
            "tolerance": rank_result.tolerance,
            "condition_number": rank_result.condition_number,
        }
        modes.append(mode)
        if rank_result.rank < system.n:
            left_vectors, _, _ = linalg.svd(pbh_matrix, full_matrices=True)
            q = left_vectors[:, -1]
            q = q / linalg.norm(q)
            eigen_residual = float(linalg.norm(q.conj().T @ A - eigenvalue * q.conj().T))
            input_residual = float(linalg.norm(q.conj().T @ B))
            failure = dict(mode)
            failure.update(
                {
                    "witness": q,
                    "eigenvector_residual": eigen_residual,
                    "input_orthogonality_residual": input_residual,
                }
            )
            failures.append(failure)
    return not failures, modes, failures


def pbh_controllability(
    system: LTISystem, tolerance_policy: TolerancePolicy | None = None
) -> Certificate:
    """Evaluate the PBH controllability criterion with exact/numerical paths."""
    policy = tolerance_policy or TolerancePolicy()
    numerical_passed, numerical_modes, numerical_failures = _numerical_pbh(system, policy)
    warnings: list[str] = []

    exact_passed: bool | None = None
    exact_modes: list[dict[str, Any]] | None = None
    exact_failures: list[dict[str, Any]] | None = None
    if system.has_exact_state_input_data:
        exact_passed, exact_modes, exact_failures = _exact_pbh(system)
        if exact_passed != numerical_passed:
            warnings.append(
                "HIGH: Exact and numerical PBH classifications disagree; the floating-point "
                "classification is tolerance-sensitive. The exact algebraic result is authoritative."
            )

    passed = exact_passed if exact_passed is not None else numerical_passed
    return Certificate(
        name="PBH controllability",
        passed=bool(passed),
        verdict="controllable" if passed else "uncontrollable",
        evidence={
            "exact_modes": exact_modes,
            "exact_failures": exact_failures,
            "numerical_modes": numerical_modes,
            "numerical_failures": numerical_failures,
        },
        diagnostics={
            "n": system.n,
            "exact_passed": exact_passed,
            "numerical_passed": numerical_passed,
        },
        warnings=warnings,
    )


def analyze_controllability(
    system: LTISystem, tolerance_policy: TolerancePolicy | None = None
) -> Certificate:
    """Build a controllability certificate and cross-check it with PBH."""
    policy = tolerance_policy or TolerancePolicy()
    numeric_matrix = np.asarray(controllability_matrix(system.A, system.B))
    numeric_rank = numerical_rank(numeric_matrix, policy)
    numerical_pivots = _numerical_pivot_columns(numeric_matrix, numeric_rank.rank)
    warnings: list[str] = []

    sensitivity = _rank_sensitivity_warning(numeric_rank)
    if sensitivity:
        warnings.append(sensitivity)

    exact_matrix: sp.Matrix | None = None
    exact_rank: int | None = None
    exact_pivots: tuple[int, ...] | None = None
    if system.has_exact_state_input_data:
        assert system.exact_A is not None and system.exact_B is not None
        exact_matrix = controllability_matrix(system.exact_A, system.exact_B)
        assert isinstance(exact_matrix, sp.MatrixBase)
        exact_rank, exact_pivots = _exact_rank_and_pivots(exact_matrix)
        if exact_rank != numeric_rank.rank:
            warnings.append(
                "HIGH: Exact controllability rank and numerical rank disagree; the floating-point "
                "classification is tolerance-sensitive. The exact algebraic rank is authoritative."
            )

    matrix_exact_passed = None if exact_rank is None else exact_rank == system.n
    matrix_numerical_passed = numeric_rank.rank == system.n
    pbh = pbh_controllability(system, policy)
    pbh_exact_passed = pbh.diagnostics["exact_passed"]
    pbh_numerical_passed = pbh.diagnostics["numerical_passed"]

    if matrix_exact_passed is not None and pbh_exact_passed is not None:
        if matrix_exact_passed != pbh_exact_passed:
            warnings.append(
                "HIGH: Controllability-matrix and PBH exact criteria disagree. This indicates an internal inconsistency."
            )
    if matrix_numerical_passed != pbh_numerical_passed:
        warnings.append(
            "HIGH: Controllability-matrix and PBH numerical criteria disagree under the active tolerance."
        )
    warnings.extend(pbh.warnings)

    passed = matrix_exact_passed if matrix_exact_passed is not None else matrix_numerical_passed
    return Certificate(
        name="Controllability",
        passed=bool(passed),
        verdict="controllable" if passed else "uncontrollable",
        evidence={
            "controllability_matrix": numeric_matrix,
            "controllability_matrix_exact": exact_matrix,
            "pivot_columns": exact_pivots if exact_pivots is not None else numerical_pivots,
            "pbh": pbh,
        },
        diagnostics={
            "n": system.n,
            "m": system.m,
            "exact_rank": exact_rank,
            "numerical_rank": numeric_rank.rank,
            "singular_values": numeric_rank.singular_values,
            "tolerance": numeric_rank.tolerance,
            "condition_number": numeric_rank.condition_number,
            "numerical_pivot_columns": numerical_pivots,
            "exact_pivot_columns": exact_pivots,
            "matrix_exact_passed": matrix_exact_passed,
            "matrix_numerical_passed": matrix_numerical_passed,
            "pbh_exact_passed": pbh_exact_passed,
            "pbh_numerical_passed": pbh_numerical_passed,
        },
        warnings=warnings,
    )
