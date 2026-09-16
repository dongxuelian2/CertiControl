"""Observability certificates for continuous-time LTI systems."""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp
from scipy import linalg

from .certificates import Certificate
from .linear_algebra import exact_rank_and_pivots, numerical_pivot_columns, rank_sensitivity_warning
from .model import LTISystem, MatrixLike
from .tolerance import TolerancePolicy, numerical_rank


def _validate_ac_shapes(A: Any, C: Any) -> tuple[int, int]:
    if len(A.shape) != 2 or len(C.shape) != 2:
        raise ValueError("A and C must both be two-dimensional matrices.")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; got shape {A.shape}.")
    if C.shape[1] != A.shape[0]:
        raise ValueError(
            f"C must have {A.shape[0]} columns to match A; got shape {C.shape}."
        )
    return int(A.shape[0]), int(C.shape[0])


def observability_matrix(A: MatrixLike, C: MatrixLike) -> np.ndarray | sp.Matrix:
    """Return ``[C; CA; ...; CA^(n-1)]`` using a recurrence.

    SymPy input yields a SymPy matrix; otherwise a NumPy array is returned.
    """
    if isinstance(A, sp.MatrixBase) or isinstance(C, sp.MatrixBase):
        A_sp = sp.Matrix(A)
        C_sp = sp.Matrix(C)
        n, _ = _validate_ac_shapes(A_sp, C_sp)
        blocks: list[sp.Matrix] = [C_sp]
        current = C_sp
        for _ in range(1, n):
            current = current * A_sp
            blocks.append(current)
        return sp.Matrix.vstack(*blocks)

    A_np = np.asarray(A)
    C_np = np.asarray(C)
    n, _ = _validate_ac_shapes(A_np, C_np)
    blocks_np: list[np.ndarray] = [C_np]
    current_np = C_np
    for _ in range(1, n):
        current_np = current_np @ A_np
        blocks_np.append(current_np)
    return np.vstack(blocks_np)


def _require_output_matrix(system: LTISystem) -> np.ndarray:
    if system.C is None:
        raise ValueError("Observability analysis requires a C output matrix.")
    return system.C


def _exact_pbh(system: LTISystem) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate exact PBH failures through the exact unobservable subspace.

    A full-rank exact observability matrix immediately certifies that no PBH
    failure exists, avoiding expensive symbolic eigendecompositions of all of
    ``A``.  If an unobservable subspace exists, it is ``A``-invariant; we
    restrict ``A`` to that smaller exact subspace, extract its eigenvectors,
    and lift them back to state-space PBH witnesses.
    """
    assert system.exact_A is not None and system.exact_C is not None
    A = sp.Matrix(system.exact_A)
    C = sp.Matrix(system.exact_C)
    obs = observability_matrix(A, C)
    assert isinstance(obs, sp.MatrixBase)
    nullspace = obs.nullspace()
    if not nullspace:
        return True, [], []

    basis_matrix = sp.Matrix.hstack(*nullspace)
    restricted_A, parameters = basis_matrix.gauss_jordan_solve(A * basis_matrix)
    if parameters.rows != 0:
        raise RuntimeError("Failed to obtain a unique restriction to the exact unobservable subspace.")

    modes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for eigenvalue, multiplicity, basis in restricted_A.eigenvects(
        simplify=False, error_when_incomplete=False
    ):
        if not basis:
            continue
        failure_dimension = len(basis)
        mode = {
            "eigenvalue": eigenvalue,
            "unobservable_algebraic_multiplicity": int(multiplicity),
            "unobservable_geometric_multiplicity": int(failure_dimension),
            "rank": int(system.n - failure_dimension),
        }
        modes.append(mode)
        witness = basis_matrix * basis[0]
        eigen_error = (A * witness - eigenvalue * witness).applyfunc(sp.simplify)
        output_error = (C * witness).applyfunc(sp.simplify)
        if eigen_error != sp.zeros(system.n, 1) or output_error != sp.zeros(C.rows, 1):
            raise RuntimeError("Exact unobservable-mode witness failed residual verification.")
        failure = dict(mode)
        failure["witness"] = witness
        failure["eigenvector_residual"] = sp.Integer(0)
        failure["output_null_residual"] = sp.Integer(0)
        failures.append(failure)
    if not failures:
        raise RuntimeError("Nontrivial exact unobservable subspace produced no PBH witness.")
    return False, modes, failures


def _numerical_pbh(
    system: LTISystem, policy: TolerancePolicy
) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    C = _require_output_matrix(system)
    A = np.asarray(system.A)
    identity = np.eye(system.n, dtype=np.result_type(A.dtype, np.complex128))
    eigenvalues = linalg.eigvals(A)
    modes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for eigenvalue in eigenvalues:
        pbh_matrix = np.vstack((eigenvalue * identity - A, C))
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
            _, _, vh = linalg.svd(pbh_matrix, full_matrices=True)
            v = vh.conj().T[:, -1]
            v = v / linalg.norm(v)
            eigen_residual = float(linalg.norm(A @ v - eigenvalue * v))
            output_residual = float(linalg.norm(C @ v))
            failure = dict(mode)
            failure.update(
                {
                    "witness": v,
                    "eigenvector_residual": eigen_residual,
                    "output_null_residual": output_residual,
                }
            )
            failures.append(failure)
    return not failures, modes, failures


def pbh_observability(
    system: LTISystem, tolerance_policy: TolerancePolicy | None = None
) -> Certificate:
    """Evaluate the PBH observability criterion with exact/numerical paths."""
    _require_output_matrix(system)
    policy = tolerance_policy or TolerancePolicy()
    numerical_passed, numerical_modes, numerical_failures = _numerical_pbh(system, policy)
    warnings: list[str] = []

    exact_passed: bool | None = None
    exact_modes: list[dict[str, Any]] | None = None
    exact_failures: list[dict[str, Any]] | None = None
    if system.has_exact_state_output_data:
        exact_passed, exact_modes, exact_failures = _exact_pbh(system)
        if exact_passed != numerical_passed:
            warnings.append(
                "HIGH: Exact and numerical PBH observability classifications disagree; the floating-point "
                "classification is tolerance-sensitive. The exact algebraic result is authoritative."
            )

    passed = exact_passed if exact_passed is not None else numerical_passed
    return Certificate(
        name="PBH observability",
        passed=bool(passed),
        verdict="observable" if passed else "unobservable",
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


def analyze_observability(
    system: LTISystem, tolerance_policy: TolerancePolicy | None = None
) -> Certificate:
    """Build an observability certificate and cross-check it with PBH."""
    C = _require_output_matrix(system)
    policy = tolerance_policy or TolerancePolicy()
    numeric_matrix = np.asarray(observability_matrix(system.A, C))
    numeric_rank = numerical_rank(numeric_matrix, policy)
    numerical_pivots = numerical_pivot_columns(numeric_matrix, numeric_rank.rank)
    warnings: list[str] = []

    sensitivity = rank_sensitivity_warning(numeric_rank, matrix_name="observability matrix")
    if sensitivity:
        warnings.append(sensitivity)

    exact_matrix: sp.Matrix | None = None
    exact_rank: int | None = None
    exact_pivots: tuple[int, ...] | None = None
    if system.has_exact_state_output_data:
        assert system.exact_A is not None and system.exact_C is not None
        exact_matrix = observability_matrix(system.exact_A, system.exact_C)
        assert isinstance(exact_matrix, sp.MatrixBase)
        exact_rank, exact_pivots = exact_rank_and_pivots(exact_matrix)
        if exact_rank != numeric_rank.rank:
            warnings.append(
                "HIGH: Exact observability rank and numerical rank disagree; the floating-point "
                "classification is tolerance-sensitive. The exact algebraic rank is authoritative."
            )

    matrix_exact_passed = None if exact_rank is None else exact_rank == system.n
    matrix_numerical_passed = numeric_rank.rank == system.n
    pbh = pbh_observability(system, policy)
    pbh_exact_passed = pbh.diagnostics["exact_passed"]
    pbh_numerical_passed = pbh.diagnostics["numerical_passed"]

    if matrix_exact_passed is not None and pbh_exact_passed is not None:
        if matrix_exact_passed != pbh_exact_passed:
            raise RuntimeError(
                "Exact observability-matrix and PBH criteria disagree; this indicates an implementation error."
            )
    if matrix_numerical_passed != pbh_numerical_passed:
        warnings.append(
            "HIGH: Observability-matrix and PBH numerical criteria disagree under the active tolerance."
        )
    warnings.extend(pbh.warnings)

    passed = matrix_exact_passed if matrix_exact_passed is not None else matrix_numerical_passed
    smallest = float(numeric_rank.singular_values[-1]) if numeric_rank.singular_values.size else 0.0
    return Certificate(
        name="Observability",
        passed=bool(passed),
        verdict="observable" if passed else "unobservable",
        evidence={
            "observability_matrix": numeric_matrix,
            "observability_matrix_exact": exact_matrix,
            "pivot_columns": exact_pivots if exact_pivots is not None else numerical_pivots,
            "pbh": pbh,
        },
        diagnostics={
            "n": system.n,
            "p": system.p,
            "exact_rank": exact_rank,
            "numerical_rank": numeric_rank.rank,
            "singular_values": numeric_rank.singular_values,
            "smallest_singular_value": smallest,
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
