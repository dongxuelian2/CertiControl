"""Continuous-time Lyapunov equation solvers and stability certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings as py_warnings

import numpy as np
import sympy as sp
from scipy import linalg

from .certificates import Certificate
from .linear_algebra import (
    exact_positive_definite,
    matrix_scale_tolerance,
    numerical_positive_definite,
)
from .model import LTISystem, MatrixLike
from .tolerance import TolerancePolicy

_VERIFICATION_MULTIPLIER = 100.0


@dataclass(frozen=True)
class LyapunovSolution:
    """Numerical solution and verification data for a continuous-time Lyapunov equation."""

    status: str
    Q: np.ndarray
    raw_P: np.ndarray | None
    P: np.ndarray | None
    symmetry_residual: float | None
    absolute_residual: float | None
    relative_residual: float | None
    solver_warnings: tuple[str, ...] = ()


def _numeric_from_sympy(matrix: sp.MatrixBase) -> np.ndarray:
    values = np.array(
        [[complex(sp.N(matrix[i, j], 17)) for j in range(matrix.cols)] for i in range(matrix.rows)],
        dtype=np.complex128,
    )
    if np.all(values.imag == 0):
        return values.real.astype(np.float64)
    return values


def _prepare_q(
    A: np.ndarray,
    Q: MatrixLike | None,
) -> tuple[np.ndarray, sp.ImmutableMatrix | None]:
    n = A.shape[0]
    if Q is None:
        numeric = np.eye(n, dtype=np.result_type(A.dtype, np.float64))
        return numeric, sp.ImmutableMatrix.eye(n)

    if isinstance(Q, sp.MatrixBase):
        symbolic = sp.ImmutableMatrix(Q)
    else:
        try:
            array_obj = np.asarray(Q, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError("Q must be a rectangular two-dimensional matrix.") from exc
        if array_obj.ndim != 2:
            raise ValueError(f"Q must be two-dimensional; got shape {array_obj.shape}.")
        try:
            symbolic = sp.ImmutableMatrix(array_obj.tolist())
        except (TypeError, ValueError) as exc:
            raise ValueError("Q contains values that cannot be interpreted as numbers.") from exc

    if symbolic.shape != (n, n):
        raise ValueError(f"Q must have shape ({n}, {n}); got shape {symbolic.shape}.")
    if not all(bool(entry.is_number) for entry in symbolic):
        raise ValueError("Q must contain only numeric entries.")
    numeric = _numeric_from_sympy(symbolic)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Q must contain only finite numeric values.")
    exact = symbolic if all(entry.is_Rational is True for entry in symbolic) else None
    return numeric, exact


def _verification_tolerance(matrix: np.ndarray) -> float:
    array = np.asarray(matrix)
    scale = max(1.0, float(linalg.norm(array, ord="fro")))
    return float(
        _VERIFICATION_MULTIPLIER
        * np.finfo(np.float64).eps
        * max(array.shape)
        * scale
    )


def _relative_residual(A: np.ndarray, P: np.ndarray, Q: np.ndarray) -> tuple[float, float]:
    AH = A.conj().T
    residual = AH @ P + P @ A + Q
    absolute = float(linalg.norm(residual, ord="fro"))
    denominator = (
        float(linalg.norm(AH @ P, ord="fro"))
        + float(linalg.norm(P @ A, ord="fro"))
        + float(linalg.norm(Q, ord="fro"))
    )
    relative = absolute / denominator if denominator > 0 else absolute
    return absolute, relative


def solve_lyapunov(
    A: MatrixLike,
    Q: MatrixLike | None = None,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> LyapunovSolution:
    """Solve ``A^* P + P A = -Q`` numerically and verify the returned solution.

    The returned ``P`` is the explicitly Hermitian-symmetrized matrix used by
    certificate checks. ``raw_P`` preserves the solver output before that step.
    Positive-definiteness is intentionally not assumed here.
    """
    policy = tolerance_policy or TolerancePolicy()
    A_array = np.asarray(A, dtype=np.complex128 if np.iscomplexobj(A) else np.float64)
    if A_array.ndim != 2 or A_array.shape[0] != A_array.shape[1]:
        raise ValueError(f"A must be square; got shape {A_array.shape}.")
    Q_array, _ = _prepare_q(A_array, Q)
    q_symmetry = float(linalg.norm(Q_array - Q_array.conj().T, ord="fro"))
    q_symmetry_tolerance = max(
        _verification_tolerance(Q_array),
        _VERIFICATION_MULTIPLIER * matrix_scale_tolerance(Q_array, policy),
    )
    if q_symmetry > q_symmetry_tolerance:
        raise ValueError("Q must be symmetric/Hermitian for Lyapunov stability analysis.")
    Q_used = (Q_array + Q_array.conj().T) / 2

    caught: list[str] = []
    try:
        with py_warnings.catch_warnings(record=True) as records:
            py_warnings.simplefilter("always")
            raw_P = linalg.solve_continuous_lyapunov(A_array.conj().T, -Q_used)
        caught = [str(record.message) for record in records]
    except (linalg.LinAlgError, ValueError) as exc:
        return LyapunovSolution(
            status="numerical_solver_failure",
            Q=Q_used,
            raw_P=None,
            P=None,
            symmetry_residual=None,
            absolute_residual=None,
            relative_residual=None,
            solver_warnings=(str(exc),),
        )

    symmetry_residual = float(linalg.norm(raw_P - raw_P.conj().T, ord="fro"))
    P = (raw_P + raw_P.conj().T) / 2
    absolute_residual, relative_residual = _relative_residual(A_array, P, Q_used)
    status = "numerical_solution_with_warning" if caught else "numerical_solution"
    return LyapunovSolution(
        status=status,
        Q=Q_used,
        raw_P=raw_P,
        P=P,
        symmetry_residual=symmetry_residual,
        absolute_residual=absolute_residual,
        relative_residual=relative_residual,
        solver_warnings=tuple(caught),
    )


def _exact_lyapunov_solve(
    A: sp.MatrixBase,
    Q: sp.MatrixBase,
) -> tuple[str, sp.Matrix | None, sp.Matrix | None]:
    """Solve the rational Lyapunov equation as an exact linear system."""
    A_mat = sp.Matrix(A)
    Q_mat = sp.Matrix(Q)
    n = A_mat.rows
    variable_count = n * n
    operator = sp.zeros(variable_count, variable_count)
    rhs = sp.zeros(variable_count, 1)

    # Column-major vectorization: index(i, j) = i + j*n.
    for j in range(n):
        for i in range(n):
            row = i + j * n
            rhs[row, 0] = -Q_mat[i, j]
            for k in range(n):
                operator[row, k + j * n] += A_mat[k, i]
                operator[row, i + k * n] += A_mat[k, j]

    rank_operator = operator.rank()
    augmented = operator.row_join(rhs)
    rank_augmented = augmented.rank()
    if rank_augmented > rank_operator:
        return "inconsistent", None, None
    if rank_operator < variable_count:
        return "nonunique", None, None

    solution_vector, parameters = operator.gauss_jordan_solve(rhs)
    if parameters.rows != 0:
        return "nonunique", None, None
    P = sp.Matrix(n, n, lambda i, j: sp.simplify(solution_vector[i + j * n]))
    residual = (A_mat.conjugate().T * P + P * A_mat + Q_mat).applyfunc(sp.simplify)
    return "unique", P, residual


def _invalid_q_certificate(
    system: LTISystem,
    Q_numeric: np.ndarray,
    Q_exact: sp.ImmutableMatrix | None,
    *,
    reason: str,
    code: str,
    diagnostics: dict[str, Any],
) -> Certificate:
    return Certificate(
        name="Continuous-time Lyapunov stability",
        passed=None,
        verdict="inconclusive",
        evidence={"Q": Q_numeric, "Q_exact": Q_exact, "P": None, "P_exact": None},
        diagnostics={
            "n": system.n,
            "solution_status": "not_attempted_invalid_q",
            "exact_certificate_passed": None,
            "numerical_certificate_passed": None,
            "warning_codes": [code],
            **diagnostics,
        },
        warnings=[reason],
    )


def analyze_lyapunov_stability(
    system: LTISystem,
    Q: MatrixLike | None = None,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> Certificate:
    """Construct exact/numerical continuous-time Lyapunov evidence for Hurwitz stability."""
    policy = tolerance_policy or TolerancePolicy()
    A = np.asarray(system.A)
    Q_numeric, Q_exact = _prepare_q(A, Q)
    warning_codes: list[str] = []
    warnings: list[str] = []

    q_symmetry_residual = float(linalg.norm(Q_numeric - Q_numeric.conj().T, ord="fro"))
    q_symmetry_tolerance = max(
        _verification_tolerance(Q_numeric),
        _VERIFICATION_MULTIPLIER * matrix_scale_tolerance(Q_numeric, policy),
    )
    if Q_exact is not None:
        exact_q_hermitian = sp.Matrix(Q_exact) == sp.Matrix(Q_exact).conjugate().T
    else:
        exact_q_hermitian = None
    if q_symmetry_residual > q_symmetry_tolerance or exact_q_hermitian is False:
        return _invalid_q_certificate(
            system,
            Q_numeric,
            Q_exact,
            reason="Q is not symmetric/Hermitian, so the classical positive-definite Lyapunov theorem does not apply.",
            code="q_not_hermitian",
            diagnostics={
                "q_symmetry_residual": q_symmetry_residual,
                "q_symmetry_tolerance": q_symmetry_tolerance,
            },
        )

    Q_used = (Q_numeric + Q_numeric.conj().T) / 2
    q_pd = numerical_positive_definite(Q_used, policy)
    exact_q_pd: bool | None = None
    exact_q_minors: tuple[sp.Expr, ...] | None = None
    if Q_exact is not None:
        exact_q_pd, exact_q_minors = exact_positive_definite(Q_exact)

    if exact_q_pd is False or (exact_q_pd is None and q_pd.verdict is not True):
        code = "q_not_positive_definite" if q_pd.verdict is False or exact_q_pd is False else "q_positive_definiteness_boundary"
        reason = (
            "Q is not positive definite, so it cannot be used for a classical Hurwitz Lyapunov certificate."
            if code == "q_not_positive_definite"
            else "Positive-definiteness of Q is tolerance-sensitive, so the Lyapunov theorem precondition is inconclusive."
        )
        return _invalid_q_certificate(
            system,
            Q_used,
            Q_exact,
            reason=reason,
            code=code,
            diagnostics={
                "q_symmetry_residual": q_symmetry_residual,
                "q_symmetry_tolerance": q_symmetry_tolerance,
                "Q_eigenvalues": q_pd.eigenvalues,
                "lambda_min_Q": q_pd.lambda_min,
                "Q_positive_definite_tolerance": q_pd.tolerance,
                "Q_numerical_positive_definite": q_pd.verdict,
                "Q_exact_positive_definite": exact_q_pd,
                "Q_leading_principal_minors": exact_q_minors,
            },
        )

    numerical_solution = solve_lyapunov(A, Q_used, tolerance_policy=policy)
    P = numerical_solution.P
    p_pd = None
    residual_ok: bool | None = None
    symmetry_ok: bool | None = None
    numerical_certificate: bool | None = None
    residual_tolerance: float | None = None
    if P is not None:
        p_pd = numerical_positive_definite(P, policy)
        residual_tolerance = float(
            _VERIFICATION_MULTIPLIER
            * np.finfo(np.float64).eps
            * max(1, system.n)
        )
        residual_ok = (
            numerical_solution.relative_residual is not None
            and numerical_solution.relative_residual <= residual_tolerance
        )
        symmetry_tolerance = _verification_tolerance(P)
        symmetry_ok = (
            numerical_solution.symmetry_residual is not None
            and numerical_solution.symmetry_residual <= symmetry_tolerance
        )
        if not residual_ok:
            warning_codes.append("lyapunov_residual_large")
            warnings.append("The numerical Lyapunov equation residual is too large for a reliable certificate.")
        if not symmetry_ok:
            warning_codes.append("p_symmetry_residual_large")
            warnings.append("The raw numerical Lyapunov solution has an unexpectedly large symmetry residual.")
        if p_pd.verdict is None:
            warning_codes.append("p_near_semidefinite")
            warnings.append("Positive-definiteness of P is tolerance-sensitive.")
        if residual_ok and symmetry_ok:
            numerical_certificate = p_pd.verdict
    else:
        warning_codes.append("solver_failure")
        warnings.append("The numerical Lyapunov solver did not return a usable solution.")

    for solver_warning in numerical_solution.solver_warnings:
        warning_codes.append("solver_warning")
        warnings.append(f"Lyapunov solver warning: {solver_warning}")

    exact_status: str | None = None
    exact_P: sp.Matrix | None = None
    exact_residual: sp.Matrix | None = None
    exact_p_symmetric: bool | None = None
    exact_p_pd: bool | None = None
    exact_p_minors: tuple[sp.Expr, ...] | None = None
    exact_certificate: bool | None = None
    if system.exact_A is not None and Q_exact is not None:
        exact_status, exact_P, exact_residual = _exact_lyapunov_solve(system.exact_A, Q_exact)
        if exact_status == "unique" and exact_P is not None and exact_residual is not None:
            exact_p_symmetric = exact_P == exact_P.conjugate().T
            residual_zero = exact_residual == sp.zeros(system.n, system.n)
            if exact_p_symmetric:
                exact_p_pd, exact_p_minors = exact_positive_definite(exact_P)
                exact_certificate = bool(exact_q_pd and exact_p_pd and residual_zero)
            else:
                warning_codes.append("exact_p_not_symmetric")
                warnings.append("The exact Lyapunov solution is not symmetric/Hermitian; no exact PD certificate is issued.")
        elif exact_q_pd is True:
            # For Q > 0, a Hurwitz A would have a unique positive-definite solution.
            exact_certificate = False

    if exact_certificate is not None and numerical_certificate is not None and exact_certificate != numerical_certificate:
        warning_codes.append("exact_numerical_disagreement")
        warnings.append(
            "Exact and numerical Lyapunov certificate classifications disagree; preserve the exact algebraic result and inspect numerical margins."
        )

    passed = exact_certificate if exact_certificate is not None else numerical_certificate
    if passed is True:
        verdict = "hurwitz_certificate"
    elif passed is False:
        verdict = "no_positive_definite_lyapunov_certificate"
    else:
        verdict = "inconclusive"

    return Certificate(
        name="Continuous-time Lyapunov stability",
        passed=passed,
        verdict=verdict,
        evidence={
            "A": A,
            "Q": Q_used,
            "Q_exact": Q_exact,
            "raw_P": numerical_solution.raw_P,
            "P": P,
            "P_exact": exact_P,
            "exact_residual_matrix": exact_residual,
        },
        diagnostics={
            "n": system.n,
            "solution_status": numerical_solution.status,
            "exact_solution_status": exact_status,
            "q_symmetry_residual": q_symmetry_residual,
            "q_symmetry_tolerance": q_symmetry_tolerance,
            "Q_eigenvalues": q_pd.eigenvalues,
            "lambda_min_Q": q_pd.lambda_min,
            "Q_positive_definite_tolerance": q_pd.tolerance,
            "Q_numerical_positive_definite": q_pd.verdict,
            "Q_exact_positive_definite": exact_q_pd,
            "Q_leading_principal_minors": exact_q_minors,
            "P_eigenvalues": None if p_pd is None else p_pd.eigenvalues,
            "lambda_min_P": None if p_pd is None else p_pd.lambda_min,
            "P_positive_definite_tolerance": None if p_pd is None else p_pd.tolerance,
            "P_numerical_positive_definite": None if p_pd is None else p_pd.verdict,
            "P_cholesky_success": None if p_pd is None else p_pd.cholesky_success,
            "P_exact_symmetric": exact_p_symmetric,
            "P_exact_positive_definite": exact_p_pd,
            "P_leading_principal_minors": exact_p_minors,
            "symmetry_residual": numerical_solution.symmetry_residual,
            "absolute_residual": numerical_solution.absolute_residual,
            "relative_residual": numerical_solution.relative_residual,
            "relative_residual_tolerance": residual_tolerance,
            "residual_ok": residual_ok,
            "symmetry_ok": symmetry_ok,
            "numerical_certificate_passed": numerical_certificate,
            "exact_certificate_passed": exact_certificate,
            "warning_codes": warning_codes,
        },
        warnings=warnings,
    )
