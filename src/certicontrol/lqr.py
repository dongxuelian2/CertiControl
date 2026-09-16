"""Continuous-time infinite-horizon LQR / CARE certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings as py_warnings

import numpy as np
import sympy as sp
from scipy import linalg

from .certificates import Certificate
from .controllability import pbh_controllability
from .linear_algebra import (
    exact_positive_definite,
    exact_positive_semidefinite,
    matrix_scale_tolerance,
    numerical_positive_definite,
    numerical_positive_semidefinite,
)
from .lyapunov import analyze_lyapunov_stability
from .model import LTISystem, MatrixLike
from .observability import pbh_observability
from .stability import analyze_spectral_stability
from .tolerance import TolerancePolicy

_RESIDUAL_MULTIPLIER = 100.0


@dataclass(frozen=True)
class CARESolution:
    """Numerical CARE solution plus independent verification quantities."""

    status: str
    raw_P: np.ndarray | None
    P: np.ndarray | None
    K: np.ndarray | None
    A_closed_loop: np.ndarray | None
    symmetry_residual: float | None
    gain_absolute_residual: float | None
    gain_relative_residual: float | None
    care_absolute_residual: float | None
    care_relative_residual: float | None
    solver_warnings: tuple[str, ...] = ()


def _numeric_from_sympy(matrix: sp.MatrixBase) -> np.ndarray:
    values = np.array(
        [[complex(sp.N(matrix[i, j], 17)) for j in range(matrix.cols)] for i in range(matrix.rows)],
        dtype=np.complex128,
    )
    if np.all(values.imag == 0):
        return values.real.astype(np.float64)
    return values


def _coerce_weight(
    value: MatrixLike,
    *,
    name: str,
    shape: tuple[int, int],
) -> tuple[np.ndarray, sp.ImmutableMatrix | None]:
    if isinstance(value, sp.MatrixBase):
        symbolic = sp.ImmutableMatrix(value)
    else:
        try:
            array_obj = np.asarray(value, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a rectangular two-dimensional matrix.") from exc
        if array_obj.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional; got shape {array_obj.shape}.")
        try:
            symbolic = sp.ImmutableMatrix(array_obj.tolist())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains values that cannot be interpreted as numbers.") from exc

    if symbolic.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got shape {symbolic.shape}.")
    if not all(bool(entry.is_number) for entry in symbolic):
        raise ValueError(f"{name} must contain only numeric entries.")
    numeric = _numeric_from_sympy(symbolic)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    exact = symbolic if all(entry.is_Rational is True for entry in symbolic) else None
    return numeric, exact


def _symmetry_tolerance(matrix: np.ndarray, policy: TolerancePolicy) -> float:
    scale_term = _RESIDUAL_MULTIPLIER * matrix_scale_tolerance(matrix, policy)
    norm_term = (
        _RESIDUAL_MULTIPLIER
        * np.finfo(np.float64).eps
        * max(matrix.shape)
        * max(1.0, float(linalg.norm(matrix, ord="fro")))
    )
    return float(max(scale_term, norm_term))


def _residual_tolerance(size: int, policy: TolerancePolicy) -> float:
    if policy.absolute is not None:
        return float(max(policy.absolute, _RESIDUAL_MULTIPLIER * np.finfo(np.float64).eps * max(1, size)))
    return float(
        policy.multiplier
        * _RESIDUAL_MULTIPLIER
        * np.finfo(np.float64).eps
        * max(1, size)
    )


def _relative_residual(residual: np.ndarray, terms: list[np.ndarray]) -> tuple[float, float]:
    absolute = float(linalg.norm(residual, ord="fro"))
    denominator = sum(float(linalg.norm(term, ord="fro")) for term in terms)
    relative = absolute / denominator if denominator > 0 else absolute
    return absolute, relative


def _validate_weight(
    value: MatrixLike,
    *,
    name: str,
    shape: tuple[int, int],
    require_pd: bool,
    policy: TolerancePolicy,
) -> dict[str, Any]:
    numeric, exact = _coerce_weight(value, name=name, shape=shape)
    symmetry_residual = float(linalg.norm(numeric - numeric.conj().T, ord="fro"))
    symmetry_tolerance = _symmetry_tolerance(numeric, policy)
    exact_hermitian = None if exact is None else sp.Matrix(exact) == sp.Matrix(exact).conjugate().T
    hermitian = symmetry_residual <= symmetry_tolerance and exact_hermitian is not False

    result: dict[str, Any] = {
        "numeric": numeric,
        "exact": exact,
        "used": None,
        "hermitian": hermitian,
        "symmetry_residual": symmetry_residual,
        "symmetry_tolerance": symmetry_tolerance,
        "numerical_status": None,
        "numerical_classification": None,
        "numerical_eigenvalues": None,
        "lambda_min": None,
        "positivity_tolerance": None,
        "exact_status": None,
        "exact_principal_minors": None,
        "final_status": False if not hermitian else None,
        "warning_codes": [],
    }
    if not hermitian:
        result["warning_codes"].append(f"{name.upper()}_NOT_SYMMETRIC")
        return result

    used = (numeric + numeric.conj().T) / 2
    result["used"] = used

    if require_pd:
        numerical = numerical_positive_definite(used, policy)
        numerical_status = numerical.verdict
        numerical_classification = (
            "POSITIVE_DEFINITE"
            if numerical.verdict is True
            else "NOT_POSITIVE_DEFINITE"
            if numerical.verdict is False
            else "TOLERANCE_SENSITIVE"
        )
        # An exactly zero floating eigenvalue is direct evidence of singularity,
        # not merely a near-boundary positive-definiteness decision.
        if exact is None and numerical.lambda_min == 0.0:
            numerical_status = False
            numerical_classification = "NOT_POSITIVE_DEFINITE"
        result.update(
            {
                "numerical_status": numerical_status,
                "numerical_classification": numerical_classification,
                "numerical_eigenvalues": numerical.eigenvalues,
                "lambda_min": numerical.lambda_min,
                "positivity_tolerance": numerical.tolerance,
                "cholesky_success": numerical.cholesky_success,
            }
        )
        if exact is not None:
            exact_status, minors = exact_positive_definite(exact)
            result["exact_status"] = exact_status
            result["exact_principal_minors"] = minors
        final = result["exact_status"] if result["exact_status"] is not None else result["numerical_status"]
        result["final_status"] = final
        if final is False:
            result["warning_codes"].append(f"{name.upper()}_NOT_PD")
        elif final is None:
            result["warning_codes"].append(f"{name.upper()}_PD_TOLERANCE_SENSITIVE")
    else:
        numerical = numerical_positive_semidefinite(used, policy)
        result.update(
            {
                "numerical_status": numerical.verdict,
                "numerical_classification": numerical.classification,
                "numerical_eigenvalues": numerical.eigenvalues,
                "lambda_min": numerical.lambda_min,
                "positivity_tolerance": numerical.tolerance,
            }
        )
        if exact is not None:
            exact_status, minors = exact_positive_semidefinite(exact)
            result["exact_status"] = exact_status
            result["exact_principal_minors"] = minors
        final = result["exact_status"] if result["exact_status"] is not None else numerical.verdict
        result["final_status"] = final
        if final is False:
            result["warning_codes"].append(f"{name.upper()}_NOT_PSD")
        elif final is None:
            result["warning_codes"].append(f"{name.upper()}_PSD_TOLERANCE_SENSITIVE")

    if (
        result["exact_status"] is not None
        and result["numerical_status"] is not None
        and result["exact_status"] != result["numerical_status"]
    ):
        result["warning_codes"].append("EXACT_NUMERICAL_DISAGREEMENT")
    return result


def _classify_exact_mode(eigenvalue: sp.Expr, tolerance: float) -> tuple[str, str, float]:
    real_expr = sp.simplify(sp.re(eigenvalue))
    numerical_real = float(sp.N(real_expr, 50))
    if real_expr.is_positive is True:
        return "nonstable", "exact_sign", numerical_real
    if real_expr.is_zero is True:
        return "nonstable", "exact_zero", numerical_real
    if real_expr.is_negative is True:
        return "stable", "exact_sign", numerical_real
    if numerical_real > tolerance:
        return "nonstable", "numerical_location_of_exact_mode", numerical_real
    if numerical_real < -tolerance:
        return "stable", "numerical_location_of_exact_mode", numerical_real
    return "boundary", "numerical_location_of_exact_mode", numerical_real


def _classify_numerical_mode(eigenvalue: complex, tolerance: float) -> tuple[str, float]:
    real_part = float(np.real(eigenvalue))
    if real_part > tolerance:
        return "nonstable", real_part
    if real_part < -tolerance:
        return "stable", real_part
    return "boundary", real_part


def analyze_stabilizability(
    system: LTISystem,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> Certificate:
    """Check whether every uncontrollable mode is strictly stable."""
    policy = tolerance_policy or TolerancePolicy()
    spectral = analyze_spectral_stability(system, tolerance_policy=policy)
    tolerance = float(spectral.diagnostics["stability_tolerance"])
    pbh = pbh_controllability(system, policy)

    numerical_offending: list[dict[str, Any]] = []
    numerical_boundary: list[dict[str, Any]] = []
    for failure in pbh.evidence["numerical_failures"] or []:
        location, real_part = _classify_numerical_mode(failure["eigenvalue"], tolerance)
        item = dict(failure)
        item["real_part"] = real_part
        item["location"] = location
        if location == "nonstable":
            numerical_offending.append(item)
        elif location == "boundary":
            numerical_boundary.append(item)
    numerical_status: bool | None
    if numerical_offending:
        numerical_status = False
    elif numerical_boundary:
        numerical_status = None
    else:
        numerical_status = True

    exact_status: bool | None = None
    exact_offending: list[dict[str, Any]] | None = None
    exact_boundary: list[dict[str, Any]] | None = None
    exact_failures = pbh.evidence.get("exact_failures")
    if exact_failures is not None:
        exact_offending = []
        exact_boundary = []
        for failure in exact_failures:
            location, source, real_part = _classify_exact_mode(failure["eigenvalue"], tolerance)
            item = dict(failure)
            item["real_part"] = real_part
            item["location"] = location
            item["location_source"] = source
            if location == "nonstable":
                exact_offending.append(item)
            elif location == "boundary":
                exact_boundary.append(item)
        if exact_offending:
            exact_status = False
        elif exact_boundary:
            exact_status = None
        else:
            exact_status = True

    passed = exact_status if exact_failures is not None else numerical_status
    warning_codes: list[str] = []
    warnings: list[str] = []
    if passed is False:
        warning_codes.append("NOT_STABILIZABLE")
        warnings.append("At least one uncontrollable mode is nondecaying or unstable.")
    elif passed is None:
        warning_codes.append("STABILIZABILITY_BOUNDARY")
        warnings.append("An uncontrollable mode lies numerically near the imaginary axis.")
    if exact_status is not None and numerical_status is not None and exact_status != numerical_status:
        warning_codes.append("EXACT_NUMERICAL_DISAGREEMENT")
        warnings.append("Exact PBH evidence and numerical stabilizability classification disagree.")

    verdict = "STABILIZABLE" if passed is True else "NOT_STABILIZABLE" if passed is False else "BOUNDARY_OR_INCONCLUSIVE"
    return Certificate(
        name="Continuous-time stabilizability",
        passed=passed,
        verdict=verdict,
        evidence={
            "pbh_controllability": pbh,
            "exact_offending_modes": exact_offending,
            "exact_boundary_modes": exact_boundary,
            "numerical_offending_modes": numerical_offending,
            "numerical_boundary_modes": numerical_boundary,
        },
        diagnostics={
            "stability_tolerance": tolerance,
            "exact_passed": exact_status,
            "numerical_passed": numerical_status,
            "warning_codes": warning_codes,
        },
        warnings=warnings,
    )


def analyze_detectability(
    system: LTISystem,
    Q: MatrixLike,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> Certificate:
    """Check LQR detectability using Q as a kernel-equivalent output map.

    For Q >= 0, ker(Q) = ker(Q^(1/2)), so PBH visibility of an eigenvector
    under Q is equivalent to visibility under the usual Q^(1/2) output map.
    """
    policy = tolerance_policy or TolerancePolicy()
    try:
        q_info = _validate_weight(Q, name="Q", shape=(system.n, system.n), require_pd=False, policy=policy)
    except ValueError as exc:
        return Certificate(
            name="Continuous-time LQR detectability",
            passed=False,
            verdict="INVALID_Q",
            evidence={},
            diagnostics={"warning_codes": ["Q_INVALID_SHAPE_OR_VALUE"]},
            warnings=[str(exc)],
        )
    if q_info["final_status"] is not True:
        code = q_info["warning_codes"][0] if q_info["warning_codes"] else "Q_PSD_TOLERANCE_SENSITIVE"
        return Certificate(
            name="Continuous-time LQR detectability",
            passed=False if q_info["final_status"] is False else None,
            verdict="INVALID_Q" if q_info["final_status"] is False else "BOUNDARY_OR_INCONCLUSIVE",
            evidence={"Q": q_info["numeric"], "Q_exact": q_info["exact"]},
            diagnostics={"Q_psd_status": q_info["final_status"], "warning_codes": [code]},
            warnings=["Detectability prerequisite requires a symmetric/Hermitian positive-semidefinite Q."],
        )

    A_input: MatrixLike = system.exact_A if system.exact_A is not None else system.A
    B_input: MatrixLike = system.exact_B if system.exact_B is not None else system.B
    C_input: MatrixLike = q_info["exact"] if q_info["exact"] is not None else q_info["used"]
    q_system = LTISystem(A_input, B_input, C=C_input)
    spectral = analyze_spectral_stability(system, tolerance_policy=policy)
    tolerance = float(spectral.diagnostics["stability_tolerance"])
    pbh = pbh_observability(q_system, policy)

    numerical_offending: list[dict[str, Any]] = []
    numerical_boundary: list[dict[str, Any]] = []
    for failure in pbh.evidence["numerical_failures"] or []:
        location, real_part = _classify_numerical_mode(failure["eigenvalue"], tolerance)
        item = dict(failure)
        item["state_cost_visibility_residual"] = item.get("output_null_residual")
        item["real_part"] = real_part
        item["location"] = location
        if location == "nonstable":
            numerical_offending.append(item)
        elif location == "boundary":
            numerical_boundary.append(item)
    numerical_status: bool | None
    if numerical_offending:
        numerical_status = False
    elif numerical_boundary:
        numerical_status = None
    else:
        numerical_status = True

    exact_status: bool | None = None
    exact_offending: list[dict[str, Any]] | None = None
    exact_boundary: list[dict[str, Any]] | None = None
    exact_failures = pbh.evidence.get("exact_failures")
    if exact_failures is not None:
        exact_offending = []
        exact_boundary = []
        for failure in exact_failures:
            location, source, real_part = _classify_exact_mode(failure["eigenvalue"], tolerance)
            item = dict(failure)
            item["state_cost_visibility_residual"] = item.get("output_null_residual")
            item["real_part"] = real_part
            item["location"] = location
            item["location_source"] = source
            if location == "nonstable":
                exact_offending.append(item)
            elif location == "boundary":
                exact_boundary.append(item)
        if exact_offending:
            exact_status = False
        elif exact_boundary:
            exact_status = None
        else:
            exact_status = True

    passed = exact_status if exact_failures is not None else numerical_status
    warning_codes: list[str] = []
    warnings: list[str] = []
    if passed is False:
        warning_codes.append("NOT_DETECTABLE")
        warnings.append("At least one nondecaying or unstable mode lies in the kernel of Q.")
    elif passed is None:
        warning_codes.append("DETECTABILITY_BOUNDARY")
        warnings.append("A Q-invisible mode lies numerically near the imaginary axis.")
    if exact_status is not None and numerical_status is not None and exact_status != numerical_status:
        warning_codes.append("EXACT_NUMERICAL_DISAGREEMENT")
        warnings.append("Exact PBH evidence and numerical detectability classification disagree.")

    verdict = "DETECTABLE" if passed is True else "NOT_DETECTABLE" if passed is False else "BOUNDARY_OR_INCONCLUSIVE"
    return Certificate(
        name="Continuous-time LQR detectability",
        passed=passed,
        verdict=verdict,
        evidence={
            "Q": q_info["used"],
            "Q_exact": q_info["exact"],
            "kernel_equivalent_output_map": q_info["used"],
            "pbh_observability": pbh,
            "exact_offending_modes": exact_offending,
            "exact_boundary_modes": exact_boundary,
            "numerical_offending_modes": numerical_offending,
            "numerical_boundary_modes": numerical_boundary,
        },
        diagnostics={
            "stability_tolerance": tolerance,
            "exact_passed": exact_status,
            "numerical_passed": numerical_status,
            "warning_codes": warning_codes,
        },
        warnings=warnings,
    )


def solve_care(
    A: MatrixLike,
    B: MatrixLike,
    Q: MatrixLike,
    R: MatrixLike,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> CARESolution:
    """Solve the standard continuous-time CARE and independently verify it."""
    policy = tolerance_policy or TolerancePolicy()
    A_array = np.asarray(A, dtype=np.complex128 if np.iscomplexobj(A) else np.float64)
    B_array = np.asarray(B, dtype=np.complex128 if np.iscomplexobj(B) else np.float64)
    if A_array.ndim != 2 or A_array.shape[0] != A_array.shape[1]:
        raise ValueError(f"A must be square; got shape {A_array.shape}.")
    n = A_array.shape[0]
    if B_array.ndim != 2 or B_array.shape[0] != n:
        raise ValueError(f"B must have {n} rows; got shape {B_array.shape}.")
    m = B_array.shape[1]
    q_info = _validate_weight(Q, name="Q", shape=(n, n), require_pd=False, policy=policy)
    r_info = _validate_weight(R, name="R", shape=(m, m), require_pd=True, policy=policy)
    if q_info["final_status"] is not True:
        raise ValueError("Q must be symmetric/Hermitian positive semidefinite for CARE synthesis.")
    if r_info["final_status"] is not True:
        raise ValueError("R must be symmetric/Hermitian positive definite for CARE synthesis.")
    Q_used = q_info["used"]
    R_used = r_info["used"]

    caught: list[str] = []
    try:
        with py_warnings.catch_warnings(record=True) as records:
            py_warnings.simplefilter("always")
            raw_P = linalg.solve_continuous_are(A_array, B_array, Q_used, R_used)
        caught = [str(record.message) for record in records]
        symmetry_residual = float(linalg.norm(raw_P - raw_P.conj().T, ord="fro"))
        P = (raw_P + raw_P.conj().T) / 2
        target = B_array.conj().T @ P
        K = linalg.solve(R_used, target, assume_a="her")
        gain_residual_matrix = R_used @ K - target
        gain_absolute, gain_relative = _relative_residual(gain_residual_matrix, [R_used @ K, target])
        PBK = P @ B_array @ K
        care_residual = A_array.conj().T @ P + P @ A_array - PBK + Q_used
        care_absolute, care_relative = _relative_residual(
            care_residual,
            [A_array.conj().T @ P, P @ A_array, PBK, Q_used],
        )
        A_closed_loop = A_array - B_array @ K
    except (linalg.LinAlgError, ValueError) as exc:
        return CARESolution(
            status="FAILED",
            raw_P=None,
            P=None,
            K=None,
            A_closed_loop=None,
            symmetry_residual=None,
            gain_absolute_residual=None,
            gain_relative_residual=None,
            care_absolute_residual=None,
            care_relative_residual=None,
            solver_warnings=(str(exc),),
        )

    return CARESolution(
        status="SOLVED_WITH_WARNING" if caught else "SOLVED",
        raw_P=raw_P,
        P=P,
        K=K,
        A_closed_loop=A_closed_loop,
        symmetry_residual=symmetry_residual,
        gain_absolute_residual=gain_absolute,
        gain_relative_residual=gain_relative,
        care_absolute_residual=care_absolute,
        care_relative_residual=care_relative,
        solver_warnings=tuple(caught),
    )


def _invalid_lqr_certificate(
    system: LTISystem,
    *,
    status: str,
    warning_codes: list[str],
    warnings: list[str],
    evidence: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    passed: bool | None = False,
) -> Certificate:
    return Certificate(
        name="Continuous-time infinite-horizon LQR",
        passed=passed,
        verdict=status,
        evidence={"A": system.A, "B": system.B, **(evidence or {})},
        diagnostics={
            "n": system.n,
            "m": system.m,
            "status": status,
            "warning_codes": warning_codes,
            **(diagnostics or {}),
        },
        warnings=warnings,
    )


def analyze_lqr(
    system: LTISystem,
    Q: MatrixLike,
    R: MatrixLike,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> Certificate:
    """Build a continuous-time infinite-horizon LQR / CARE certificate."""
    policy = tolerance_policy or TolerancePolicy()
    try:
        q_info = _validate_weight(Q, name="Q", shape=(system.n, system.n), require_pd=False, policy=policy)
        r_info = _validate_weight(R, name="R", shape=(system.m, system.m), require_pd=True, policy=policy)
    except ValueError as exc:
        return _invalid_lqr_certificate(
            system,
            status="INVALID_INPUT",
            warning_codes=["INVALID_WEIGHT_DIMENSIONS_OR_VALUES"],
            warnings=[str(exc)],
        )

    weight_codes = list(q_info["warning_codes"]) + list(r_info["warning_codes"])
    weight_warnings: list[str] = []
    if q_info["final_status"] is False:
        weight_warnings.append("Q must be symmetric/Hermitian positive semidefinite.")
    elif q_info["final_status"] is None:
        weight_warnings.append("Positive-semidefiniteness of Q is tolerance-sensitive.")
    if r_info["final_status"] is False:
        weight_warnings.append("R must be symmetric/Hermitian positive definite.")
    elif r_info["final_status"] is None:
        weight_warnings.append("Positive-definiteness of R is tolerance-sensitive.")

    common_weight_diagnostics = {
        "Q_hermitian": q_info["hermitian"],
        "Q_symmetry_residual": q_info["symmetry_residual"],
        "Q_symmetry_tolerance": q_info["symmetry_tolerance"],
        "Q_psd_status": q_info["final_status"],
        "Q_numerical_classification": q_info["numerical_classification"],
        "Q_eigenvalues": q_info["numerical_eigenvalues"],
        "Q_lambda_min": q_info["lambda_min"],
        "Q_psd_tolerance": q_info["positivity_tolerance"],
        "Q_exact_psd_status": q_info["exact_status"],
        "Q_exact_principal_minors": q_info["exact_principal_minors"],
        "R_hermitian": r_info["hermitian"],
        "R_symmetry_residual": r_info["symmetry_residual"],
        "R_symmetry_tolerance": r_info["symmetry_tolerance"],
        "R_pd_status": r_info["final_status"],
        "R_eigenvalues": r_info["numerical_eigenvalues"],
        "R_lambda_min": r_info["lambda_min"],
        "R_pd_tolerance": r_info["positivity_tolerance"],
        "R_exact_pd_status": r_info["exact_status"],
        "R_exact_leading_principal_minors": r_info["exact_principal_minors"],
    }
    weight_evidence = {
        "Q": q_info["used"] if q_info["used"] is not None else q_info["numeric"],
        "Q_exact": q_info["exact"],
        "R": r_info["used"] if r_info["used"] is not None else r_info["numeric"],
        "R_exact": r_info["exact"],
    }

    if q_info["final_status"] is False or r_info["final_status"] is False:
        return _invalid_lqr_certificate(
            system,
            status="INVALID_INPUT",
            warning_codes=weight_codes,
            warnings=weight_warnings,
            evidence=weight_evidence,
            diagnostics=common_weight_diagnostics,
        )
    if q_info["final_status"] is None or r_info["final_status"] is None:
        return _invalid_lqr_certificate(
            system,
            status="BOUNDARY_OR_INCONCLUSIVE",
            warning_codes=weight_codes,
            warnings=weight_warnings,
            evidence=weight_evidence,
            diagnostics=common_weight_diagnostics,
            passed=None,
        )

    stabilizability = analyze_stabilizability(system, tolerance_policy=policy)
    detectability = analyze_detectability(
        system,
        q_info["exact"] if q_info["exact"] is not None else q_info["used"],
        tolerance_policy=policy,
    )
    if stabilizability.passed is False or detectability.passed is False:
        codes = list(stabilizability.diagnostics.get("warning_codes", [])) + list(detectability.diagnostics.get("warning_codes", []))
        return _invalid_lqr_certificate(
            system,
            status="PREREQUISITE_FAILURE",
            warning_codes=codes,
            warnings=stabilizability.warnings + detectability.warnings,
            evidence={**weight_evidence, "stabilizability": stabilizability, "detectability": detectability},
            diagnostics={
                **common_weight_diagnostics,
                "stabilizable": stabilizability.passed,
                "detectable": detectability.passed,
            },
        )
    if stabilizability.passed is None or detectability.passed is None:
        codes = list(stabilizability.diagnostics.get("warning_codes", [])) + list(detectability.diagnostics.get("warning_codes", []))
        return _invalid_lqr_certificate(
            system,
            status="BOUNDARY_OR_INCONCLUSIVE",
            warning_codes=codes,
            warnings=stabilizability.warnings + detectability.warnings,
            evidence={**weight_evidence, "stabilizability": stabilizability, "detectability": detectability},
            diagnostics={
                **common_weight_diagnostics,
                "stabilizable": stabilizability.passed,
                "detectable": detectability.passed,
            },
            passed=None,
        )

    care = solve_care(system.A, system.B, q_info["used"], r_info["used"], tolerance_policy=policy)
    if care.P is None or care.K is None or care.A_closed_loop is None:
        return _invalid_lqr_certificate(
            system,
            status="SOLVER_FAILURE",
            warning_codes=["CARE_SOLVER_FAILED"],
            warnings=["The continuous-time CARE solver failed.", *care.solver_warnings],
            evidence={**weight_evidence, "stabilizability": stabilizability, "detectability": detectability},
            diagnostics={
                **common_weight_diagnostics,
                "stabilizable": True,
                "detectable": True,
                "solver_status": care.status,
            },
        )

    P = care.P
    K = care.K
    Acl = care.A_closed_loop
    residual_tolerance = _residual_tolerance(system.n, policy)
    p_symmetry_tolerance = _symmetry_tolerance(P, policy)
    p_symmetry_ok = bool(care.symmetry_residual is not None and care.symmetry_residual <= p_symmetry_tolerance)
    p_psd = numerical_positive_semidefinite(P, policy)
    care_residual_ok = bool(care.care_relative_residual is not None and care.care_relative_residual <= residual_tolerance)
    gain_consistency_ok = bool(care.gain_relative_residual is not None and care.gain_relative_residual <= residual_tolerance)

    q_used = q_info["used"]
    r_used = r_info["used"]
    KHRK = K.conj().T @ r_used @ K
    q_closed = q_used + KHRK
    closed_residual = Acl.conj().T @ P + P @ Acl + q_closed
    closed_absolute, closed_relative = _relative_residual(
        closed_residual,
        [Acl.conj().T @ P, P @ Acl, q_used, KHRK],
    )
    closed_identity_ok = closed_relative <= residual_tolerance

    closed_system = LTISystem(Acl, system.B)
    closed_spectral = analyze_spectral_stability(closed_system, tolerance_policy=policy)
    q_closed_pd = numerical_positive_definite(q_closed, policy)
    closed_lyapunov = None
    care_p_vs_lyapunov_p_relative = None
    if q_closed_pd.verdict is True:
        closed_lyapunov = analyze_lyapunov_stability(closed_system, q_closed, tolerance_policy=policy)
        lyap_P = closed_lyapunov.evidence.get("P")
        if lyap_P is not None:
            difference = float(linalg.norm(P - lyap_P, ord="fro"))
            scale = max(1.0, float(linalg.norm(P, ord="fro")), float(linalg.norm(lyap_P, ord="fro")))
            care_p_vs_lyapunov_p_relative = difference / scale

    warning_codes = list(weight_codes)
    warnings = list(weight_warnings)
    warning_codes.extend(stabilizability.diagnostics.get("warning_codes", []))
    warning_codes.extend(detectability.diagnostics.get("warning_codes", []))
    warnings.extend(stabilizability.warnings)
    warnings.extend(detectability.warnings)
    if care.solver_warnings:
        warning_codes.append("CARE_SOLVER_WARNING")
        warnings.extend(f"CARE solver warning: {message}" for message in care.solver_warnings)
    if not p_symmetry_ok:
        warning_codes.append("P_NOT_SYMMETRIC")
        warnings.append("The raw CARE solution has an unexpectedly large Hermitian symmetry residual.")
    if p_psd.verdict is False:
        warning_codes.append("P_NOT_PSD")
        warnings.append("The symmetrized CARE solution P is not positive semidefinite.")
    elif p_psd.verdict is None:
        warning_codes.append("P_PSD_TOLERANCE_SENSITIVE")
        warnings.append("Positive-semidefiniteness of P is tolerance-sensitive.")
    if not care_residual_ok:
        warning_codes.append("CARE_RESIDUAL_LARGE")
        warnings.append("The independently computed CARE residual is too large.")
    if not gain_consistency_ok:
        warning_codes.append("GAIN_CONSISTENCY_RESIDUAL_LARGE")
        warnings.append("K is not numerically consistent with R K = B^* P under the active tolerance.")
    if closed_spectral.passed is False:
        warning_codes.append("CLOSED_LOOP_NOT_HURWITZ")
        warnings.append("The CARE candidate does not produce a Hurwitz closed loop.")
    elif closed_spectral.passed is None:
        warning_codes.append("CLOSED_LOOP_BOUNDARY")
        warnings.append("Closed-loop poles lie numerically near the imaginary axis.")
    if not closed_identity_ok:
        warning_codes.append("CLOSED_LOOP_IDENTITY_RESIDUAL_LARGE")
        warnings.append("The closed-loop Riccati/Lyapunov identity residual is too large.")
    if care_residual_ok != closed_identity_ok:
        warning_codes.append("INTERNAL_CONSISTENCY_WARNING")
        warnings.append("CARE and closed-loop identity residual checks disagree.")

    boundary = p_psd.verdict is None or closed_spectral.passed is None
    hard_failure = (
        not p_symmetry_ok
        or p_psd.verdict is False
        or not care_residual_ok
        or not gain_consistency_ok
        or closed_spectral.passed is False
        or not closed_identity_ok
    )
    if hard_failure:
        passed: bool | None = False
        status = "NUMERICAL_VERIFICATION_FAILURE"
    elif boundary:
        passed = None
        status = "BOUNDARY_OR_INCONCLUSIVE"
    else:
        passed = True
        status = "VALID_STABILIZING_LQR_CERTIFICATE"

    input_exact = bool(
        system.exact_A is not None
        and system.exact_B is not None
        and q_info["exact"] is not None
        and r_info["exact"] is not None
    )

    return Certificate(
        name="Continuous-time infinite-horizon LQR",
        passed=passed,
        verdict=status,
        evidence={
            "A": system.A,
            "B": system.B,
            "Q": q_used,
            "Q_exact": q_info["exact"],
            "R": r_used,
            "R_exact": r_info["exact"],
            "raw_P": care.raw_P,
            "P": P,
            "K": K,
            "A_closed_loop": Acl,
            "Q_closed_loop": q_closed,
            "stabilizability": stabilizability,
            "detectability": detectability,
            "closed_loop_spectral": closed_spectral,
            "closed_loop_lyapunov_cross_certificate": closed_lyapunov,
        },
        diagnostics={
            **common_weight_diagnostics,
            "status": status,
            "input_exact": input_exact,
            "care_solution_exact": False,
            "stabilizable": stabilizability.passed,
            "detectable": detectability.passed,
            "solver_status": care.status,
            "P_symmetry_residual": care.symmetry_residual,
            "P_symmetry_tolerance": p_symmetry_tolerance,
            "P_symmetry_ok": p_symmetry_ok,
            "P_eigenvalues": p_psd.eigenvalues,
            "P_min_eigenvalue": p_psd.lambda_min,
            "P_psd_tolerance": p_psd.tolerance,
            "P_psd_status": p_psd.verdict,
            "P_psd_classification": p_psd.classification,
            "gain_absolute_residual": care.gain_absolute_residual,
            "gain_relative_residual": care.gain_relative_residual,
            "gain_consistency_ok": gain_consistency_ok,
            "care_absolute_residual": care.care_absolute_residual,
            "care_relative_residual": care.care_relative_residual,
            "care_tolerance": residual_tolerance,
            "care_residual_ok": care_residual_ok,
            "K_shape": K.shape,
            "closed_loop_eigenvalues": closed_spectral.diagnostics["eigenvalues"],
            "closed_loop_spectral_abscissa": closed_spectral.diagnostics["spectral_abscissa"],
            "closed_loop_spectral_margin": closed_spectral.diagnostics["spectral_margin"],
            "closed_loop_hurwitz_status": closed_spectral.passed,
            "closed_loop_identity_absolute_residual": closed_absolute,
            "closed_loop_identity_relative_residual": closed_relative,
            "closed_loop_identity_residual_ok": closed_identity_ok,
            "Q_closed_loop_eigenvalues": q_closed_pd.eigenvalues,
            "Q_closed_loop_lambda_min": q_closed_pd.lambda_min,
            "Q_closed_loop_positive_definite": q_closed_pd.verdict,
            "care_P_vs_closed_loop_lyapunov_P_relative": care_p_vs_lyapunov_p_relative,
            "warning_codes": warning_codes,
        },
        warnings=warnings,
    )