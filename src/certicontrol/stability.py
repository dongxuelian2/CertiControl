"""Continuous-time spectral and combined Hurwitz stability analysis."""

from __future__ import annotations

import numpy as np
from scipy import linalg

from .certificates import Certificate
from .lyapunov import analyze_lyapunov_stability
from .model import LTISystem, MatrixLike
from .tolerance import TolerancePolicy

_SMALL_MARGIN_FACTOR = 100.0


def _spectral_tolerance(A: np.ndarray, policy: TolerancePolicy) -> float:
    if policy.absolute is not None:
        return float(policy.absolute)
    scale = max(1.0, float(linalg.norm(A, ord=2)))
    return float(
        policy.multiplier
        * scale
        * max(1, A.shape[0])
        * np.finfo(np.float64).eps
    )


def analyze_spectral_stability(
    system: LTISystem,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> Certificate:
    """Classify continuous-time Hurwitz stability from the spectrum of ``A``.

    The classification is deliberately three-way: Hurwitz, unstable, or
    boundary/tolerance-sensitive. No claim about general marginal Lyapunov
    stability is made at the imaginary-axis boundary.
    """
    policy = tolerance_policy or TolerancePolicy()
    A = np.asarray(system.A)
    eigenvalues = linalg.eigvals(A)
    real_parts = np.real(eigenvalues)
    spectral_abscissa = float(np.max(real_parts))
    tolerance = _spectral_tolerance(A, policy)
    distance_to_axis = float(np.min(np.abs(real_parts)))
    warning_codes: list[str] = []
    warnings: list[str] = []

    if spectral_abscissa < -tolerance:
        passed: bool | None = True
        verdict = "HURWITZ"
        spectral_margin = -spectral_abscissa
        if spectral_margin <= _SMALL_MARGIN_FACTOR * tolerance:
            warning_codes.append("small_stability_margin")
            warnings.append(
                "The system is close to the imaginary-axis stability boundary; the spectral verdict may be numerically sensitive."
            )
    elif spectral_abscissa > tolerance:
        passed = False
        verdict = "UNSTABLE"
        spectral_margin = None
    else:
        passed = None
        verdict = "BOUNDARY_OR_INCONCLUSIVE"
        spectral_margin = -spectral_abscissa if spectral_abscissa < 0 else 0.0
        warning_codes.append("spectral_boundary")
        warnings.append(
            "One or more eigenvalues lie on or numerically near the imaginary axis; Hurwitz classification is tolerance-sensitive."
        )

    dominant = [
        complex(value)
        for value in eigenvalues
        if abs(float(np.real(value)) - spectral_abscissa) <= tolerance
    ]

    characteristic_polynomial = None
    if system.exact_A is not None:
        characteristic_polynomial = system.exact_A.charpoly().as_expr()

    return Certificate(
        name="Continuous-time spectral stability",
        passed=passed,
        verdict=verdict,
        evidence={
            "A": A,
            "characteristic_polynomial_exact": characteristic_polynomial,
        },
        diagnostics={
            "n": system.n,
            "eigenvalues": eigenvalues,
            "real_parts": real_parts,
            "spectral_abscissa": spectral_abscissa,
            "stability_tolerance": tolerance,
            "spectral_margin": spectral_margin,
            "distance_to_imaginary_axis": distance_to_axis,
            "dominant_eigenvalues": dominant,
            "warning_codes": warning_codes,
        },
        warnings=warnings,
    )


def analyze_stability(
    system: LTISystem,
    Q: MatrixLike | None = None,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> Certificate:
    """Cross-check spectral Hurwitz diagnostics against a Lyapunov certificate."""
    policy = tolerance_policy or TolerancePolicy()
    spectral = analyze_spectral_stability(system, tolerance_policy=policy)
    lyapunov = analyze_lyapunov_stability(system, Q, tolerance_policy=policy)

    spectral_pass = spectral.passed
    lyapunov_pass = lyapunov.passed
    exact_lyapunov_pass = lyapunov.diagnostics.get("exact_certificate_passed")
    warning_codes: list[str] = []
    warnings: list[str] = []

    if spectral_pass is True and lyapunov_pass is True:
        cross_check = "CONSISTENT_PASS"
    elif spectral_pass is False and lyapunov_pass is False:
        cross_check = "CONSISTENT_FAIL"
    elif spectral_pass is None:
        if exact_lyapunov_pass is True:
            cross_check = "EXACT_PASS_NUMERICAL_BOUNDARY"
        elif exact_lyapunov_pass is False:
            cross_check = "EXACT_FAIL_NUMERICAL_BOUNDARY"
        else:
            cross_check = "BOUNDARY_OR_INCONCLUSIVE"
    elif lyapunov_pass is None:
        cross_check = "BOUNDARY_OR_INCONCLUSIVE"
    else:
        cross_check = "NUMERICAL_DISAGREEMENT"
        warning_codes.append("spectral_lyapunov_disagreement")
        warnings.append(
            "Spectral and Lyapunov classifications disagree; inspect margins, residuals, and tolerance diagnostics."
        )

    # Exact rational Lyapunov evidence is authoritative when available. For
    # numerical-only data, require consistency unless one criterion is simply
    # unavailable; never turn a spectral boundary into a numerical PASS.
    if exact_lyapunov_pass is not None:
        passed: bool | None = bool(exact_lyapunov_pass)
    elif spectral_pass is None:
        passed = None
    elif lyapunov_pass is None:
        passed = spectral_pass
    elif spectral_pass == lyapunov_pass:
        passed = spectral_pass
    else:
        passed = None

    if passed is True:
        verdict = "HURWITZ"
    elif passed is False:
        verdict = "NOT_HURWITZ"
    else:
        verdict = "BOUNDARY_OR_INCONCLUSIVE"

    return Certificate(
        name="Continuous-time Hurwitz stability",
        passed=passed,
        verdict=verdict,
        evidence={
            "spectral": spectral,
            "lyapunov": lyapunov,
        },
        diagnostics={
            "cross_check": cross_check,
            "spectral_passed": spectral_pass,
            "lyapunov_passed": lyapunov_pass,
            "exact_lyapunov_passed": exact_lyapunov_pass,
            "warning_codes": warning_codes,
        },
        warnings=warnings,
    )
