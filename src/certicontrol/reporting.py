"""Reporting helpers for aggregated CertiControl analyses.

Reporting consumes already computed certificates; it never recomputes control
properties.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import sympy as sp

from .analysis import (
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_INCONCLUSIVE,
    STATUS_NOT_RUN,
    STATUS_PASS,
    AnalysisRecord,
    SystemAnalysis,
)


WARNING_MESSAGES: dict[str, str] = {
    "REACHABLE_RANK_TOLERANCE_SENSITIVE": "Reachable-subspace rank is sensitive to the active numerical tolerance.",
    "UNOBSERVABLE_NULLITY_TOLERANCE_SENSITIVE": "Unobservable-subspace nullity is sensitive to the active numerical tolerance.",
    "SUBSPACE_INTERSECTION_TOLERANCE_SENSITIVE": "The dimension of R ∩ N is sensitive to the active numerical tolerance.",
    "KALMAN_BASIS_ILL_CONDITIONED": "The Kalman coordinate transformation is severely ill-conditioned.",
    "EXACT_NUMERICAL_DIMENSION_DISAGREEMENT": "Exact and numerical structural dimensions disagree; inspect exact evidence and numerical conditioning.",
    "CARE_RESIDUAL_LARGE": "The independently verified CARE residual is too large.",
    "CLOSED_LOOP_IDENTITY_RESIDUAL_LARGE": "The LQR closed-loop identity residual is too large.",
    "CLOSED_LOOP_NOT_HURWITZ": "The computed feedback does not produce a Hurwitz closed loop.",
    "CLOSED_LOOP_BOUNDARY": "Closed-loop poles are numerically near the imaginary axis.",
    "P_PSD_TOLERANCE_SENSITIVE": "Positive-semidefiniteness of the CARE solution is tolerance-sensitive.",
    "spectral_boundary": "One or more poles lie on or numerically near the imaginary axis.",
    "small_stability_margin": "The Hurwitz spectral margin is small relative to the active tolerance.",
}


def format_warning(code: str, context: str | None = None) -> str:
    message = WARNING_MESSAGES.get(code, code.replace("_", " ").strip().capitalize())
    return f"{context}: {message}" if context else message


def format_scalar(value: Any) -> str:
    """Format scalars consistently for reports and the Streamlit UI."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, sp.Rational):
        return str(value)
    if isinstance(value, sp.Basic):
        return str(value)
    if isinstance(value, complex):
        real = 0.0 if abs(value.real) < 5e-15 else float(value.real)
        imag = 0.0 if abs(value.imag) < 5e-15 else float(value.imag)
        if imag == 0.0:
            return format_scalar(real)
        if real == 0.0:
            if imag == 1.0:
                return "1j"
            if imag == -1.0:
                return "-1j"
            return f"{format_scalar(imag)}j"
        sign = "+" if imag >= 0 else "-"
        magnitude = abs(imag)
        imag_text = "j" if magnitude == 1 else f"{format_scalar(magnitude)}j"
        return f"{format_scalar(real)}{sign}{imag_text}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value == 0:
            return "0"
        nearest = round(value)
        if abs(value - nearest) <= 1e-12 * max(1.0, abs(value)):
            return str(int(nearest))
        magnitude = abs(value)
        if magnitude < 1e-4 or magnitude >= 1e6:
            return f"{value:.6e}".replace("e+", "e")
        return f"{value:.8g}"
    return str(value)



def _latex_scalar(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, sp.Basic):
        return sp.latex(value)
    return format_scalar(value)


def _matrix_rows(matrix: Any) -> list[list[Any]]:
    if isinstance(matrix, sp.MatrixBase):
        return [[matrix[i, j] for j in range(matrix.cols)] for i in range(matrix.rows)]
    array = np.asarray(matrix)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"matrix_to_latex expects a vector or two-dimensional matrix; got shape {array.shape}.")
    return array.tolist()


def matrix_to_latex(matrix: Any) -> str:
    """Render NumPy/SymPy matrices, including empty matrices, as LaTeX."""
    rows = _matrix_rows(matrix)
    if not rows:
        return r"\begin{bmatrix}\end{bmatrix}"
    if len(rows[0]) == 0:
        return r"\begin{bmatrix}\end{bmatrix}"
    body = r" \\ ".join(" & ".join(_latex_scalar(item) for item in row) for row in rows)
    return rf"\begin{{bmatrix}} {body} \end{{bmatrix}}"


def _matrix_source(analysis: SystemAnalysis, name: str) -> Any | None:
    system = analysis.system
    exact = getattr(system, f"exact_{name}")
    numeric = getattr(system, name)
    if numeric is None:
        return None
    return exact if exact is not None else numeric


def _status_line(record: AnalysisRecord) -> str:
    if record.status == STATUS_NOT_RUN:
        return f"**Status:** NOT_RUN — {record.reason or 'Not requested.'}"
    if record.status == STATUS_ERROR:
        return f"**Status:** ERROR — {record.error or 'Unknown error.'}"
    verdict = "" if record.certificate is None else f"  \n**Verdict:** `{record.certificate.verdict}`"
    return f"**Status:** {record.status}{verdict}"


def _fmt_diag(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.ndim == 1:
            return ", ".join(format_scalar(item) for item in array.tolist())
    return format_scalar(value)


def _controllability_markdown(record: AnalysisRecord) -> str:
    lines = ["## Controllability", "", _status_line(record), ""]
    cert = record.certificate
    if cert is None:
        return "\n".join(lines)
    d = cert.diagnostics
    rank = d.get("exact_rank") if d.get("exact_rank") is not None else d.get("numerical_rank")
    lines += [f"- Rank: `{rank} / {d.get('n')}`", f"- Numerical rank: `{d.get('numerical_rank')}`"]
    if d.get("exact_rank") is not None:
        lines.append(f"- Exact algebraic rank: `{d.get('exact_rank')}`")
    lines += [
        f"- PBH exact: `{_fmt_diag(d.get('pbh_exact_passed'))}`",
        f"- PBH numerical: `{_fmt_diag(d.get('pbh_numerical_passed'))}`",
        f"- Active rank tolerance: `{_fmt_diag(d.get('tolerance'))}`",
        "",
        "### Controllability Matrix",
        "",
        "$$",
        matrix_to_latex(cert.evidence.get("controllability_matrix_exact") if cert.evidence.get("controllability_matrix_exact") is not None else cert.evidence["controllability_matrix"]),
        "$$",
        "",
        f"Singular values: `{_fmt_diag(d.get('singular_values'))}`",
    ]
    pbh = cert.evidence.get("pbh")
    if pbh is not None and not pbh.passed:
        failures = pbh.evidence.get("exact_failures") or pbh.evidence.get("numerical_failures") or []
        if failures:
            failure = failures[0]
            lines += ["", "### PBH Failure Witness", "", f"Offending eigenvalue: `{_fmt_diag(failure.get('eigenvalue'))}`"]
            if failure.get("witness") is not None:
                lines += ["", "$$", matrix_to_latex(failure["witness"]), "$$"]
    return "\n".join(lines)


def _observability_markdown(record: AnalysisRecord) -> str:
    lines = ["## Observability", "", _status_line(record), ""]
    cert = record.certificate
    if cert is None:
        return "\n".join(lines)
    d = cert.diagnostics
    rank = d.get("exact_rank") if d.get("exact_rank") is not None else d.get("numerical_rank")
    lines += [f"- Rank: `{rank} / {d.get('n')}`", f"- Numerical rank: `{d.get('numerical_rank')}`"]
    if d.get("exact_rank") is not None:
        lines.append(f"- Exact algebraic rank: `{d.get('exact_rank')}`")
    lines += [
        f"- PBH exact: `{_fmt_diag(d.get('pbh_exact_passed'))}`",
        f"- PBH numerical: `{_fmt_diag(d.get('pbh_numerical_passed'))}`",
        "",
        "### Observability Matrix",
        "",
        "$$",
        matrix_to_latex(cert.evidence.get("observability_matrix_exact") if cert.evidence.get("observability_matrix_exact") is not None else cert.evidence["observability_matrix"]),
        "$$",
        "",
        f"Singular values: `{_fmt_diag(d.get('singular_values'))}`",
    ]
    pbh = cert.evidence.get("pbh")
    if pbh is not None and not pbh.passed:
        failures = pbh.evidence.get("exact_failures") or pbh.evidence.get("numerical_failures") or []
        if failures:
            failure = failures[0]
            lines += ["", "### PBH Failure Witness", "", f"Offending eigenvalue: `{_fmt_diag(failure.get('eigenvalue'))}`"]
            if failure.get("witness") is not None:
                lines += ["", "$$", matrix_to_latex(failure["witness"]), "$$"]
    return "\n".join(lines)


def _stability_markdown(record: AnalysisRecord) -> str:
    lines = ["## Stability / Lyapunov", "", _status_line(record), ""]
    cert = record.certificate
    if cert is None:
        return "\n".join(lines)
    spectral = cert.evidence.get("spectral")
    lyapunov = cert.evidence.get("lyapunov")
    if spectral is not None:
        d = spectral.diagnostics
        lines += [
            f"- Spectral abscissa: `{_fmt_diag(d.get('spectral_abscissa'))}`",
            f"- Spectral margin: `{_fmt_diag(d.get('spectral_margin'))}`",
            f"- Eigenvalues: `{_fmt_diag(d.get('eigenvalues'))}`",
        ]
    if lyapunov is not None:
        d = lyapunov.diagnostics
        lines += [
            "",
            "### Lyapunov Certificate",
            "",
            f"- Numerical certificate: `{_fmt_diag(d.get('numerical_certificate_passed'))}`",
            f"- Exact certificate: `{_fmt_diag(d.get('exact_certificate_passed'))}`",
            f"- Relative equation residual: `{_fmt_diag(d.get('relative_residual'))}`",
            f"- lambda_min(P): `{_fmt_diag(d.get('lambda_min_P'))}`",
        ]
        P = lyapunov.evidence.get("P_exact") if lyapunov.evidence.get("P_exact") is not None else lyapunov.evidence.get("P")
        if P is not None:
            lines += ["", "$$", rf"P = {matrix_to_latex(P)}", "$$"]
    return "\n".join(lines)


def _lqr_markdown(record: AnalysisRecord) -> str:
    lines = ["## LQR / CARE", "", _status_line(record), ""]
    cert = record.certificate
    if cert is None:
        return "\n".join(lines)
    d = cert.diagnostics
    lines += [
        f"- Stabilizable: `{_fmt_diag(d.get('stabilizable'))}`",
        f"- Detectable: `{_fmt_diag(d.get('detectable'))}`",
        f"- CARE relative residual: `{_fmt_diag(d.get('care_relative_residual'))}`",
        f"- Closed-loop identity relative residual: `{_fmt_diag(d.get('closed_loop_identity_relative_residual'))}`",
        f"- Closed-loop spectral abscissa: `{_fmt_diag(d.get('closed_loop_spectral_abscissa'))}`",
        f"- Closed-loop Hurwitz: `{_fmt_diag(d.get('closed_loop_hurwitz_status'))}`",
    ]
    for label, key in (("P", "P"), ("K", "K"), ("A_{cl}", "A_closed_loop")):
        matrix = cert.evidence.get(key)
        if matrix is not None:
            lines += ["", "$$", rf"{label} = {matrix_to_latex(matrix)}", "$$"]
    return "\n".join(lines)


def _kalman_markdown(record: AnalysisRecord) -> str:
    lines = ["## Kalman Structural Decomposition", "", _status_line(record), ""]
    cert = record.certificate
    if cert is None:
        return "\n".join(lines)
    d = cert.diagnostics
    lines += [
        f"- Reachable dimension: `{d.get('reachable_dimension')}`",
        f"- Unobservable dimension: `{d.get('unobservable_dimension')}`",
        f"- Intersection dimension: `{d.get('intersection_dimension')}`",
        f"- Controllable / observable: `{d.get('controllable_observable_dimension')}`",
        f"- Controllable / unobservable: `{d.get('controllable_unobservable_dimension')}`",
        f"- Uncontrollable / observable: `{d.get('uncontrollable_observable_dimension')}`",
        f"- Uncontrollable / unobservable: `{d.get('uncontrollable_unobservable_dimension')}`",
        f"- cond(T): `{_fmt_diag(d.get('transformation_condition_number'))}`",
        f"- A structural-zero residual: `{_fmt_diag(d.get('A_structural_zero_relative_residual'))}`",
        f"- B reachable-support residual: `{_fmt_diag(d.get('B_structural_zero_relative_residual'))}`",
        f"- C unobservable-annihilation residual: `{_fmt_diag(d.get('C_structural_zero_relative_residual'))}`",
    ]
    for label, key in (("T", "T"), ("\\bar A", "A_transformed"), ("\\bar B", "B_transformed"), ("\\bar C", "C_transformed")):
        matrix = cert.evidence.get(key)
        if matrix is not None:
            lines += ["", "$$", rf"{label} = {matrix_to_latex(matrix)}", "$$"]
    return "\n".join(lines)


def to_markdown(analysis: SystemAnalysis) -> str:
    """Render an already-computed ``SystemAnalysis`` as Markdown."""
    system = analysis.system
    lines = [
        "# CertiControl Analysis Report",
        "",
        f"- CertiControl version: `{analysis.version}`",
        f"- Generated at: `{analysis.generated_at}`",
        "- Scope: `continuous-time finite-dimensional LTI`",
        f"- States / inputs / outputs: `{system.n} / {system.m} / {system.p if system.p is not None else 'not specified'}`",
        f"- Tolerance: `absolute={analysis.tolerance_policy.absolute}, multiplier={analysis.tolerance_policy.multiplier}`",
        "",
        "## System",
        "",
    ]
    for name in ("A", "B", "C", "D"):
        matrix = _matrix_source(analysis, name)
        if matrix is not None:
            lines += ["$$", rf"{name} = {matrix_to_latex(matrix)}", "$$", ""]

    lines += [
        "## System Summary",
        "",
        "| Analysis | Status |",
        "|---|---|",
    ]
    for key, title in (
        ("controllability", "Controllability"),
        ("observability", "Observability"),
        ("stability", "Hurwitz stability"),
        ("kalman", "Kalman structure"),
        ("lqr", "LQR / CARE"),
    ):
        lines.append(f"| {title} | {analysis.status(key)} |")
    lines += [""]

    sections = {
        "controllability": _controllability_markdown,
        "observability": _observability_markdown,
        "stability": _stability_markdown,
        "lqr": _lqr_markdown,
        "kalman": _kalman_markdown,
    }
    for key in ("controllability", "observability", "stability", "lqr", "kalman"):
        record = analysis.records.get(key)
        if record is not None:
            lines += [sections[key](record), ""]

    lines += ["## Warnings", ""]
    if analysis.warnings:
        lines.extend(f"- {warning}" for warning in analysis.warnings)
    else:
        lines.append("No warnings reported.")
    lines += [
        "",
        "---",
        "CertiControl certificates are inspectable computational evidence. They are not formal proofs or safety-critical engineering certification.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def to_json_dict(analysis: SystemAnalysis) -> dict[str, Any]:
    """Return a JSON-compatible report dictionary."""
    return analysis.to_json_dict()


def to_json(analysis: SystemAnalysis, *, indent: int = 2) -> str:
    """Serialize a complete analysis report to JSON text."""
    return json.dumps(to_json_dict(analysis), indent=indent, ensure_ascii=False)