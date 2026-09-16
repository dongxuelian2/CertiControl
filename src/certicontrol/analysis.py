"""High-level orchestration for CertiControl analysis sessions.

This module composes existing mathematical certificates.  It deliberately does
not reimplement any control-theoretic calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any

import numpy as np
import sympy as sp

from .certificates import Certificate
from .controllability import analyze_controllability
from .decomposition import analyze_kalman_decomposition
from .lqr import analyze_lqr
from .model import LTISystem, MatrixLike
from .observability import analyze_observability
from .stability import analyze_stability
from .tolerance import TolerancePolicy


STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"
STATUS_NOT_RUN = "NOT_RUN"
STATUS_ERROR = "ERROR"


def status_from_certificate(certificate: Certificate) -> str:
    """Map the tri-state ``Certificate.passed`` value to a product status."""
    if certificate.passed is True:
        return STATUS_PASS
    if certificate.passed is False:
        return STATUS_FAIL
    return STATUS_INCONCLUSIVE


def _json_safe(value: Any) -> Any:
    if isinstance(value, Certificate):
        return value.to_dict(json_safe=True)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        if abs(value.imag) <= 1e-15:
            return float(value.real)
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, sp.MatrixBase):
        return [[str(value[i, j]) for j in range(value.cols)] for i in range(value.rows)]
    if isinstance(value, sp.Basic):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _package_version() -> str:
    try:
        return importlib_metadata.version("certicontrol")
    except importlib_metadata.PackageNotFoundError:
        return "0.2.0"


@dataclass(frozen=True)
class AnalysisOptions:
    """Select which existing certificates should be orchestrated."""

    run_controllability: bool = True
    run_observability: bool = True
    run_stability: bool = True
    run_kalman: bool = True
    run_lqr: bool = True


@dataclass
class AnalysisRecord:
    """One analysis slot, including explicit not-run/error states."""

    name: str
    status: str
    certificate: Certificate | None = None
    reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "error": self.error,
            "certificate": None if self.certificate is None else self.certificate.to_dict(json_safe=True),
        }


@dataclass
class SystemAnalysis:
    """Aggregate result for one continuous-time LTI system."""

    system: LTISystem
    records: dict[str, AnalysisRecord]
    tolerance_policy: TolerancePolicy
    generated_at: str
    version: str = field(default_factory=_package_version)
    warnings: list[str] = field(default_factory=list)
    warning_codes: list[dict[str, str]] = field(default_factory=list)

    @property
    def certificates(self) -> dict[str, Certificate]:
        """Only certificates that actually ran."""
        return {
            key: record.certificate
            for key, record in self.records.items()
            if record.certificate is not None
        }

    def status(self, name: str) -> str:
        record = self.records.get(name)
        return STATUS_NOT_RUN if record is None else record.status

    def summary(self) -> dict[str, Any]:
        kalman = self.certificates.get("kalman")
        lqr = self.certificates.get("lqr")
        result: dict[str, Any] = {
            "states": self.system.n,
            "inputs": self.system.m,
            "outputs": self.system.p,
            "controllability": self.status("controllability"),
            "observability": self.status("observability"),
            "hurwitz": self.status("stability"),
            "kalman": self.status("kalman"),
            "lqr": self.status("lqr"),
        }
        if kalman is not None:
            d = kalman.diagnostics
            result.update(
                {
                    "reachable_dimension": d.get("reachable_dimension"),
                    "unobservable_dimension": d.get("unobservable_dimension"),
                    "intersection_dimension": d.get("intersection_dimension"),
                    "controllable_observable_dimension": d.get("controllable_observable_dimension"),
                    "controllable_unobservable_dimension": d.get("controllable_unobservable_dimension"),
                    "uncontrollable_observable_dimension": d.get("uncontrollable_observable_dimension"),
                    "uncontrollable_unobservable_dimension": d.get("uncontrollable_unobservable_dimension"),
                }
            )
        if lqr is not None:
            result.update(
                {
                    "stabilizable": lqr.diagnostics.get("stabilizable"),
                    "detectable": lqr.diagnostics.get("detectable"),
                    "closed_loop_hurwitz": lqr.diagnostics.get("closed_loop_hurwitz_status"),
                }
            )
        return result

    def to_json_dict(self) -> dict[str, Any]:
        """Return a complete JSON-compatible analysis payload."""
        exact_flags = {
            "A": self.system.exact_A is not None,
            "B": self.system.exact_B is not None,
            "C": self.system.C is not None and self.system.exact_C is not None,
            "D": self.system.D is not None and self.system.exact_D is not None,
        }
        system_payload = {
            "n": self.system.n,
            "m": self.system.m,
            "p": self.system.p,
            "A": _json_safe(self.system.exact_A if self.system.exact_A is not None else self.system.A),
            "B": _json_safe(self.system.exact_B if self.system.exact_B is not None else self.system.B),
            "C": None
            if self.system.C is None
            else _json_safe(self.system.exact_C if self.system.exact_C is not None else self.system.C),
            "D": None
            if self.system.D is None
            else _json_safe(self.system.exact_D if self.system.exact_D is not None else self.system.D),
            "exact_rational_input": exact_flags,
        }
        payload = {
            "metadata": {
                "certicontrol_version": self.version,
                "generated_at": self.generated_at,
                "scope": "continuous-time finite-dimensional LTI",
                "tolerance_policy": {
                    "absolute": self.tolerance_policy.absolute,
                    "multiplier": self.tolerance_policy.multiplier,
                },
                "analyses_performed": [
                    name for name, record in self.records.items() if record.certificate is not None
                ],
            },
            "system": system_payload,
            "summary": _json_safe(self.summary()),
            "analyses": {name: record.to_dict() for name, record in self.records.items()},
            "warnings": list(self.warnings),
            "warning_codes": list(self.warning_codes),
        }
        return _json_safe(payload)



def _collect_certificate_warnings(
    certificate: Certificate,
    source: str,
    *,
    seen: set[int] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Collect warnings/codes from nested certificates without changing them."""
    if seen is None:
        seen = set()
    if id(certificate) in seen:
        return [], []
    seen.add(id(certificate))
    warnings = [f"[{source}] {message}" for message in certificate.warnings]
    codes = [
        {"source": source, "code": str(code)}
        for code in certificate.diagnostics.get("warning_codes", [])
    ]

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Certificate):
            nested_warnings, nested_codes = _collect_certificate_warnings(
                value, path, seen=seen
            )
            warnings.extend(nested_warnings)
            codes.extend(nested_codes)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(certificate.evidence, source)
    return warnings, codes


def _not_run(name: str, reason: str) -> AnalysisRecord:
    return AnalysisRecord(name=name, status=STATUS_NOT_RUN, reason=reason)


def _run_certificate(name: str, function: Any) -> AnalysisRecord:
    try:
        certificate = function()
    except Exception as exc:  # isolate one module without losing other results
        return AnalysisRecord(name=name, status=STATUS_ERROR, error=f"{type(exc).__name__}: {exc}")
    return AnalysisRecord(
        name=name,
        status=status_from_certificate(certificate),
        certificate=certificate,
    )


def analyze_system(
    system: LTISystem,
    *,
    options: AnalysisOptions | None = None,
    Q: MatrixLike | None = None,
    R: MatrixLike | None = None,
    tolerance_policy: TolerancePolicy | None = None,
    timestamp: str | None = None,
) -> SystemAnalysis:
    """Run selected CertiControl certificates and aggregate partial results.

    Missing optional inputs produce ``NOT_RUN`` records.  A failure inside one
    analysis is isolated to that record so that already valid certificates from
    other modules remain available to reporting/UI code.
    """
    if not isinstance(system, LTISystem):
        raise TypeError("analyze_system expects a validated LTISystem instance.")
    options = options or AnalysisOptions()
    policy = tolerance_policy or TolerancePolicy()
    records: dict[str, AnalysisRecord] = {}

    if options.run_controllability:
        records["controllability"] = _run_certificate(
            "controllability", lambda: analyze_controllability(system, policy)
        )
    else:
        records["controllability"] = _not_run("controllability", "Disabled by analysis options.")

    if not options.run_observability:
        records["observability"] = _not_run("observability", "Disabled by analysis options.")
    elif system.C is None:
        records["observability"] = _not_run("observability", "C is required for observability analysis.")
    else:
        records["observability"] = _run_certificate(
            "observability", lambda: analyze_observability(system, policy)
        )

    if options.run_stability:
        records["stability"] = _run_certificate(
            "stability", lambda: analyze_stability(system, tolerance_policy=policy)
        )
    else:
        records["stability"] = _not_run("stability", "Disabled by analysis options.")

    if not options.run_kalman:
        records["kalman"] = _not_run("kalman", "Disabled by analysis options.")
    elif system.C is None:
        records["kalman"] = _not_run("kalman", "C is required for full Kalman decomposition.")
    else:
        records["kalman"] = _run_certificate(
            "kalman", lambda: analyze_kalman_decomposition(system, tolerance_policy=policy)
        )

    if not options.run_lqr:
        records["lqr"] = _not_run("lqr", "Disabled by analysis options.")
    elif Q is None or R is None:
        records["lqr"] = _not_run("lqr", "Both Q and R are required for LQR analysis.")
    else:
        records["lqr"] = _run_certificate(
            "lqr", lambda: analyze_lqr(system, Q, R, tolerance_policy=policy)
        )

    warnings: list[str] = []
    warning_codes: list[dict[str, str]] = []
    for key, record in records.items():
        if record.certificate is not None:
            nested_warnings, nested_codes = _collect_certificate_warnings(record.certificate, key)
            warnings.extend(nested_warnings)
            warning_codes.extend(nested_codes)
        elif record.error:
            warnings.append(f"[{key}] Analysis error: {record.error}")

    warnings = list(dict.fromkeys(warnings))
    warning_codes = [dict(item) for item in {tuple(sorted(item.items())) for item in warning_codes}]
    warning_codes.sort(key=lambda item: (item.get("source", ""), item.get("code", "")))

    generated_at = timestamp or datetime.now(timezone.utc).isoformat()
    return SystemAnalysis(
        system=system,
        records=records,
        tolerance_policy=policy,
        generated_at=generated_at,
        warnings=warnings,
        warning_codes=warning_codes,
    )