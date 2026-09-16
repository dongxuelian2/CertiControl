"""Structured certificate objects shared by analysis routines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp


def _json_safe(value: Any) -> Any:
    if isinstance(value, Certificate):
        return _json_safe(value.to_dict())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
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


@dataclass
class Certificate:
    """Machine-readable result with evidence, diagnostics, and warnings."""

    name: str
    passed: bool | None
    verdict: str
    evidence: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, *, json_safe: bool = False) -> dict[str, Any]:
        """Return a dictionary representation, optionally JSON-compatible."""
        data: dict[str, Any] = {
            "name": self.name,
            "passed": self.passed,
            "verdict": self.verdict,
            "evidence": self.evidence,
            "diagnostics": self.diagnostics,
            "warnings": list(self.warnings),
        }
        return _json_safe(data) if json_safe else data
