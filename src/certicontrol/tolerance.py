"""Centralized floating-point tolerance and rank diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg


@dataclass(frozen=True)
class TolerancePolicy:
    """Policy for numerical rank decisions based on singular values.

    If ``absolute`` is omitted, the threshold is
    ``multiplier * sigma_max * max(m, n) * eps`` using float64 machine epsilon.
    """

    absolute: float | None = None
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.absolute is not None and self.absolute < 0:
            raise ValueError("absolute tolerance must be non-negative.")
        if self.multiplier <= 0:
            raise ValueError("tolerance multiplier must be positive.")

    def threshold(self, singular_values: np.ndarray, shape: tuple[int, int]) -> float:
        """Return the actual singular-value threshold used for this matrix."""
        if self.absolute is not None:
            return float(self.absolute)
        sigma_max = float(singular_values[0]) if singular_values.size else 0.0
        return float(self.multiplier * sigma_max * max(shape) * np.finfo(np.float64).eps)


@dataclass(frozen=True)
class NumericalRankResult:
    """SVD-based numerical rank diagnostics."""

    rank: int
    singular_values: np.ndarray
    tolerance: float
    condition_number: float


def numerical_rank(matrix: np.ndarray, policy: TolerancePolicy | None = None) -> NumericalRankResult:
    """Compute numerical rank and diagnostics under one tolerance policy."""
    policy = policy or TolerancePolicy()
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"rank input must be two-dimensional; got shape {array.shape}.")
    singular_values = linalg.svdvals(array)
    tolerance = policy.threshold(singular_values, array.shape)
    rank = int(np.count_nonzero(singular_values > tolerance))
    if singular_values.size == 0 or singular_values[-1] == 0:
        condition_number = float("inf")
    else:
        condition_number = float(singular_values[0] / singular_values[-1])
    return NumericalRankResult(rank, singular_values, tolerance, condition_number)
