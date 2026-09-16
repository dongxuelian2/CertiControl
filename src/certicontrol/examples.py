"""Small systems useful in documentation and tests."""

from __future__ import annotations

from .model import LTISystem


def controllable_second_order() -> LTISystem:
    """Return a standard controllable second-order SISO system."""
    return LTISystem([[0, 1], [-2, -3]], [[0], [1]])


def uncontrollable_second_order() -> LTISystem:
    """Return a two-state system whose second mode is unreachable."""
    return LTISystem([[0, 0], [0, 1]], [[1], [0]])
