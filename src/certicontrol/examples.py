"""Small systems useful in documentation and tests."""

from __future__ import annotations

from .model import LTISystem


def controllable_second_order() -> LTISystem:
    """Return a standard controllable second-order SISO system."""
    return LTISystem([[0, 1], [-2, -3]], [[0], [1]])


def uncontrollable_second_order() -> LTISystem:
    """Return a two-state system whose second mode is unreachable."""
    return LTISystem([[0, 0], [0, 1]], [[1], [0]])


def observable_second_order() -> LTISystem:
    """Return a standard observable second-order SISO-output system."""
    return LTISystem([[0, 1], [-2, -3]], [[0], [1]], C=[[1, 0]])


def unobservable_second_order() -> LTISystem:
    """Return a two-state system whose second mode is invisible at the output."""
    return LTISystem([[-1, 0], [0, -2]], [[1], [0]], C=[[1, 0]])
