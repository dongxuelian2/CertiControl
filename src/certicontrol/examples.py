"""Curated deterministic systems for documentation and the interactive app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from .model import LTISystem, MatrixLike


@dataclass(frozen=True)
class ExampleSystem:
    name: str
    description: str
    system: LTISystem
    default_Q: MatrixLike | None = None
    default_R: MatrixLike | None = None
    recommended_absolute_tolerance: float | None = None


def _source_matrix(system: LTISystem, name: str) -> Any | None:
    numeric = getattr(system, name)
    if numeric is None:
        return None
    exact = getattr(system, f"exact_{name}")
    return exact if exact is not None else numeric


def matrix_to_input_text(matrix: Any | None) -> str:
    """Render a matrix in the plain-text syntax accepted by ``parse_matrix``."""
    if matrix is None:
        return ""
    if isinstance(matrix, sp.MatrixBase):
        rows = [[str(matrix[i, j]) for j in range(matrix.cols)] for i in range(matrix.rows)]
    else:
        array = np.asarray(matrix)
        rows = [[str(item) for item in row] for row in array.tolist()]
    return "\n".join(" ".join(row) for row in rows)


def example_matrix_text(example: ExampleSystem) -> dict[str, str]:
    return {
        name: matrix_to_input_text(_source_matrix(example.system, name))
        for name in ("A", "B", "C", "D")
    }


def curated_examples() -> dict[str, ExampleSystem]:
    """Return deterministic examples used by both docs and the Streamlit demo."""
    stable = LTISystem([[0, 1], [-2, -3]], [[0], [1]], C=[[1, 0]], D=[[0]])

    uncontrollable = LTISystem(
        [[-1, 0], [0, 1]],
        [[1], [0]],
        C=[[1, 0], [0, 1]],
        D=[[0], [0]],
    )

    unobservable = LTISystem(
        [[-1, 0], [0, -2]],
        [[1, 0], [0, 1]],
        C=[[1, 0]],
        D=[[0, 0]],
    )

    unstable_controllable = LTISystem(
        [[0, 1], [1, 0]],
        [[0], [1]],
        C=[[1, 0]],
        D=[[0]],
    )

    stabilizable_not_controllable = LTISystem(
        [[-2, 0], [0, 1]],
        [[0], [1]],
        C=[[1, 0], [0, 1]],
        D=[[0], [0]],
    )

    four_part_A = sp.Matrix(
        [
            [-1, 0, sp.Rational(1, 3), 0],
            [sp.Rational(1, 5), -2, sp.Rational(1, 4), sp.Rational(1, 2)],
            [0, 0, -3, 0],
            [0, 0, sp.Rational(2, 5), -4],
        ]
    )
    four_part = LTISystem(
        four_part_A,
        sp.Matrix([[1], [1], [0], [0]]),
        C=sp.Matrix([[1, 0, 1, 0]]),
        D=sp.Matrix([[0]]),
    )

    tolerance_sensitive = LTISystem(
        np.diag([0.0, 1.0]),
        np.array([[1.0], [2e-7]]),
        C=np.eye(2),
        D=np.zeros((2, 1)),
    )

    return {
        "Stable minimal second-order": ExampleSystem(
            name="Stable minimal second-order",
            description="A controllable, observable, Hurwitz two-state reference system with exact rational data.",
            system=stable,
            default_Q=sp.eye(2),
            default_R=sp.Matrix([[1]]),
        ),
        "Uncontrollable mode": ExampleSystem(
            name="Uncontrollable mode",
            description="Contains an unstable input-decoupled mode, so PBH controllability returns a witness.",
            system=uncontrollable,
        ),
        "Unobservable mode": ExampleSystem(
            name="Unobservable mode",
            description="The second stable state is invisible at the output, producing an unobservable PBH witness.",
            system=unobservable,
        ),
        "Unstable but controllable": ExampleSystem(
            name="Unstable but controllable",
            description="Open-loop unstable but controllable/observable; LQR moves the closed-loop poles into the left half-plane.",
            system=unstable_controllable,
            default_Q=sp.eye(2),
            default_R=sp.Matrix([[1]]),
        ),
        "Stabilizable not controllable": ExampleSystem(
            name="Stabilizable not controllable",
            description="One unreachable mode is already stable, so the system is not controllable but is stabilizable.",
            system=stabilizable_not_controllable,
            default_Q=sp.eye(2),
            default_R=sp.Matrix([[1]]),
        ),
        "Four-part Kalman structure": ExampleSystem(
            name="Four-part Kalman structure",
            description="Exact rational four-state fixture with co=cu=uo=uu=1.",
            system=four_part,
        ),
        "Tolerance-sensitive": ExampleSystem(
            name="Tolerance-sensitive",
            description="A nearly unreachable direction becomes ambiguous under a deliberate 1e-6 absolute tolerance.",
            system=tolerance_sensitive,
            recommended_absolute_tolerance=1e-6,
        ),
    }


def get_example(name: str) -> ExampleSystem:
    examples = curated_examples()
    try:
        return examples[name]
    except KeyError as exc:
        raise KeyError(f"Unknown CertiControl example {name!r}.") from exc


# Backward-compatible small helper functions retained from the earlier examples module.
def controllable_second_order() -> LTISystem:
    return LTISystem([[0, 1], [-2, -3]], [[0], [1]])


def uncontrollable_second_order() -> LTISystem:
    return LTISystem([[0, 0], [0, 1]], [[1], [0]])


def observable_second_order() -> LTISystem:
    return LTISystem([[0, 1], [-2, -3]], [[0], [1]], C=[[1, 0]])


def unobservable_second_order() -> LTISystem:
    return LTISystem([[-1, 0], [0, -2]], [[1], [0]], C=[[1, 0]])