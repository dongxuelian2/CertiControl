"""Core data model for continuous-time LTI systems."""

from __future__ import annotations

from typing import Any, Sequence, TypeAlias

import numpy as np
import sympy as sp

MatrixLike: TypeAlias = np.ndarray | sp.MatrixBase | Sequence[Sequence[Any]]


def _sympy_to_numeric(matrix: sp.MatrixBase) -> np.ndarray:
    values = np.array(
        [[complex(sp.N(matrix[i, j], 17)) for j in range(matrix.cols)] for i in range(matrix.rows)],
        dtype=np.complex128,
    )
    if np.all(values.imag == 0):
        return values.real.astype(np.float64)
    return values


def _coerce_matrix(value: MatrixLike, name: str) -> tuple[np.ndarray, sp.ImmutableMatrix | None]:
    """Normalize a matrix while retaining exact rational data when possible."""
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

    if symbolic.rows == 0 or symbolic.cols == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not all(bool(entry.is_number) for entry in symbolic):
        raise ValueError(f"{name} must contain only numeric entries.")

    try:
        numeric = _sympy_to_numeric(symbolic)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains entries that cannot be converted to finite numeric values.") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite numeric values.")

    exact = symbolic if all(entry.is_Rational is True for entry in symbolic) else None
    return numeric, exact


class LTISystem:
    """Continuous-time finite-dimensional LTI system ``xdot = Ax + Bu``.

    Numeric matrices are always retained for floating-point diagnostics.  If a
    supplied matrix consists only of exact integers/rationals, an exact SymPy
    copy is retained as well.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray | None
    D: np.ndarray | None

    def __init__(
        self,
        A: MatrixLike,
        B: MatrixLike,
        C: MatrixLike | None = None,
        D: MatrixLike | None = None,
    ) -> None:
        self.A, self._A_exact = _coerce_matrix(A, "A")
        self.B, self._B_exact = _coerce_matrix(B, "B")
        self.C, self._C_exact = (None, None) if C is None else _coerce_matrix(C, "C")
        self.D, self._D_exact = (None, None) if D is None else _coerce_matrix(D, "D")
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A must be square; got shape {self.A.shape}.")
        n = self.A.shape[0]
        if self.B.shape[0] != n:
            raise ValueError(
                f"B must have the same number of rows as A ({n}); got shape {self.B.shape}."
            )
        if self.C is not None and self.C.shape[1] != n:
            raise ValueError(
                f"C must have {n} columns to match the state dimension; got shape {self.C.shape}."
            )
        if self.D is not None and self.D.shape[1] != self.B.shape[1]:
            raise ValueError(
                "D must have the same number of columns as B "
                f"({self.B.shape[1]}); got shape {self.D.shape}."
            )
        if self.C is not None and self.D is not None and self.C.shape[0] != self.D.shape[0]:
            raise ValueError(
                "C and D must have the same number of output rows; "
                f"got C{self.C.shape} and D{self.D.shape}."
            )

    @property
    def n(self) -> int:
        """Number of states."""
        return int(self.A.shape[0])

    @property
    def m(self) -> int:
        """Number of inputs."""
        return int(self.B.shape[1])

    @property
    def p(self) -> int | None:
        """Number of outputs, when C or D is provided."""
        if self.C is not None:
            return int(self.C.shape[0])
        if self.D is not None:
            return int(self.D.shape[0])
        return None

    @property
    def exact_A(self) -> sp.ImmutableMatrix | None:
        return self._A_exact

    @property
    def exact_B(self) -> sp.ImmutableMatrix | None:
        return self._B_exact

    @property
    def exact_C(self) -> sp.ImmutableMatrix | None:
        return self._C_exact

    @property
    def exact_D(self) -> sp.ImmutableMatrix | None:
        return self._D_exact

    @property
    def has_exact_state_input_data(self) -> bool:
        """Whether A and B are available as exact rational matrices."""
        return self._A_exact is not None and self._B_exact is not None

    @property
    def has_exact_state_output_data(self) -> bool:
        """Whether A and C are available as exact rational matrices."""
        return self._A_exact is not None and self._C_exact is not None
