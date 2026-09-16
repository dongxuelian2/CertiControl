"""Reachability, observability, and Kalman structural decomposition certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings as py_warnings

import numpy as np
import sympy as sp
from scipy import linalg

from .certificates import Certificate
from .controllability import analyze_controllability, controllability_matrix
from .model import LTISystem, MatrixLike
from .observability import analyze_observability, observability_matrix
from .tolerance import NumericalRankResult, TolerancePolicy, numerical_rank

_RESIDUAL_MULTIPLIER = 200.0
_ILL_CONDITIONED_THRESHOLD = 1.0 / np.sqrt(np.finfo(np.float64).eps)
_RANK_SENSITIVITY_FACTOR = 10.0


@dataclass(frozen=True)
class _SubspaceComputation:
    numerical_basis: np.ndarray
    numerical_dimension: int
    singular_values: np.ndarray
    tolerance: float
    exact_basis: sp.Matrix | None
    exact_dimension: int | None
    dimension: int
    warning_codes: tuple[str, ...]
    warnings: tuple[str, ...]


def _as_numeric(matrix: MatrixLike) -> np.ndarray:
    if isinstance(matrix, sp.MatrixBase):
        values = np.array(
            [[complex(sp.N(matrix[i, j], 17)) for j in range(matrix.cols)] for i in range(matrix.rows)],
            dtype=np.complex128,
        )
        if np.all(values.imag == 0):
            return values.real.astype(np.float64)
        return values
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"matrix must be two-dimensional; got shape {array.shape}.")
    return array


def _matrix_exact_if_rational(matrix: MatrixLike) -> sp.ImmutableMatrix | None:
    if isinstance(matrix, sp.MatrixBase):
        symbolic = sp.ImmutableMatrix(matrix)
    else:
        array = np.asarray(matrix, dtype=object)
        if array.ndim != 2:
            return None
        try:
            symbolic = sp.ImmutableMatrix(array.tolist())
        except (TypeError, ValueError):
            return None
    return symbolic if all(entry.is_Rational is True for entry in symbolic) else None


def _empty_numeric_basis(n: int, dtype: np.dtype[Any] | type = float) -> np.ndarray:
    return np.empty((n, 0), dtype=dtype)


def _empty_exact_basis(n: int) -> sp.Matrix:
    return sp.zeros(n, 0)


def _rank_tolerance_sensitive(result: NumericalRankResult) -> bool:
    if result.tolerance <= 0 or result.singular_values.size == 0:
        return False
    positive = result.singular_values[result.singular_values > 0]
    if positive.size == 0:
        return False
    near = positive[
        (positive >= result.tolerance / _RANK_SENSITIVITY_FACTOR)
        & (positive <= result.tolerance * _RANK_SENSITIVITY_FACTOR)
    ]
    return bool(near.size)


def _residual_tolerance(n: int) -> float:
    return float(_RESIDUAL_MULTIPLIER * np.finfo(np.float64).eps * max(1, n))


def _relative_norm(residual: np.ndarray, *references: np.ndarray) -> tuple[float, float]:
    absolute = float(linalg.norm(residual, ord="fro"))
    denominator = sum(float(linalg.norm(reference, ord="fro")) for reference in references)
    relative = absolute / denominator if denominator > 0 else absolute
    return absolute, relative


def _projector_residual(basis: np.ndarray, values: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return values.copy()
    return values - basis @ (basis.conj().T @ values)


def _numerical_columnspace(matrix: np.ndarray, policy: TolerancePolicy) -> tuple[np.ndarray, NumericalRankResult]:
    array = np.asarray(matrix)
    result = numerical_rank(array, policy)
    if result.rank == 0:
        return _empty_numeric_basis(array.shape[0], np.result_type(array.dtype, float)), result
    u, _, _ = linalg.svd(array, full_matrices=False)
    return u[:, : result.rank], result


def _numerical_nullspace(matrix: np.ndarray, policy: TolerancePolicy) -> tuple[np.ndarray, NumericalRankResult]:
    array = np.asarray(matrix)
    result = numerical_rank(array, policy)
    _, _, vh = linalg.svd(array, full_matrices=True)
    ncols = array.shape[1]
    if result.rank >= ncols:
        return _empty_numeric_basis(ncols, np.result_type(array.dtype, float)), result
    return vh.conj().T[:, result.rank :], result


def _exact_columnspace(matrix: sp.MatrixBase) -> sp.Matrix:
    basis = sp.Matrix(matrix).columnspace()
    return sp.Matrix.hstack(*basis) if basis else _empty_exact_basis(matrix.rows)


def _exact_nullspace(matrix: sp.MatrixBase) -> sp.Matrix:
    basis = sp.Matrix(matrix).nullspace()
    return sp.Matrix.hstack(*basis) if basis else _empty_exact_basis(matrix.cols)


def _exact_membership(container: sp.Matrix, values: sp.Matrix) -> bool:
    if values.cols == 0:
        return True
    if container.cols == 0:
        return values == sp.zeros(values.rows, values.cols)
    return container.row_join(values).rank() == container.rank()


def _reachable_computation(A: MatrixLike, B: MatrixLike, policy: TolerancePolicy) -> _SubspaceComputation:
    A_num = _as_numeric(A)
    B_num = _as_numeric(B)
    matrix_num = np.asarray(controllability_matrix(A_num, B_num))
    numerical_basis, rank_result = _numerical_columnspace(matrix_num, policy)
    warning_codes: list[str] = []
    warnings: list[str] = []
    if _rank_tolerance_sensitive(rank_result):
        warning_codes.append("REACHABLE_RANK_TOLERANCE_SENSITIVE")
        warnings.append("Reachable-subspace dimension is sensitive to the active singular-value tolerance.")

    A_exact = _matrix_exact_if_rational(A)
    B_exact = _matrix_exact_if_rational(B)
    exact_basis: sp.Matrix | None = None
    exact_dimension: int | None = None
    if A_exact is not None and B_exact is not None:
        matrix_exact = controllability_matrix(A_exact, B_exact)
        assert isinstance(matrix_exact, sp.MatrixBase)
        exact_basis = _exact_columnspace(matrix_exact)
        exact_dimension = exact_basis.cols
        if exact_dimension != rank_result.rank:
            warning_codes.append("EXACT_NUMERICAL_DIMENSION_DISAGREEMENT")
            warnings.append(
                "Exact and numerical reachable-subspace dimensions disagree; the exact algebraic dimension is authoritative."
            )
    dimension = exact_dimension if exact_dimension is not None else rank_result.rank
    return _SubspaceComputation(
        numerical_basis=numerical_basis,
        numerical_dimension=rank_result.rank,
        singular_values=rank_result.singular_values,
        tolerance=rank_result.tolerance,
        exact_basis=exact_basis,
        exact_dimension=exact_dimension,
        dimension=int(dimension),
        warning_codes=tuple(warning_codes),
        warnings=tuple(warnings),
    )


def _unobservable_computation(A: MatrixLike, C: MatrixLike, policy: TolerancePolicy) -> _SubspaceComputation:
    A_num = _as_numeric(A)
    C_num = _as_numeric(C)
    matrix_num = np.asarray(observability_matrix(A_num, C_num))
    numerical_basis, rank_result = _numerical_nullspace(matrix_num, policy)
    numerical_dimension = numerical_basis.shape[1]
    warning_codes: list[str] = []
    warnings: list[str] = []
    if _rank_tolerance_sensitive(rank_result):
        warning_codes.append("UNOBSERVABLE_NULLITY_TOLERANCE_SENSITIVE")
        warnings.append("Unobservable-subspace dimension is sensitive to the active singular-value tolerance.")

    A_exact = _matrix_exact_if_rational(A)
    C_exact = _matrix_exact_if_rational(C)
    exact_basis: sp.Matrix | None = None
    exact_dimension: int | None = None
    if A_exact is not None and C_exact is not None:
        matrix_exact = observability_matrix(A_exact, C_exact)
        assert isinstance(matrix_exact, sp.MatrixBase)
        exact_basis = _exact_nullspace(matrix_exact)
        exact_dimension = exact_basis.cols
        if exact_dimension != numerical_dimension:
            warning_codes.append("EXACT_NUMERICAL_DIMENSION_DISAGREEMENT")
            warnings.append(
                "Exact and numerical unobservable-subspace dimensions disagree; the exact algebraic dimension is authoritative."
            )
    dimension = exact_dimension if exact_dimension is not None else numerical_dimension
    return _SubspaceComputation(
        numerical_basis=numerical_basis,
        numerical_dimension=numerical_dimension,
        singular_values=rank_result.singular_values,
        tolerance=rank_result.tolerance,
        exact_basis=exact_basis,
        exact_dimension=exact_dimension,
        dimension=int(dimension),
        warning_codes=tuple(warning_codes),
        warnings=tuple(warnings),
    )


def _subspace_residuals_reachable(A: np.ndarray, B: np.ndarray, basis: np.ndarray) -> dict[str, float]:
    invariance = _projector_residual(basis, A @ basis)
    input_residual = _projector_residual(basis, B)
    inv_abs, inv_rel = _relative_norm(invariance, A @ basis)
    input_abs, input_rel = _relative_norm(input_residual, B)
    return {
        "reachable_invariance_residual": inv_abs,
        "reachable_invariance_relative_residual": inv_rel,
        "input_inclusion_residual": input_abs,
        "input_inclusion_relative_residual": input_rel,
    }


def _subspace_residuals_unobservable(A: np.ndarray, C: np.ndarray, basis: np.ndarray) -> dict[str, float]:
    invariance = _projector_residual(basis, A @ basis)
    output = C @ basis
    inv_abs, inv_rel = _relative_norm(invariance, A @ basis)
    output_abs, output_rel = _relative_norm(output, C, basis)
    return {
        "unobservable_invariance_residual": inv_abs,
        "unobservable_invariance_relative_residual": inv_rel,
        "output_annihilation_residual": output_abs,
        "output_annihilation_relative_residual": output_rel,
    }


def reachable_subspace(
    A: MatrixLike,
    B: MatrixLike,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> Certificate:
    """Return a certificate for the reachable subspace im[C(A,B)]."""
    policy = tolerance_policy or TolerancePolicy()
    system = LTISystem(A, B)
    source_A = system.exact_A if system.exact_A is not None else system.A
    source_B = system.exact_B if system.exact_B is not None else system.B
    data = _reachable_computation(source_A, source_B, policy)
    residuals = _subspace_residuals_reachable(system.A, system.B, data.numerical_basis)
    tol = _residual_tolerance(system.n)
    warning_codes = list(data.warning_codes)
    warnings = list(data.warnings)
    dimension_sensitive = "REACHABLE_RANK_TOLERANCE_SENSITIVE" in warning_codes and data.exact_dimension is None
    hard_failure = (
        not dimension_sensitive
        and (
            residuals["reachable_invariance_relative_residual"] > tol
            or residuals["input_inclusion_relative_residual"] > tol
        )
    )
    if residuals["reachable_invariance_relative_residual"] > tol:
        warning_codes.append("REACHABLE_INVARIANCE_RESIDUAL_LARGE")
        warnings.append("The numerical reachable basis is not sufficiently A-invariant.")
    if residuals["input_inclusion_relative_residual"] > tol:
        warning_codes.append("INPUT_INCLUSION_RESIDUAL_LARGE")
        warnings.append("The input image is not sufficiently contained in the computed reachable subspace.")

    exact_invariance = None
    exact_input_inclusion = None
    if data.exact_basis is not None and system.exact_A is not None and system.exact_B is not None:
        exact_invariance = _exact_membership(data.exact_basis, sp.Matrix(system.exact_A) * data.exact_basis)
        exact_input_inclusion = _exact_membership(data.exact_basis, sp.Matrix(system.exact_B))
        if not exact_invariance or not exact_input_inclusion:
            hard_failure = True

    passed: bool | None = False if hard_failure else None if dimension_sensitive else True
    verdict = "VALID" if passed is True else "INVALID_SUBSPACE" if passed is False else "BOUNDARY_OR_INCONCLUSIVE"
    return Certificate(
        name="Reachable subspace",
        passed=passed,
        verdict=verdict,
        evidence={
            "basis": data.numerical_basis,
            "numerical_basis": data.numerical_basis,
            "exact_basis": data.exact_basis,
        },
        diagnostics={
            "state_dimension": system.n,
            "dimension": data.dimension,
            "numerical_dimension": data.numerical_dimension,
            "exact_dimension": data.exact_dimension,
            "singular_values": data.singular_values,
            "tolerance": data.tolerance,
            **residuals,
            "exact_A_invariant": exact_invariance,
            "exact_input_inclusion": exact_input_inclusion,
            "residual_tolerance": tol,
            "warning_codes": list(dict.fromkeys(warning_codes)),
        },
        warnings=list(dict.fromkeys(warnings)),
    )


def unobservable_subspace(
    A: MatrixLike,
    C: MatrixLike,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> Certificate:
    """Return a certificate for the canonical unobservable subspace ker O(A,C)."""
    policy = tolerance_policy or TolerancePolicy()
    A_num = _as_numeric(A)
    dummy_B = np.zeros((A_num.shape[0], 1), dtype=np.result_type(A_num.dtype, float))
    A_exact = _matrix_exact_if_rational(A)
    C_exact = _matrix_exact_if_rational(C)
    system = LTISystem(
        A_exact if A_exact is not None else A_num,
        sp.zeros(A_num.shape[0], 1) if A_exact is not None else dummy_B,
        C=C_exact if C_exact is not None else C,
    )
    source_A = system.exact_A if system.exact_A is not None else system.A
    source_C = system.exact_C if system.exact_C is not None else system.C
    assert source_C is not None
    data = _unobservable_computation(source_A, source_C, policy)
    assert system.C is not None
    residuals = _subspace_residuals_unobservable(system.A, system.C, data.numerical_basis)
    tol = _residual_tolerance(system.n)
    warning_codes = list(data.warning_codes)
    warnings = list(data.warnings)
    dimension_sensitive = "UNOBSERVABLE_NULLITY_TOLERANCE_SENSITIVE" in warning_codes and data.exact_dimension is None
    hard_failure = (
        not dimension_sensitive
        and (
            residuals["unobservable_invariance_relative_residual"] > tol
            or residuals["output_annihilation_relative_residual"] > tol
        )
    )
    if residuals["unobservable_invariance_relative_residual"] > tol:
        warning_codes.append("UNOBSERVABLE_INVARIANCE_RESIDUAL_LARGE")
        warnings.append("The numerical unobservable basis is not sufficiently A-invariant.")
    if residuals["output_annihilation_relative_residual"] > tol:
        warning_codes.append("OUTPUT_ANNIHILATION_RESIDUAL_LARGE")
        warnings.append("C does not sufficiently annihilate the computed unobservable basis.")

    exact_invariance = None
    exact_output_annihilation = None
    if data.exact_basis is not None and system.exact_A is not None and system.exact_C is not None:
        exact_invariance = _exact_membership(data.exact_basis, sp.Matrix(system.exact_A) * data.exact_basis)
        exact_output_annihilation = (
            sp.Matrix(system.exact_C) * data.exact_basis
            == sp.zeros(system.exact_C.rows, data.exact_basis.cols)
        )
        if not exact_invariance or not exact_output_annihilation:
            hard_failure = True

    passed: bool | None = False if hard_failure else None if dimension_sensitive else True
    verdict = "VALID" if passed is True else "INVALID_SUBSPACE" if passed is False else "BOUNDARY_OR_INCONCLUSIVE"
    return Certificate(
        name="Unobservable subspace",
        passed=passed,
        verdict=verdict,
        evidence={
            "basis": data.numerical_basis,
            "numerical_basis": data.numerical_basis,
            "exact_basis": data.exact_basis,
        },
        diagnostics={
            "state_dimension": system.n,
            "dimension": data.dimension,
            "numerical_dimension": data.numerical_dimension,
            "exact_dimension": data.exact_dimension,
            "singular_values": data.singular_values,
            "tolerance": data.tolerance,
            **residuals,
            "exact_A_invariant": exact_invariance,
            "exact_output_annihilation": exact_output_annihilation,
            "residual_tolerance": tol,
            "warning_codes": list(dict.fromkeys(warning_codes)),
        },
        warnings=list(dict.fromkeys(warnings)),
    )


def _exact_intersection(R: sp.Matrix, N: sp.Matrix) -> sp.Matrix:
    n = R.rows
    if R.cols == 0 or N.cols == 0:
        return _empty_exact_basis(n)
    coupled = R.row_join(-N)
    nullspace = coupled.nullspace()
    if not nullspace:
        return _empty_exact_basis(n)
    candidates: list[sp.Matrix] = []
    current = _empty_exact_basis(n)
    current_rank = 0
    for coeff in nullspace:
        vector = R * coeff[: R.cols, :]
        if vector == sp.zeros(n, 1):
            continue
        trial = current.row_join(vector)
        rank = trial.rank()
        if rank > current_rank:
            candidates.append(vector)
            current = trial
            current_rank = rank
    return sp.Matrix.hstack(*candidates) if candidates else _empty_exact_basis(n)


def _numerical_intersection(
    R: np.ndarray,
    N: np.ndarray,
    policy: TolerancePolicy,
) -> tuple[np.ndarray, int, NumericalRankResult, np.ndarray]:
    n = R.shape[0]
    combined = np.hstack((R, N))
    rank_result = numerical_rank(combined, policy)
    if R.shape[1] == 0 or N.shape[1] == 0:
        return (
            _empty_numeric_basis(n, np.result_type(R.dtype, N.dtype, float)),
            0,
            rank_result,
            np.empty((0,)),
        )
    intersection_dimension = int(R.shape[1] + N.shape[1] - rank_result.rank)
    cross = R.conj().T @ N
    u, cosines, _ = linalg.svd(cross, full_matrices=False)
    if intersection_dimension <= 0:
        basis = _empty_numeric_basis(n, np.result_type(R.dtype, N.dtype, float))
    else:
        basis = R @ u[:, :intersection_dimension]
        basis, _ = linalg.qr(basis, mode="economic")
        basis = basis[:, :intersection_dimension]
    return basis, intersection_dimension, rank_result, cosines


def _exact_added_columns(existing: sp.Matrix, candidates: sp.Matrix, count: int) -> sp.Matrix:
    if count == 0:
        return _empty_exact_basis(existing.rows)
    current = sp.Matrix(existing)
    rank = current.rank()
    selected: list[sp.Matrix] = []
    for j in range(candidates.cols):
        candidate = candidates[:, j]
        trial = current.row_join(candidate)
        new_rank = trial.rank()
        if new_rank > rank:
            selected.append(candidate)
            current = trial
            rank = new_rank
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError("Failed to extend an exact basis to the requested dimension.")
    return sp.Matrix.hstack(*selected)


def _numerical_complement_within(space: np.ndarray, subspace: np.ndarray, count: int) -> np.ndarray:
    n = space.shape[0]
    if count == 0:
        return _empty_numeric_basis(n, np.result_type(space.dtype, subspace.dtype, float))
    projected = _projector_residual(subspace, space)
    u, _, _ = linalg.svd(projected, full_matrices=False)
    return u[:, :count]


def _numerical_sum_complement(R: np.ndarray, N: np.ndarray, rank_sum: int, count: int) -> np.ndarray:
    n = R.shape[0]
    if count == 0:
        return _empty_numeric_basis(n, np.result_type(R.dtype, N.dtype, float))
    combined = np.hstack((R, N))
    u, _, _ = linalg.svd(combined, full_matrices=True)
    return u[:, rank_sum : rank_sum + count]


def _block_slices(dimensions: dict[str, int]) -> tuple[dict[str, tuple[int, int]], dict[str, slice]]:
    metadata: dict[str, tuple[int, int]] = {}
    slices: dict[str, slice] = {}
    start = 0
    for name in ("co", "cu", "uo", "uu"):
        stop = start + int(dimensions[name])
        metadata[name] = (start, stop)
        slices[name] = slice(start, stop)
        start = stop
    return metadata, slices


def _forbidden_A_mask(n: int, slices: dict[str, slice]) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    for row, col in (
        ("co", "cu"),
        ("co", "uu"),
        ("uo", "co"),
        ("uo", "cu"),
        ("uo", "uu"),
        ("uu", "co"),
        ("uu", "cu"),
    ):
        mask[slices[row], slices[col]] = True
    return mask


def _numeric_transform(
    T: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    # Ill-conditioning is reported explicitly in diagnostics; suppress SciPy's
    # duplicate runtime warning while still solving rather than inverting T.
    with py_warnings.catch_warnings():
        py_warnings.simplefilter("ignore", linalg.LinAlgWarning)
        Abar = linalg.solve(T, A @ T)
        Bbar = linalg.solve(T, B)
    Cbar = None if C is None else C @ T
    return Abar, Bbar, Cbar


def _exact_solve_left(T: sp.Matrix, rhs: sp.Matrix) -> sp.Matrix:
    solution, parameters = T.gauss_jordan_solve(rhs)
    if parameters.rows != 0:
        raise RuntimeError("Exact state transformation did not have a unique solution.")
    return sp.Matrix(solution)


def _exact_transform(
    T: sp.Matrix,
    A: sp.Matrix,
    B: sp.Matrix,
    C: sp.Matrix | None,
) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix | None]:
    Abar = _exact_solve_left(T, A * T)
    Bbar = _exact_solve_left(T, B)
    Cbar = None if C is None else C * T
    return Abar, Bbar, Cbar


def _matrix_block_zero_exact(matrix: sp.Matrix, row_slice: slice, col_slice: slice) -> bool:
    block = matrix[row_slice, col_slice]
    return block == sp.zeros(block.rows, block.cols)


def _exact_kalman_zero_pattern(
    Abar: sp.Matrix,
    Bbar: sp.Matrix,
    Cbar: sp.Matrix,
    slices: dict[str, slice],
) -> tuple[bool, bool, bool]:
    a_ok = all(
        _matrix_block_zero_exact(Abar, slices[row], slices[col])
        for row, col in (
            ("co", "cu"),
            ("co", "uu"),
            ("uo", "co"),
            ("uo", "cu"),
            ("uo", "uu"),
            ("uu", "co"),
            ("uu", "cu"),
        )
    )
    lower_rows = list(range(slices["uo"].start, slices["uo"].stop)) + list(
        range(slices["uu"].start, slices["uu"].stop)
    )
    b_ok = all(Bbar[i, j] == 0 for i in lower_rows for j in range(Bbar.cols))
    forbidden_cols = list(range(slices["cu"].start, slices["cu"].stop)) + list(
        range(slices["uu"].start, slices["uu"].stop)
    )
    c_ok = all(Cbar[i, j] == 0 for i in range(Cbar.rows) for j in forbidden_cols)
    return a_ok, b_ok, c_ok


def _transformation_residuals(
    T: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray | None,
    Abar: np.ndarray,
    Bbar: np.ndarray,
    Cbar: np.ndarray | None,
) -> dict[str, float]:
    resA = T @ Abar - A @ T
    a_abs, a_rel = _relative_norm(resA, T @ Abar, A @ T)
    resB = T @ Bbar - B
    b_abs, b_rel = _relative_norm(resB, T @ Bbar, B)
    if C is None or Cbar is None:
        c_abs = c_rel = 0.0
    else:
        resC = Cbar - C @ T
        c_abs, c_rel = _relative_norm(resC, Cbar, C @ T)
    return {
        "similarity_residual_A": a_abs,
        "similarity_relative_residual_A": a_rel,
        "similarity_residual_B": b_abs,
        "similarity_relative_residual_B": b_rel,
        "similarity_residual_C": c_abs,
        "similarity_relative_residual_C": c_rel,
    }


def _kalman_structural_residuals(
    Abar: np.ndarray,
    Bbar: np.ndarray,
    Cbar: np.ndarray,
    slices: dict[str, slice],
) -> dict[str, float]:
    mask = _forbidden_A_mask(Abar.shape[0], slices)
    forbidden_A = np.where(mask, Abar, 0)
    a_abs = float(linalg.norm(forbidden_A, ord="fro"))
    a_scale = float(linalg.norm(Abar, ord="fro"))
    a_rel = a_abs / a_scale if a_scale > 0 else a_abs

    rows = list(range(slices["uo"].start, slices["uo"].stop)) + list(
        range(slices["uu"].start, slices["uu"].stop)
    )
    forbidden_B = Bbar[rows, :] if rows else np.empty((0, Bbar.shape[1]), dtype=Bbar.dtype)
    b_abs = float(linalg.norm(forbidden_B, ord="fro"))
    b_scale = float(linalg.norm(Bbar, ord="fro"))
    b_rel = b_abs / b_scale if b_scale > 0 else b_abs

    cols = list(range(slices["cu"].start, slices["cu"].stop)) + list(
        range(slices["uu"].start, slices["uu"].stop)
    )
    forbidden_C = Cbar[:, cols] if cols else np.empty((Cbar.shape[0], 0), dtype=Cbar.dtype)
    c_abs = float(linalg.norm(forbidden_C, ord="fro"))
    c_scale = float(linalg.norm(Cbar, ord="fro"))
    c_rel = c_abs / c_scale if c_scale > 0 else c_abs
    return {
        "A_structural_zero_residual": a_abs,
        "A_structural_zero_relative_residual": a_rel,
        "B_structural_zero_residual": b_abs,
        "B_structural_zero_relative_residual": b_rel,
        "C_structural_zero_residual": c_abs,
        "C_structural_zero_relative_residual": c_rel,
    }


def _basis_columns_numeric(matrix: sp.Matrix) -> np.ndarray:
    return _as_numeric(matrix)


def analyze_reachability_decomposition(
    system: LTISystem,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> Certificate:
    """Construct and verify a reachability decomposition x = T_R z."""
    policy = tolerance_policy or TolerancePolicy()
    source_A = system.exact_A if system.exact_A is not None else system.A
    source_B = system.exact_B if system.exact_B is not None else system.B
    rdata = _reachable_computation(source_A, source_B, policy)
    n = system.n
    exact_T = exact_Abar = exact_Bbar = exact_Cbar = None
    exact_zero_A = exact_zero_B = None
    exact_similarity_A = exact_similarity_B = exact_similarity_C = None

    if rdata.exact_basis is not None and system.exact_A is not None and system.exact_B is not None:
        complement = _exact_added_columns(rdata.exact_basis, sp.eye(n), n - rdata.exact_basis.cols)
        exact_T = rdata.exact_basis.row_join(complement)
        exact_C = sp.Matrix(system.exact_C) if system.exact_C is not None else None
        exact_Abar, exact_Bbar, exact_Cbar = _exact_transform(
            exact_T, sp.Matrix(system.exact_A), sp.Matrix(system.exact_B), exact_C
        )
        r = rdata.exact_basis.cols
        exact_zero_A = exact_Abar[r:, :r] == sp.zeros(n - r, r)
        exact_zero_B = exact_Bbar[r:, :] == sp.zeros(n - r, exact_Bbar.cols)
        exact_similarity_A = exact_T * exact_Abar == sp.Matrix(system.exact_A) * exact_T
        exact_similarity_B = exact_T * exact_Bbar == sp.Matrix(system.exact_B)
        exact_similarity_C = (
            None
            if exact_Cbar is None or system.exact_C is None
            else exact_Cbar == sp.Matrix(system.exact_C) * exact_T
        )
        T = _basis_columns_numeric(exact_T)
        reachable_dimension = r
    else:
        reachable_dimension = rdata.numerical_dimension
        R = rdata.numerical_basis
        if reachable_dimension == n:
            complement = _empty_numeric_basis(n, R.dtype)
        elif reachable_dimension == 0:
            complement = np.eye(n, dtype=np.result_type(system.A.dtype, float))
        else:
            u, _, _ = linalg.svd(R, full_matrices=True)
            complement = u[:, reachable_dimension:]
        T = np.hstack((R, complement))

    Abar, Bbar, Cbar = _numeric_transform(T, system.A, system.B, system.C)
    rr = numerical_rank(T, policy)
    condition = float(np.linalg.cond(T))
    transform_residuals = _transformation_residuals(T, system.A, system.B, system.C, Abar, Bbar, Cbar)
    r = reachable_dimension
    lower_A = Abar[r:, :r]
    lower_B = Bbar[r:, :]
    a_zero_abs = float(linalg.norm(lower_A, ord="fro"))
    b_zero_abs = float(linalg.norm(lower_B, ord="fro"))
    a_scale = float(linalg.norm(Abar, ord="fro"))
    b_scale = float(linalg.norm(Bbar, ord="fro"))
    a_zero_rel = a_zero_abs / a_scale if a_scale > 0 else a_zero_abs
    b_zero_rel = b_zero_abs / b_scale if b_scale > 0 else b_zero_abs
    tol = _residual_tolerance(n)
    codes = list(rdata.warning_codes)
    warnings = list(rdata.warnings)
    hard_failure = rr.rank != n or max(
        transform_residuals["similarity_relative_residual_A"],
        transform_residuals["similarity_relative_residual_B"],
        a_zero_rel,
        b_zero_rel,
    ) > tol
    if rr.rank != n:
        codes.append("KALMAN_BASIS_RANK_DEFICIENT")
        warnings.append("Reachability-decomposition transformation is rank deficient.")
    if condition > _ILL_CONDITIONED_THRESHOLD:
        codes.append("KALMAN_BASIS_ILL_CONDITIONED")
        warnings.append("Reachability-decomposition transformation is severely ill-conditioned.")
    if a_zero_rel > tol or b_zero_rel > tol:
        codes.append("STRUCTURAL_ZERO_RESIDUAL_LARGE")
        warnings.append("Reachability-decomposition zero blocks exceed the verification tolerance.")
    if max(
        transform_residuals["similarity_relative_residual_A"],
        transform_residuals["similarity_relative_residual_B"],
    ) > tol:
        codes.append("TRANSFORMATION_RESIDUAL_LARGE")
        warnings.append("Reachability-decomposition similarity equations exceed the verification tolerance.")

    exact_valid = (
        exact_T is not None
        and exact_zero_A is True
        and exact_zero_B is True
        and exact_similarity_A is True
        and exact_similarity_B is True
        and (exact_similarity_C is not False)
    )
    if exact_T is not None:
        hard_failure = not exact_valid
    boundary = "REACHABLE_RANK_TOLERANCE_SENSITIVE" in codes and rdata.exact_dimension is None
    passed: bool | None = False if hard_failure else None if boundary else True
    verdict = "VALID" if passed is True else "INVALID_DECOMPOSITION" if passed is False else "BOUNDARY_OR_INCONCLUSIVE"
    return Certificate(
        name="Reachability decomposition",
        passed=passed,
        verdict=verdict,
        evidence={
            "reachable_basis": rdata.numerical_basis,
            "reachable_basis_exact": rdata.exact_basis,
            "T": T,
            "T_exact": exact_T,
            "A_transformed": Abar,
            "B_transformed": Bbar,
            "C_transformed": Cbar,
            "D_transformed": None if system.D is None else system.D.copy(),
            "A_transformed_exact": exact_Abar,
            "B_transformed_exact": exact_Bbar,
            "C_transformed_exact": exact_Cbar,
            "D_transformed_exact": system.exact_D,
        },
        diagnostics={
            "state_dimension": n,
            "reachable_dimension": reachable_dimension,
            "reachable_numerical_dimension": rdata.numerical_dimension,
            "reachable_exact_dimension": rdata.exact_dimension,
            "transformation_rank": rr.rank,
            "transformation_condition_number": condition,
            **transform_residuals,
            "A_lower_left_zero_residual": a_zero_abs,
            "A_lower_left_zero_relative_residual": a_zero_rel,
            "B_lower_zero_residual": b_zero_abs,
            "B_lower_zero_relative_residual": b_zero_rel,
            "exact_A_lower_left_zero": exact_zero_A,
            "exact_B_lower_zero": exact_zero_B,
            "exact_similarity_A": exact_similarity_A,
            "exact_similarity_B": exact_similarity_B,
            "exact_similarity_C": exact_similarity_C,
            "residual_tolerance": tol,
            "warning_codes": list(dict.fromkeys(codes)),
            "basis_is_canonical": False,
        },
        warnings=list(dict.fromkeys(warnings)),
    )


def analyze_observability_decomposition(
    system: LTISystem,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> Certificate:
    """Construct and verify an observability decomposition x = T_O z."""
    if system.C is None:
        raise ValueError("C is required for observability decomposition.")
    policy = tolerance_policy or TolerancePolicy()
    source_A = system.exact_A if system.exact_A is not None else system.A
    source_C = system.exact_C if system.exact_C is not None else system.C
    ndata = _unobservable_computation(source_A, source_C, policy)
    n = system.n
    exact_T = exact_Abar = exact_Bbar = exact_Cbar = None
    exact_zero_A = exact_zero_C = None
    exact_similarity_A = exact_similarity_B = exact_similarity_C = None

    if ndata.exact_basis is not None and system.exact_A is not None and system.exact_C is not None:
        observable_dimension = n - ndata.exact_basis.cols
        observable_complement = _exact_added_columns(ndata.exact_basis, sp.eye(n), observable_dimension)
        exact_T = observable_complement.row_join(ndata.exact_basis)
        exact_B = sp.Matrix(system.exact_B) if system.exact_B is not None else sp.Matrix(system.B)
        exact_Abar, exact_Bbar, exact_Cbar = _exact_transform(
            exact_T, sp.Matrix(system.exact_A), exact_B, sp.Matrix(system.exact_C)
        )
        exact_zero_A = exact_Abar[:observable_dimension, observable_dimension:] == sp.zeros(
            observable_dimension, n - observable_dimension
        )
        exact_zero_C = exact_Cbar[:, observable_dimension:] == sp.zeros(
            exact_Cbar.rows, n - observable_dimension
        )
        exact_similarity_A = exact_T * exact_Abar == sp.Matrix(system.exact_A) * exact_T
        exact_similarity_B = (
            None if system.exact_B is None else exact_T * exact_Bbar == sp.Matrix(system.exact_B)
        )
        exact_similarity_C = exact_Cbar == sp.Matrix(system.exact_C) * exact_T
        T = _basis_columns_numeric(exact_T)
        unobservable_dimension = ndata.exact_basis.cols
    else:
        unobservable_dimension = ndata.numerical_dimension
        N = ndata.numerical_basis
        if unobservable_dimension == 0:
            observable_complement = np.eye(n, dtype=np.result_type(system.A.dtype, float))
        elif unobservable_dimension == n:
            observable_complement = _empty_numeric_basis(n, N.dtype)
        else:
            u, _, _ = linalg.svd(N, full_matrices=True)
            observable_complement = u[:, unobservable_dimension:]
        T = np.hstack((observable_complement, N))
        observable_dimension = n - unobservable_dimension

    Abar, Bbar, Cbar = _numeric_transform(T, system.A, system.B, system.C)
    assert Cbar is not None
    rr = numerical_rank(T, policy)
    condition = float(np.linalg.cond(T))
    transform_residuals = _transformation_residuals(T, system.A, system.B, system.C, Abar, Bbar, Cbar)
    upper_right = Abar[:observable_dimension, observable_dimension:]
    right_C = Cbar[:, observable_dimension:]
    a_zero_abs = float(linalg.norm(upper_right, ord="fro"))
    c_zero_abs = float(linalg.norm(right_C, ord="fro"))
    a_scale = float(linalg.norm(Abar, ord="fro"))
    c_scale = float(linalg.norm(Cbar, ord="fro"))
    a_zero_rel = a_zero_abs / a_scale if a_scale > 0 else a_zero_abs
    c_zero_rel = c_zero_abs / c_scale if c_scale > 0 else c_zero_abs
    tol = _residual_tolerance(n)
    codes = list(ndata.warning_codes)
    warnings = list(ndata.warnings)
    hard_failure = rr.rank != n or max(
        transform_residuals["similarity_relative_residual_A"],
        transform_residuals["similarity_relative_residual_C"],
        a_zero_rel,
        c_zero_rel,
    ) > tol
    if rr.rank != n:
        codes.append("KALMAN_BASIS_RANK_DEFICIENT")
        warnings.append("Observability-decomposition transformation is rank deficient.")
    if condition > _ILL_CONDITIONED_THRESHOLD:
        codes.append("KALMAN_BASIS_ILL_CONDITIONED")
        warnings.append("Observability-decomposition transformation is severely ill-conditioned.")
    if a_zero_rel > tol or c_zero_rel > tol:
        codes.append("STRUCTURAL_ZERO_RESIDUAL_LARGE")
        warnings.append("Observability-decomposition zero blocks exceed the verification tolerance.")
    if max(
        transform_residuals["similarity_relative_residual_A"],
        transform_residuals["similarity_relative_residual_C"],
    ) > tol:
        codes.append("TRANSFORMATION_RESIDUAL_LARGE")
        warnings.append("Observability-decomposition similarity equations exceed the verification tolerance.")

    exact_valid = (
        exact_T is not None
        and exact_zero_A is True
        and exact_zero_C is True
        and exact_similarity_A is True
        and exact_similarity_C is True
        and (exact_similarity_B is not False)
    )
    if exact_T is not None:
        hard_failure = not exact_valid
    boundary = "UNOBSERVABLE_NULLITY_TOLERANCE_SENSITIVE" in codes and ndata.exact_dimension is None
    passed: bool | None = False if hard_failure else None if boundary else True
    verdict = "VALID" if passed is True else "INVALID_DECOMPOSITION" if passed is False else "BOUNDARY_OR_INCONCLUSIVE"
    return Certificate(
        name="Observability decomposition",
        passed=passed,
        verdict=verdict,
        evidence={
            "unobservable_basis": ndata.numerical_basis,
            "unobservable_basis_exact": ndata.exact_basis,
            "T": T,
            "T_exact": exact_T,
            "A_transformed": Abar,
            "B_transformed": Bbar,
            "C_transformed": Cbar,
            "D_transformed": None if system.D is None else system.D.copy(),
            "A_transformed_exact": exact_Abar,
            "B_transformed_exact": exact_Bbar,
            "C_transformed_exact": exact_Cbar,
            "D_transformed_exact": system.exact_D,
        },
        diagnostics={
            "state_dimension": n,
            "observable_complement_dimension": observable_dimension,
            "unobservable_dimension": unobservable_dimension,
            "unobservable_numerical_dimension": ndata.numerical_dimension,
            "unobservable_exact_dimension": ndata.exact_dimension,
            "transformation_rank": rr.rank,
            "transformation_condition_number": condition,
            **transform_residuals,
            "A_upper_right_zero_residual": a_zero_abs,
            "A_upper_right_zero_relative_residual": a_zero_rel,
            "C_right_zero_residual": c_zero_abs,
            "C_right_zero_relative_residual": c_zero_rel,
            "exact_A_upper_right_zero": exact_zero_A,
            "exact_C_right_zero": exact_zero_C,
            "exact_similarity_A": exact_similarity_A,
            "exact_similarity_B": exact_similarity_B,
            "exact_similarity_C": exact_similarity_C,
            "residual_tolerance": tol,
            "warning_codes": list(dict.fromkeys(codes)),
            "basis_is_canonical": False,
        },
        warnings=list(dict.fromkeys(warnings)),
    )


def analyze_kalman_decomposition(
    system: LTISystem,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> Certificate:
    """Construct a four-part Kalman structural decomposition certificate.

    The canonical structural objects are R, N, R∩N, and their dimensions.
    Complement bases and the particular state transformation T are generally
    non-unique and are therefore explicitly marked non-canonical.
    """
    if system.C is None:
        raise ValueError("C is required for full Kalman decomposition.")
    policy = tolerance_policy or TolerancePolicy()
    n = system.n
    source_A_r = system.exact_A if system.exact_A is not None else system.A
    source_B = system.exact_B if system.exact_B is not None else system.B
    source_A_o = system.exact_A if system.exact_A is not None else system.A
    source_C = system.exact_C if system.exact_C is not None else system.C
    rdata = _reachable_computation(source_A_r, source_B, policy)
    ndata = _unobservable_computation(source_A_o, source_C, policy)

    intersection_num, intersection_num_dim, intersection_rank, principal_cosines = _numerical_intersection(
        rdata.numerical_basis, ndata.numerical_basis, policy
    )
    codes = list(rdata.warning_codes) + list(ndata.warning_codes)
    warnings = list(rdata.warnings) + list(ndata.warnings)
    intersection_sensitive = _rank_tolerance_sensitive(intersection_rank)
    if intersection_sensitive:
        codes.append("SUBSPACE_INTERSECTION_TOLERANCE_SENSITIVE")
        warnings.append("The numerical dimension of R ∩ N is sensitive to the active tolerance.")

    intersection_in_R = _projector_residual(rdata.numerical_basis, intersection_num)
    intersection_in_N = _projector_residual(ndata.numerical_basis, intersection_num)
    intersection_R_abs, intersection_R_rel = _relative_norm(intersection_in_R, intersection_num)
    intersection_N_abs, intersection_N_rel = _relative_norm(intersection_in_N, intersection_num)

    exact_intersection = None
    exact_intersection_dim = None
    exact_intersection_in_R = None
    exact_intersection_in_N = None
    if rdata.exact_basis is not None and ndata.exact_basis is not None:
        exact_intersection = _exact_intersection(rdata.exact_basis, ndata.exact_basis)
        exact_intersection_dim = exact_intersection.cols
        exact_intersection_in_R = _exact_membership(rdata.exact_basis, exact_intersection)
        exact_intersection_in_N = _exact_membership(ndata.exact_basis, exact_intersection)
        if exact_intersection_dim != intersection_num_dim:
            codes.append("EXACT_NUMERICAL_DIMENSION_DISAGREEMENT")
            warnings.append(
                "Exact and numerical R ∩ N dimensions disagree; exact algebraic dimensions are authoritative."
            )

    exact_reachable_invariance = None
    exact_input_inclusion = None
    exact_unobservable_invariance = None
    exact_output_annihilation = None
    if rdata.exact_basis is not None and system.exact_A is not None and system.exact_B is not None:
        exact_reachable_invariance = _exact_membership(
            rdata.exact_basis, sp.Matrix(system.exact_A) * rdata.exact_basis
        )
        exact_input_inclusion = _exact_membership(rdata.exact_basis, sp.Matrix(system.exact_B))
    if ndata.exact_basis is not None and system.exact_A is not None and system.exact_C is not None:
        exact_unobservable_invariance = _exact_membership(
            ndata.exact_basis, sp.Matrix(system.exact_A) * ndata.exact_basis
        )
        exact_output_annihilation = (
            sp.Matrix(system.exact_C) * ndata.exact_basis
            == sp.zeros(system.exact_C.rows, ndata.exact_basis.cols)
        )

    use_exact = (
        exact_intersection is not None
        and rdata.exact_basis is not None
        and ndata.exact_basis is not None
        and system.exact_A is not None
        and system.exact_B is not None
        and system.exact_C is not None
    )

    exact_T = exact_Abar = exact_Bbar = exact_Cbar = None
    exact_zero_A = exact_zero_B = exact_zero_C = None
    exact_similarity_A = exact_similarity_B = exact_similarity_C = None

    if use_exact:
        R_exact = rdata.exact_basis
        N_exact = ndata.exact_basis
        I_exact = exact_intersection
        assert R_exact is not None and N_exact is not None and I_exact is not None
        d_cu = I_exact.cols
        d_co = R_exact.cols - d_cu
        d_uu = N_exact.cols - d_cu
        rank_sum_exact = R_exact.row_join(N_exact).rank()
        d_uo = n - rank_sum_exact
        T_cu_exact = I_exact
        T_co_exact = _exact_added_columns(T_cu_exact, R_exact, d_co)
        T_uu_exact = _exact_added_columns(T_cu_exact, N_exact, d_uu)
        existing = T_co_exact.row_join(T_cu_exact).row_join(T_uu_exact)
        T_uo_exact = _exact_added_columns(existing, sp.eye(n), d_uo)
        exact_T = T_co_exact.row_join(T_cu_exact).row_join(T_uo_exact).row_join(T_uu_exact)
        if exact_T.rank() != n:
            raise RuntimeError("Exact Kalman basis construction produced a rank-deficient transformation.")
        exact_Abar, exact_Bbar, exact_Cbar = _exact_transform(
            exact_T, sp.Matrix(system.exact_A), sp.Matrix(system.exact_B), sp.Matrix(system.exact_C)
        )
        dimensions = {"co": d_co, "cu": d_cu, "uo": d_uo, "uu": d_uu}
        _, exact_slices = _block_slices(dimensions)
        exact_zero_A, exact_zero_B, exact_zero_C = _exact_kalman_zero_pattern(
            exact_Abar, exact_Bbar, exact_Cbar, exact_slices
        )
        exact_similarity_A = exact_T * exact_Abar == sp.Matrix(system.exact_A) * exact_T
        exact_similarity_B = exact_T * exact_Bbar == sp.Matrix(system.exact_B)
        exact_similarity_C = exact_Cbar == sp.Matrix(system.exact_C) * exact_T
        T = _basis_columns_numeric(exact_T)
        T_co = _basis_columns_numeric(T_co_exact)
        T_cu = _basis_columns_numeric(T_cu_exact)
        T_uo = _basis_columns_numeric(T_uo_exact)
        T_uu = _basis_columns_numeric(T_uu_exact)
        reachable_dimension = R_exact.cols
        unobservable_dimension = N_exact.cols
        intersection_dimension = I_exact.cols
        rank_sum = rank_sum_exact
    else:
        reachable_dimension = rdata.numerical_dimension
        unobservable_dimension = ndata.numerical_dimension
        intersection_dimension = intersection_num_dim
        rank_sum = intersection_rank.rank
        d_cu = intersection_dimension
        d_co = reachable_dimension - d_cu
        d_uu = unobservable_dimension - d_cu
        d_uo = n - rank_sum
        dimensions = {"co": d_co, "cu": d_cu, "uo": d_uo, "uu": d_uu}
        T_cu = intersection_num
        T_co = _numerical_complement_within(rdata.numerical_basis, T_cu, d_co)
        T_uu = _numerical_complement_within(ndata.numerical_basis, T_cu, d_uu)
        T_uo = _numerical_sum_complement(
            rdata.numerical_basis, ndata.numerical_basis, rank_sum, d_uo
        )
        T = np.hstack((T_co, T_cu, T_uo, T_uu))

    if any(value < 0 for value in dimensions.values()) or sum(dimensions.values()) != n:
        return Certificate(
            name="Kalman structural decomposition",
            passed=False,
            verdict="INVALID_DECOMPOSITION",
            evidence={
                "reachable_basis": rdata.numerical_basis,
                "unobservable_basis": ndata.numerical_basis,
            },
            diagnostics={
                "state_dimension": n,
                "controllable_observable_dimension": dimensions.get("co"),
                "controllable_unobservable_dimension": dimensions.get("cu"),
                "uncontrollable_observable_dimension": dimensions.get("uo"),
                "uncontrollable_unobservable_dimension": dimensions.get("uu"),
                "warning_codes": [*codes, "INVALID_DIMENSION_IDENTITY"],
            },
            warnings=[*warnings, "Computed Kalman block dimensions are inconsistent with the state dimension."],
        )

    metadata_slices, slices = _block_slices(dimensions)
    Abar, Bbar, Cbar = _numeric_transform(T, system.A, system.B, system.C)
    assert Cbar is not None
    transform_rank = numerical_rank(T, policy)
    condition = float(np.linalg.cond(T))
    transform_residuals = _transformation_residuals(
        T, system.A, system.B, system.C, Abar, Bbar, Cbar
    )
    structural_residuals = _kalman_structural_residuals(Abar, Bbar, Cbar, slices)
    reachable_residuals = _subspace_residuals_reachable(
        system.A, system.B, rdata.numerical_basis
    )
    unobservable_residuals = _subspace_residuals_unobservable(
        system.A, system.C, ndata.numerical_basis
    )
    tol = _residual_tolerance(n)

    if transform_rank.rank != n:
        codes.append("KALMAN_BASIS_RANK_DEFICIENT")
        warnings.append("The assembled Kalman transformation is rank deficient.")
    if condition > _ILL_CONDITIONED_THRESHOLD:
        codes.append("KALMAN_BASIS_ILL_CONDITIONED")
        warnings.append("The Kalman transformation is severely ill-conditioned.")
    if (
        reachable_residuals["reachable_invariance_relative_residual"] > tol
        or reachable_residuals["input_inclusion_relative_residual"] > tol
    ):
        codes.append("REACHABLE_INVARIANCE_RESIDUAL_LARGE")
        warnings.append("Reachable-subspace invariance or input inclusion residual is too large.")
    if (
        unobservable_residuals["unobservable_invariance_relative_residual"] > tol
        or unobservable_residuals["output_annihilation_relative_residual"] > tol
    ):
        codes.append("UNOBSERVABLE_INVARIANCE_RESIDUAL_LARGE")
        warnings.append("Unobservable-subspace invariance or output-annihilation residual is too large.")
    if max(
        transform_residuals["similarity_relative_residual_A"],
        transform_residuals["similarity_relative_residual_B"],
        transform_residuals["similarity_relative_residual_C"],
    ) > tol:
        codes.append("TRANSFORMATION_RESIDUAL_LARGE")
        warnings.append("One or more state-transformation equations exceed the residual tolerance.")
    if max(
        structural_residuals["A_structural_zero_relative_residual"],
        structural_residuals["B_structural_zero_relative_residual"],
        structural_residuals["C_structural_zero_relative_residual"],
    ) > tol:
        codes.append("STRUCTURAL_ZERO_RESIDUAL_LARGE")
        warnings.append("One or more theoretically required Kalman zero blocks exceed the residual tolerance.")

    co_controllability = None
    co_observability = None
    if dimensions["co"] > 0:
        s = slices["co"]
        if use_exact and exact_Abar is not None and exact_Bbar is not None and exact_Cbar is not None:
            co_system = LTISystem(exact_Abar[s, s], exact_Bbar[s, :], C=exact_Cbar[:, s])
        else:
            co_system = LTISystem(Abar[s, s], Bbar[s, :], C=Cbar[:, s])
        co_controllability = analyze_controllability(co_system, policy)
        co_observability = analyze_observability(co_system, policy)
        if not co_controllability.passed or not co_observability.passed:
            codes.append("CO_BLOCK_NOT_MINIMAL")
            warnings.append(
                "The controllable/observable diagonal block failed a controllability or observability cross-check."
            )

    exact_failure = use_exact and not all(
        value is True
        for value in (
            exact_zero_A,
            exact_zero_B,
            exact_zero_C,
            exact_similarity_A,
            exact_similarity_B,
            exact_similarity_C,
            exact_reachable_invariance,
            exact_input_inclusion,
            exact_unobservable_invariance,
            exact_output_annihilation,
            exact_intersection_in_R,
            exact_intersection_in_N,
        )
    )
    numerical_boundary = (
        not use_exact
        and (
            "REACHABLE_RANK_TOLERANCE_SENSITIVE" in codes
            or "UNOBSERVABLE_NULLITY_TOLERANCE_SENSITIVE" in codes
            or "SUBSPACE_INTERSECTION_TOLERANCE_SENSITIVE" in codes
        )
    )
    transform_failure = max(
        transform_residuals["similarity_relative_residual_A"],
        transform_residuals["similarity_relative_residual_B"],
        transform_residuals["similarity_relative_residual_C"],
    ) > tol
    structural_or_subspace_failure = max(
        structural_residuals["A_structural_zero_relative_residual"],
        structural_residuals["B_structural_zero_relative_residual"],
        structural_residuals["C_structural_zero_relative_residual"],
        reachable_residuals["reachable_invariance_relative_residual"],
        reachable_residuals["input_inclusion_relative_residual"],
        unobservable_residuals["unobservable_invariance_relative_residual"],
        unobservable_residuals["output_annihilation_relative_residual"],
        intersection_R_rel,
        intersection_N_rel,
    ) > tol
    co_failure = (
        (co_controllability is not None and not co_controllability.passed)
        or (co_observability is not None and not co_observability.passed)
    )
    if use_exact:
        # Exact rational structural identities are authoritative. Numerical
        # transforms may be unreliable for a severely ill-conditioned exact T;
        # retain those diagnostics as warnings without overriding exact truth.
        hard_failure = bool(exact_failure or co_failure)
    else:
        hard_failure = (
            transform_rank.rank != n
            or transform_failure
            or (structural_or_subspace_failure and not numerical_boundary)
            or co_failure
        )
    passed: bool | None = False if hard_failure else None if numerical_boundary else True
    if passed is False:
        verdict = "INVALID_DECOMPOSITION"
    elif passed is None:
        verdict = "BOUNDARY_OR_INCONCLUSIVE"
    elif codes:
        verdict = "VALID_WITH_NUMERICAL_WARNINGS"
    else:
        verdict = "VALID"

    return Certificate(
        name="Kalman structural decomposition",
        passed=passed,
        verdict=verdict,
        evidence={
            "reachable_basis": (
                _basis_columns_numeric(rdata.exact_basis)
                if use_exact and rdata.exact_basis is not None
                else rdata.numerical_basis
            ),
            "reachable_basis_numerical": rdata.numerical_basis,
            "reachable_basis_exact": rdata.exact_basis,
            "unobservable_basis": (
                _basis_columns_numeric(ndata.exact_basis)
                if use_exact and ndata.exact_basis is not None
                else ndata.numerical_basis
            ),
            "unobservable_basis_numerical": ndata.numerical_basis,
            "unobservable_basis_exact": ndata.exact_basis,
            "intersection_basis": T_cu,
            "intersection_basis_numerical": intersection_num,
            "intersection_basis_exact": exact_intersection,
            "T_co": T_co,
            "T_cu": T_cu,
            "T_uo": T_uo,
            "T_uu": T_uu,
            "T": T,
            "T_exact": exact_T,
            "A_transformed": Abar,
            "B_transformed": Bbar,
            "C_transformed": Cbar,
            "D_transformed": None if system.D is None else system.D.copy(),
            "A_transformed_exact": exact_Abar,
            "B_transformed_exact": exact_Bbar,
            "C_transformed_exact": exact_Cbar,
            "D_transformed_exact": system.exact_D,
            "co_block_controllability": co_controllability,
            "co_block_observability": co_observability,
        },
        diagnostics={
            "state_dimension": n,
            "reachable_dimension": reachable_dimension,
            "reachable_numerical_dimension": rdata.numerical_dimension,
            "reachable_exact_dimension": rdata.exact_dimension,
            "unobservable_dimension": unobservable_dimension,
            "unobservable_numerical_dimension": ndata.numerical_dimension,
            "unobservable_exact_dimension": ndata.exact_dimension,
            "intersection_dimension": intersection_dimension,
            "intersection_numerical_dimension": intersection_num_dim,
            "intersection_exact_dimension": exact_intersection_dim,
            "sum_subspace_dimension": rank_sum,
            "controllable_observable_dimension": dimensions["co"],
            "controllable_unobservable_dimension": dimensions["cu"],
            "uncontrollable_observable_dimension": dimensions["uo"],
            "uncontrollable_unobservable_dimension": dimensions["uu"],
            "reachable_observable_dimension": dimensions["co"],
            "dimensions_sum_to_state_dimension": sum(dimensions.values()) == n,
            "block_slices": metadata_slices,
            "basis_is_canonical": False,
            "canonical_subspaces": (
                "reachable",
                "unobservable",
                "reachable_intersection_unobservable",
            ),
            "intersection_principal_cosines": principal_cosines,
            "intersection_rank_singular_values": intersection_rank.singular_values,
            "intersection_tolerance": intersection_rank.tolerance,
            "intersection_reachable_residual": intersection_R_abs,
            "intersection_reachable_relative_residual": intersection_R_rel,
            "intersection_unobservable_residual": intersection_N_abs,
            "intersection_unobservable_relative_residual": intersection_N_rel,
            "exact_intersection_in_reachable": exact_intersection_in_R,
            "exact_intersection_in_unobservable": exact_intersection_in_N,
            "exact_reachable_invariance": exact_reachable_invariance,
            "exact_input_inclusion": exact_input_inclusion,
            "exact_unobservable_invariance": exact_unobservable_invariance,
            "exact_output_annihilation": exact_output_annihilation,
            "transformation_rank": transform_rank.rank,
            "transformation_condition_number": condition,
            **reachable_residuals,
            **unobservable_residuals,
            **transform_residuals,
            **structural_residuals,
            "exact_A_structural_zero_pattern": exact_zero_A,
            "exact_B_structural_zero_pattern": exact_zero_B,
            "exact_C_structural_zero_pattern": exact_zero_C,
            "exact_similarity_A": exact_similarity_A,
            "exact_similarity_B": exact_similarity_B,
            "exact_similarity_C": exact_similarity_C,
            "co_block_controllable": None if co_controllability is None else co_controllability.passed,
            "co_block_observable": None if co_observability is None else co_observability.passed,
            "residual_tolerance": tol,
            "warning_codes": list(dict.fromkeys(codes)),
        },
        warnings=list(dict.fromkeys(warnings)),
    )