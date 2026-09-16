import json

import numpy as np
import pytest
import sympy as sp

from certicontrol import LTISystem, TolerancePolicy
from certicontrol.decomposition import (
    analyze_kalman_decomposition,
    analyze_observability_decomposition,
    analyze_reachability_decomposition,
    reachable_subspace,
    unobservable_subspace,
)


def _dims(cert):
    d = cert.diagnostics
    return (
        d["controllable_observable_dimension"],
        d["controllable_unobservable_dimension"],
        d["uncontrollable_observable_dimension"],
        d["uncontrollable_unobservable_dimension"],
    )


def _all_four_exact_system():
    A = sp.Matrix(
        [
            [-1, 0, sp.Rational(1, 3), 0],
            [sp.Rational(1, 5), -2, sp.Rational(1, 4), sp.Rational(1, 2)],
            [0, 0, -3, 0],
            [0, 0, sp.Rational(2, 5), -4],
        ]
    )
    B = sp.Matrix([[1], [1], [0], [0]])
    C = sp.Matrix([[1, 0, 1, 0]])
    return LTISystem(A, B, C=C)


def test_reachable_subspace_fully_reachable_exact_and_numerical():
    cert = reachable_subspace([[0, 1], [-2, -3]], [[0], [1]])
    assert cert.passed is True
    assert cert.diagnostics["dimension"] == 2
    assert cert.diagnostics["exact_dimension"] == 2
    assert cert.diagnostics["numerical_dimension"] == 2
    assert cert.diagnostics["exact_A_invariant"] is True
    assert cert.diagnostics["exact_input_inclusion"] is True
    assert cert.diagnostics["reachable_invariance_residual"] < 1e-12
    assert cert.diagnostics["input_inclusion_residual"] < 1e-12


def test_reachable_subspace_partial_and_zero_input():
    partial = reachable_subspace(np.diag([-1.0, -2.0]), np.array([[1.0], [0.0]]))
    assert partial.diagnostics["dimension"] == 1
    zero = reachable_subspace(np.diag([-1.0, -2.0]), np.zeros((2, 1)))
    assert zero.passed is True
    assert zero.diagnostics["dimension"] == 0
    assert zero.evidence["numerical_basis"].shape == (2, 0)


def test_unobservable_subspace_fully_observable_and_partial():
    observable = unobservable_subspace([[0, 1], [-2, -3]], [[1, 0]])
    assert observable.passed is True
    assert observable.diagnostics["dimension"] == 0
    partial = unobservable_subspace(np.diag([-1.0, -2.0]), np.array([[1.0, 0.0]]))
    assert partial.diagnostics["dimension"] == 1
    assert partial.diagnostics["unobservable_invariance_residual"] < 1e-12
    assert partial.diagnostics["output_annihilation_residual"] < 1e-12


def test_completely_unobservable_subspace_handles_zero_c():
    cert = unobservable_subspace([[0, 1], [-2, -3]], [[0, 0]])
    assert cert.passed is True
    assert cert.diagnostics["dimension"] == 2
    assert cert.evidence["numerical_basis"].shape == (2, 2)
    assert cert.diagnostics["exact_output_annihilation"] is True


def test_exact_rational_subspaces_preserve_exact_bases():
    A = sp.Matrix([[0, sp.Rational(1, 2)], [0, -1]])
    B = sp.Matrix([[1], [0]])
    C = sp.Matrix([[0, sp.Rational(2, 3)]])
    r = reachable_subspace(A, B)
    n = unobservable_subspace(A, C)
    assert isinstance(r.evidence["exact_basis"], sp.MatrixBase)
    assert isinstance(n.evidence["exact_basis"], sp.MatrixBase)
    assert r.diagnostics["exact_dimension"] == 1
    assert n.diagnostics["exact_dimension"] == 1


def test_reachability_decomposition_has_correct_zero_orientation():
    system = LTISystem(np.diag([-1.0, -2.0]), np.array([[1.0], [0.0]]), C=[[1.0, 1.0]])
    cert = analyze_reachability_decomposition(system)
    assert cert.passed is True
    assert cert.diagnostics["reachable_dimension"] == 1
    Abar = cert.evidence["A_transformed"]
    Bbar = cert.evidence["B_transformed"]
    assert np.linalg.norm(Abar[1:, :1]) < 1e-12
    assert np.linalg.norm(Bbar[1:, :]) < 1e-12
    assert cert.diagnostics["A_lower_left_zero_relative_residual"] < cert.diagnostics["residual_tolerance"]


def test_reachability_decomposition_works_without_c():
    system = LTISystem([[0, 1], [0, 0]], [[0], [1]])
    cert = analyze_reachability_decomposition(system)
    assert cert.passed is True
    assert cert.evidence["C_transformed"] is None


def test_observability_decomposition_has_correct_zero_orientation():
    system = LTISystem(np.diag([-1.0, -2.0]), [[1.0], [1.0]], C=[[1.0, 0.0]])
    cert = analyze_observability_decomposition(system)
    assert cert.passed is True
    assert cert.diagnostics["unobservable_dimension"] == 1
    Abar = cert.evidence["A_transformed"]
    Cbar = cert.evidence["C_transformed"]
    assert np.linalg.norm(Abar[:1, 1:]) < 1e-12
    assert np.linalg.norm(Cbar[:, 1:]) < 1e-12


def test_observability_decomposition_requires_c():
    with pytest.raises(ValueError, match="C is required"):
        analyze_observability_decomposition(LTISystem([[0, 1], [0, 0]], [[0], [1]]))


def test_full_kalman_requires_c():
    with pytest.raises(ValueError, match="C is required"):
        analyze_kalman_decomposition(LTISystem([[0, 1], [0, 0]], [[0], [1]]))


def test_fully_controllable_observable_system_has_only_co_block():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]], C=[[1, 0]])
    cert = analyze_kalman_decomposition(system)
    assert cert.passed is True
    assert _dims(cert) == (2, 0, 0, 0)
    assert cert.diagnostics["intersection_dimension"] == 0
    assert cert.diagnostics["co_block_controllable"] is True
    assert cert.diagnostics["co_block_observable"] is True


def test_controllable_but_not_observable_dimensions():
    system = LTISystem(np.diag([-1.0, -2.0]), np.array([[1.0], [1.0]]), C=[[1.0, 0.0]])
    cert = analyze_kalman_decomposition(system)
    assert cert.passed is True
    assert _dims(cert) == (1, 1, 0, 0)


def test_observable_but_not_controllable_dimensions():
    system = LTISystem(np.diag([-1.0, -2.0]), np.array([[1.0], [0.0]]), C=[[1.0, 1.0]])
    cert = analyze_kalman_decomposition(system)
    assert cert.passed is True
    assert _dims(cert) == (1, 0, 1, 0)


def test_neither_controllable_nor_observable_simple_system():
    system = LTISystem(np.diag([-1.0, -2.0]), np.array([[1.0], [0.0]]), C=[[0.0, 1.0]])
    cert = analyze_kalman_decomposition(system)
    assert cert.passed is True
    assert _dims(cert) == (0, 1, 1, 0)
    assert cert.diagnostics["intersection_dimension"] == 1


def test_all_four_kalman_classes_nonzero_exact_fixture():
    cert = analyze_kalman_decomposition(_all_four_exact_system())
    assert cert.passed is True
    assert cert.verdict == "VALID"
    assert cert.diagnostics["reachable_dimension"] == 2
    assert cert.diagnostics["unobservable_dimension"] == 2
    assert cert.diagnostics["intersection_dimension"] == 1
    assert _dims(cert) == (1, 1, 1, 1)
    assert cert.diagnostics["dimensions_sum_to_state_dimension"] is True


def test_all_four_fixture_does_not_overconstrain_allowed_off_diagonal_blocks():
    cert = analyze_kalman_decomposition(_all_four_exact_system())
    Abar = cert.evidence["A_transformed_exact"]
    assert Abar[1, 0] != 0
    assert Abar[0, 2] != 0
    assert Abar[1, 3] != 0
    assert Abar[3, 2] != 0
    assert cert.diagnostics["exact_A_structural_zero_pattern"] is True


def test_all_four_exact_intersection_is_verified_in_both_subspaces():
    cert = analyze_kalman_decomposition(_all_four_exact_system())
    assert cert.diagnostics["exact_intersection_in_reachable"] is True
    assert cert.diagnostics["exact_intersection_in_unobservable"] is True
    assert cert.evidence["intersection_basis_exact"].cols == 1
    assert cert.diagnostics["intersection_reachable_residual"] < 1e-12
    assert cert.diagnostics["intersection_unobservable_residual"] < 1e-12


def test_exact_structural_certificate_has_exact_zero_relations():
    cert = analyze_kalman_decomposition(_all_four_exact_system())
    assert cert.evidence["T_exact"].rank() == 4
    assert cert.diagnostics["exact_A_structural_zero_pattern"] is True
    assert cert.diagnostics["exact_B_structural_zero_pattern"] is True
    assert cert.diagnostics["exact_C_structural_zero_pattern"] is True
    assert cert.diagnostics["exact_similarity_A"] is True
    assert cert.diagnostics["exact_similarity_B"] is True
    assert cert.diagnostics["exact_similarity_C"] is True
    assert cert.diagnostics["exact_reachable_invariance"] is True
    assert cert.diagnostics["exact_unobservable_invariance"] is True


def test_nontrivial_exact_similarity_transform_preserves_all_four_dimensions():
    base = _all_four_exact_system()
    A, B, C = sp.Matrix(base.exact_A), sp.Matrix(base.exact_B), sp.Matrix(base.exact_C)
    S = sp.Matrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 2]])
    assert S.det() != 0
    transformed = LTISystem(S.inv() * A * S, S.inv() * B, C=C * S)
    cert = analyze_kalman_decomposition(transformed)
    assert cert.passed is True
    assert _dims(cert) == (1, 1, 1, 1)
    assert cert.diagnostics["exact_A_structural_zero_pattern"] is True


def test_numeric_similarity_invariance_of_all_structural_dimensions():
    A = np.diag([-1.0, -2.0, -3.0, -4.0])
    B = np.array([[1.0], [1.0], [0.0], [0.0]])
    C = np.array([[1.0, 0.0, 1.0, 0.0]])
    original = analyze_kalman_decomposition(LTISystem(A, B, C=C))
    S = np.array([[2.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.5, 0.2], [0.0, 0.0, 0.3, 1.0]])
    A2 = np.linalg.solve(S, A @ S)
    B2 = np.linalg.solve(S, B)
    C2 = C @ S
    transformed = analyze_kalman_decomposition(LTISystem(A2, B2, C=C2))
    assert original.passed is True and transformed.passed is True
    assert _dims(original) == _dims(transformed)
    assert original.diagnostics["reachable_dimension"] == transformed.diagnostics["reachable_dimension"]
    assert original.diagnostics["unobservable_dimension"] == transformed.diagnostics["unobservable_dimension"]
    assert original.diagnostics["intersection_dimension"] == transformed.diagnostics["intersection_dimension"]


def test_structural_duality_swaps_cu_and_uo():
    A = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0])
    B = np.array([[1.0], [1.0], [1.0], [0.0], [0.0]])
    C = np.array([[1.0, 0.0, 0.0, 1.0, 0.0]])
    original = analyze_kalman_decomposition(LTISystem(A, B, C=C))
    dual = analyze_kalman_decomposition(LTISystem(A.T, C.T, C=B.T))
    assert _dims(original) == (1, 2, 1, 1)
    assert _dims(dual) == (1, 1, 2, 1)
    assert original.diagnostics["controllable_observable_dimension"] == dual.diagnostics["controllable_observable_dimension"]
    assert original.diagnostics["controllable_unobservable_dimension"] == dual.diagnostics["uncontrollable_observable_dimension"]
    assert original.diagnostics["uncontrollable_observable_dimension"] == dual.diagnostics["controllable_unobservable_dimension"]
    assert original.diagnostics["uncontrollable_unobservable_dimension"] == dual.diagnostics["uncontrollable_unobservable_dimension"]


def test_co_block_cross_check_is_controllable_and_observable():
    cert = analyze_kalman_decomposition(_all_four_exact_system())
    assert cert.evidence["co_block_controllability"].passed is True
    assert cert.evidence["co_block_observability"].passed is True
    assert cert.diagnostics["co_block_controllable"] is True
    assert cert.diagnostics["co_block_observable"] is True


def test_zero_input_and_zero_output_edge_cases_do_not_crash():
    zero_b = analyze_kalman_decomposition(
        LTISystem(np.diag([-1.0, -2.0]), np.zeros((2, 1)), C=np.eye(2))
    )
    assert zero_b.passed is True
    assert _dims(zero_b) == (0, 0, 2, 0)
    zero_c = analyze_kalman_decomposition(
        LTISystem(np.diag([-1.0, -2.0]), np.array([[1.0], [1.0]]), C=np.zeros((1, 2)))
    )
    assert zero_c.passed is True
    assert _dims(zero_c) == (0, 2, 0, 0)


def test_mimo_multioutput_and_d_are_supported():
    A = sp.diag(-1, -2, -3, -4)
    B = sp.Matrix([[1, 0], [0, 1], [0, 0], [0, 0]])
    C = sp.Matrix([[1, 0, 1, 0], [0, 0, 2, 0]])
    D = sp.Matrix([[sp.Rational(1, 2), 0], [0, sp.Rational(2, 5)]])
    cert = analyze_kalman_decomposition(LTISystem(A, B, C=C, D=D))
    assert cert.passed is True
    assert _dims(cert) == (1, 1, 1, 1)
    np.testing.assert_allclose(cert.evidence["D_transformed"], np.array(D, dtype=float))
    assert cert.evidence["D_transformed_exact"] == D


def test_numerical_near_reachability_boundary_is_inconclusive_not_false_pass():
    A = np.diag([0.0, 1.0])
    B = np.array([[1.0], [2e-7]])
    C = np.eye(2)
    cert = analyze_kalman_decomposition(
        LTISystem(A, B, C=C), tolerance_policy=TolerancePolicy(absolute=1e-6)
    )
    assert cert.passed is None
    assert cert.verdict == "BOUNDARY_OR_INCONCLUSIVE"
    assert "REACHABLE_RANK_TOLERANCE_SENSITIVE" in cert.diagnostics["warning_codes"]
    assert cert.diagnostics["reachable_numerical_dimension"] == 1
    assert cert.diagnostics["intersection_tolerance"] == 1e-6


def test_exact_dimension_overrides_numerical_tolerance_disagreement():
    tiny = sp.Rational(1, 10**20)
    A = sp.diag(0, 1)
    B = sp.Matrix([[1], [tiny]])
    C = sp.eye(2)
    cert = analyze_kalman_decomposition(LTISystem(A, B, C=C))
    assert cert.passed is True
    assert cert.diagnostics["reachable_exact_dimension"] == 2
    assert cert.diagnostics["reachable_numerical_dimension"] == 1
    assert cert.diagnostics["reachable_dimension"] == 2
    assert "EXACT_NUMERICAL_DIMENSION_DISAGREEMENT" in cert.diagnostics["warning_codes"]


def test_block_slices_and_noncanonical_metadata_are_structured():
    cert = analyze_kalman_decomposition(_all_four_exact_system())
    assert cert.diagnostics["block_slices"] == {
        "co": (0, 1), "cu": (1, 2), "uo": (2, 3), "uu": (3, 4)
    }
    assert cert.diagnostics["basis_is_canonical"] is False
    assert "reachable" in cert.diagnostics["canonical_subspaces"]


def test_kalman_transformation_equations_and_structural_residuals_pass():
    cert = analyze_kalman_decomposition(_all_four_exact_system())
    tol = cert.diagnostics["residual_tolerance"]
    for key in (
        "similarity_relative_residual_A",
        "similarity_relative_residual_B",
        "similarity_relative_residual_C",
        "A_structural_zero_relative_residual",
        "B_structural_zero_relative_residual",
        "C_structural_zero_relative_residual",
    ):
        assert cert.diagnostics[key] <= tol
    assert cert.diagnostics["transformation_rank"] == 4
    assert np.isfinite(cert.diagnostics["transformation_condition_number"])


def test_complex_numeric_decomposition_uses_conjugate_inner_products():
    A = np.diag([-1.0 + 1.0j, -2.0 - 0.5j])
    B = np.array([[1.0 + 0j], [0.0 + 0j]])
    C = np.array([[1.0 + 0j, 1.0j]])
    cert = analyze_kalman_decomposition(LTISystem(A, B, C=C))
    assert cert.passed is True
    assert _dims(cert) == (1, 0, 1, 0)
    assert cert.diagnostics["reachable_invariance_residual"] < 1e-12


def test_kalman_certificate_is_json_safe_including_empty_and_exact_matrices():
    cert = analyze_kalman_decomposition(_all_four_exact_system())
    payload = cert.to_dict(json_safe=True)
    assert payload["diagnostics"]["controllable_observable_dimension"] == 1
    json.dumps(payload)


def test_intersection_near_boundary_reports_machine_readable_warning():
    A = -np.eye(2)
    B = np.array([[1.0], [0.0]])
    eps = 2e-7
    C = np.array([[-eps, 1.0]])
    cert = analyze_kalman_decomposition(
        LTISystem(A, B, C=C), tolerance_policy=TolerancePolicy(absolute=1e-6)
    )
    assert cert.passed is None
    assert "SUBSPACE_INTERSECTION_TOLERANCE_SENSITIVE" in cert.diagnostics["warning_codes"]
    assert cert.diagnostics["intersection_numerical_dimension"] == 1
    assert cert.diagnostics["intersection_rank_singular_values"][-1] < cert.diagnostics["intersection_tolerance"]


def test_exact_reachability_decomposition_verifies_exact_zero_blocks():
    system = _all_four_exact_system()
    cert = analyze_reachability_decomposition(system)
    assert cert.passed is True
    assert cert.diagnostics["reachable_exact_dimension"] == 2
    assert cert.diagnostics["exact_A_lower_left_zero"] is True
    assert cert.diagnostics["exact_B_lower_zero"] is True
    assert cert.evidence["T_exact"].rank() == 4


def test_exact_observability_decomposition_verifies_exact_zero_blocks():
    system = _all_four_exact_system()
    cert = analyze_observability_decomposition(system)
    assert cert.passed is True
    assert cert.diagnostics["unobservable_exact_dimension"] == 2
    assert cert.diagnostics["exact_A_upper_right_zero"] is True
    assert cert.diagnostics["exact_C_right_zero"] is True
    assert cert.evidence["T_exact"].rank() == 4