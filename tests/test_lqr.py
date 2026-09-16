import json

import numpy as np
import pytest
import sympy as sp
from scipy import linalg

from certicontrol import (
    LTISystem,
    TolerancePolicy,
    analyze_controllability,
    analyze_detectability,
    analyze_lqr,
    analyze_observability,
    analyze_stabilizability,
    solve_care,
)


def test_scalar_unstable_open_loop_matches_analytic_stabilizing_care_solution():
    system = LTISystem([[1]], [[1]])
    cert = analyze_lqr(system, [[1]], [[1]])
    expected = 1.0 + np.sqrt(2.0)
    assert cert.passed is True
    np.testing.assert_allclose(cert.evidence["P"], [[expected]], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cert.evidence["K"], [[expected]], rtol=1e-12, atol=1e-12)
    assert cert.diagnostics["closed_loop_spectral_abscissa"] == pytest.approx(-np.sqrt(2.0))


def test_scalar_stable_open_loop_matches_analytic_solution():
    system = LTISystem([[-1]], [[1]])
    cert = analyze_lqr(system, [[1]], [[1]])
    expected = -1.0 + np.sqrt(2.0)
    assert cert.passed is True
    np.testing.assert_allclose(cert.evidence["P"], [[expected]], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cert.evidence["K"], [[expected]], rtol=1e-12, atol=1e-12)


def test_textbook_two_state_lqr_certificate_is_fully_verified():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]])
    cert = analyze_lqr(system, sp.eye(2), [[1]])
    assert cert.passed is True
    assert cert.verdict == "VALID_STABILIZING_LQR_CERTIFICATE"
    assert cert.diagnostics["stabilizable"] is True
    assert cert.diagnostics["detectable"] is True
    assert cert.diagnostics["care_residual_ok"] is True
    assert cert.diagnostics["gain_consistency_ok"] is True
    assert cert.diagnostics["closed_loop_hurwitz_status"] is True
    assert cert.diagnostics["closed_loop_identity_residual_ok"] is True
    assert cert.evidence["K"].shape == (1, 2)


def test_low_level_care_reports_independent_residuals():
    solution = solve_care(
        np.array([[0.0, 1.0], [-2.0, -3.0]]),
        np.array([[0.0], [1.0]]),
        np.eye(2),
        np.array([[1.0]]),
    )
    assert solution.status in {"SOLVED", "SOLVED_WITH_WARNING"}
    assert solution.P is not None
    assert solution.K is not None
    assert solution.care_relative_residual is not None and solution.care_relative_residual < 1e-12
    assert solution.gain_relative_residual is not None and solution.gain_relative_residual < 1e-12


def test_q_positive_semidefinite_singular_is_valid():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]])
    cert = analyze_lqr(system, [[1, 0], [0, 0]], [[1]])
    assert cert.passed is True
    assert cert.diagnostics["Q_psd_status"] is True
    assert cert.diagnostics["Q_numerical_classification"] == "POSITIVE_SEMIDEFINITE"
    assert cert.diagnostics["Q_exact_psd_status"] is True


def test_valid_lqr_can_have_semidefinite_not_definite_p():
    system = LTISystem([[-1, 0], [0, -2]], [[1], [0]])
    cert = analyze_lqr(system, [[1, 0], [0, 0]], [[1]])
    assert cert.passed is True
    assert cert.diagnostics["P_psd_status"] is True
    assert cert.diagnostics["P_psd_classification"] == "POSITIVE_SEMIDEFINITE"
    assert cert.diagnostics["P_min_eigenvalue"] == pytest.approx(0.0, abs=1e-14)


def test_q_indefinite_is_rejected_before_care():
    system = LTISystem([[-1, 0], [0, -2]], [[1], [0]])
    cert = analyze_lqr(system, [[1, 0], [0, -1]], [[1]])
    assert cert.passed is False
    assert cert.verdict == "INVALID_INPUT"
    assert "Q_NOT_PSD" in cert.diagnostics["warning_codes"]


def test_q_nonsymmetric_is_rejected_without_silent_symmetrization():
    system = LTISystem([[-1, 0], [0, -2]], [[1], [0]])
    cert = analyze_lqr(system, [[1, 1], [0, 1]], [[1]])
    assert cert.passed is False
    assert cert.verdict == "INVALID_INPUT"
    assert "Q_NOT_SYMMETRIC" in cert.diagnostics["warning_codes"]


@pytest.mark.parametrize(
    "R, expected_code",
    [
        ([[1, 0], [0, 0]], "R_NOT_PD"),
        ([[1, 0], [0, -1]], "R_NOT_PD"),
        ([[1, 1], [0, 1]], "R_NOT_SYMMETRIC"),
    ],
)
def test_invalid_r_cases_are_rejected(R, expected_code):
    system = LTISystem([[-1, 0], [0, -2]], np.eye(2))
    cert = analyze_lqr(system, np.eye(2), R)
    assert cert.passed is False
    assert cert.verdict == "INVALID_INPUT"
    assert expected_code in cert.diagnostics["warning_codes"]


def test_wrong_q_dimension_returns_clear_structured_failure():
    system = LTISystem(np.eye(2), np.ones((2, 1)))
    cert = analyze_lqr(system, np.eye(3), [[1]])
    assert cert.passed is False
    assert cert.verdict == "INVALID_INPUT"
    assert "Q must have shape" in cert.warnings[0]


def test_wrong_r_dimension_returns_clear_structured_failure():
    system = LTISystem(np.eye(2), np.eye(2))
    cert = analyze_lqr(system, np.eye(2), [[1]])
    assert cert.passed is False
    assert cert.verdict == "INVALID_INPUT"
    assert "R must have shape" in cert.warnings[0]


def test_controllable_system_is_stabilizable():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]])
    cert = analyze_stabilizability(system)
    assert cert.passed is True
    assert cert.verdict == "STABILIZABLE"


def test_uncontrollable_but_stabilizable_is_not_rejected():
    system = LTISystem([[-2, 0], [0, 1]], [[0], [1]])
    controllability = analyze_controllability(system)
    stabilizability = analyze_stabilizability(system)
    lqr = analyze_lqr(system, np.eye(2), [[1]])
    assert controllability.passed is False
    assert stabilizability.passed is True
    assert lqr.passed is True


def test_unstable_uncontrollable_mode_is_not_stabilizable_and_has_witness():
    system = LTISystem([[1, 0], [0, -2]], [[0], [1]])
    cert = analyze_stabilizability(system)
    assert cert.passed is False
    assert cert.verdict == "NOT_STABILIZABLE"
    failures = cert.evidence["exact_offending_modes"]
    assert failures
    assert failures[0]["eigenvalue"] == 1
    assert failures[0]["input_orthogonality_residual"] == 0


def test_nonstabilizable_problem_is_rejected_before_care():
    system = LTISystem([[1, 0], [0, -2]], [[0], [1]])
    cert = analyze_lqr(system, np.eye(2), [[1]])
    assert cert.passed is False
    assert cert.verdict == "PREREQUISITE_FAILURE"
    assert cert.diagnostics["stabilizable"] is False


def test_stabilizability_boundary_is_inconclusive_for_numerical_data():
    system = LTISystem(
        np.array([[1e-16, 0.0], [0.0, -1.0]]),
        np.array([[0.0], [1.0]]),
    )
    cert = analyze_stabilizability(system)
    assert cert.passed is None
    assert "STABILIZABILITY_BOUNDARY" in cert.diagnostics["warning_codes"]


def test_observable_under_q_implies_detectable():
    A = [[0, 1], [-2, -3]]
    B = [[0], [1]]
    Q = [[1, 0], [0, 1]]
    system = LTISystem(A, B)
    q_observability = analyze_observability(LTISystem(A, B, C=Q))
    detectability = analyze_detectability(system, Q)
    assert q_observability.passed is True
    assert detectability.passed is True


def test_unobservable_under_q_but_detectable_when_hidden_mode_is_stable():
    A = [[-2, 0], [0, 1]]
    B = [[1], [1]]
    Q = [[0, 0], [0, 1]]
    system = LTISystem(A, B)
    q_observability = analyze_observability(LTISystem(A, B, C=Q))
    detectability = analyze_detectability(system, Q)
    assert q_observability.passed is False
    assert detectability.passed is True


def test_unstable_unpenalized_mode_is_not_detectable_and_has_witness():
    system = LTISystem([[1, 0], [0, -1]], np.eye(2))
    Q = [[0, 0], [0, 1]]
    cert = analyze_detectability(system, Q)
    assert cert.passed is False
    failures = cert.evidence["exact_offending_modes"]
    assert failures
    assert failures[0]["eigenvalue"] == 1
    assert failures[0]["state_cost_visibility_residual"] == 0


def test_nondetectable_problem_is_rejected_before_care():
    system = LTISystem([[1, 0], [0, -1]], np.eye(2))
    cert = analyze_lqr(system, [[0, 0], [0, 1]], np.eye(2))
    assert cert.passed is False
    assert cert.verdict == "PREREQUISITE_FAILURE"
    assert cert.diagnostics["stabilizable"] is True
    assert cert.diagnostics["detectable"] is False


def test_detectability_boundary_is_inconclusive_for_numerical_data():
    system = LTISystem(
        np.array([[1e-16, 0.0], [0.0, -1.0]]),
        np.eye(2),
    )
    cert = analyze_detectability(system, np.diag([0.0, 1.0]))
    assert cert.passed is None
    assert "DETECTABILITY_BOUNDARY" in cert.diagnostics["warning_codes"]


def test_mimo_lqr_with_nonidentity_r():
    A = np.array([[1.0, 0.2], [0.0, -1.0]])
    B = np.eye(2)
    Q = np.diag([2.0, 1.0])
    R = np.array([[2.0, 0.2], [0.2, 1.0]])
    cert = analyze_lqr(LTISystem(A, B), Q, R)
    assert cert.passed is True
    assert cert.evidence["K"].shape == (2, 2)
    assert cert.diagnostics["care_relative_residual"] < 1e-12
    assert cert.diagnostics["closed_loop_hurwitz_status"] is True


def test_closed_loop_identity_is_independently_verified():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]])
    cert = analyze_lqr(system, np.eye(2), [[1]])
    P = cert.evidence["P"]
    K = cert.evidence["K"]
    Acl = cert.evidence["A_closed_loop"]
    Q = cert.evidence["Q"]
    R = cert.evidence["R"]
    residual = Acl.T @ P + P @ Acl + Q + K.T @ R @ K
    np.testing.assert_allclose(residual, np.zeros((2, 2)), atol=1e-12)
    assert cert.diagnostics["closed_loop_identity_residual_ok"] is True


def test_closed_loop_hurwitz_cross_check_and_margin_are_reported():
    cert = analyze_lqr(LTISystem([[1]], [[1]]), [[1]], [[1]])
    assert cert.diagnostics["closed_loop_hurwitz_status"] is True
    assert cert.diagnostics["closed_loop_spectral_abscissa"] < 0
    assert cert.diagnostics["closed_loop_spectral_margin"] > 0


def test_similarity_covariance_of_p_k_and_closed_loop():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.diag([2.0, 1.0])
    R = np.array([[1.5]])
    T = np.array([[2.0, 1.0], [1.0, 1.0]])
    A_prime = np.linalg.solve(T, A @ T)
    B_prime = np.linalg.solve(T, B)
    Q_prime = T.T @ Q @ T

    original = analyze_lqr(LTISystem(A, B), Q, R)
    transformed = analyze_lqr(LTISystem(A_prime, B_prime), Q_prime, R)
    assert original.passed is True and transformed.passed is True
    np.testing.assert_allclose(transformed.evidence["P"], T.T @ original.evidence["P"] @ T, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(transformed.evidence["K"], original.evidence["K"] @ T, rtol=1e-10, atol=1e-10)
    expected_acl = np.linalg.solve(T, original.evidence["A_closed_loop"] @ T)
    np.testing.assert_allclose(transformed.evidence["A_closed_loop"], expected_acl, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        np.sort_complex(transformed.diagnostics["closed_loop_eigenvalues"]),
        np.sort_complex(original.diagnostics["closed_loop_eigenvalues"]),
        atol=1e-10,
    )


def test_orthogonal_transform_preserves_identity_q():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    B = np.array([[0.0], [1.0]])
    theta = 0.41
    T = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    transformed = analyze_lqr(LTISystem(T.T @ A @ T, T.T @ B), np.eye(2), [[1]])
    original = analyze_lqr(LTISystem(A, B), np.eye(2), [[1]])
    assert transformed.passed is True
    np.testing.assert_allclose(transformed.evidence["Q"], np.eye(2), atol=1e-14)
    np.testing.assert_allclose(transformed.evidence["P"], T.T @ original.evidence["P"] @ T, atol=1e-11)


def test_larger_scalar_r_reduces_feedback_gain_in_simple_case():
    system = LTISystem([[1]], [[1]])
    gain_r1 = float(analyze_lqr(system, [[1]], [[1]]).evidence["K"][0, 0])
    gain_r100 = float(analyze_lqr(system, [[1]], [[100]]).evidence["K"][0, 0])
    assert gain_r100 < gain_r1


def test_solver_failure_is_captured_as_structured_status(monkeypatch):
    def fail(*args, **kwargs):
        raise linalg.LinAlgError("synthetic CARE failure")

    monkeypatch.setattr("certicontrol.lqr.linalg.solve_continuous_are", fail)
    cert = analyze_lqr(LTISystem([[0, 1], [-2, -3]], [[0], [1]]), np.eye(2), [[1]])
    assert cert.passed is False
    assert cert.verdict == "SOLVER_FAILURE"
    assert "CARE_SOLVER_FAILED" in cert.diagnostics["warning_codes"]


def test_exact_inputs_are_recorded_but_care_solution_is_numerical():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]])
    cert = analyze_lqr(system, sp.eye(2), sp.Matrix([[1]]))
    assert cert.passed is True
    assert cert.diagnostics["input_exact"] is True
    assert cert.diagnostics["care_solution_exact"] is False
    assert cert.evidence["Q_exact"] == sp.eye(2)
    assert cert.evidence["R_exact"] == sp.Matrix([[1]])


def test_complex_lqr_uses_conjugate_transpose_consistently():
    A = np.array([[-1.0 + 1.0j]])
    B = np.array([[1.0 + 0.0j]])
    Q = np.array([[1.0 + 0.0j]])
    R = np.array([[1.0 + 0.0j]])
    cert = analyze_lqr(LTISystem(A, B), Q, R)
    assert cert.passed is True
    P = cert.evidence["P"]
    K = cert.evidence["K"]
    residual = A.conj().T @ P + P @ A - P @ B @ K + Q
    np.testing.assert_allclose(residual, np.zeros((1, 1)), atol=1e-12)


def test_moderately_conditioned_similarity_still_has_small_residuals():
    A = np.array([[1.0, 0.0], [0.0, -2.0]])
    B = np.eye(2)
    Q = np.eye(2)
    R = np.diag([1.0, 3.0])
    T = np.array([[6.0, 1.0], [0.5, 0.8]])
    A_prime = np.linalg.solve(T, A @ T)
    B_prime = np.linalg.solve(T, B)
    Q_prime = T.T @ Q @ T
    cert = analyze_lqr(LTISystem(A_prime, B_prime), Q_prime, R)
    assert cert.passed is True
    assert cert.diagnostics["care_relative_residual"] < 1e-11
    assert cert.diagnostics["closed_loop_identity_relative_residual"] < 1e-11


def test_lqr_certificate_is_json_safe():
    cert = analyze_lqr(LTISystem([[0, 1], [-2, -3]], [[0], [1]]), sp.eye(2), [[1]])
    data = cert.to_dict(json_safe=True)
    assert data["diagnostics"]["status"] == "VALID_STABILIZING_LQR_CERTIFICATE"
    assert isinstance(data["evidence"]["P"], list)
    json.dumps(data)


def test_exact_detectability_diagnostics_are_preserved_in_high_level_lqr():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]])
    cert = analyze_lqr(system, sp.diag(1, 0), sp.Matrix([[1]]))
    detectability = cert.evidence["detectability"]
    assert detectability.diagnostics["exact_passed"] is True
    assert detectability.evidence["Q_exact"] == sp.diag(1, 0)


def test_numerically_borderline_q_psd_returns_inconclusive_not_false_pass():
    system = LTISystem(np.diag([-1.0, -2.0]), np.eye(2))
    Q = np.diag([1.0, -1e-18])
    cert = analyze_lqr(system, Q, np.eye(2))
    assert cert.passed is None
    assert cert.verdict == "BOUNDARY_OR_INCONCLUSIVE"
    assert "Q_PSD_TOLERANCE_SENSITIVE" in cert.diagnostics["warning_codes"]


def test_numerically_borderline_r_pd_returns_inconclusive():
    system = LTISystem(np.diag([-1.0, -2.0]), np.eye(2))
    R = np.diag([1.0, 1e-18])
    cert = analyze_lqr(system, np.eye(2), R)
    assert cert.passed is None
    assert cert.verdict == "BOUNDARY_OR_INCONCLUSIVE"
    assert "R_PD_TOLERANCE_SENSITIVE" in cert.diagnostics["warning_codes"]


def test_care_residual_alone_does_not_define_stabilizing_solution(monkeypatch):
    from certicontrol import CARESolution

    bad_p = np.array([[1.0 - np.sqrt(2.0)]])
    bad_k = bad_p.copy()
    bad_acl = np.array([[1.0]]) - bad_k

    def return_antistabilizing_solution(*args, **kwargs):
        return CARESolution(
            status="SOLVED",
            raw_P=bad_p,
            P=bad_p,
            K=bad_k,
            A_closed_loop=bad_acl,
            symmetry_residual=0.0,
            gain_absolute_residual=0.0,
            gain_relative_residual=0.0,
            care_absolute_residual=0.0,
            care_relative_residual=0.0,
        )

    monkeypatch.setattr("certicontrol.lqr.solve_care", return_antistabilizing_solution)
    cert = analyze_lqr(LTISystem([[1.0]], [[1.0]]), [[1.0]], [[1.0]])
    assert cert.passed is False
    assert cert.verdict == "NUMERICAL_VERIFICATION_FAILURE"
    assert cert.diagnostics["care_residual_ok"] is True
    assert cert.diagnostics["closed_loop_hurwitz_status"] is False
    assert "CLOSED_LOOP_NOT_HURWITZ" in cert.diagnostics["warning_codes"]


def test_strict_closed_loop_cost_matrix_attaches_phase3_lyapunov_cross_certificate():
    cert = analyze_lqr(LTISystem([[0, 1], [-2, -3]], [[0], [1]]), np.eye(2), [[1]])
    cross = cert.evidence["closed_loop_lyapunov_cross_certificate"]
    assert cross is not None
    assert cross.passed is True
    assert cert.diagnostics["Q_closed_loop_positive_definite"] is True
    assert cert.diagnostics["care_P_vs_closed_loop_lyapunov_P_relative"] < 1e-12


def test_care_and_closed_loop_identity_residuals_are_mutually_consistent():
    cert = analyze_lqr(LTISystem([[0, 1], [-2, -3]], [[0], [1]]), np.eye(2), [[1]])
    assert cert.diagnostics["care_residual_ok"] is True
    assert cert.diagnostics["closed_loop_identity_residual_ok"] is True
    assert abs(
        cert.diagnostics["care_absolute_residual"]
        - cert.diagnostics["closed_loop_identity_absolute_residual"]
    ) < 1e-12
    assert "INTERNAL_CONSISTENCY_WARNING" not in cert.diagnostics["warning_codes"]


def test_numerical_singular_r_is_rejected_not_treated_as_valid_boundary():
    system = LTISystem(np.diag([-1.0, -2.0]), np.eye(2))
    cert = analyze_lqr(system, np.eye(2), np.diag([1.0, 0.0]))
    assert cert.passed is False
    assert cert.verdict == "INVALID_INPUT"
    assert "R_NOT_PD" in cert.diagnostics["warning_codes"]


def test_zero_q_and_zero_gain_are_valid_for_already_hurwitz_uncontrolled_system():
    system = LTISystem([[-1, 0], [0, -2]], [[0], [0]])
    cert = analyze_lqr(system, sp.zeros(2), [[1]])
    assert cert.passed is True
    assert cert.diagnostics["stabilizable"] is True
    assert cert.diagnostics["detectable"] is True
    np.testing.assert_allclose(cert.evidence["P"], np.zeros((2, 2)), atol=1e-14)
    np.testing.assert_allclose(cert.evidence["K"], np.zeros((1, 2)), atol=1e-14)
    assert cert.diagnostics["P_psd_classification"] == "POSITIVE_SEMIDEFINITE"