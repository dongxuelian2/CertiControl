import json

import numpy as np
import pytest
import sympy as sp

from certicontrol import (
    LTISystem,
    analyze_lyapunov_stability,
    analyze_stability,
    solve_lyapunov,
)


def test_exact_scalar_stable_certificate_is_one_quarter():
    system = LTISystem([[-2]], [[1]])
    cert = analyze_lyapunov_stability(system)
    assert cert.passed is True
    assert cert.diagnostics["exact_solution_status"] == "unique"
    assert cert.evidence["P_exact"] == sp.Matrix([[sp.Rational(1, 4)]])
    assert cert.diagnostics["P_exact_positive_definite"] is True
    assert cert.diagnostics["P_leading_principal_minors"] == (sp.Rational(1, 4),)
    assert cert.evidence["exact_residual_matrix"] == sp.zeros(1)


def test_exact_scalar_unstable_equation_solves_but_certificate_fails():
    system = LTISystem([[1]], [[1]])
    cert = analyze_lyapunov_stability(system)
    assert cert.diagnostics["exact_solution_status"] == "unique"
    assert cert.evidence["P_exact"] == sp.Matrix([[sp.Rational(-1, 2)]])
    assert cert.diagnostics["P_exact_positive_definite"] is False
    assert cert.passed is False
    assert cert.verdict == "no_positive_definite_lyapunov_certificate"


def test_exact_textbook_2x2_certificate_and_principal_minors():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]])
    cert = analyze_lyapunov_stability(system)
    expected = sp.Matrix([[sp.Rational(5, 4), sp.Rational(1, 4)], [sp.Rational(1, 4), sp.Rational(1, 4)]])
    assert cert.evidence["P_exact"] == expected
    assert cert.evidence["exact_residual_matrix"] == sp.zeros(2)
    assert cert.diagnostics["P_exact_symmetric"] is True
    assert cert.diagnostics["P_exact_positive_definite"] is True
    assert cert.diagnostics["P_leading_principal_minors"] == (sp.Rational(5, 4), sp.Rational(1, 4))
    assert cert.diagnostics["exact_certificate_passed"] is True


def test_exact_and_numerical_lyapunov_paths_agree_on_textbook_system():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]])
    cert = analyze_lyapunov_stability(system)
    assert cert.diagnostics["exact_certificate_passed"] is True
    assert cert.diagnostics["numerical_certificate_passed"] is True
    np.testing.assert_allclose(
        cert.evidence["P"],
        np.array(cert.evidence["P_exact"], dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_numerical_textbook_2x2_residual_and_pd_margin():
    system = LTISystem(np.array([[0.0, 1.0], [-2.0, -3.0]]), np.array([[0.0], [1.0]]))
    cert = analyze_lyapunov_stability(system)
    assert cert.passed is True
    assert cert.diagnostics["numerical_certificate_passed"] is True
    assert cert.diagnostics["lambda_min_P"] > cert.diagnostics["P_positive_definite_tolerance"]
    assert cert.diagnostics["relative_residual"] <= cert.diagnostics["relative_residual_tolerance"]
    assert cert.diagnostics["symmetry_ok"] is True


def test_custom_q_is_used_and_preserved_in_certificate():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    Q = np.diag([2.0, 3.0])
    system = LTISystem(A, np.array([[0.0], [1.0]]))
    cert = analyze_lyapunov_stability(system, Q)
    np.testing.assert_allclose(cert.evidence["Q"], Q)
    P = cert.evidence["P"]
    np.testing.assert_allclose(A.T @ P + P @ A + Q, np.zeros((2, 2)), atol=1e-12)
    assert cert.passed is True


def test_custom_exact_q_retains_exact_certificate():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]])
    Q = sp.diag(2, 3)
    cert = analyze_lyapunov_stability(system, Q)
    assert cert.evidence["Q_exact"] == Q
    assert cert.diagnostics["Q_exact_positive_definite"] is True
    assert cert.diagnostics["Q_leading_principal_minors"] == (sp.Integer(2), sp.Integer(6))
    assert cert.diagnostics["exact_certificate_passed"] is True


def test_wrong_shape_q_raises_clear_error():
    system = LTISystem(np.eye(2), np.ones((2, 1)))
    with pytest.raises(ValueError, match="Q must have shape"):
        analyze_lyapunov_stability(system, np.eye(3))


def test_nonsymmetric_q_is_rejected_as_certificate_precondition():
    system = LTISystem([[-1, 0], [0, -2]], [[1], [0]])
    cert = analyze_lyapunov_stability(system, [[1, 1], [0, 1]])
    assert cert.passed is None
    assert cert.diagnostics["solution_status"] == "not_attempted_invalid_q"
    assert "q_not_hermitian" in cert.diagnostics["warning_codes"]


def test_indefinite_q_is_rejected_as_certificate_precondition():
    system = LTISystem([[-1, 0], [0, -2]], [[1], [0]])
    cert = analyze_lyapunov_stability(system, [[1, 0], [0, -1]])
    assert cert.passed is None
    assert cert.diagnostics["solution_status"] == "not_attempted_invalid_q"
    assert "q_not_positive_definite" in cert.diagnostics["warning_codes"]


def test_unstable_system_has_solved_equation_but_no_pd_certificate():
    system = LTISystem([[1, 0], [0, -2]], [[1], [0]])
    cert = analyze_stability(system)
    lyap = cert.evidence["lyapunov"]
    assert lyap.diagnostics["exact_solution_status"] == "unique"
    assert lyap.passed is False
    assert cert.evidence["spectral"].passed is False
    assert cert.diagnostics["cross_check"] == "CONSISTENT_FAIL"


def test_imaginary_axis_operator_degeneracy_is_handled_without_crash():
    system = LTISystem([[0, -1], [1, 0]], [[1], [0]])
    cert = analyze_lyapunov_stability(system)
    assert cert.diagnostics["exact_solution_status"] in {"inconsistent", "nonunique"}
    assert cert.passed is False


def test_lyapunov_certificate_covariance_under_nonorthogonal_similarity():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    Q = np.diag([2.0, 3.0])
    B = np.array([[0.0], [1.0]])
    T = np.array([[2.0, 1.0], [1.0, 1.0]])
    A_prime = np.linalg.solve(T, A @ T)
    B_prime = np.linalg.solve(T, B)
    Q_prime = T.T @ Q @ T

    original = analyze_lyapunov_stability(LTISystem(A, B), Q)
    transformed = analyze_lyapunov_stability(LTISystem(A_prime, B_prime), Q_prime)
    expected_P_prime = T.T @ original.evidence["P"] @ T
    np.testing.assert_allclose(transformed.evidence["P"], expected_P_prime, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        A_prime.T @ expected_P_prime + expected_P_prime @ A_prime + Q_prime,
        np.zeros((2, 2)),
        atol=1e-11,
    )
    assert transformed.passed is True


def test_orthogonal_similarity_preserves_identity_q_and_covariant_p():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    B = np.array([[0.0], [1.0]])
    theta = 0.37
    T = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A_prime = T.T @ A @ T
    B_prime = T.T @ B

    original = analyze_lyapunov_stability(LTISystem(A, B))
    transformed = analyze_lyapunov_stability(LTISystem(A_prime, B_prime))
    np.testing.assert_allclose(transformed.evidence["Q"], np.eye(2), atol=1e-14)
    np.testing.assert_allclose(transformed.evidence["P"], T.T @ original.evidence["P"] @ T, atol=1e-12)
    assert transformed.passed is True


def test_solve_lyapunov_exposes_raw_symmetrized_and_residual_data():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    solution = solve_lyapunov(A)
    assert solution.status in {"numerical_solution", "numerical_solution_with_warning"}
    assert solution.raw_P is not None
    assert solution.P is not None
    assert solution.symmetry_residual is not None
    assert solution.relative_residual is not None
    np.testing.assert_allclose(solution.P, solution.P.T.conj(), atol=1e-14)


def test_complex_input_uses_conjugate_transpose_lyapunov_equation():
    A = np.array([[-1.0 + 1.0j]])
    system = LTISystem(A, np.array([[1.0 + 0.0j]]))
    cert = analyze_lyapunov_stability(system)
    assert cert.passed is True
    P = cert.evidence["P"]
    Q = cert.evidence["Q"]
    np.testing.assert_allclose(A.conj().T @ P + P @ A + Q, np.zeros((1, 1)), atol=1e-12)


def test_stability_certificate_json_safe_with_exact_and_numerical_evidence():
    cert = analyze_stability(LTISystem([[0, 1], [-2, -3]], [[0], [1]]))
    data = cert.to_dict(json_safe=True)
    assert data["diagnostics"]["cross_check"] == "CONSISTENT_PASS"
    json.dumps(data)
