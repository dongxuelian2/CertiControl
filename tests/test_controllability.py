import json

import numpy as np
import sympy as sp

from certicontrol import (
    LTISystem,
    TolerancePolicy,
    analyze_controllability,
    controllability_matrix,
    parse_matrix,
    pbh_controllability,
)


def test_standard_second_order_system_is_controllable():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]])
    certificate = analyze_controllability(system)
    assert certificate.passed
    assert certificate.verdict == "controllable"
    assert certificate.diagnostics["exact_rank"] == 2
    assert certificate.diagnostics["numerical_rank"] == 2
    assert certificate.diagnostics["pbh_exact_passed"] is True
    assert certificate.diagnostics["pbh_numerical_passed"] is True


def test_controllability_matrix_uses_expected_block_structure_siso():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    B = np.array([[0.0], [1.0]])
    matrix = controllability_matrix(A, B)
    np.testing.assert_allclose(matrix, np.array([[0.0, 1.0], [1.0, -3.0]]))


def test_controllability_matrix_supports_mimo():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.eye(2)
    matrix = controllability_matrix(A, B)
    assert matrix.shape == (2, 4)
    assert np.linalg.matrix_rank(matrix) == 2


def test_uncontrollable_system_produces_exact_and_numerical_pbh_witness():
    system = LTISystem([[0, 0], [0, 1]], [[1], [0]])
    certificate = analyze_controllability(system)
    assert not certificate.passed
    assert certificate.diagnostics["exact_rank"] == 1

    pbh = certificate.evidence["pbh"]
    exact_failures = pbh.evidence["exact_failures"]
    numerical_failures = pbh.evidence["numerical_failures"]
    assert exact_failures
    assert exact_failures[0]["witness"] is not None
    assert exact_failures[0]["eigenvector_residual"] == 0
    assert exact_failures[0]["input_orthogonality_residual"] == 0
    assert numerical_failures
    assert min(item["eigenvector_residual"] for item in numerical_failures) < 1e-10
    assert min(item["input_orthogonality_residual"] for item in numerical_failures) < 1e-10


def test_rational_parser_preserves_exact_controllability_rank():
    A = parse_matrix("0, 1/2\n0, 0")
    B = parse_matrix("0\n1")
    system = LTISystem(A, B)
    certificate = analyze_controllability(system)
    assert certificate.passed
    assert certificate.diagnostics["exact_rank"] == 2
    exact_matrix = certificate.evidence["controllability_matrix_exact"]
    assert exact_matrix == sp.Matrix([[0, sp.Rational(1, 2)], [1, 0]])


def test_exact_numerical_rank_conflict_generates_warning():
    tiny = sp.Rational(1, 10**20)
    system = LTISystem(sp.diag(0, 1), sp.diag(1, tiny))
    certificate = analyze_controllability(system)
    assert certificate.passed  # exact algebraic conclusion
    assert certificate.diagnostics["exact_rank"] == 2
    assert certificate.diagnostics["numerical_rank"] == 1
    assert any("Exact controllability rank and numerical rank disagree" in w for w in certificate.warnings)
    assert any("Exact and numerical PBH classifications disagree" in w for w in certificate.warnings)


def test_custom_absolute_tolerance_is_recorded_and_used():
    system = LTISystem([[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1e-6]])
    certificate = analyze_controllability(system, TolerancePolicy(absolute=1e-4))
    assert certificate.diagnostics["tolerance"] == 1e-4
    assert certificate.diagnostics["numerical_rank"] == 1
    assert not certificate.passed


def test_matrix_and_pbh_criteria_agree_across_fixtures():
    fixtures = [
        LTISystem([[0, 1], [-2, -3]], [[0], [1]]),
        LTISystem([[0, 0], [0, 1]], [[1], [0]]),
        LTISystem([[0, 1, 0], [0, 0, 1], [-1, -2, -3]], [[0], [0], [1]]),
        LTISystem([[1, 0], [0, 2]], [[1, 0], [0, 1]]),
    ]
    for system in fixtures:
        certificate = analyze_controllability(system)
        assert certificate.diagnostics["matrix_exact_passed"] == certificate.diagnostics["pbh_exact_passed"]
        assert certificate.diagnostics["matrix_numerical_passed"] == certificate.diagnostics["pbh_numerical_passed"]


def test_similarity_invariance_of_controllability_rank():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    B = np.array([[0.0], [1.0]])
    T = np.array([[2.0, 1.0], [1.0, 1.0]])
    T_inv = np.linalg.inv(T)
    transformed = LTISystem(T_inv @ A @ T, T_inv @ B)
    original = LTISystem(A, B)
    rank_original = analyze_controllability(original).diagnostics["numerical_rank"]
    rank_transformed = analyze_controllability(transformed).diagnostics["numerical_rank"]
    assert rank_original == rank_transformed == 2


def test_pbh_public_api_reports_uncontrollable_mode():
    system = LTISystem([[2.0, 0.0], [0.0, -1.0]], [[1.0], [0.0]])
    certificate = pbh_controllability(system)
    assert not certificate.passed
    assert certificate.evidence["numerical_failures"]
    offending = [f["eigenvalue"] for f in certificate.evidence["numerical_failures"]]
    assert any(abs(value + 1) < 1e-10 for value in offending)


def test_json_safe_certificate_representation():
    certificate = analyze_controllability(LTISystem([[0, 1], [-2, -3]], [[0], [1]]))
    data = certificate.to_dict(json_safe=True)
    assert data["diagnostics"]["exact_rank"] == 2
    assert isinstance(data["evidence"]["controllability_matrix"], list)
    json.dumps(data)
