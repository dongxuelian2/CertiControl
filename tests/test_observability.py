import json

import numpy as np
import pytest
import sympy as sp

from certicontrol import (
    LTISystem,
    TolerancePolicy,
    analyze_controllability,
    analyze_observability,
    observability_matrix,
    parse_matrix,
    pbh_controllability,
    pbh_observability,
)


def test_standard_second_order_system_is_observable():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]], C=[[1, 0]])
    certificate = analyze_observability(system)
    assert certificate.passed
    assert certificate.verdict == "observable"
    assert certificate.diagnostics["exact_rank"] == 2
    assert certificate.diagnostics["numerical_rank"] == 2
    assert certificate.diagnostics["pbh_exact_passed"] is True
    assert certificate.diagnostics["pbh_numerical_passed"] is True


def test_observability_matrix_uses_expected_block_structure_siso():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    C = np.array([[1.0, 0.0]])
    matrix = observability_matrix(A, C)
    np.testing.assert_allclose(matrix, np.eye(2))


def test_observability_matrix_supports_mimo_output():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    C = np.eye(2)
    matrix = observability_matrix(A, C)
    assert matrix.shape == (4, 2)
    assert np.linalg.matrix_rank(matrix) == 2


def test_unobservable_system_produces_exact_and_numerical_pbh_witness():
    system = LTISystem([[-1, 0], [0, -2]], [[1], [0]], C=[[1, 0]])
    certificate = analyze_observability(system)
    assert not certificate.passed
    assert certificate.diagnostics["exact_rank"] == 1

    pbh = certificate.evidence["pbh"]
    exact_failures = pbh.evidence["exact_failures"]
    numerical_failures = pbh.evidence["numerical_failures"]
    assert exact_failures
    exact_failure = next(item for item in exact_failures if item["eigenvalue"] == -2)
    exact_witness = sp.Matrix(exact_failure["witness"])
    assert exact_witness[0] == 0
    assert exact_witness[1] != 0
    assert exact_failure["eigenvector_residual"] == 0
    assert exact_failure["output_null_residual"] == 0

    assert numerical_failures
    numerical_failure = min(
        numerical_failures,
        key=lambda item: abs(item["eigenvalue"] + 2),
    )
    assert abs(numerical_failure["eigenvalue"] + 2) < 1e-10
    assert numerical_failure["eigenvector_residual"] < 1e-10
    assert numerical_failure["output_null_residual"] < 1e-10


def test_rational_parser_preserves_exact_observability_rank():
    A = parse_matrix("0, 1/2\n0, 0")
    C = parse_matrix("2/3, 0")
    system = LTISystem(A, parse_matrix("0\n1"), C=C)
    certificate = analyze_observability(system)
    assert certificate.passed
    assert certificate.diagnostics["exact_rank"] == 2
    exact_matrix = certificate.evidence["observability_matrix_exact"]
    assert exact_matrix == sp.Matrix(
        [[sp.Rational(2, 3), 0], [0, sp.Rational(1, 3)]]
    )


def test_exact_numerical_observability_rank_conflict_generates_warning():
    tiny = sp.Rational(1, 10**20)
    system = LTISystem(sp.diag(0, 1), sp.eye(2), C=sp.Matrix([[1, tiny]]))
    certificate = analyze_observability(system)
    assert certificate.passed  # exact algebraic conclusion
    assert certificate.diagnostics["exact_rank"] == 2
    assert certificate.diagnostics["numerical_rank"] == 1
    assert certificate.diagnostics["smallest_singular_value"] < certificate.diagnostics["tolerance"]
    assert any("Exact observability rank and numerical rank disagree" in w for w in certificate.warnings)
    assert any("Exact and numerical PBH observability classifications disagree" in w for w in certificate.warnings)


def test_custom_absolute_tolerance_is_recorded_and_used_for_observability():
    system = LTISystem(
        [[0.0, 0.0], [0.0, 1.0]],
        [[1.0], [0.0]],
        C=[[1.0, 1e-6]],
    )
    certificate = analyze_observability(system, TolerancePolicy(absolute=1e-4))
    assert certificate.diagnostics["tolerance"] == 1e-4
    assert certificate.diagnostics["numerical_rank"] == 1
    assert not certificate.passed


def test_observability_matrix_and_pbh_criteria_agree_across_fixtures():
    fixtures = [
        LTISystem([[0, 1], [-2, -3]], [[0], [1]], C=[[1, 0]]),
        LTISystem([[-1, 0], [0, -2]], [[1], [0]], C=[[1, 0]]),
        LTISystem([[0, 1, 0], [0, 0, 1], [-1, -2, -3]], [[0], [0], [1]], C=[[1, 0, 0]]),
        LTISystem([[1, 0], [0, 2]], [[1], [1]], C=[[1, 0], [0, 1]]),
    ]
    for system in fixtures:
        certificate = analyze_observability(system)
        assert certificate.diagnostics["matrix_exact_passed"] == certificate.diagnostics["pbh_exact_passed"]
        assert certificate.diagnostics["matrix_numerical_passed"] == certificate.diagnostics["pbh_numerical_passed"]


def test_controllability_observability_duality_across_multiple_fixtures():
    fixtures = [
        (
            np.array([[0.0, 1.0], [-2.0, -3.0]]),
            np.array([[1.0, 0.0]]),
        ),
        (
            np.diag([-1.0, -2.0]),
            np.array([[1.0, 0.0]]),
        ),
        (
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -2.0, -3.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ),
    ]
    for A, C in fixtures:
        obs_system = LTISystem(A, np.ones((A.shape[0], 1)), C=C)
        dual_system = LTISystem(A.T, C.T)
        obs = analyze_observability(obs_system)
        ctrl = analyze_controllability(dual_system)
        assert obs.diagnostics["numerical_rank"] == ctrl.diagnostics["numerical_rank"]
        assert obs.diagnostics["matrix_numerical_passed"] == ctrl.diagnostics["matrix_numerical_passed"]
        obs_pbh = pbh_observability(obs_system)
        ctrl_pbh = pbh_controllability(dual_system)
        assert obs_pbh.diagnostics["numerical_passed"] == ctrl_pbh.diagnostics["numerical_passed"]


def test_exact_rational_controllability_observability_duality():
    A = sp.Matrix([[0, sp.Rational(1, 2)], [sp.Rational(2, 3), 1]])
    C = sp.Matrix([[sp.Rational(2, 3), sp.Rational(1, 2)]])
    obs_system = LTISystem(A, sp.Matrix([[0], [1]]), C=C)
    dual_system = LTISystem(A.T, C.T)
    obs = analyze_observability(obs_system)
    ctrl = analyze_controllability(dual_system)
    assert obs.diagnostics["exact_rank"] == ctrl.diagnostics["exact_rank"]
    assert obs.diagnostics["matrix_exact_passed"] == ctrl.diagnostics["matrix_exact_passed"]
    assert obs.diagnostics["pbh_exact_passed"] == ctrl.diagnostics["pbh_exact_passed"]


def test_similarity_invariance_of_observability_rank_and_verdict():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    T = np.array([[2.0, 1.0], [1.0, 1.0]])
    T_inv = np.linalg.inv(T)
    original = analyze_observability(LTISystem(A, B, C=C))
    transformed = analyze_observability(LTISystem(T_inv @ A @ T, T_inv @ B, C=C @ T))
    assert original.diagnostics["numerical_rank"] == transformed.diagnostics["numerical_rank"] == 2
    assert original.verdict == transformed.verdict == "observable"


def test_similarity_invariance_under_seeded_well_conditioned_transforms():
    rng = np.random.default_rng(20260916)
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -2.0, -3.0]])
    B = np.array([[0.0], [0.0], [1.0]])
    C = np.array([[1.0, 0.0, 0.0]])
    original = analyze_observability(LTISystem(A, B, C=C))
    for _ in range(5):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        transformed = analyze_observability(LTISystem(q.T @ A @ q, q.T @ B, C=C @ q))
        assert transformed.diagnostics["numerical_rank"] == original.diagnostics["numerical_rank"]
        assert transformed.verdict == original.verdict


def test_pbh_observability_public_api_reports_unobservable_mode():
    system = LTISystem([[2.0, 0.0], [0.0, -1.0]], [[1.0], [0.0]], C=[[1.0, 0.0]])
    certificate = pbh_observability(system)
    assert not certificate.passed
    assert certificate.evidence["numerical_failures"]
    offending = [f["eigenvalue"] for f in certificate.evidence["numerical_failures"]]
    assert any(abs(value + 1) < 1e-10 for value in offending)


def test_observability_certificate_json_safe_representation():
    certificate = analyze_observability(
        LTISystem([[0, 1], [-2, -3]], [[0], [1]], C=[[1, 0]])
    )
    data = certificate.to_dict(json_safe=True)
    assert data["diagnostics"]["exact_rank"] == 2
    assert isinstance(data["evidence"]["observability_matrix"], list)
    json.dumps(data)


def test_observability_requires_c_matrix():
    with pytest.raises(ValueError, match="requires a C output matrix"):
        analyze_observability(LTISystem([[0, 1], [-2, -3]], [[0], [1]]))
