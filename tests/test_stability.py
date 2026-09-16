import json

import numpy as np

from certicontrol import (
    LTISystem,
    TolerancePolicy,
    analyze_spectral_stability,
    analyze_stability,
)


def _system(A):
    A = np.asarray(A)
    return LTISystem(A, np.ones((A.shape[0], 1)))


def test_spectral_stability_textbook_hurwitz_system():
    cert = analyze_spectral_stability(_system([[0.0, 1.0], [-2.0, -3.0]]))
    assert cert.passed is True
    assert cert.verdict == "HURWITZ"
    np.testing.assert_allclose(np.sort(cert.diagnostics["real_parts"]), [-2.0, -1.0])
    assert cert.diagnostics["spectral_abscissa"] == -1.0
    assert cert.diagnostics["spectral_margin"] == 1.0


def test_stable_complex_eigenvalues_are_hurwitz_and_json_safe():
    system = _system([[-1.0, -2.0], [2.0, -1.0]])
    cert = analyze_stability(system)
    spectral = cert.evidence["spectral"]
    assert cert.passed is True
    assert spectral.diagnostics["spectral_abscissa"] == -1.0
    eigenvalues = spectral.diagnostics["eigenvalues"]
    assert any(abs(value.imag - 2.0) < 1e-12 for value in eigenvalues)
    assert any(abs(value.imag + 2.0) < 1e-12 for value in eigenvalues)
    json.dumps(cert.to_dict(json_safe=True))


def test_positive_eigenvalue_is_unstable():
    cert = analyze_spectral_stability(_system([[1.0, 0.0], [0.0, -2.0]]))
    assert cert.passed is False
    assert cert.verdict == "UNSTABLE"
    assert cert.diagnostics["spectral_abscissa"] == 1.0


def test_zero_eigenvalue_is_boundary_not_hurwitz():
    cert = analyze_spectral_stability(_system([[0.0, 0.0], [0.0, -1.0]]))
    assert cert.passed is None
    assert cert.verdict == "BOUNDARY_OR_INCONCLUSIVE"
    assert "spectral_boundary" in cert.diagnostics["warning_codes"]


def test_imaginary_axis_pair_is_boundary_not_false_stable():
    cert = analyze_spectral_stability(_system([[0.0, -1.0], [1.0, 0.0]]))
    assert cert.passed is None
    assert abs(cert.diagnostics["spectral_abscissa"]) <= cert.diagnostics["stability_tolerance"]
    assert "spectral_boundary" in cert.diagnostics["warning_codes"]


def test_near_boundary_negative_eigenvalue_reports_tolerance_warning():
    system = _system([[-1e-16, 0.0], [0.0, -1.0]])
    cert = analyze_spectral_stability(system)
    assert cert.passed is None
    assert cert.diagnostics["spectral_margin"] > 0
    assert cert.diagnostics["stability_tolerance"] > cert.diagnostics["spectral_margin"]
    assert "spectral_boundary" in cert.diagnostics["warning_codes"]


def test_custom_spectral_tolerance_can_force_boundary_classification():
    cert = analyze_spectral_stability(
        _system([[-1e-4, 0.0], [0.0, -1.0]]),
        tolerance_policy=TolerancePolicy(absolute=1e-3),
    )
    assert cert.passed is None
    assert cert.diagnostics["stability_tolerance"] == 1e-3


def test_defective_jordan_block_is_hurwitz():
    cert = analyze_stability(_system([[-1.0, 1.0], [0.0, -1.0]]))
    assert cert.passed is True
    assert cert.diagnostics["cross_check"] == "CONSISTENT_PASS"


def test_similarity_invariance_of_spectral_hurwitz_verdict():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    T = np.array([[2.0, 1.0], [1.0, 1.0]])
    transformed_A = np.linalg.solve(T, A @ T)
    original = analyze_spectral_stability(_system(A))
    transformed = analyze_spectral_stability(_system(transformed_A))
    assert original.passed == transformed.passed is True
    np.testing.assert_allclose(
        np.sort_complex(original.diagnostics["eigenvalues"]),
        np.sort_complex(transformed.diagnostics["eigenvalues"]),
        atol=1e-12,
    )


def test_seeded_well_conditioned_stable_similarity_family_passes():
    rng = np.random.default_rng(20260916)
    diagonal = np.diag([-0.5, -1.5, -3.0])
    for _ in range(5):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        A = q @ diagonal @ q.T
        cert = analyze_stability(_system(A))
        assert cert.passed is True
        assert cert.diagnostics["cross_check"] == "CONSISTENT_PASS"


def test_boundary_numerical_only_system_does_not_become_false_pass():
    system = _system([[-1e-16, 0.0], [0.0, -1.0]])
    cert = analyze_stability(system)
    assert cert.evidence["spectral"].passed is None
    assert cert.passed is None
    assert cert.diagnostics["cross_check"] == "BOUNDARY_OR_INCONCLUSIVE"
