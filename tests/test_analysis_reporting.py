import json

import numpy as np
import sympy as sp

from certicontrol import AnalysisOptions, LTISystem, TolerancePolicy, analyze_system
from certicontrol.examples import curated_examples
from certicontrol.reporting import format_scalar, format_warning, matrix_to_latex, to_json, to_markdown


def test_full_analysis_pipeline_runs_all_certificates():
    example = curated_examples()["Stable minimal second-order"]
    analysis = analyze_system(
        example.system,
        Q=example.default_Q,
        R=example.default_R,
        timestamp="2026-09-16T00:00:00+00:00",
    )
    assert analysis.status("controllability") == "PASS"
    assert analysis.status("observability") == "PASS"
    assert analysis.status("stability") == "PASS"
    assert analysis.status("kalman") == "PASS"
    assert analysis.status("lqr") == "PASS"
    assert set(analysis.certificates) == {
        "controllability", "observability", "stability", "kalman", "lqr"
    }


def test_partial_ab_analysis_does_not_fail_without_c_or_lqr_weights():
    system = LTISystem([[0, 1], [-2, -3]], [[0], [1]])
    analysis = analyze_system(system, timestamp="fixed")
    assert analysis.status("controllability") == "PASS"
    assert analysis.status("stability") == "PASS"
    assert analysis.status("observability") == "NOT_RUN"
    assert analysis.status("kalman") == "NOT_RUN"
    assert analysis.status("lqr") == "NOT_RUN"
    assert "C is required" in analysis.records["observability"].reason


def test_analysis_options_can_disable_modules_without_marking_failure():
    system = LTISystem([[-1]], [[1]], C=[[1]])
    options = AnalysisOptions(
        run_controllability=False,
        run_observability=False,
        run_stability=True,
        run_kalman=False,
        run_lqr=False,
    )
    analysis = analyze_system(system, options=options, timestamp="fixed")
    assert analysis.status("controllability") == "NOT_RUN"
    assert analysis.status("observability") == "NOT_RUN"
    assert analysis.status("kalman") == "NOT_RUN"
    assert analysis.status("lqr") == "NOT_RUN"
    assert analysis.status("stability") == "PASS"


def test_stabilizable_not_controllable_remains_valid_lqr():
    example = curated_examples()["Stabilizable not controllable"]
    analysis = analyze_system(example.system, Q=example.default_Q, R=example.default_R, timestamp="fixed")
    assert analysis.status("controllability") == "FAIL"
    assert analysis.status("lqr") == "PASS"
    assert analysis.certificates["lqr"].diagnostics["stabilizable"] is True


def test_four_part_summary_exposes_all_structural_dimensions():
    example = curated_examples()["Four-part Kalman structure"]
    options = AnalysisOptions(run_lqr=False)
    analysis = analyze_system(example.system, options=options, timestamp="fixed")
    summary = analysis.summary()
    assert (
        summary["controllable_observable_dimension"],
        summary["controllable_unobservable_dimension"],
        summary["uncontrollable_observable_dimension"],
        summary["uncontrollable_unobservable_dimension"],
    ) == (1, 1, 1, 1)


def test_tolerance_sensitive_example_preserves_inconclusive_kalman_status():
    example = curated_examples()["Tolerance-sensitive"]
    analysis = analyze_system(
        example.system,
        options=AnalysisOptions(run_lqr=False),
        tolerance_policy=TolerancePolicy(absolute=1e-6),
        timestamp="fixed",
    )
    assert analysis.status("kalman") == "INCONCLUSIVE"
    codes = {(item["source"], item["code"]) for item in analysis.warning_codes}
    assert ("kalman", "REACHABLE_RANK_TOLERANCE_SENSITIVE") in codes


def test_json_report_roundtrip_is_safe():
    example = curated_examples()["Stable minimal second-order"]
    analysis = analyze_system(example.system, Q=example.default_Q, R=example.default_R, timestamp="fixed")
    payload = analysis.to_json_dict()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["metadata"]["generated_at"] == "fixed"
    assert decoded["summary"]["controllability"] == "PASS"
    assert "kalman" in decoded["analyses"]


def test_to_json_returns_valid_json_text():
    system = LTISystem([[-1]], [[1]], C=[[1]])
    analysis = analyze_system(system, options=AnalysisOptions(run_lqr=False), timestamp="fixed")
    decoded = json.loads(to_json(analysis))
    assert decoded["system"]["n"] == 1


def test_markdown_contains_product_sections_and_statuses():
    example = curated_examples()["Stable minimal second-order"]
    analysis = analyze_system(example.system, Q=example.default_Q, R=example.default_R, timestamp="fixed")
    report = to_markdown(analysis)
    assert "# CertiControl Analysis Report" in report
    assert "## Controllability" in report
    assert "## Observability" in report
    assert "## Stability / Lyapunov" in report
    assert "## LQR / CARE" in report
    assert "## Kalman Structural Decomposition" in report
    assert "## Warnings" in report


def test_markdown_partial_analysis_keeps_not_run_sections():
    analysis = analyze_system(LTISystem([[-1]], [[1]]), timestamp="fixed")
    report = to_markdown(analysis)
    assert "## Observability" in report
    assert "**Status:** NOT_RUN" in report


def test_matrix_to_latex_handles_rational_complex_and_empty_matrices():
    rational = matrix_to_latex(sp.Matrix([[sp.Rational(1, 2), 1], [0, -2]]))
    assert r"\begin{bmatrix}" in rational
    assert r"\frac{1}{2}" in rational
    complex_matrix = matrix_to_latex(np.array([[1 + 2j, 0]]))
    assert "1+2j" in complex_matrix
    assert matrix_to_latex(sp.zeros(2, 0)) == r"\begin{bmatrix}\end{bmatrix}"


def test_matrix_to_latex_accepts_witness_vectors():
    rendered = matrix_to_latex(np.array([1.0, 2.0]))
    assert "1" in rendered and "2" in rendered


def test_scalar_formatter_is_consistent():
    assert format_scalar(0.0) == "0"
    assert format_scalar(1.0) == "1"
    assert format_scalar(sp.Rational(1, 2)) == "1/2"
    assert "e-" in format_scalar(2.3e-12)
    assert format_scalar(1 + 2j) == "1+2j"


def test_warning_formatter_uses_human_message_and_context():
    text = format_warning("KALMAN_BASIS_ILL_CONDITIONED", "kalman")
    assert text.startswith("kalman:")
    assert "ill-conditioned" in text


def test_report_metadata_contains_version_tolerance_and_performed_analyses():
    analysis = analyze_system(
        LTISystem([[-1.0]], [[1.0]], C=[[1.0]]),
        options=AnalysisOptions(run_lqr=False),
        tolerance_policy=TolerancePolicy(absolute=1e-8),
        timestamp="fixed",
    )
    metadata = analysis.to_json_dict()["metadata"]
    assert metadata["certicontrol_version"]
    assert metadata["tolerance_policy"]["absolute"] == 1e-8
    assert "stability" in metadata["analyses_performed"]