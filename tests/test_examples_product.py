import sympy as sp

from certicontrol import AnalysisOptions, analyze_system
from certicontrol.examples import curated_examples, example_matrix_text, get_example


def test_curated_example_library_has_required_seven_cases():
    examples = curated_examples()
    required = {
        "Stable minimal second-order",
        "Uncontrollable mode",
        "Unobservable mode",
        "Unstable but controllable",
        "Stabilizable not controllable",
        "Four-part Kalman structure",
        "Tolerance-sensitive",
    }
    assert required <= set(examples)
    assert len(examples) >= 7


def test_examples_have_descriptions_and_valid_systems():
    for example in curated_examples().values():
        assert example.description.strip()
        assert example.system.n >= 1
        assert example.system.m >= 1


def test_example_text_preserves_exact_rational_entries():
    example = get_example("Four-part Kalman structure")
    texts = example_matrix_text(example)
    assert "1/3" in texts["A"]
    assert "2/5" in texts["A"]


def test_uncontrollable_example_produces_pbh_failure_witness():
    example = get_example("Uncontrollable mode")
    analysis = analyze_system(example.system, options=AnalysisOptions(run_lqr=False), timestamp="fixed")
    cert = analysis.certificates["controllability"]
    assert cert.passed is False
    pbh = cert.evidence["pbh"]
    failures = pbh.evidence["exact_failures"] or pbh.evidence["numerical_failures"]
    assert failures
    assert failures[0]["witness"] is not None


def test_unobservable_example_produces_pbh_failure_witness():
    example = get_example("Unobservable mode")
    analysis = analyze_system(example.system, options=AnalysisOptions(run_lqr=False), timestamp="fixed")
    cert = analysis.certificates["observability"]
    assert cert.passed is False
    pbh = cert.evidence["pbh"]
    failures = pbh.evidence["exact_failures"] or pbh.evidence["numerical_failures"]
    assert failures
    assert failures[0]["witness"] is not None


def test_unstable_controllable_example_has_lqr_defaults():
    example = get_example("Unstable but controllable")
    assert isinstance(example.default_Q, sp.MatrixBase)
    assert isinstance(example.default_R, sp.MatrixBase)