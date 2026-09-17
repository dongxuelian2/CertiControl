import json
from pathlib import Path

from certicontrol import AnalysisOptions, TolerancePolicy, analyze_system
from certicontrol.examples import curated_examples, get_example
from certicontrol.reporting import to_json, to_markdown


ROOT = Path(__file__).resolve().parents[1]


def _analysis_for(name: str):
    example = get_example(name)
    policy = (
        TolerancePolicy(absolute=example.recommended_absolute_tolerance)
        if example.recommended_absolute_tolerance is not None
        else TolerancePolicy()
    )
    return analyze_system(
        example.system,
        Q=example.default_Q,
        R=example.default_R,
        tolerance_policy=policy,
        timestamp="release-smoke",
    )


def test_release_artifacts_exist():
    required = [
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "requirements.txt",
        "docs/DEMO_SCRIPT.md",
        "docs/DEVPOST_SUBMISSION.md",
        "docs/SUBMISSION_CHECKLIST.md",
        "docs/RESUME_BULLETS.md",
        "docs/images/README.md",
    ]
    for relative in required:
        assert (ROOT / relative).is_file(), relative


def test_readme_has_judge_first_release_markers_and_valid_internal_targets():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Inspect the evidence behind linear control analysis." in readme
    assert "actions/workflows/tests.yml/badge.svg" in readme
    assert "## Why CertiControl?" in readme
    assert "## Interactive demo" in readme
    assert "## Quick start" in readme
    assert "## Hackathon" in readme
    for relative in [
        "LICENSE",
        "docs/DEVPOST_SUBMISSION.md",
        "docs/DEMO_SCRIPT.md",
        "docs/SUBMISSION_CHECKLIST.md",
        "docs/RESUME_BULLETS.md",
    ]:
        assert (ROOT / relative).exists(), relative


def test_streamlit_cloud_requirements_reuse_pyproject_app_extra():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").strip()
    assert requirements == ".[app]"


def test_all_curated_examples_complete_without_analysis_error_and_export():
    examples = curated_examples()
    assert len(examples) == 7
    for name, example in examples.items():
        policy = (
            TolerancePolicy(absolute=example.recommended_absolute_tolerance)
            if example.recommended_absolute_tolerance is not None
            else TolerancePolicy()
        )
        analysis = analyze_system(
            example.system,
            Q=example.default_Q,
            R=example.default_R,
            tolerance_policy=policy,
            timestamp="release-smoke",
        )
        assert all(record.status != "ERROR" for record in analysis.records.values()), name
        markdown = to_markdown(analysis)
        encoded = to_json(analysis, indent=2)
        assert markdown.startswith("# CertiControl Analysis Report")
        assert json.loads(encoded)["metadata"]["scope"] == "continuous-time finite-dimensional LTI"


def test_curated_example_truth_table():
    stable = _analysis_for("Stable minimal second-order")
    assert stable.status("controllability") == "PASS"
    assert stable.status("observability") == "PASS"
    assert stable.status("stability") == "PASS"

    uncontrollable = _analysis_for("Uncontrollable mode")
    assert uncontrollable.status("controllability") == "FAIL"

    unobservable = _analysis_for("Unobservable mode")
    assert unobservable.status("observability") == "FAIL"

    unstable = _analysis_for("Unstable but controllable")
    assert unstable.status("stability") == "FAIL"
    assert unstable.status("lqr") == "PASS"
    assert unstable.certificates["lqr"].diagnostics["closed_loop_hurwitz_status"] is True

    stabilizable = _analysis_for("Stabilizable not controllable")
    assert stabilizable.status("controllability") == "FAIL"
    assert stabilizable.status("lqr") == "PASS"
    assert stabilizable.certificates["lqr"].diagnostics["stabilizable"] is True

    kalman = _analysis_for("Four-part Kalman structure")
    d = kalman.certificates["kalman"].diagnostics
    assert (
        d["controllable_observable_dimension"],
        d["controllable_unobservable_dimension"],
        d["uncontrollable_observable_dimension"],
        d["uncontrollable_unobservable_dimension"],
    ) == (1, 1, 1, 1)

    sensitive = _analysis_for("Tolerance-sensitive")
    assert sensitive.warnings or sensitive.warning_codes


def test_pbh_demo_witnesses_have_small_exact_residuals():
    uncontrollable = _analysis_for("Uncontrollable mode")
    pbh_c = uncontrollable.certificates["controllability"].evidence["pbh"]
    failure_c = pbh_c.evidence["exact_failures"][0]
    assert failure_c["eigenvector_residual"] == 0
    assert failure_c["input_orthogonality_residual"] == 0

    unobservable = _analysis_for("Unobservable mode")
    pbh_o = unobservable.certificates["observability"].evidence["pbh"]
    failure_o = pbh_o.evidence["exact_failures"][0]
    assert failure_o["eigenvector_residual"] == 0
    assert failure_o["output_null_residual"] == 0


def test_export_smoke_writes_readable_markdown_and_parseable_json(tmp_path):
    analysis = _analysis_for("Four-part Kalman structure")
    markdown_path = tmp_path / "certicontrol_report.md"
    json_path = tmp_path / "certicontrol_certificate.json"
    markdown_path.write_text(to_markdown(analysis), encoding="utf-8")
    json_path.write_text(to_json(analysis, indent=2), encoding="utf-8")

    markdown = markdown_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "## Structural Decomposition" in markdown
    assert payload["analyses"]["kalman"]["status"] == "PASS"
    assert "1/3" in json_path.read_text(encoding="utf-8")


def test_submission_docs_keep_external_human_items_explicitly_unfinished():
    checklist = (ROOT / "docs/SUBMISSION_CHECKLIST.md").read_text(encoding="utf-8")
    devpost = (ROOT / "docs/DEVPOST_SUBMISSION.md").read_text(encoding="utf-8")
    assert "[ ] `docs/images/overview.png`" in checklist
    assert "[ ] Deploy to Streamlit Community Cloud" in checklist
    assert "[ ] Record a 90–120 second video" in checklist
    assert "[TODO: add final team member names" in devpost
    assert "[TODO: deploy and insert verified URL]" in devpost
