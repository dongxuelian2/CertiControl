from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_runs_real_streamlit_health_check():
    workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    assert "streamlit run app.py" in workflow
    assert "/_stcore/health" in workflow
    assert "--server.headless true" in workflow


def test_readme_does_not_reference_uncaptured_screenshot_files():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("overview.png", "lqr.png", "kalman.png", "stability.png"):
        if name in readme and f"](docs/images/{name})" in readme:
            assert (ROOT / "docs/images" / name).is_file(), name


def test_screenshot_manifest_requires_real_app_captures():
    manifest = (ROOT / "docs/images/README.md").read_text(encoding="utf-8")
    assert "real screenshots captured from the running CertiControl Streamlit application" in manifest
    assert "Do not commit generated UI mockups" in manifest
