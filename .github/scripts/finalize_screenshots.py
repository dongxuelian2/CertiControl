from pathlib import Path


def main() -> None:
    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8")
    text = text.replace("docs/images/overview.webp", "docs/images/overview.png")
    text = text.replace("docs/images/lqr.webp", "docs/images/lqr.png")
    text = text.replace("docs/images/kalman.webp", "docs/images/kalman.png")
    text = text.replace(
        "The screenshots above are real captures from the running Streamlit application; "
        "repository copies are resized/compressed for fast README rendering.",
        "The screenshots above are full-resolution real captures from the running Streamlit application.",
    )
    readme.write_text(text, encoding="utf-8")

    manifest = Path("docs/images/README.md")
    manifest.write_text(
        """# Screenshot Assets

This directory contains **real screenshots captured from the running CertiControl Streamlit application**.

Current assets:

- `overview.png` — full-resolution Overview/status screen using the four-part Kalman example.
- `lqr.png` — full-resolution LQR certificate and open/closed-loop pole comparison.
- `kalman.png` — full-resolution four-part Kalman structural decomposition.
- `stability.png` — optional future/Devpost stability capture if committed later.

The PNG files are direct browser screenshots of the real Streamlit UI. Do not commit generated UI mockups as if they were screenshots of the real application.

For Devpost and other presentation surfaces, these PNGs can be reused directly.
""",
        encoding="utf-8",
    )

    for path in Path("docs/images").glob("*.webp"):
        path.unlink()


if __name__ == "__main__":
    main()
