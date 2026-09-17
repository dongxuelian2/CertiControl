# Screenshot Assets

This directory is reserved for **real screenshots captured from the running CertiControl Streamlit application**.

Planned files:

- `overview.png` — Overview/status screen.
- `lqr.png` — LQR evidence and open/closed-loop pole comparison.
- `kalman.png` — four-part Kalman structure, transformation, and residuals.
- `stability.png` — optional Lyapunov/stability evidence.

Do not commit generated UI mockups as if they were screenshots of the real application.

Recommended capture sequence:

1. Start the app with `streamlit run app.py`.
2. Use the exact curated example named in `docs/SUBMISSION_CHECKLIST.md`.
3. Run analysis and expand only the evidence needed for the shot.
4. Capture at a readable desktop width with the app title and relevant tab visible.
5. Save the PNG with the exact filename above.
6. Only after the file exists, add it to README or Devpost.
