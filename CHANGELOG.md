# Changelog

All notable project-level changes are summarized here. CertiControl is currently a pre-1.0 toolkit; this file records release milestones rather than reconstructing every development commit.

## 0.2.0 — 2026-09-17

### Certificate core

- Added controllability certificates with exact/numerical rank, PBH cross-checks, and failure witnesses.
- Added observability certificates with PBH witnesses, duality checks, and similarity invariance.
- Added continuous-time spectral stability and Lyapunov certificates, including exact rational verification where supported.
- Added continuous-time infinite-horizon LQR / CARE certificates with stabilizability, detectability, residual verification, gain consistency, and closed-loop Hurwitz checks.
- Added reachability, observability, and full four-part Kalman structural decomposition certificates with transformation and zero-block verification.

### Product layer

- Added unified `SystemAnalysis` orchestration with explicit `PASS`, `FAIL`, `INCONCLUSIVE`, `NOT_RUN`, and `ERROR` states.
- Added Markdown and JSON certificate exports.
- Added seven deterministic curated examples.
- Added a Streamlit application with evidence panels and Plotly pole visualizations.
- Added Python 3.10 / 3.13 GitHub Actions coverage and product-layer integration tests.

### Release-candidate packaging

- Polished repository documentation for rapid judge/portfolio review.
- Added hackathon submission draft, demo script, submission checklist, and resume-ready project descriptions.
- Added Streamlit Community Cloud installation metadata and an automated Streamlit startup health check.
- Added deterministic release tests for all curated examples and export paths.

## 0.1.0

- Initial package skeleton and early certificate-oriented continuous-time LTI analysis APIs.
