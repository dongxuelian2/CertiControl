# InfinityX Submission Checklist

Target deadline supplied for the submission workflow: **2026-09-25 23:45 IST**.

This checklist deliberately separates repository/program work from external human submission work. A box is checked only when the item is actually available or verified.

## Repository / release candidate

- [x] GitHub repository is public.
- [x] MIT `LICENSE` is present.
- [x] Phase 6 baseline CI is green on Python 3.10 and 3.13.
- [x] README explains the problem, certificate-first distinction, run commands, scope, and limitations.
- [x] CI badge is included in README.
- [x] `CHANGELOG.md` is present for version `0.2.0`.
- [x] Streamlit app entry point is root-level `app.py`.
- [x] Streamlit Community Cloud dependency entry point is prepared in `requirements.txt`.
- [x] Seven curated deterministic examples are included.
- [x] Markdown and JSON export paths exist.
- [x] Devpost draft is prepared.
- [x] Demo recording script is prepared.
- [x] Resume/portfolio descriptions are prepared.
- [ ] Final Phase 7 GitHub Actions run is green on Python 3.10 and 3.13.
- [ ] Final release-branch link audit completed after the last commit.

## Screenshots

Capture screenshots only from the real running Streamlit application.

- [ ] `docs/images/overview.png` — Overview using **Stable minimal second-order** or **Four-part Kalman structure**.
- [ ] `docs/images/lqr.png` — LQR tab using **Unstable but controllable**, with open/closed poles and residuals visible.
- [ ] `docs/images/kalman.png` — Kalman Structure tab showing `co = cu = uo = uu = 1`.
- [ ] Optional `docs/images/stability.png` — Stability/Lyapunov evidence.
- [ ] Insert only captured, existing images into README / submission text.

## Live demo

- [ ] Deploy to Streamlit Community Cloud or another public host.
- [ ] Confirm the deployed app loads without secrets.
- [ ] Load **Four-part Kalman structure** and run analysis on the deployed app.
- [ ] Load **Unstable but controllable** and confirm LQR visualization on the deployed app.
- [ ] Test Markdown download.
- [ ] Test JSON download.
- [ ] Add the verified live URL to README and `DEVPOST_SUBMISSION.md`.

## Demo video

- [ ] Record a 90–120 second video using `docs/DEMO_SCRIPT.md`.
- [ ] Ensure the video shows a PBH witness, Lyapunov evidence, LQR verification, and Kalman structure.
- [ ] Ensure no unsupported features are claimed.
- [ ] Upload the video to the final hosting location.
- [ ] Add the verified video URL to `DEVPOST_SUBMISSION.md`.

## Devpost / submission form

- [x] Project Name drafted.
- [x] Tagline drafted.
- [x] Problem Statement drafted.
- [x] Solution Description drafted.
- [x] Key Features drafted.
- [x] Technology Stack drafted.
- [x] Technical architecture / How It Works drafted.
- [x] Challenges / Accomplishments / Learnings drafted.
- [ ] Add final team member names and roles.
- [ ] Add real screenshots.
- [ ] Add live demo URL if deployed.
- [ ] Add demo video URL.
- [ ] Verify source-repository link from the submission form.
- [ ] Review AI-assisted-development disclosure if the form asks for it.
- [ ] Submit the final Devpost entry.

## GitHub presentation metadata

The current connector does not expose repository-metadata editing, so these remain manual:

- [ ] Set GitHub About description to: `Certificate-based analysis for continuous-time LTI control systems with exact and numerical verification.`
- [ ] Add topics: `control-systems`, `control-theory`, `linear-systems`, `lti-systems`, `lyapunov`, `lqr`, `kalman-decomposition`, `python`.
- [ ] Add the verified live-demo URL as the repository homepage if deployed.

## Final link check

Immediately before submission, open every public link in a logged-out/private browser window:

- [ ] GitHub repository.
- [ ] CI badge / Actions page.
- [ ] Live demo.
- [ ] Demo video.
- [ ] Every screenshot used in Devpost.
- [ ] Any portfolio/resume link pointing back to the project.
