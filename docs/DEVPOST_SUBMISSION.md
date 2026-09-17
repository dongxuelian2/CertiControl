# Devpost Submission Draft

This file is copy-ready source text for the InfinityX Global Hackathon 2K26 submission. Replace bracketed TODO fields before final submission.

## Project Name

CertiControl

## Tagline

Inspect the evidence behind linear control analysis.

## Problem

Classical control software is excellent at computation, but many analysis workflows expose only the final numerical answer: a Boolean controllability result, a stability verdict, or a feedback gain. That makes it harder to inspect *why* a criterion passed, identify the exact mode responsible for failure, or recognize when a conclusion is sensitive to numerical tolerance.

The problem is especially visible in small state-space models used for teaching, prototyping, and engineering sanity checks. Exact algebraic structure and floating-point diagnostics are often separated, while equivalent control-theory criteria are rarely presented together as evidence for the same conclusion.

## Solution

CertiControl is a certificate-based toolkit for continuous-time finite-dimensional LTI systems. It converts classical linear-control criteria into inspectable computational certificates containing evidence, failure witnesses, residuals, margins, cross-checks, and structural transformations.

Instead of returning only `Controllable: True`, CertiControl can expose the controllability matrix, exact and numerical ranks, singular values, active tolerance, a PBH cross-check, and—when the test fails—an offending eigenmode with a witness and residuals. The same certificate-first idea extends to observability, Lyapunov stability, continuous-time LQR/CARE, and the four-part Kalman structural decomposition.

A Streamlit interface lets a user load curated examples or enter matrices directly, run selected analyses, inspect concise verdicts before expanding the mathematical evidence, and export the result as Markdown or JSON.

## Key Features

- **Certificate-first analysis** — conclusions are accompanied by the evidence used to support them.
- **Exact + numerical dual computation** — rational inputs preserve exact SymPy evidence where supported while NumPy/SciPy diagnostics provide numerical cross-checks.
- **Explicit PBH failure witnesses** — failed controllability or observability tests identify an offending mode and expose the relevant residuals.
- **Independent Lyapunov and CARE verification** — solver output is checked with positivity conditions, equation residuals, gain consistency, closed-loop stability, and equivalent identities.
- **Kalman structural decomposition** — reachable/unobservable subspaces, their intersection, four structural dimensions, coordinate transformation, conditioning, and required zero blocks are verified.
- **Interactive reporting and export** — Streamlit and Plotly provide an inspectable UI with Markdown and JSON certificate downloads.

## Technology Stack

- Python
- NumPy
- SciPy
- SymPy
- Streamlit
- Plotly
- pytest
- GitHub Actions

## How It Works

```text
Streamlit UI
    ↓
Unified analysis / reporting
    ↓
Certificate objects
    ↓
Control-theory modules
    ↓
Exact SymPy + numerical NumPy/SciPy
```

The UI does not reimplement the mathematics. It calls the same package APIs used by the tests and reporting layer. Missing optional inputs produce explicit `NOT_RUN` states rather than crashing unrelated analyses.

The numerical path uses scale-aware tolerances, singular values, residuals, eigenvalue margins, and conditioning diagnostics. Where exact rational paths are implemented, exact algebraic results remain authoritative if floating-point calculations become ill-conditioned; disagreement is surfaced as a warning rather than hidden.

## Challenges

One challenge was avoiding false confidence from successful numerical solvers. A CARE solver, for example, may return a matrix, but CertiControl only accepts a stabilizing LQR certificate after independently checking prerequisites, the Riccati residual, gain consistency, positive semidefiniteness of the solution, the closed-loop identity, and Hurwitz stability.

Another challenge was the Kalman decomposition. The structural subspaces are canonical, but complement bases and the final transformation are generally non-unique. Tests therefore verify spans, dimensions, invariance, similarity equations, conditioning, and only the theoretically required zero blocks rather than comparing one hard-coded transformation matrix.

A third challenge was keeping numerical ambiguity honest. Near-rank-loss and near-imaginary-axis cases are allowed to remain tolerance-sensitive or inconclusive instead of being forced into a clean-looking pass/fail result.

## Accomplishments

- Built a unified certificate model across controllability, observability, stability/Lyapunov, LQR/CARE, and Kalman decomposition.
- Preserved exact rational computation where it is mathematically reliable while retaining numerical diagnostics and cross-checks.
- Added reproducible examples demonstrating PBH witnesses, stabilizable-but-not-controllable systems, closed-loop LQR verification, four-part Kalman structure, and tolerance sensitivity.
- Built an offline Streamlit interface with evidence panels, pole visualizations, warning aggregation, and exportable reports.
- Maintained automated tests on Python 3.10 and 3.13 through GitHub Actions.

## What We Learned

The project reinforced that numerical correctness is not the same as inspectability. Many classical control results already have natural certificate objects: PBH witnesses, equation residuals, spectral margins, principal minors, invariant subspaces, and coordinate identities. Exposing those objects directly makes failures easier to debug and makes numerical assumptions easier to see.

We also learned that exact and numerical methods are most useful when they are not forced into one representation. Exact arithmetic is powerful for small rational systems, while numerical linear algebra provides conditioning and tolerance information that exact arithmetic does not capture.

## Real-World Value

CertiControl is useful for:

- teaching and inspecting state-space concepts;
- debugging small state-space models;
- understanding why a design prerequisite fails;
- recognizing tolerance-sensitive calculations;
- creating reproducible control-theory examples;
- rapid sanity checking of small continuous-time LTI models.

The current scope targets small finite-dimensional systems, typically around `n <= 10`. It is not presented as safety-critical engineering certification.

## What's Next

After the hackathon, natural extensions include discrete-time analysis, minimal realization, pole placement, and observer design. These are intentionally outside the current submission so the release candidate remains focused on the certificate-first core.

## GitHub

https://github.com/dongxuelian2/CertiControl

## Demo

- Live demo: **[TODO: deploy and insert verified URL]**
- Demo video: **[TODO: record and insert final video URL]**
- Recording script: `docs/DEMO_SCRIPT.md`

## Screenshots

**[TODO: capture real application screenshots for Overview, LQR, and Kalman Structure.]**

Do not replace this TODO with generated mockups; screenshots should come from the running Streamlit application.

## Team

**[TODO: add final team member names and roles exactly as they should appear on the submission.]**

## AI-Assisted Development Disclosure

AI-assisted development tools were used during implementation. The shipped application itself does not call an AI model or require an AI API. CertiControl's runtime control analysis is implemented with deterministic numerical and symbolic Python libraries.
