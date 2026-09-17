# CertiControl Demo Script

Target length: **90–120 seconds**.

The goal is not to teach control theory. The goal is to show the difference between a black-box answer and an inspectable certificate.

## Recording setup

- Start from the Streamlit app with the sidebar visible.
- Use the curated examples; do not type matrices live unless necessary.
- Keep the browser zoom at a level where verdicts, residuals, and plots are readable.
- Do not claim formal verification or industrial certification.

## Shot list

### 0–10 s — Problem and solution

**Screen:** CertiControl title / Overview.

**Voiceover:**

> Classical control tools can tell us that a system is controllable, observable, or stable. But a final Boolean or gain matrix often hides the evidence behind that conclusion. CertiControl turns those classical criteria into inspectable computational certificates.

### 10–30 s — Overview

**Action:** Load **Stable minimal second-order**, run analysis, show Overview.

**Voiceover:**

> A single analysis combines controllability, observability, stability, optional LQR, and Kalman structure. Each module has an explicit status, while warnings and tolerance-sensitive conclusions stay visible instead of being rounded away.

### 30–50 s — PBH witness

**Action:** Load **Uncontrollable mode**, run analysis, open Controllability and the failure evidence.

**Voiceover:**

> When a criterion fails, CertiControl does not stop at “false.” Here the PBH test returns the offending eigenmode, a left-eigenvector witness, and the residuals showing that this mode is decoupled from the input.

### 50–72 s — Lyapunov evidence

**Action:** Return to **Stable minimal second-order**, open Stability.

**Voiceover:**

> Stability analysis reports the spectral abscissa and margin, then independently checks a Lyapunov equation. With exact rational data, supported paths retain exact algebraic evidence; numerical residuals remain available as a cross-check.

### 72–98 s — LQR verification

**Action:** Load **Unstable but controllable**, run analysis with its default Q and R, open LQR and the pole plot.

**Voiceover:**

> For LQR, a Riccati solver returning a matrix is not treated as a certificate. CertiControl checks the weighting matrices, stabilizability, detectability, the CARE residual, gain consistency, the closed-loop identity, and finally verifies that the closed-loop poles are Hurwitz.

### 98–113 s — Kalman structure

**Action:** Load **Four-part Kalman structure**, open Kalman Structure.

**Voiceover:**

> The structural decomposition exposes how controllability and observability coexist in the same state space. This example has one state in each Kalman class. The transformation, conditioning, similarity equations, invariant subspaces, and required zero blocks are all checked.

### 113–120 s — Export and close

**Action:** Open Export and briefly show the Markdown / JSON buttons, then the GitHub repository.

**Voiceover:**

> Every result can be exported as Markdown or JSON for inspection and reuse. CertiControl is open source, runs offline, and focuses on making classical linear-control calculations explain their evidence.

## Optional 15-second fallback ending

If the recording is running long, skip the detailed Kalman matrix view and use:

> Finally, the Kalman tab verifies a structural state-space decomposition, and the full certificate can be exported as Markdown or JSON. The repository includes the source, tests, and reproducible examples.

## Claims to avoid

Do **not** say:

- formally verified;
- proven safe for industrial deployment;
- new LQR or Lyapunov algorithm;
- AI-powered control platform;
- minimal realization or simulation support.

The accurate language is **inspectable computational evidence**, **exact algebraic certificate where supported**, and **numerically verified residual/cross-check**.
