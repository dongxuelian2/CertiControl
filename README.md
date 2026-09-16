# CertiControl

CertiControl is a certificate-based toolkit for continuous-time LTI systems that exposes the mathematical evidence behind controllability, observability, stability, LQR synthesis, and Kalman structural decomposition.

It is designed for small teaching and analysis problems where a Boolean answer is not enough: certificates retain exact algebraic evidence when available, numerical residuals and margins, PBH witnesses, active tolerances, cross-checks, and structural transformations.

## Why CertiControl

Typical control software can tell you that a system is controllable, stable, or has an LQR gain. CertiControl is built around a different question:

> **What evidence can be inspected to justify that conclusion?**

The package therefore keeps the intermediate mathematical objects that matter: controllability/observability matrices, exact ranks, singular values, PBH failure modes, Lyapunov matrices, CARE residuals, closed-loop identities, invariant subspaces, and Kalman-coordinate zero patterns.

## Features

- **Exact + numerical dual paths** for rational state-space data where the mathematics supports exact computation.
- **Controllability and observability certificates** with matrix-rank and PBH cross-checks.
- **Failure witnesses** for uncontrollable and unobservable modes.
- **Continuous-time Hurwitz analysis** with spectral abscissa, margins, and boundary-aware classification.
- **Lyapunov certificates** with exact rational solves, positive-definiteness evidence, and equation residuals.
- **Continuous-time infinite-horizon LQR / CARE** with Q/R validation, stabilizability, detectability, independently verified CARE residuals, and a closed-loop Riccati/Lyapunov identity.
- **Kalman structural decomposition** with reachable/unobservable subspaces, their intersection, exact structural zeros, transformation residuals, and conditioning diagnostics.
- **Unified analysis sessions** that isolate unavailable or failed modules instead of discarding successful certificates.
- **Exportable Markdown and JSON reports**.
- **Offline Streamlit application** with curated examples and pole visualizations.

## Quick Start

Clone the repository and install the interactive-app extra:

```bash
git clone https://github.com/dongxuelian2/CertiControl.git
cd CertiControl
python -m pip install -e ".[app]"
streamlit run app.py
```

For library-only use, the core dependencies remain NumPy, SciPy, and SymPy:

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[test,app]"
pytest -q
```

## Interactive App

The Streamlit app requires no network service and no cloud backend. It provides:

- text input for `A`, `B`, optional `C`, and optional `D` using the same safe parser as the Python library;
- exact rational input such as `1/2` without pre-converting it to floating point;
- automatic or custom numerical rank tolerance;
- a seven-system curated example selector;
- Overview, Controllability, Observability, Stability, LQR, Kalman Structure, and Export tabs;
- PBH failure witnesses;
- open-loop pole plots and LQR open-vs-closed-loop pole plots;
- Kalman four-part structural dimensions and transformed matrices;
- downloadable Markdown and JSON certificates.

A useful demo path is **Four-part Kalman structure**, which gives

```text
Controllable / Observable       1
Controllable / Unobservable     1
Uncontrollable / Observable     1
Uncontrollable / Unobservable   1
```

The **Unstable but controllable** example provides a separate LQR demo in which the open-loop system is not Hurwitz but the certified LQR closed loop is Hurwitz.

## Python API Example

```python
from certicontrol import (
    AnalysisOptions,
    LTISystem,
    analyze_system,
    to_markdown,
)

system = LTISystem(
    A=[[0, 1], [-2, -3]],
    B=[[0], [1]],
    C=[[1, 0]],
    D=[[0]],
)

analysis = analyze_system(
    system,
    Q=[[1, 0], [0, 1]],
    R=[[1]],
    options=AnalysisOptions(run_kalman=True, run_lqr=True),
)

print(analysis.summary())
print(to_markdown(analysis))
```

If `C` is omitted, controllability and stability can still run; observability and full Kalman decomposition are explicitly reported as `NOT_RUN`. Likewise, LQR is `NOT_RUN` until both Q and R are supplied.

## Certificates

### Controllability

CertiControl computes

\[
\mathcal C=[B,AB,\ldots,A^{n-1}B]
\]

with exact rank/RREF data when rational input is available, SVD rank diagnostics otherwise, and an independent PBH check. If PBH fails, the certificate retains an offending left-eigenvector witness satisfying

\[
q^T A=\lambda q^T,\qquad q^T B=0
\]

(or the conjugate-transpose form for complex numerical data), together with residuals.

### Observability

The observability certificate uses

\[
\mathcal O=\begin{bmatrix}C\\CA\\\vdots\\CA^{n-1}\end{bmatrix}
\]

plus PBH observability. Failure evidence includes an unobservable right-eigenvector satisfying

\[
Av=\lambda v,\qquad Cv=0.
\]

### Stability / Lyapunov

Spectral analysis reports the continuous-time spectral abscissa and a three-state Hurwitz classification. The Lyapunov path checks

\[
A^*P+PA=-Q
\]

including symmetry/Hermitian residuals, positive-definiteness evidence, and exact zero residuals for supported rational systems. Imaginary-axis and tolerance-sensitive cases are not silently promoted to PASS.

### LQR / CARE

For

\[
J=\int_0^\infty(x^*Qx+u^*Ru)\,dt,
\]

CertiControl validates `Q \succeq 0`, `R \succ 0`, stabilizability, and detectability before accepting a CARE solution. It then verifies

\[
A^*P+PA-PBR^{-1}B^*P+Q=0
\]

without explicitly forming `R^{-1}`, computes `u=-Kx`, checks the closed-loop poles, and independently verifies

\[
A_{cl}^*P+PA_{cl}+Q+K^*RK=0.
\]

A successful SciPy CARE return alone is therefore not treated as a certificate.

### Kalman Structural Decomposition

The structural phase computes the canonical subspaces

\[
\mathcal R=\operatorname{im}\mathcal C,\qquad
\mathcal N=\ker\mathcal O,\qquad
\mathcal R\cap\mathcal N,
\]

and constructs a non-unique coordinate transformation

\[
x=Tz
\]

ordered as controllable/observable, controllable/unobservable, uncontrollable/observable, uncontrollable/unobservable. CertiControl verifies the invariant-subspace relations, similarity equations, input/output support, and only those transformed zero blocks that Kalman decomposition actually requires. The particular complement bases and `T` are generally non-canonical; the subspaces and dimensions are the structural invariants.

## Reports

`SystemAnalysis` can be exported without rerunning any mathematical analysis:

```python
from certicontrol import to_json, to_markdown

markdown_text = to_markdown(analysis)
json_text = to_json(analysis, indent=2)
```

Reports include system matrices, CertiControl version, timestamp, active tolerance policy, analyses performed, nested certificate evidence, machine-readable warning codes, and human-readable warnings.

## Testing

The GitHub Actions workflow tests Python 3.10 and Python 3.13. The suite includes the mathematical regression tests from Phases 1–5 plus reporting, unified-analysis, curated-example, visualization-helper, JSON roundtrip, partial-input, and app-import smoke tests.

## Mathematical Scope

Current scope:

- continuous-time;
- finite-dimensional LTI state-space systems;
- typically `n <= 10` for interactive numerical work;
- exact rational paths where explicitly supported by a certificate module.

Not currently implemented:

- discrete-time systems;
- transfer functions and frequency-domain analysis;
- minimal realization products or model reduction;
- pole placement or observer synthesis;
- Kalman filtering / LQG;
- MPC, H2, H-infinity, or robust-control synthesis;
- time-domain simulation UI;
- formal proof export.

**Kalman decomposition** in this project means the structural decomposition of a linear realization, not Kalman filtering.

## Certificate Semantics and Limitations

`PASS`, `FAIL`, and `INCONCLUSIVE` retain the tri-state semantics of the mathematical certificates; an analysis omitted because required input is missing is `NOT_RUN`. Numerical rank, subspace, stability, and conditioning conclusions remain tolerance-dependent, and ill-conditioned transformations are reported rather than hidden.

CertiControl certificates are **inspectable computational evidence**. They are not formal proofs or safety-critical engineering certification.

## License

MIT. See `LICENSE`.