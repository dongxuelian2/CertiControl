# CertiControl

CertiControl is a certificate-based analysis toolkit for continuous-time finite-dimensional LTI systems.

## Current scope

CertiControl currently implements controllability, observability, and continuous-time Hurwitz/Lyapunov certificate slices.

### Controllability

- validated `LTISystem(A, B, C=None, D=None)` data model;
- safe integer, floating-point, and rational matrix parsing;
- centralized SVD rank tolerance policy;
- MIMO controllability matrix construction;
- exact SymPy rank/RREF pivots when rational data are available;
- numerical singular values, rank, condition diagnostics, and actual tolerance;
- PBH controllability cross-check;
- exact or residual-bearing numerical witnesses for uncontrollable modes;
- warnings when exact and floating-point classifications disagree.

### Observability

- SISO/MIMO observability matrix construction;
- exact SymPy rank/RREF pivots when rational data are available;
- numerical singular values, rank, condition diagnostics, and actual tolerance;
- PBH observability cross-check;
- exact or residual-bearing witnesses for unobservable modes;
- controllability--observability duality and similarity-invariance tests.

LQR, Riccati equations, decomposition, discrete-time analysis, and UI work remain intentionally outside the current scope.

## Install

```bash
python -m pip install -e .
```

For development tests:

```bash
python -m pip install -e ".[test]"
pytest
```

## Example

```python
from certicontrol import LTISystem, analyze_controllability, parse_matrix

A = parse_matrix("""
0  1
-2 -3
""")
B = parse_matrix("""
0
1
""")

system = LTISystem(A, B)
certificate = analyze_controllability(system)

print(certificate.verdict)                       # controllable
print(certificate.diagnostics["exact_rank"])   # 2
print(certificate.diagnostics["numerical_rank"])  # 2
print(certificate.diagnostics["singular_values"])
print(certificate.diagnostics["tolerance"])
```

## Observability example

```python
from certicontrol import LTISystem, analyze_observability

system = LTISystem(
    A=[[-1, 0], [0, -2]],
    B=[[1], [0]],
    C=[[1, 0]],
)
certificate = analyze_observability(system)

print(certificate.verdict)                        # unobservable
print(certificate.diagnostics["exact_rank"])    # 1
print(certificate.diagnostics["pbh_exact_passed"])  # False
print(certificate.evidence["pbh"].evidence["exact_failures"])  # witness data
```

## Stability and Lyapunov Certificates

For finite-dimensional continuous-time LTI systems, Hurwitz stability is equivalent to asymptotic/exponential stability. CertiControl reports both spectral evidence and a checkable Lyapunov certificate rather than only a Boolean.

```python
from certicontrol import LTISystem, analyze_stability

system = LTISystem(
    A=[[0, 1], [-2, -3]],
    B=[[0], [1]],
)
certificate = analyze_stability(system)

spectral = certificate.evidence["spectral"]
lyapunov = certificate.evidence["lyapunov"]

print(spectral.diagnostics["spectral_abscissa"])  # -1.0
print(certificate.verdict)                         # HURWITZ
print(lyapunov.evidence["P"])                    # numerical P
print(lyapunov.diagnostics["lambda_min_P"])
print(lyapunov.diagnostics["relative_residual"])
print(lyapunov.evidence["P_exact"])              # exact rational P when available
```

With the default `Q = I`, the Lyapunov evidence checks `A^T P + P A = -Q` for real systems (or `A^* P + P A = -Q` for complex systems), verifies positive definiteness, and records residuals and exact leading principal minors when rational data are available. Boundary eigenvalues near the imaginary axis are reported as tolerance-sensitive rather than being mislabeled as asymptotically stable.

## What "certificate" means here

A CertiControl certificate is **computational/checkable evidence**: matrices, exact algebraic ranks where available, singular values, tolerances, PBH cross-checks, witnesses, and residuals. It is not a formal proof object and is not safety-critical certification.
