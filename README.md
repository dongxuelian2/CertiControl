# CertiControl

CertiControl is a certificate-based analysis toolkit for continuous-time finite-dimensional LTI systems.

## Current scope

Phase 1 implements a complete controllability certificate slice:

- validated `LTISystem(A, B, C=None, D=None)` data model;
- safe integer, floating-point, and rational matrix parsing;
- centralized SVD rank tolerance policy;
- MIMO controllability matrix construction;
- exact SymPy rank/RREF pivots when rational data are available;
- numerical singular values, rank, condition diagnostics, and actual tolerance;
- PBH controllability cross-check;
- exact or residual-bearing numerical witnesses for uncontrollable modes;
- warnings when exact and floating-point classifications disagree.

Observability, Lyapunov analysis, LQR, Kalman decomposition, and UI work are intentionally outside this phase.

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

## What "certificate" means here

A CertiControl certificate is **computational/checkable evidence**: matrices, exact algebraic ranks where available, singular values, tolerances, PBH cross-checks, witnesses, and residuals. It is not a formal proof object and is not safety-critical certification.
