"""CertiControl public API."""

from .certificates import Certificate
from .controllability import analyze_controllability, controllability_matrix, pbh_controllability
from .lyapunov import LyapunovSolution, analyze_lyapunov_stability, solve_lyapunov
from .lqr import CARESolution, analyze_detectability, analyze_lqr, analyze_stabilizability, solve_care
from .model import LTISystem
from .observability import analyze_observability, observability_matrix, pbh_observability
from .parsing import parse_matrix
from .stability import analyze_spectral_stability, analyze_stability
from .tolerance import NumericalRankResult, TolerancePolicy, numerical_rank

__all__ = [
    "CARESolution",
    "Certificate",
    "LTISystem",
    "LyapunovSolution",
    "NumericalRankResult",
    "TolerancePolicy",
    "analyze_controllability",
    "analyze_detectability",
    "analyze_lqr",
    "analyze_observability",
    "analyze_lyapunov_stability",
    "analyze_spectral_stability",
    "analyze_stability",
    "analyze_stabilizability",
    "controllability_matrix",
    "numerical_rank",
    "observability_matrix",
    "parse_matrix",
    "pbh_controllability",
    "pbh_observability",
    "solve_care",
    "solve_lyapunov",
]