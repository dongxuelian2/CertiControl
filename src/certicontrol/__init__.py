"""CertiControl public API."""

from .analysis import AnalysisOptions, AnalysisRecord, SystemAnalysis, analyze_system, status_from_certificate
from .certificates import Certificate
from .controllability import analyze_controllability, controllability_matrix, pbh_controllability
from .decomposition import analyze_kalman_decomposition, analyze_observability_decomposition, analyze_reachability_decomposition, reachable_subspace, unobservable_subspace
from .examples import ExampleSystem, curated_examples, get_example
from .lyapunov import LyapunovSolution, analyze_lyapunov_stability, solve_lyapunov
from .lqr import CARESolution, analyze_detectability, analyze_lqr, analyze_stabilizability, solve_care
from .model import LTISystem
from .observability import analyze_observability, observability_matrix, pbh_observability
from .parsing import parse_matrix
from .reporting import matrix_to_latex, to_json, to_json_dict, to_markdown
from .stability import analyze_spectral_stability, analyze_stability
from .tolerance import NumericalRankResult, TolerancePolicy, numerical_rank

__version__ = "0.2.0"

__all__ = [
    "AnalysisOptions",
    "AnalysisRecord",
    "CARESolution",
    "Certificate",
    "ExampleSystem",
    "LTISystem",
    "LyapunovSolution",
    "NumericalRankResult",
    "SystemAnalysis",
    "TolerancePolicy",
    "analyze_controllability",
    "analyze_detectability",
    "analyze_kalman_decomposition",
    "analyze_lqr",
    "analyze_observability_decomposition",
    "analyze_observability",
    "analyze_reachability_decomposition",
    "analyze_lyapunov_stability",
    "analyze_spectral_stability",
    "analyze_stability",
    "analyze_stabilizability",
    "analyze_system",
    "controllability_matrix",
    "curated_examples",
    "get_example",
    "matrix_to_latex",
    "numerical_rank",
    "observability_matrix",
    "parse_matrix",
    "pbh_controllability",
    "pbh_observability",
    "reachable_subspace",
    "solve_care",
    "solve_lyapunov",
    "status_from_certificate",
    "to_json",
    "to_json_dict",
    "to_markdown",
    "unobservable_subspace",
]