"""
MetalBayes - Three-objective Bayesian optimization framework
For optimizing three objectives: Adhesion, Coverage, Uniformity
Supports independent optimization for organic and oxide formulations
"""

from .optimizer import OrganicOptimizer, OxideOptimizer
from .surrogate import GaussianProcessSurrogate
from .acquisition import EVHIAcquisition

__version__ = "0.1.0"
__all__ = [
    "OrganicOptimizer",
    "OxideOptimizer",
    "GaussianProcessSurrogate",
    "EVHIAcquisition",
]
