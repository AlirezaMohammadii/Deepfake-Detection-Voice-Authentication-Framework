"""
Advanced Bayesian Inference Engines

Provides multiple inference algorithms for Bayesian networks including
variational inference, MCMC, and exact inference.
"""

# Conditional imports to handle missing dependencies gracefully
try:
    from .variational_inference import VariationalInferenceEngine
    VARIATIONAL_AVAILABLE = True
except ImportError:
    VARIATIONAL_AVAILABLE = False

try:
    from .mcmc_inference import MCMCInferenceEngine
    MCMC_AVAILABLE = True
except ImportError:
    MCMC_AVAILABLE = False

__all__ = [
    'VariationalInferenceEngine',
    'MCMCInferenceEngine',
    'VARIATIONAL_AVAILABLE',
    'MCMC_AVAILABLE'
] 