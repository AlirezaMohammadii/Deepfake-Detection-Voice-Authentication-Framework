"""
Bayesian Networks Integration for Physics-Based Deepfake Detection

This module provides comprehensive Bayesian network capabilities including:
- Temporal modeling with Dynamic Bayesian Networks
- Hierarchical analysis with multi-level modeling
- Causal inference using Pearl's framework
- Uncertainty quantification and confidence calibration
- User adaptation and personalization

Main Components:
- core: Core Bayesian engine and configuration
- networks: Specialized Bayesian network implementations
- inference: Advanced inference algorithms
- utils: Utility modules for temporal tracking and user management
"""

from .core.bayesian_engine import BayesianDeepfakeEngine, BayesianConfig, BayesianDetectionResult
from .utils.bayesian_config_loader import load_bayesian_config

__version__ = "1.0.0"
__author__ = "Physics-Based Deepfake Detection Team"

__all__ = [
    'BayesianDeepfakeEngine',
    'BayesianConfig', 
    'BayesianDetectionResult',
    'load_bayesian_config'
] 