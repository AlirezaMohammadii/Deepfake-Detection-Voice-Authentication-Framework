"""
Specialized Bayesian Network Implementations

Contains temporal, hierarchical, and causal Bayesian networks
for deepfake detection analysis.
"""

# Conditional imports to handle missing pgmpy gracefully
try:
    from .temporal_bn import TemporalBayesianNetwork
    from .hierarchical_bn import HierarchicalBayesianNetwork
    from .causal_bn import CausalBayesianNetwork
    NETWORKS_AVAILABLE = True
except ImportError:
    NETWORKS_AVAILABLE = False

__all__ = [
    'TemporalBayesianNetwork',
    'HierarchicalBayesianNetwork', 
    'CausalBayesianNetwork',
    'NETWORKS_AVAILABLE'
] 