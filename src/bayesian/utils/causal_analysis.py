"""
Causal Analysis Engine for Bayesian Networks
Implements causal inference and intervention analysis for deepfake detection
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)

class CausalRelationType(Enum):
    """Types of causal relationships"""
    DIRECT = "direct"           # X -> Y
    INDIRECT = "indirect"       # X -> Z -> Y
    COMMON_CAUSE = "common_cause"  # Z -> X, Z -> Y
    COLLIDER = "collider"       # X -> Z, Y -> Z
    SPURIOUS = "spurious"       # No causal relationship

@dataclass
class CausalRelationship:
    """Represents a causal relationship between variables"""
    cause: str
    effect: str
    relationship_type: CausalRelationType
    strength: float
    confidence: float
    confounders: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"{self.cause} -> {self.effect} ({self.relationship_type.value}, strength={self.strength:.3f})"

@dataclass
class InterventionResult:
    """Result of a causal intervention"""
    intervention_variable: str
    intervention_value: float
    pre_intervention_distribution: Dict[str, float]
    post_intervention_distribution: Dict[str, float]
    causal_effect: float
    confidence_interval: Tuple[float, float]
    significance: float

class CausalDiscoveryMethod(ABC):
    """Abstract base class for causal discovery methods"""
    
    @abstractmethod
    def discover_causal_structure(self, data: Dict[str, List[float]]) -> List[CausalRelationship]:
        """Discover causal relationships from data"""
        pass

class CorrelationBasedDiscovery(CausalDiscoveryMethod):
    """Simple correlation-based causal discovery"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def discover_causal_structure(self, data: Dict[str, List[float]]) -> List[CausalRelationship]:
        """Discover causal relationships using correlation analysis"""
        relationships = []
        variables = list(data.keys())
        
        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                try:
                    # Calculate correlation
                    values1 = np.array(data[var1])
                    values2 = np.array(data[var2])
                    
                    if len(values1) > 1 and len(values2) > 1:
                        correlation = np.corrcoef(values1, values2)[0, 1]
                        
                        if abs(correlation) > self.threshold:
                            # Assume direction based on physics knowledge
                            if self._is_physics_causal(var1, var2):
                                cause, effect = var1, var2
                            else:
                                cause, effect = var2, var1
                            
                            relationship = CausalRelationship(
                                cause=cause,
                                effect=effect,
                                relationship_type=CausalRelationType.DIRECT,
                                strength=abs(correlation),
                                confidence=min(0.95, abs(correlation) + 0.2)
                            )
                            relationships.append(relationship)
                
                except Exception as e:
                    logger.warning(f"Could not compute correlation between {var1} and {var2}: {e}")
        
        return relationships
    
    def _is_physics_causal(self, var1: str, var2: str) -> bool:
        """Determine causal direction based on physics knowledge"""
        # Simple heuristics based on physics features
        physics_hierarchy = [
            'delta_ft', 'delta_fr', 'delta_fv',  # Basic dynamics
            'velocity_magnitude', 'doppler_shift'  # Derived quantities
        ]
        
        try:
            idx1 = physics_hierarchy.index(var1)
            idx2 = physics_hierarchy.index(var2)
            return idx1 < idx2  # Earlier in hierarchy causes later
        except ValueError:
            # If not in hierarchy, use alphabetical order as fallback
            return var1 < var2

class PearlCausalEngine:
    """
    Implements Pearl's causal hierarchy for deepfake detection
    
    Level 1: Association (P(y|x)) - Traditional correlations
    Level 2: Intervention (P(y|do(x))) - What happens if we intervene
    Level 3: Counterfactuals (P(y_x|x',y')) - What would have happened
    """
    
    def __init__(self, discovery_method: Optional[CausalDiscoveryMethod] = None):
        self.discovery_method = discovery_method or CorrelationBasedDiscovery()
        self.causal_graph = {}
        self.learned_relationships = []
        
    def learn_causal_structure(self, data: Dict[str, List[float]]) -> None:
        """Learn causal structure from observational data"""
        logger.info("Learning causal structure from data...")
        
        # Discover causal relationships
        self.learned_relationships = self.discovery_method.discover_causal_structure(data)
        
        # Build causal graph
        self.causal_graph = self._build_graph(self.learned_relationships)
        
        logger.info(f"Discovered {len(self.learned_relationships)} causal relationships")
        for rel in self.learned_relationships:
            logger.debug(f"  {rel}")
    
    def _build_graph(self, relationships: List[CausalRelationship]) -> Dict[str, List[str]]:
        """Build adjacency list representation of causal graph"""
        graph = {}
        
        for rel in relationships:
            if rel.cause not in graph:
                graph[rel.cause] = []
            graph[rel.cause].append(rel.effect)
        
        return graph
    
    def estimate_causal_effect(self, 
                             cause: str, 
                             effect: str, 
                             data: Dict[str, List[float]],
                             confounders: Optional[List[str]] = None) -> float:
        """
        Estimate causal effect using adjustment formula
        P(y|do(x)) = Σ_z P(y|x,z) * P(z)
        """
        try:
            cause_values = np.array(data[cause])
            effect_values = np.array(data[effect])
            
            if confounders:
                # Adjust for confounders using stratification
                return self._adjust_for_confounders(cause_values, effect_values, data, confounders)
            else:
                # Simple linear effect estimation
                if len(cause_values) > 1 and len(effect_values) > 1:
                    # Use linear regression coefficient as causal effect estimate
                    X = np.column_stack([np.ones(len(cause_values)), cause_values])
                    beta = np.linalg.lstsq(X, effect_values, rcond=None)[0]
                    return beta[1]  # Slope coefficient
                else:
                    return 0.0
        
        except Exception as e:
            logger.warning(f"Could not estimate causal effect {cause} -> {effect}: {e}")
            return 0.0
    
    def _adjust_for_confounders(self, 
                              cause_values: np.ndarray,
                              effect_values: np.ndarray,
                              data: Dict[str, List[float]],
                              confounders: List[str]) -> float:
        """Adjust for confounders using stratification"""
        try:
            # Simple adjustment: include confounders in regression
            X_matrix = [np.ones(len(cause_values)), cause_values]
            
            for confounder in confounders:
                if confounder in data:
                    confounder_values = np.array(data[confounder])
                    if len(confounder_values) == len(cause_values):
                        X_matrix.append(confounder_values)
            
            X = np.column_stack(X_matrix)
            beta = np.linalg.lstsq(X, effect_values, rcond=None)[0]
            
            return beta[1]  # Coefficient for the cause variable
            
        except Exception as e:
            logger.warning(f"Confounder adjustment failed: {e}")
            return 0.0
    
    def do_intervention(self, 
                       intervention_var: str, 
                       intervention_value: float,
                       target_var: str,
                       data: Dict[str, List[float]]) -> InterventionResult:
        """
        Perform Pearl's do-intervention: P(target|do(intervention_var = value))
        """
        logger.info(f"Performing intervention: do({intervention_var} = {intervention_value})")
        
        # Pre-intervention distribution
        pre_distribution = self._estimate_distribution(data)
        
        # Simulate intervention by modifying the data
        intervention_data = data.copy()
        intervention_data[intervention_var] = [intervention_value] * len(data[intervention_var])
        
        # Post-intervention distribution
        post_distribution = self._estimate_distribution(intervention_data)
        
        # Calculate causal effect
        causal_effect = (post_distribution.get(target_var, 0) - 
                        pre_distribution.get(target_var, 0))
        
        # Estimate confidence interval (simplified)
        std_effect = abs(causal_effect) * 0.1  # 10% uncertainty
        confidence_interval = (causal_effect - 1.96 * std_effect, 
                             causal_effect + 1.96 * std_effect)
        
        return InterventionResult(
            intervention_variable=intervention_var,
            intervention_value=intervention_value,
            pre_intervention_distribution=pre_distribution,
            post_intervention_distribution=post_distribution,
            causal_effect=causal_effect,
            confidence_interval=confidence_interval,
            significance=abs(causal_effect) / max(std_effect, 1e-6)
        )
    
    def _estimate_distribution(self, data: Dict[str, List[float]]) -> Dict[str, float]:
        """Estimate mean values for each variable"""
        return {var: np.mean(values) for var, values in data.items() if values}
    
    def counterfactual_analysis(self, 
                              observed_data: Dict[str, float],
                              counterfactual_var: str,
                              counterfactual_value: float) -> Dict[str, float]:
        """
        Perform counterfactual analysis: What would have happened if X=x instead of X=x'?
        """
        logger.info(f"Counterfactual analysis: {counterfactual_var} = {counterfactual_value}")
        
        # This is a simplified implementation
        # In practice, would require more sophisticated structural equation modeling
        
        counterfactual_outcomes = observed_data.copy()
        
        # Propagate counterfactual change through causal graph
        if counterfactual_var in self.causal_graph:
            for effect_var in self.causal_graph[counterfactual_var]:
                # Find causal relationship
                rel = self._find_relationship(counterfactual_var, effect_var)
                if rel:
                    # Estimate counterfactual effect
                    change = (counterfactual_value - observed_data.get(counterfactual_var, 0))
                    effect_change = change * rel.strength
                    counterfactual_outcomes[effect_var] = (
                        observed_data.get(effect_var, 0) + effect_change
                    )
        
        return counterfactual_outcomes
    
    def _find_relationship(self, cause: str, effect: str) -> Optional[CausalRelationship]:
        """Find causal relationship between two variables"""
        for rel in self.learned_relationships:
            if rel.cause == cause and rel.effect == effect:
                return rel
        return None
    
    def get_causal_explanation(self, 
                             feature_values: Dict[str, float],
                             target_outcome: str) -> Dict[str, Any]:
        """
        Generate causal explanation for a given outcome
        """
        explanation = {
            'target_outcome': target_outcome,
            'causal_factors': [],
            'intervention_recommendations': [],
            'counterfactual_scenarios': []
        }
        
        # Find all variables that causally influence the target
        causal_factors = []
        for rel in self.learned_relationships:
            if rel.effect == target_outcome:
                factor_value = feature_values.get(rel.cause, 0)
                causal_factors.append({
                    'variable': rel.cause,
                    'value': factor_value,
                    'causal_strength': rel.strength,
                    'relationship_type': rel.relationship_type.value,
                    'contribution': factor_value * rel.strength
                })
        
        explanation['causal_factors'] = sorted(
            causal_factors, 
            key=lambda x: abs(x['contribution']), 
            reverse=True
        )
        
        # Generate intervention recommendations
        for factor in explanation['causal_factors'][:3]:  # Top 3 factors
            recommendation = {
                'variable': factor['variable'],
                'current_value': factor['value'],
                'recommended_change': 'decrease' if factor['contribution'] > 0 else 'increase',
                'expected_effect': abs(factor['contribution']) * 0.5
            }
            explanation['intervention_recommendations'].append(recommendation)
        
        # Generate counterfactual scenarios
        for factor in explanation['causal_factors'][:2]:  # Top 2 factors
            counterfactual_value = factor['value'] * 0.5  # 50% reduction
            counterfactual_outcome = self.counterfactual_analysis(
                feature_values, factor['variable'], counterfactual_value
            )
            
            scenario = {
                'variable': factor['variable'],
                'original_value': factor['value'],
                'counterfactual_value': counterfactual_value,
                'predicted_outcome': counterfactual_outcome.get(target_outcome, 0)
            }
            explanation['counterfactual_scenarios'].append(scenario)
        
        return explanation

class CausalAnalysisEngine:
    """
    Main causal analysis engine for deepfake detection
    Integrates causal discovery, intervention analysis, and counterfactual reasoning
    """
    
    def __init__(self, 
                 discovery_method: Optional[CausalDiscoveryMethod] = None,
                 enable_interventions: bool = True,
                 enable_counterfactuals: bool = True):
        """
        Initialize causal analysis engine
        
        Args:
            discovery_method: Method for causal discovery
            enable_interventions: Whether to enable intervention analysis
            enable_counterfactuals: Whether to enable counterfactual analysis
        """
        self.causal_engine = PearlCausalEngine(discovery_method)
        self.enable_interventions = enable_interventions
        self.enable_counterfactuals = enable_counterfactuals
        self.temporal_data = []
        
        logger.info(f"CausalAnalysisEngine initialized: "
                   f"interventions={enable_interventions}, counterfactuals={enable_counterfactuals}")
    
    def add_temporal_data(self, features: Dict[str, float], timestamp: float = None) -> None:
        """Add temporal data point for causal analysis"""
        if timestamp is None:
            timestamp = time.time()
        
        self.temporal_data.append({
            'timestamp': timestamp,
            'features': features.copy()
        })
        
        # Keep only recent data (last 100 samples)
        if len(self.temporal_data) > 100:
            self.temporal_data = self.temporal_data[-100:]
    
    def analyze_causal_structure(self) -> Dict[str, Any]:
        """Analyze causal structure from temporal data"""
        if len(self.temporal_data) < 10:
            return {
                'causal_relationships': [],
                'analysis': 'insufficient_data',
                'num_samples': len(self.temporal_data)
            }
        
        # Convert temporal data to format for causal discovery
        data_dict = {}
        for sample in self.temporal_data:
            for feature, value in sample['features'].items():
                if feature not in data_dict:
                    data_dict[feature] = []
                data_dict[feature].append(value)
        
        # Learn causal structure
        self.causal_engine.learn_causal_structure(data_dict)
        
        # Prepare results
        relationships_data = []
        for rel in self.causal_engine.learned_relationships:
            relationships_data.append({
                'cause': rel.cause,
                'effect': rel.effect,
                'type': rel.relationship_type.value,
                'strength': rel.strength,
                'confidence': rel.confidence
            })
        
        return {
            'causal_relationships': relationships_data,
            'num_relationships': len(relationships_data),
            'causal_graph': self.causal_engine.causal_graph,
            'analysis': 'completed',
            'num_samples': len(self.temporal_data)
        }
    
    def perform_intervention_analysis(self, 
                                    intervention_var: str,
                                    intervention_value: float,
                                    target_var: str = 'spoof_probability') -> Optional[Dict[str, Any]]:
        """Perform intervention analysis if enabled"""
        if not self.enable_interventions or len(self.temporal_data) < 5:
            return None
        
        # Convert temporal data for intervention
        data_dict = {}
        for sample in self.temporal_data:
            for feature, value in sample['features'].items():
                if feature not in data_dict:
                    data_dict[feature] = []
                data_dict[feature].append(value)
        
        # Perform intervention
        result = self.causal_engine.do_intervention(
            intervention_var, intervention_value, target_var, data_dict
        )
        
        return {
            'intervention_variable': result.intervention_variable,
            'intervention_value': result.intervention_value,
            'causal_effect': result.causal_effect,
            'confidence_interval': result.confidence_interval,
            'significance': result.significance,
            'interpretation': self._interpret_intervention(result)
        }
    
    def _interpret_intervention(self, result: InterventionResult) -> str:
        """Interpret intervention result"""
        effect_magnitude = abs(result.causal_effect)
        
        if effect_magnitude < 0.01:
            return "negligible_effect"
        elif effect_magnitude < 0.1:
            return "small_effect"
        elif effect_magnitude < 0.3:
            return "moderate_effect"
        else:
            return "large_effect"
    
    def generate_counterfactual_explanations(self, 
                                           current_features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Generate counterfactual explanations if enabled"""
        if not self.enable_counterfactuals or not self.causal_engine.learned_relationships:
            return None
        
        explanations = []
        
        # Generate counterfactuals for top causal factors
        for rel in self.causal_engine.learned_relationships[:3]:
            if rel.cause in current_features:
                original_value = current_features[rel.cause]
                
                # Try different counterfactual values
                for multiplier in [0.5, 0.8, 1.2, 1.5]:
                    counterfactual_value = original_value * multiplier
                    
                    counterfactual_outcome = self.causal_engine.counterfactual_analysis(
                        current_features, rel.cause, counterfactual_value
                    )
                    
                    explanations.append({
                        'variable': rel.cause,
                        'original_value': original_value,
                        'counterfactual_value': counterfactual_value,
                        'predicted_changes': counterfactual_outcome,
                        'causal_relationship': {
                            'effect': rel.effect,
                            'strength': rel.strength,
                            'type': rel.relationship_type.value
                        }
                    })
        
        return {
            'counterfactual_scenarios': explanations,
            'num_scenarios': len(explanations)
        }
    
    def get_causal_insights(self, current_features: Dict[str, float]) -> Dict[str, Any]:
        """Get comprehensive causal insights"""
        insights = {
            'causal_structure_analysis': self.analyze_causal_structure(),
            'current_features': current_features
        }
        
        # Add intervention analysis if enabled
        if self.enable_interventions and self.causal_engine.learned_relationships:
            # Try intervention on the strongest causal factor
            strongest_rel = max(self.causal_engine.learned_relationships, 
                              key=lambda r: r.strength, default=None)
            
            if strongest_rel and strongest_rel.cause in current_features:
                current_value = current_features[strongest_rel.cause]
                intervention_value = current_value * 0.5  # 50% reduction
                
                intervention_result = self.perform_intervention_analysis(
                    strongest_rel.cause, intervention_value, strongest_rel.effect
                )
                insights['intervention_analysis'] = intervention_result
        
        # Add counterfactual explanations if enabled
        if self.enable_counterfactuals:
            counterfactual_explanations = self.generate_counterfactual_explanations(current_features)
            if counterfactual_explanations:
                insights['counterfactual_explanations'] = counterfactual_explanations
        
        return insights
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get causal analysis statistics"""
        return {
            'temporal_data_points': len(self.temporal_data),
            'learned_relationships': len(self.causal_engine.learned_relationships),
            'causal_graph_size': len(self.causal_engine.causal_graph),
            'interventions_enabled': self.enable_interventions,
            'counterfactuals_enabled': self.enable_counterfactuals
        }

class CausalFeatureAnalyzer:
    """
    Feature analyzer with causal reasoning capabilities
    Wrapper class that provides causal analysis functionality for test_runner.py
    """
    
    def __init__(self, enable_interventions: bool = True, enable_counterfactuals: bool = True):
        """
        Initialize causal feature analyzer
        
        Args:
            enable_interventions: Whether to enable intervention analysis
            enable_counterfactuals: Whether to enable counterfactual analysis
        """
        self.causal_engine = CausalAnalysisEngine(
            enable_interventions=enable_interventions,
            enable_counterfactuals=enable_counterfactuals
        )
        
        logger.info(f"CausalFeatureAnalyzer initialized")
    
    def analyze_features(self, features: Dict[str, float], user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze features using causal reasoning
        
        Args:
            features: Dictionary of extracted features
            user_context: Optional user context information
            
        Returns:
            Dictionary with causal analysis results
        """
        # Add temporal data point
        self.causal_engine.add_temporal_data(features)
        
        # Get comprehensive causal insights
        insights = self.causal_engine.get_causal_insights(features)
        
        return {
            'causal_insights': insights,
            'temporal_data_points': len(self.causal_engine.temporal_data),
            'analysis_timestamp': time.time() if 'time' in globals() else 0.0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return self.causal_engine.get_statistics()
    
    def reset(self):
        """Reset the analyzer state"""
        self.causal_engine.temporal_data.clear()
        self.causal_engine.causal_engine.learned_relationships.clear()
        self.causal_engine.causal_engine.causal_graph.clear()
        
        logger.info("CausalFeatureAnalyzer reset") 