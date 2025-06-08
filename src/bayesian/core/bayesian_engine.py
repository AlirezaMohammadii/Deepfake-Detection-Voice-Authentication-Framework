"""
Core Bayesian Engine for Deepfake Detection
Implements the main probabilistic reasoning framework
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Conditional imports for Bayesian libraries
try:
    import pgmpy
    from pgmpy.models import BayesianNetwork, DynamicBayesianNetwork as DBN
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination, ApproxInference
    from pgmpy.estimators import HillClimbSearch, K2Score, BayesianEstimator
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    logging.warning("pgmpy not available. Some Bayesian features will be limited.")

@dataclass
class BayesianConfig:
    """Configuration for Bayesian components"""
    enable_temporal_modeling: bool = True
    enable_hierarchical_modeling: bool = True
    enable_causal_analysis: bool = True
    inference_method: str = "variational"  # variational, mcmc, exact
    temporal_window_size: int = 10
    max_inference_time: float = 5.0
    uncertainty_threshold: float = 0.1
    
@dataclass
class BayesianDetectionResult:
    """Result container for Bayesian detection analysis"""
    spoof_probability: float
    confidence_score: float
    uncertainty_metrics: Dict[str, float]
    causal_explanations: Dict[str, float]
    temporal_consistency: Optional[float] = None
    feature_dependencies: Optional[Dict[str, List[str]]] = None
    counterfactual_analysis: Optional[Dict[str, float]] = None
    processing_time: float = 0.0
    
class ConfidenceLevel(Enum):
    """Confidence levels for detection decisions"""
    VERY_HIGH = "very_high"    # >95%
    HIGH = "high"              # 85-95%
    MEDIUM = "medium"          # 70-85%
    LOW = "low"                # 50-70%
    VERY_LOW = "very_low"      # <50%

class BayesianDeepfakeEngine:
    """
    Main Bayesian reasoning engine for deepfake detection
    Integrates multiple BN architectures for comprehensive analysis
    """
    
    def __init__(self, config: Optional[BayesianConfig] = None):
        self.config = config or BayesianConfig()
        self.logger = logging.getLogger(__name__)
        
        # Check for required dependencies
        if not PGMPY_AVAILABLE:
            self.logger.warning("pgmpy not available. Using simplified Bayesian analysis.")
        
        # Initialize BN components
        self._initialize_bayesian_networks()
        self._initialize_inference_engines()
        
        # State management
        self.temporal_state = {}
        self.user_contexts = {}
        self.feature_cache = {}
        
    def _initialize_bayesian_networks(self):
        """Initialize all Bayesian network components"""
        try:
            if self.config.enable_temporal_modeling and PGMPY_AVAILABLE:
                from ..networks.temporal_bn import TemporalBayesianNetwork
                self.temporal_bn = TemporalBayesianNetwork(
                    window_size=self.config.temporal_window_size
                )
            else:
                self.temporal_bn = None
                
            if self.config.enable_hierarchical_modeling and PGMPY_AVAILABLE:
                from ..networks.hierarchical_bn import HierarchicalBayesianNetwork
                self.hierarchical_bn = HierarchicalBayesianNetwork()
            else:
                self.hierarchical_bn = None
                
            if self.config.enable_causal_analysis and PGMPY_AVAILABLE:
                from ..networks.causal_bn import CausalBayesianNetwork
                self.causal_bn = CausalBayesianNetwork()
            else:
                self.causal_bn = None
                
            # Feature dependency network (simplified version available without pgmpy)
            self.feature_bn = self._create_feature_dependency_network()
            
        except ImportError as e:
            self.logger.warning(f"Could not initialize all Bayesian networks: {e}")
            self.temporal_bn = None
            self.hierarchical_bn = None
            self.causal_bn = None
            self.feature_bn = None
            
    def _initialize_inference_engines(self):
        """Initialize inference engines based on configuration"""
        try:
            if self.config.inference_method == "variational":
                from ..inference.variational_inference import VariationalInferenceEngine
                self.inference_engine = VariationalInferenceEngine()
            elif self.config.inference_method == "mcmc" and PGMPY_AVAILABLE:
                from ..inference.mcmc_inference import MCMCInferenceEngine
                self.inference_engine = MCMCInferenceEngine()
            else:  # exact or fallback
                self.inference_engine = self._create_simple_inference_engine()
        except ImportError:
            self.inference_engine = self._create_simple_inference_engine()
            
    def _create_simple_inference_engine(self):
        """Create simplified inference engine when advanced libraries unavailable"""
        class SimpleInferenceEngine:
            def __init__(self):
                self.name = "simple"
                
            async def infer(self, evidence, prior=None):
                # Simple probabilistic inference based on feature values
                return self._simple_probabilistic_inference(evidence)
                
            def _simple_probabilistic_inference(self, evidence):
                # Simplified Bayesian inference using heuristics
                total_evidence = sum(evidence.values()) / len(evidence)
                probability = 1.0 / (1.0 + np.exp(-total_evidence))  # Sigmoid
                confidence = abs(probability - 0.5) * 2  # Distance from uncertainty
                return {
                    'probability': probability,
                    'confidence': confidence,
                    'uncertainty': 1.0 - confidence
                }
        
        return SimpleInferenceEngine()
            
    def _create_feature_dependency_network(self):
        """Create feature dependency network"""
        class FeatureDependencyNetwork:
            def __init__(self):
                # Define known physics feature dependencies
                self.dependencies = {
                    'delta_fr_revised': ['algorithm_artifacts', 'synthesis_process'],
                    'delta_ft_revised': ['recording_conditions', 'motion_patterns'],
                    'delta_fv_revised': ['temporal_stability', 'synthesis_quality'],
                    'authenticity': ['delta_fr_revised', 'delta_ft_revised', 'delta_fv_revised']
                }
                
            def analyze_dependencies(self, features):
                return self.dependencies
                
        return FeatureDependencyNetwork()

    async def analyze_audio_probabilistic(self, 
                                        physics_features: Dict[str, torch.Tensor],
                                        temporal_sequence: List[Dict],
                                        user_context: Optional[Dict] = None,
                                        audio_metadata: Optional[Dict] = None) -> BayesianDetectionResult:
        """
        Comprehensive Bayesian analysis of audio features
        
        Args:
            physics_features: Current physics features from VoiceRadar
            temporal_sequence: Historical feature sequence
            user_context: User-specific information
            audio_metadata: Additional audio metadata
            
        Returns:
            Comprehensive Bayesian analysis result
        """
        start_time = time.time()
        
        try:
            # Convert physics features to discrete states for BN processing
            discretized_features = self._discretize_physics_features(physics_features)
            
            # Initialize results
            spoof_probability = 0.5
            confidence_score = 0.0
            uncertainty_metrics = {'total_uncertainty': 1.0}
            causal_explanations = {}
            temporal_consistency = None
            feature_dependencies = None
            
            # Temporal analysis
            if self.config.enable_temporal_modeling and self.temporal_bn and temporal_sequence:
                try:
                    temporal_analysis = await self._analyze_temporal_sequence(
                        discretized_features, temporal_sequence
                    )
                    temporal_consistency = temporal_analysis.get('temporal_consistency', 0.5)
                    spoof_probability = temporal_analysis.get('sequence_probability', spoof_probability)
                except Exception as e:
                    self.logger.warning(f"Temporal analysis failed: {e}")
            
            # Hierarchical analysis
            if self.config.enable_hierarchical_modeling and self.hierarchical_bn:
                try:
                    hierarchical_analysis = await self._analyze_hierarchical(
                        discretized_features, user_context, audio_metadata
                    )
                    hierarchical_prob = hierarchical_analysis.get('hierarchical_authenticity', 0.5)
                    spoof_probability = (spoof_probability + (1.0 - hierarchical_prob)) / 2
                    confidence_score = max(confidence_score, hierarchical_analysis.get('audio_level_confidence', 0.0))
                except Exception as e:
                    self.logger.warning(f"Hierarchical analysis failed: {e}")
            
            # Feature dependency analysis
            if self.feature_bn:
                try:
                    dependency_analysis = await self._analyze_feature_dependencies(discretized_features)
                    feature_dependencies = dependency_analysis.get('dependencies')
                except Exception as e:
                    self.logger.warning(f"Dependency analysis failed: {e}")
            
            # Causal analysis
            if self.config.enable_causal_analysis and self.causal_bn:
                try:
                    causal_analysis = await self._perform_causal_analysis(discretized_features)
                    causal_explanations = causal_analysis.get('causal_explanations', {})
                    # Adjust probability based on causal evidence
                    causal_prob = causal_analysis.get('observational_probability', spoof_probability)
                    spoof_probability = (spoof_probability + causal_prob) / 2
                except Exception as e:
                    self.logger.warning(f"Causal analysis failed: {e}")
            
            # Fallback analysis using simple methods
            if spoof_probability == 0.5 and confidence_score == 0.0:
                simple_result = await self._simple_bayesian_analysis(physics_features)
                spoof_probability = simple_result['spoof_probability']
                confidence_score = simple_result['confidence_score']
                uncertainty_metrics = simple_result['uncertainty_metrics']
                causal_explanations = simple_result['causal_explanations']
            
            # Ensure probability is in valid range
            spoof_probability = max(0.0, min(1.0, spoof_probability))
            confidence_score = max(0.0, min(1.0, confidence_score))
            
            # Calculate uncertainty metrics
            if uncertainty_metrics.get('total_uncertainty', 1.0) == 1.0:
                uncertainty_metrics = self._calculate_uncertainty_metrics(
                    spoof_probability, confidence_score, physics_features
                )
            
            # Create result
            result = BayesianDetectionResult(
                spoof_probability=spoof_probability,
                confidence_score=confidence_score,
                uncertainty_metrics=uncertainty_metrics,
                causal_explanations=causal_explanations,
                temporal_consistency=temporal_consistency,
                feature_dependencies=feature_dependencies,
                processing_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Bayesian analysis failed: {e}")
            return self._create_fallback_result(time.time() - start_time)
    
    def _discretize_physics_features(self, physics_features: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """Convert continuous physics features to discrete states"""
        discretized = {}
        
        for feature_name, feature_value in physics_features.items():
            if torch.is_tensor(feature_value):
                value = feature_value.item()
            else:
                value = float(feature_value)
            
            # Feature-specific discretization based on empirical analysis
            if 'delta_fr' in feature_name:
                # Rotational dynamics (most discriminative)
                if value < 6.5:
                    discretized[feature_name] = 'low'
                elif value < 7.5:
                    discretized[feature_name] = 'medium'
                else:
                    discretized[feature_name] = 'high'
                    
            elif 'delta_ft' in feature_name:
                # Translational dynamics
                if value < 0.06:
                    discretized[feature_name] = 'low'
                elif value < 0.08:
                    discretized[feature_name] = 'medium'
                else:
                    discretized[feature_name] = 'high'
                    
            elif 'delta_fv' in feature_name:
                # Vibrational dynamics
                if value < 1.0:
                    discretized[feature_name] = 'low'
                elif value < 1.5:
                    discretized[feature_name] = 'medium'
                else:
                    discretized[feature_name] = 'high'
                    
            else:
                # Generic discretization
                if value < 0.33:
                    discretized[feature_name] = 'low'
                elif value < 0.67:
                    discretized[feature_name] = 'medium'
                else:
                    discretized[feature_name] = 'high'
        
        return discretized

    async def _analyze_temporal_sequence(self, discretized_features: Dict[str, str], 
                                       temporal_sequence: List[Dict]) -> Dict[str, float]:
        """Analyze temporal sequence for authenticity"""
        if not self.temporal_bn:
            return {'temporal_consistency': 0.5, 'sequence_probability': 0.5}
        
        try:
            return await self.temporal_bn.analyze_sequence([discretized_features] + temporal_sequence)
        except Exception as e:
            self.logger.warning(f"Temporal sequence analysis failed: {e}")
            return {'temporal_consistency': 0.5, 'sequence_probability': 0.5}
    
    async def _analyze_hierarchical(self, discretized_features: Dict[str, str],
                                  user_context: Optional[Dict] = None,
                                  audio_metadata: Optional[Dict] = None) -> Dict[str, float]:
        """Perform hierarchical Bayesian analysis"""
        if not self.hierarchical_bn:
            return {'hierarchical_authenticity': 0.5, 'audio_level_confidence': 0.5}
        
        try:
            return await self.hierarchical_bn.analyze_hierarchical(
                discretized_features, user_context, audio_metadata
            )
        except Exception as e:
            self.logger.warning(f"Hierarchical analysis failed: {e}")
            return {'hierarchical_authenticity': 0.5, 'audio_level_confidence': 0.5}
    
    async def _analyze_feature_dependencies(self, discretized_features: Dict[str, str]) -> Dict[str, Any]:
        """Analyze feature dependencies"""
        if not self.feature_bn:
            return {'dependencies': {}}
        
        try:
            dependencies = self.feature_bn.analyze_dependencies(discretized_features)
            return {'dependencies': dependencies}
        except Exception as e:
            self.logger.warning(f"Feature dependency analysis failed: {e}")
            return {'dependencies': {}}
    
    async def _perform_causal_analysis(self, discretized_features: Dict[str, str]) -> Dict[str, Any]:
        """Perform causal intervention analysis"""
        if not self.causal_bn:
            return {'causal_explanations': {}, 'observational_probability': 0.5}
        
        try:
            return await self.causal_bn.perform_causal_analysis(discretized_features)
        except Exception as e:
            self.logger.warning(f"Causal analysis failed: {e}")
            return {'causal_explanations': {}, 'observational_probability': 0.5}
    
    async def _simple_bayesian_analysis(self, physics_features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Simplified Bayesian analysis when advanced methods unavailable"""
        
        # Extract feature values
        feature_values = {}
        for name, value in physics_features.items():
            if torch.is_tensor(value):
                feature_values[name] = value.item()
            else:
                feature_values[name] = float(value)
        
        # Simple probabilistic model based on empirical observations
        spoof_indicators = 0.0
        confidence_contributions = []
        
        # Rotational dynamics (most discriminative)
        if 'delta_fr_revised' in feature_values:
            delta_fr = feature_values['delta_fr_revised']
            if delta_fr > 7.5:  # High rotation indicates TTS
                spoof_indicators += 0.4
                confidence_contributions.append(0.8)
            elif delta_fr > 7.2:  # Medium rotation
                spoof_indicators += 0.2
                confidence_contributions.append(0.6)
            else:  # Low rotation indicates genuine
                spoof_indicators += 0.0
                confidence_contributions.append(0.7)
        
        # Translational dynamics
        if 'delta_ft_revised' in feature_values:
            delta_ft = feature_values['delta_ft_revised']
            if delta_ft > 0.08:  # High translation
                spoof_indicators += 0.1
                confidence_contributions.append(0.4)
            else:
                confidence_contributions.append(0.3)
        
        # Vibrational dynamics
        if 'delta_fv_revised' in feature_values:
            delta_fv = feature_values['delta_fv_revised']
            if delta_fv > 1.5:  # High vibration
                spoof_indicators += 0.2
                confidence_contributions.append(0.5)
            else:
                confidence_contributions.append(0.4)
        
        # Total dynamics
        if 'delta_f_total_revised' in feature_values:
            delta_total = feature_values['delta_f_total_revised']
            if delta_total > 8.0:  # High total dynamics
                spoof_indicators += 0.3
                confidence_contributions.append(0.7)
        
        # Calculate final probabilities
        spoof_probability = min(1.0, spoof_indicators)
        confidence_score = np.mean(confidence_contributions) if confidence_contributions else 0.5
        
        # Calculate uncertainty metrics
        uncertainty = 1.0 - confidence_score
        uncertainty_metrics = {
            'total_uncertainty': uncertainty,
            'epistemic_uncertainty': uncertainty * 0.6,
            'aleatoric_uncertainty': uncertainty * 0.4
        }
        
        # Simple causal explanations
        causal_explanations = {}
        total_influence = sum([0.4 if 'delta_fr_revised' in feature_values else 0,
                              0.2 if 'delta_ft_revised' in feature_values else 0,
                              0.3 if 'delta_fv_revised' in feature_values else 0,
                              0.1])  # baseline
        
        if 'delta_fr_revised' in feature_values:
            causal_explanations['delta_fr_influence'] = 0.4 / total_influence
        if 'delta_ft_revised' in feature_values:
            causal_explanations['delta_ft_influence'] = 0.2 / total_influence
        if 'delta_fv_revised' in feature_values:
            causal_explanations['delta_fv_influence'] = 0.3 / total_influence
        
        return {
            'spoof_probability': spoof_probability,
            'confidence_score': confidence_score,
            'uncertainty_metrics': uncertainty_metrics,
            'causal_explanations': causal_explanations
        }
    
    def _calculate_uncertainty_metrics(self, spoof_probability: float, 
                                     confidence_score: float,
                                     physics_features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate comprehensive uncertainty metrics"""
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = 1.0 - confidence_score
        
        # Aleatoric uncertainty (data uncertainty)
        # Based on how close to decision boundary
        decision_uncertainty = 1.0 - abs(spoof_probability - 0.5) * 2
        aleatoric = decision_uncertainty * 0.5
        
        # Feature-based uncertainty
        feature_uncertainty = 0.0
        if physics_features:
            feature_vars = []
            for name, value in physics_features.items():
                if torch.is_tensor(value):
                    # Simulate uncertainty based on feature magnitude
                    feature_vars.append(abs(value.item()) * 0.1)
            
            if feature_vars:
                feature_uncertainty = np.mean(feature_vars)
        
        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic**2 + aleatoric**2 + feature_uncertainty**2)
        total_uncertainty = min(1.0, total_uncertainty)
        
        return {
            'total_uncertainty': total_uncertainty,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'feature_uncertainty': feature_uncertainty
        }
    
    def _create_fallback_result(self, processing_time: float = 0.0) -> BayesianDetectionResult:
        """Create fallback result when analysis fails"""
        return BayesianDetectionResult(
            spoof_probability=0.5,
            confidence_score=0.0,
            uncertainty_metrics={'total_uncertainty': 1.0},
            causal_explanations={},
            temporal_consistency=None,
            feature_dependencies=None,
            counterfactual_analysis=None,
            processing_time=processing_time
        )
    
    def get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Convert confidence score to categorical level"""
        if confidence_score >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.85:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def update_temporal_state(self, user_id: str, features: Dict[str, Any]) -> None:
        """Update temporal state for user"""
        if user_id not in self.temporal_state:
            self.temporal_state[user_id] = []
        
        self.temporal_state[user_id].append({
            'timestamp': time.time(),
            'features': features
        })
        
        # Keep only recent history
        max_history = self.config.temporal_window_size * 2
        if len(self.temporal_state[user_id]) > max_history:
            self.temporal_state[user_id] = self.temporal_state[user_id][-max_history:]
    
    def get_temporal_context(self, user_id: str) -> List[Dict]:
        """Get temporal context for user"""
        return self.temporal_state.get(user_id, [])
    
    def clear_temporal_state(self, user_id: Optional[str] = None) -> None:
        """Clear temporal state for user or all users"""
        if user_id:
            self.temporal_state.pop(user_id, None)
        else:
            self.temporal_state.clear() 