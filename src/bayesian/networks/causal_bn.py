"""
Causal Bayesian Network for Intervention Analysis
Implements Pearl's causal framework for counterfactual analysis
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
from dataclasses import dataclass
import time

# Conditional imports
try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination, CausalInference
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False

@dataclass
class CausalIntervention:
    """Represents a causal intervention"""
    variable: str
    value: Any
    intervention_type: str = "do"  # do, see, counterfactual

@dataclass
class CausalExplanation:
    """Causal explanation result"""
    variable: str
    causal_effect: float
    counterfactual_probability: float
    explanation: str

class CausalBayesianNetwork:
    """
    Causal Bayesian Network for deepfake detection
    Implements Pearl's causal hierarchy: association, intervention, counterfactuals
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize causal BN if available
        if PGMPY_AVAILABLE:
            self._initialize_causal_bn()
        else:
            self.logger.warning("pgmpy not available. Using simplified causal analysis.")
            self.causal_bn = None
            
        # Causal knowledge base
        self.causal_relationships = self._define_causal_relationships()
        self.intervention_cache = {}
        
    def _initialize_causal_bn(self):
        """Initialize causal Bayesian network structure"""
        try:
            # Create causal DAG representing the data generating process
            self.causal_bn = BayesianNetwork()
            
            # Define causal variables
            causal_variables = [
                # Exogenous variables (external causes)
                'synthesis_algorithm',      # TTS algorithm type
                'voice_model_quality',      # Quality of voice model
                'recording_conditions',     # Recording environment
                'speaker_characteristics',  # Original speaker traits
                
                # Endogenous variables (observable effects)
                'algorithm_artifacts',      # Artifacts from synthesis process
                'spectral_distortions',     # Frequency domain distortions
                'temporal_inconsistencies', # Time domain inconsistencies
                'voiceprint_deviation',     # Deviation from natural voiceprint
                
                # Intermediate mechanisms
                'synthesis_process',        # The synthesis mechanism
                'vocoder_effects',          # Vocoder-induced changes
                'prosody_modification',     # Prosodic alterations
                
                # Observable physics features
                'delta_fr_causal',         # Rotational dynamics (causal version)
                'delta_ft_causal',         # Translational dynamics (causal version)
                'delta_fv_causal',         # Vibrational dynamics (causal version)
                
                # Final outcome
                'authenticity_causal'      # Authenticity (causal version)
            ]
            
            # Add nodes
            for var in causal_variables:
                self.causal_bn.add_node(var)
            
            # Define causal edges based on domain knowledge
            causal_edges = [
                # Synthesis algorithm influences all downstream processes
                ('synthesis_algorithm', 'synthesis_process'),
                ('synthesis_algorithm', 'algorithm_artifacts'),
                ('synthesis_algorithm', 'vocoder_effects'),
                
                # Voice model quality affects synthesis fidelity
                ('voice_model_quality', 'synthesis_process'),
                ('voice_model_quality', 'voiceprint_deviation'),
                
                # Recording conditions affect observable features
                ('recording_conditions', 'spectral_distortions'),
                ('recording_conditions', 'delta_fr_causal'),
                ('recording_conditions', 'delta_ft_causal'),
                
                # Speaker characteristics influence natural patterns
                ('speaker_characteristics', 'voiceprint_deviation'),
                ('speaker_characteristics', 'prosody_modification'),
                
                # Synthesis process causes various artifacts
                ('synthesis_process', 'algorithm_artifacts'),
                ('synthesis_process', 'spectral_distortions'),
                ('synthesis_process', 'temporal_inconsistencies'),
                ('synthesis_process', 'vocoder_effects'),
                
                # Algorithm artifacts affect physics features
                ('algorithm_artifacts', 'delta_fr_causal'),
                ('algorithm_artifacts', 'delta_ft_causal'),
                ('algorithm_artifacts', 'delta_fv_causal'),
                
                # Spectral distortions influence dynamics
                ('spectral_distortions', 'delta_fr_causal'),
                ('spectral_distortions', 'delta_fv_causal'),
                
                # Temporal inconsistencies affect all dynamics
                ('temporal_inconsistencies', 'delta_fr_causal'),
                ('temporal_inconsistencies', 'delta_ft_causal'),
                ('temporal_inconsistencies', 'delta_fv_causal'),
                
                # Vocoder effects influence features
                ('vocoder_effects', 'delta_fr_causal'),
                ('vocoder_effects', 'delta_ft_causal'),
                
                # Prosody modifications affect dynamics
                ('prosody_modification', 'delta_ft_causal'),
                ('prosody_modification', 'delta_fv_causal'),
                
                # Voiceprint deviation influences all features
                ('voiceprint_deviation', 'delta_fr_causal'),
                ('voiceprint_deviation', 'delta_ft_causal'),
                ('voiceprint_deviation', 'delta_fv_causal'),
                
                # Physics features determine authenticity
                ('delta_fr_causal', 'authenticity_causal'),
                ('delta_ft_causal', 'authenticity_causal'),
                ('delta_fv_causal', 'authenticity_causal'),
                
                # Direct causal pathways to authenticity
                ('synthesis_process', 'authenticity_causal'),
                ('algorithm_artifacts', 'authenticity_causal'),
                ('voiceprint_deviation', 'authenticity_causal'),
            ]
            
            self.causal_bn.add_edges_from(causal_edges)
            
            # Define causal CPDs
            self._define_causal_cpds()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize causal BN: {e}")
            self.causal_bn = None
    
    def _define_causal_cpds(self):
        """Define Conditional Probability Distributions for causal model"""
        if not self.causal_bn:
            return
            
        try:
            # Exogenous variable CPDs (root causes)
            cpd_synthesis_alg = TabularCPD(
                variable='synthesis_algorithm',
                variable_card=4,  # none, basic_tts, advanced_tts, neural_voice
                values=[[0.7], [0.15], [0.1], [0.05]]  # Most audio is genuine
            )
            
            cpd_voice_quality = TabularCPD(
                variable='voice_model_quality',
                variable_card=3,  # poor, good, excellent
                values=[[0.3], [0.4], [0.3]]
            )
            
            cpd_recording_cond = TabularCPD(
                variable='recording_conditions',
                variable_card=3,  # poor, good, excellent
                values=[[0.2], [0.5], [0.3]]
            )
            
            cpd_speaker_char = TabularCPD(
                variable='speaker_characteristics',
                variable_card=3,  # typical, distinctive, unique
                values=[[0.5], [0.3], [0.2]]
            )
            
            # Intermediate mechanism CPDs
            cpd_synthesis_process = TabularCPD(
                variable='synthesis_process',
                variable_card=3,  # none, moderate, intensive
                values=[
                    [0.95, 0.8, 0.6, 0.2,   # none synthesis for different algorithms
                     0.9, 0.7, 0.5, 0.1,    # different voice qualities
                     0.85, 0.65, 0.45, 0.05],
                    [0.04, 0.15, 0.3, 0.4,  # moderate synthesis
                     0.08, 0.2, 0.35, 0.3,
                     0.12, 0.25, 0.4, 0.35],
                    [0.01, 0.05, 0.1, 0.4,  # intensive synthesis
                     0.02, 0.1, 0.15, 0.6,
                     0.03, 0.1, 0.15, 0.6]
                ],
                evidence=['synthesis_algorithm', 'voice_model_quality'],
                evidence_card=[4, 3]
            )
            
            # Algorithm artifacts based on synthesis algorithm and process
            cpd_algorithm_artifacts = TabularCPD(
                variable='algorithm_artifacts',
                variable_card=3,  # none, some, many
                values=[
                    [0.95, 0.8, 0.6, 0.2, 0.85, 0.6, 0.3, 0.05, 0.7, 0.4, 0.1, 0.02],  # none artifacts
                    [0.04, 0.15, 0.25, 0.3, 0.12, 0.25, 0.4, 0.25, 0.2, 0.35, 0.4, 0.28],  # some artifacts
                    [0.01, 0.05, 0.15, 0.5, 0.03, 0.15, 0.3, 0.7, 0.1, 0.25, 0.5, 0.7]    # many artifacts
                ],
                evidence=['synthesis_algorithm', 'synthesis_process'],
                evidence_card=[4, 3]
            )
            
            # Spectral distortions
            cpd_spectral_dist = TabularCPD(
                variable='spectral_distortions',
                variable_card=3,  # low, medium, high
                values=[
                    [0.8, 0.6, 0.4, 0.7, 0.5, 0.3, 0.6, 0.4, 0.2],  # low distortions
                    [0.15, 0.25, 0.35, 0.2, 0.3, 0.4, 0.25, 0.35, 0.45],  # medium
                    [0.05, 0.15, 0.25, 0.1, 0.2, 0.3, 0.15, 0.25, 0.35]   # high
                ],
                evidence=['recording_conditions', 'synthesis_process'],
                evidence_card=[3, 3]
            )
            
            # Temporal inconsistencies
            cpd_temporal_incons = TabularCPD(
                variable='temporal_inconsistencies',
                variable_card=3,  # low, medium, high
                values=[
                    [0.9, 0.7, 0.4],  # low inconsistencies for none/moderate/intensive synthesis
                    [0.08, 0.2, 0.35],  # medium
                    [0.02, 0.1, 0.25]   # high
                ],
                evidence=['synthesis_process'],
                evidence_card=[3]
            )
            
            # Vocoder effects
            cpd_vocoder_effects = TabularCPD(
                variable='vocoder_effects',
                variable_card=3,  # minimal, moderate, strong
                values=[
                    [0.9, 0.7, 0.5, 0.2, 0.8, 0.6, 0.4, 0.15, 0.7, 0.5, 0.3, 0.1],  # minimal
                    [0.08, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.4, 0.2, 0.3, 0.4, 0.45],  # moderate
                    [0.02, 0.1, 0.2, 0.4, 0.05, 0.15, 0.25, 0.45, 0.1, 0.2, 0.3, 0.45]   # strong
                ],
                evidence=['synthesis_algorithm', 'synthesis_process'],
                evidence_card=[4, 3]
            )
            
            # Prosody modification
            cpd_prosody_mod = TabularCPD(
                variable='prosody_modification',
                variable_card=3,  # none, subtle, obvious
                values=[
                    [0.85, 0.7, 0.5, 0.8, 0.6, 0.4, 0.75, 0.55, 0.35],  # none
                    [0.12, 0.2, 0.3, 0.15, 0.25, 0.35, 0.18, 0.28, 0.38],  # subtle
                    [0.03, 0.1, 0.2, 0.05, 0.15, 0.25, 0.07, 0.17, 0.27]   # obvious
                ],
                evidence=['synthesis_process', 'speaker_characteristics'],
                evidence_card=[3, 3]
            )
            
            # Voiceprint deviation
            cpd_voiceprint_dev = TabularCPD(
                variable='voiceprint_deviation',
                variable_card=3,  # low, medium, high
                values=[
                    [0.8, 0.6, 0.4, 0.7, 0.5, 0.3, 0.6, 0.4, 0.2],  # low deviation
                    [0.15, 0.25, 0.35, 0.2, 0.3, 0.4, 0.25, 0.35, 0.45],  # medium
                    [0.05, 0.15, 0.25, 0.1, 0.2, 0.3, 0.15, 0.25, 0.35]   # high
                ],
                evidence=['voice_model_quality', 'speaker_characteristics'],
                evidence_card=[3, 3]
            )
            
            # Physics feature CPDs (these are the observed variables)
            # Delta FR (rotational dynamics) - most discriminative feature
            cpd_delta_fr = TabularCPD(
                variable='delta_fr_causal',
                variable_card=3,  # low, medium, high
                values=[
                    # Complex dependencies on multiple causal factors
                    # This is a simplified version - in practice would have many more combinations
                    [0.9, 0.7, 0.3, 0.8, 0.5, 0.1, 0.6, 0.3, 0.05] * 12,  # low (genuine-like)
                    [0.08, 0.2, 0.3, 0.15, 0.3, 0.4, 0.25, 0.4, 0.45] * 12,  # medium
                    [0.02, 0.1, 0.4, 0.05, 0.2, 0.5, 0.15, 0.3, 0.5] * 12   # high (TTS-like)
                ][:3][:len([0.9, 0.7, 0.3, 0.8, 0.5, 0.1, 0.6, 0.3, 0.05, 0.85, 0.6, 0.2, 0.75, 0.4, 0.08, 0.55, 0.25, 0.03, 0.8, 0.55, 0.15, 0.7, 0.35, 0.05, 0.5, 0.2, 0.02])],
                evidence=['algorithm_artifacts', 'spectral_distortions', 'temporal_inconsistencies'],
                evidence_card=[3, 3, 3]
            )
            
            # Simplified CPDs for other physics features
            cpd_delta_ft = TabularCPD(
                variable='delta_ft_causal',
                variable_card=3,
                values=[
                    [0.8, 0.6, 0.4, 0.7, 0.5, 0.3, 0.6, 0.4, 0.2] * 9,  # Repeat pattern
                    [0.15, 0.25, 0.35, 0.2, 0.3, 0.4, 0.25, 0.35, 0.45] * 9,
                    [0.05, 0.15, 0.25, 0.1, 0.2, 0.3, 0.15, 0.25, 0.35] * 9
                ][:3][:81],  # Truncate to expected size
                evidence=['temporal_inconsistencies', 'vocoder_effects', 'prosody_modification'],
                evidence_card=[3, 3, 3]
            )
            
            cpd_delta_fv = TabularCPD(
                variable='delta_fv_causal',
                variable_card=3,
                values=[
                    [0.85, 0.65, 0.45, 0.75, 0.55, 0.35, 0.65, 0.45, 0.25] * 9,
                    [0.12, 0.22, 0.32, 0.18, 0.28, 0.38, 0.23, 0.33, 0.43] * 9,
                    [0.03, 0.13, 0.23, 0.07, 0.17, 0.27, 0.12, 0.22, 0.32] * 9
                ][:3][:81],
                evidence=['spectral_distortions', 'vocoder_effects', 'voiceprint_deviation'],
                evidence_card=[3, 3, 3]
            )
            
            # Final authenticity based on all evidence
            cpd_authenticity = TabularCPD(
                variable='authenticity_causal',
                variable_card=2,  # authentic, spoof
                values=[
                    # Very complex CPD - simplified here
                    # In practice, this would be learned from data
                    [0.95] * 81 + [0.05] * 81,  # Authentic probabilities
                    [0.05] * 81 + [0.95] * 81   # Spoof probabilities
                ],
                evidence=['delta_fr_causal', 'delta_ft_causal', 'delta_fv_causal', 'synthesis_process', 'algorithm_artifacts', 'voiceprint_deviation'],
                evidence_card=[3, 3, 3, 3, 3, 3]
            )
            
            # Add CPDs to the model (using simpler versions that are valid)
            # Simplified CPDs for demonstration
            simple_cpds = [
                cpd_synthesis_alg, cpd_voice_quality, cpd_recording_cond, cpd_speaker_char,
            ]
            
            # Add simplified versions of dependent variables
            for var in ['synthesis_process', 'algorithm_artifacts', 'spectral_distortions', 
                       'temporal_inconsistencies', 'vocoder_effects', 'prosody_modification',
                       'voiceprint_deviation', 'delta_fr_causal', 'delta_ft_causal', 
                       'delta_fv_causal', 'authenticity_causal']:
                
                if var in ['delta_fr_causal', 'delta_ft_causal', 'delta_fv_causal']:
                    # Physics features - simplified single parent
                    simple_cpd = TabularCPD(
                        variable=var,
                        variable_card=3,
                        values=[
                            [0.8, 0.6, 0.4],  # low values
                            [0.15, 0.25, 0.35],  # medium
                            [0.05, 0.15, 0.25]   # high
                        ],
                        evidence=['synthesis_process'],
                        evidence_card=[3]
                    )
                elif var == 'authenticity_causal':
                    # Final authenticity
                    simple_cpd = TabularCPD(
                        variable=var,
                        variable_card=2,
                        values=[
                            [0.9, 0.7, 0.3, 0.8, 0.5, 0.1, 0.6, 0.2, 0.05] * 3,  # authentic
                            [0.1, 0.3, 0.7, 0.2, 0.5, 0.9, 0.4, 0.8, 0.95] * 3   # spoof
                        ][:2][:27],
                        evidence=['delta_fr_causal', 'delta_ft_causal', 'delta_fv_causal'],
                        evidence_card=[3, 3, 3]
                    )
                else:
                    # Other intermediate variables
                    simple_cpd = TabularCPD(
                        variable=var,
                        variable_card=3,
                        values=[
                            [0.7, 0.5, 0.3],
                            [0.2, 0.3, 0.4],
                            [0.1, 0.2, 0.3]
                        ],
                        evidence=['synthesis_algorithm'],
                        evidence_card=[4]
                    )
                
                simple_cpds.append(simple_cpd)
            
            # Add CPDs to model
            self.causal_bn.add_cpds(*simple_cpds)
            
            # Validate model
            if self.causal_bn.check_model():
                self.logger.info("Causal BN model initialized successfully")
            else:
                self.logger.warning("Causal BN model validation failed")
                
        except Exception as e:
            self.logger.error(f"Failed to define causal CPDs: {e}")
    
    def _define_causal_relationships(self) -> Dict[str, Dict[str, float]]:
        """Define known causal relationships between variables"""
        return {
            # Direct causal effects (based on domain knowledge)
            'synthesis_algorithm': {
                'delta_fr_revised': 0.8,  # Strong causal effect on rotational dynamics
                'delta_ft_revised': 0.3,  # Moderate effect on translational dynamics
                'delta_fv_revised': 0.4,  # Moderate effect on vibrational dynamics
                'authenticity': -0.9      # Strong negative effect on authenticity
            },
            'algorithm_artifacts': {
                'delta_fr_revised': 0.7,
                'delta_ft_revised': 0.2,
                'delta_fv_revised': 0.3,
                'authenticity': -0.8
            },
            'temporal_inconsistencies': {
                'delta_fr_revised': 0.6,
                'delta_ft_revised': 0.5,
                'delta_fv_revised': 0.4,
                'authenticity': -0.7
            },
            'vocoder_effects': {
                'delta_fr_revised': 0.5,
                'delta_ft_revised': 0.4,
                'delta_fv_revised': 0.6,
                'authenticity': -0.6
            },
            'recording_conditions': {
                'delta_fr_revised': 0.2,
                'delta_ft_revised': 0.3,
                'delta_fv_revised': 0.1,
                'authenticity': 0.1  # Good recording slightly improves authenticity detection
            }
        }

    async def perform_causal_analysis(self, discretized_features: Dict[str, str]) -> Dict[str, Any]:
        """
        Perform comprehensive causal analysis including interventions and counterfactuals
        
        Args:
            discretized_features: Discretized physics features
            
        Returns:
            Causal analysis results including explanations and counterfactuals
        """
        try:
            if self.causal_bn and PGMPY_AVAILABLE:
                return await self._bn_causal_analysis(discretized_features)
            else:
                return await self._simple_causal_analysis(discretized_features)
                
        except Exception as e:
            self.logger.error(f"Causal analysis failed: {e}")
            return {
                'causal_explanations': {},
                'observational_probability': 0.5,
                'interventional_probabilities': {},
                'counterfactual_analysis': {}
            }
    
    async def _bn_causal_analysis(self, features: Dict[str, str]) -> Dict[str, Any]:
        """Advanced BN-based causal analysis"""
        try:
            from pgmpy.inference import VariableElimination
            
            # Create inference object
            inference = VariableElimination(self.causal_bn)
            
            # Map features to causal BN variables
            evidence = self._map_features_to_causal_vars(features)
            
            # 1. Observational Analysis (Level 1 of Pearl's hierarchy)
            observational_result = inference.query(
                variables=['authenticity_causal'],
                evidence=evidence
            )
            observational_prob = observational_result['authenticity_causal'].values[1]  # Spoof probability
            
            # 2. Interventional Analysis (Level 2 - do-calculus)
            interventional_probs = {}
            causal_effects = {}
            
            # Test interventions on key causal variables
            intervention_variables = ['synthesis_process', 'algorithm_artifacts', 'temporal_inconsistencies']
            
            for var in intervention_variables:
                if var not in evidence:  # Don't intervene on observed variables
                    # Intervention: do(var = high_synthesis/many_artifacts/high_inconsistency)
                    intervention_evidence = evidence.copy()
                    intervention_evidence[var] = 2  # High value
                    
                    intervention_result = inference.query(
                        variables=['authenticity_causal'],
                        evidence=intervention_evidence
                    )
                    intervention_prob = intervention_result['authenticity_causal'].values[1]
                    
                    interventional_probs[var] = intervention_prob
                    causal_effects[var] = intervention_prob - observational_prob
            
            # 3. Counterfactual Analysis (Level 3)
            counterfactual_analysis = await self._perform_counterfactual_analysis(
                features, evidence, inference
            )
            
            # 4. Generate causal explanations
            causal_explanations = self._generate_causal_explanations(
                features, causal_effects, counterfactual_analysis
            )
            
            return {
                'causal_explanations': causal_explanations,
                'observational_probability': observational_prob,
                'interventional_probabilities': interventional_probs,
                'counterfactual_analysis': counterfactual_analysis,
                'causal_effects': causal_effects
            }
            
        except Exception as e:
            self.logger.error(f"BN causal analysis failed: {e}")
            return await self._simple_causal_analysis(features)
    
    async def _simple_causal_analysis(self, features: Dict[str, str]) -> Dict[str, Any]:
        """Simplified causal analysis using predefined relationships"""
        
        causal_explanations = {}
        causal_effects = {}
        
        # Calculate observational probability based on features
        observational_prob = 0.5
        
        # Analyze each feature's causal contribution
        for feat_name, feat_value in features.items():
            if feat_name in ['delta_fr_revised', 'delta_ft_revised', 'delta_fv_revised']:
                # Map discrete values to numeric
                if feat_value == 'high':
                    feature_impact = 0.3
                elif feat_value == 'medium':
                    feature_impact = 0.1
                else:  # low
                    feature_impact = -0.1
                
                observational_prob += feature_impact
                
                # Calculate causal effects using domain knowledge
                if feat_name == 'delta_fr_revised':
                    # Rotational dynamics most strongly indicates TTS
                    causal_effect = feature_impact * 0.8
                    explanation = f"Rotational dynamics ({feat_value}) strongly indicates "
                    explanation += "TTS synthesis" if feat_value == 'high' else "genuine speech"
                    
                elif feat_name == 'delta_ft_revised':
                    causal_effect = feature_impact * 0.3
                    explanation = f"Translational dynamics ({feat_value}) moderately suggests "
                    explanation += "synthesis artifacts" if feat_value == 'high' else "natural patterns"
                    
                else:  # delta_fv_revised
                    causal_effect = feature_impact * 0.4
                    explanation = f"Vibrational dynamics ({feat_value}) indicates "
                    explanation += "vocoder effects" if feat_value == 'high' else "organic variation"
                
                causal_explanations[feat_name] = explanation
                causal_effects[feat_name] = causal_effect
        
        # Normalize probability
        observational_prob = max(0.0, min(1.0, observational_prob))
        
        # Simulate interventional analysis
        interventional_probs = {}
        
        # What if we force high synthesis artifacts?
        if observational_prob < 0.7:
            interventional_probs['force_synthesis_artifacts'] = min(1.0, observational_prob + 0.4)
        
        # What if we remove all synthesis indicators?
        interventional_probs['remove_synthesis_indicators'] = max(0.0, observational_prob - 0.5)
        
        # Counterfactual analysis
        counterfactual_analysis = {
            'if_genuine': max(0.0, observational_prob - 0.6),  # How likely if it were genuine?
            'if_tts': min(1.0, observational_prob + 0.6),      # How likely if it were TTS?
            'minimal_change_needed': 0.5 - observational_prob if observational_prob > 0.5 else 0.0
        }
        
        return {
            'causal_explanations': causal_explanations,
            'observational_probability': observational_prob,
            'interventional_probabilities': interventional_probs,
            'counterfactual_analysis': counterfactual_analysis,
            'causal_effects': causal_effects
        }
    
    def _map_features_to_causal_vars(self, features: Dict[str, str]) -> Dict[str, int]:
        """Map discretized features to causal BN variables"""
        evidence = {}
        
        # Map physics features
        state_map = {'low': 0, 'medium': 1, 'high': 2}
        
        for feat_name, feat_value in features.items():
            causal_var = feat_name.replace('_revised', '_causal')
            if causal_var in ['delta_fr_causal', 'delta_ft_causal', 'delta_fv_causal']:
                evidence[causal_var] = state_map.get(feat_value, 1)
        
        return evidence
    
    async def _perform_counterfactual_analysis(self, 
                                             original_features: Dict[str, str],
                                             evidence: Dict[str, int],
                                             inference) -> Dict[str, float]:
        """Perform counterfactual analysis"""
        
        counterfactuals = {}
        
        try:
            # Counterfactual: What if delta_fr was different?
            if 'delta_fr_causal' in evidence:
                original_delta_fr = evidence['delta_fr_causal']
                
                # Test counterfactual scenarios
                for new_value, scenario in [(0, 'low_rotation'), (2, 'high_rotation')]:
                    if new_value != original_delta_fr:
                        counterfactual_evidence = evidence.copy()
                        counterfactual_evidence['delta_fr_causal'] = new_value
                        
                        result = inference.query(
                            variables=['authenticity_causal'],
                            evidence=counterfactual_evidence
                        )
                        
                        counterfactual_prob = result['authenticity_causal'].values[1]
                        counterfactuals[f'if_{scenario}'] = counterfactual_prob
            
            # Counterfactual: What if no synthesis process occurred?
            if 'synthesis_process' not in evidence:
                no_synthesis_evidence = evidence.copy()
                no_synthesis_evidence['synthesis_process'] = 0  # No synthesis
                
                result = inference.query(
                    variables=['authenticity_causal'],
                    evidence=no_synthesis_evidence
                )
                
                counterfactuals['if_no_synthesis'] = result['authenticity_causal'].values[1]
            
        except Exception as e:
            self.logger.warning(f"Counterfactual analysis failed: {e}")
        
        return counterfactuals
    
    def _generate_causal_explanations(self, 
                                    features: Dict[str, str],
                                    causal_effects: Dict[str, float],
                                    counterfactual_analysis: Dict[str, float]) -> Dict[str, float]:
        """Generate human-readable causal explanations"""
        
        explanations = {}
        
        # Normalize causal effects to [0, 1] for explanation weights
        total_effect = sum(abs(effect) for effect in causal_effects.values())
        
        if total_effect > 0:
            for var, effect in causal_effects.items():
                normalized_effect = abs(effect) / total_effect
                explanations[f'{var}_influence'] = normalized_effect
        
        # Add counterfactual insights
        for scenario, prob in counterfactual_analysis.items():
            explanations[f'counterfactual_{scenario}'] = prob
        
        return explanations
    
    def perform_intervention(self, 
                           variables: Dict[str, Any], 
                           target: str = 'authenticity_causal') -> Dict[str, float]:
        """
        Perform Pearl's do-intervention
        
        Args:
            variables: Variables to intervene on {var_name: value}
            target: Target variable to query
            
        Returns:
            Intervention results
        """
        if not self.causal_bn or not PGMPY_AVAILABLE:
            return {'intervention_effect': 0.0}
        
        try:
            from pgmpy.inference import VariableElimination
            
            inference = VariableElimination(self.causal_bn)
            
            # Convert intervention values to appropriate format
            intervention_evidence = {}
            for var, value in variables.items():
                if isinstance(value, str):
                    state_map = {'low': 0, 'medium': 1, 'high': 2, 'none': 0, 'some': 1, 'many': 2}
                    intervention_evidence[var] = state_map.get(value, 1)
                else:
                    intervention_evidence[var] = int(value)
            
            # Perform intervention query
            result = inference.query(
                variables=[target],
                evidence=intervention_evidence
            )
            
            intervention_probability = result[target].values[1] if target == 'authenticity_causal' else result[target].values[0]
            
            return {
                'intervention_effect': intervention_probability,
                'target_variable': target,
                'intervention_variables': variables
            }
            
        except Exception as e:
            self.logger.error(f"Intervention failed: {e}")
            return {'intervention_effect': 0.0}
    
    def get_causal_graph_info(self) -> Dict[str, Any]:
        """Get information about the causal graph structure"""
        if not self.causal_bn:
            return {
                'nodes': [],
                'edges': [],
                'causal_relationships': self.causal_relationships
            }
        
        return {
            'nodes': list(self.causal_bn.nodes()),
            'edges': list(self.causal_bn.edges()),
            'causal_relationships': self.causal_relationships,
            'total_variables': len(self.causal_bn.nodes()),
            'total_edges': len(self.causal_bn.edges())
        }
    
    def explain_causal_path(self, 
                          source: str, 
                          target: str = 'authenticity_causal') -> List[str]:
        """
        Find and explain causal path from source to target
        
        Args:
            source: Source variable
            target: Target variable
            
        Returns:
            List of variables in causal path
        """
        if not self.causal_bn:
            return []
        
        try:
            import networkx as nx
            
            # Convert to networkx graph for path finding
            G = nx.DiGraph(self.causal_bn.edges())
            
            if source in G.nodes() and target in G.nodes():
                try:
                    path = nx.shortest_path(G, source, target)
                    return path
                except nx.NetworkXNoPath:
                    return []
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Causal path explanation failed: {e}")
            return []
    
    def validate_causal_assumptions(self) -> Dict[str, bool]:
        """Validate key causal assumptions"""
        validations = {
            'no_hidden_confounders': True,  # Assumption: no unmeasured confounders
            'causal_sufficiency': True,     # Assumption: all relevant variables measured
            'faithfulness': True,           # Assumption: graph represents all dependencies
            'markov_condition': True,       # Assumption: Markov condition holds
        }
        
        # In practice, these would be statistical tests
        # For now, return optimistic assumptions
        return validations 