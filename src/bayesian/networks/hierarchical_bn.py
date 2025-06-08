"""
Hierarchical Bayesian Network for Multi-Level Analysis
Implements hierarchical modeling with user, session, and audio-level components
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
from dataclasses import dataclass, field
import time

# Conditional imports
try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False

@dataclass
class UserProfile:
    """User profile for hierarchical modeling"""
    user_id: str
    voice_characteristics: Dict[str, float] = field(default_factory=dict)
    historical_authenticity: float = 0.5
    baseline_features: Dict[str, float] = field(default_factory=dict)
    adaptation_parameters: Dict[str, float] = field(default_factory=dict)
    update_count: int = 0
    last_updated: float = 0.0

@dataclass
class SessionContext:
    """Session context for hierarchical analysis"""
    session_id: str
    user_id: str
    recording_conditions: Dict[str, Any] = field(default_factory=dict)
    device_characteristics: Dict[str, Any] = field(default_factory=dict)
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    session_authenticity: float = 0.5
    sample_count: int = 0

class HierarchicalBayesianNetwork:
    """
    Hierarchical Bayesian Network for multi-level deepfake detection
    Models dependencies across user, session, and audio levels
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # User profiles and session contexts
        self.user_profiles: Dict[str, UserProfile] = {}
        self.session_contexts: Dict[str, SessionContext] = {}
        
        # Initialize hierarchical BN if available
        if PGMPY_AVAILABLE:
            self._initialize_hierarchical_bn()
        else:
            self.logger.warning("pgmpy not available. Using simplified hierarchical analysis.")
            self.hierarchical_bn = None
            
        # Hierarchical parameters
        self.user_weight = 0.3      # Weight for user-level evidence
        self.session_weight = 0.4   # Weight for session-level evidence
        self.audio_weight = 0.3     # Weight for audio-level evidence
        
        # Adaptation parameters
        self.adaptation_rate = 0.1
        self.min_samples_for_adaptation = 5
        
    def _initialize_hierarchical_bn(self):
        """Initialize hierarchical Bayesian network structure"""
        try:
            # Create hierarchical BN with three levels
            self.hierarchical_bn = BayesianNetwork()
            
            # Define hierarchical variables
            hierarchical_variables = [
                # User level
                'user_authenticity_prior',
                'user_voice_characteristics',
                'user_behavior_pattern',
                
                # Session level
                'session_authenticity',
                'recording_quality',
                'device_type',
                'environmental_noise',
                
                # Audio level
                'audio_authenticity',
                'delta_fr_audio',
                'delta_ft_audio',
                'delta_fv_audio',
                'spectral_features',
                'temporal_consistency'
            ]
            
            # Add nodes
            for var in hierarchical_variables:
                self.hierarchical_bn.add_node(var)
            
            # Define hierarchical edges
            hierarchical_edges = [
                # User -> Session dependencies
                ('user_authenticity_prior', 'session_authenticity'),
                ('user_voice_characteristics', 'session_authenticity'),
                ('user_behavior_pattern', 'session_authenticity'),
                
                # Session -> Audio dependencies
                ('session_authenticity', 'audio_authenticity'),
                ('recording_quality', 'audio_authenticity'),
                ('device_type', 'spectral_features'),
                ('environmental_noise', 'spectral_features'),
                
                # Audio-level internal dependencies
                ('delta_fr_audio', 'audio_authenticity'),
                ('delta_ft_audio', 'audio_authenticity'),
                ('delta_fv_audio', 'audio_authenticity'),
                ('spectral_features', 'audio_authenticity'),
                ('temporal_consistency', 'audio_authenticity'),
                
                # Cross-level influences
                ('user_voice_characteristics', 'spectral_features'),
                ('recording_quality', 'delta_fr_audio'),
                ('recording_quality', 'delta_ft_audio'),
                ('recording_quality', 'delta_fv_audio'),
            ]
            
            self.hierarchical_bn.add_edges_from(hierarchical_edges)
            
            # Define CPDs
            self._define_hierarchical_cpds()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hierarchical BN: {e}")
            self.hierarchical_bn = None
    
    def _define_hierarchical_cpds(self):
        """Define Conditional Probability Distributions for hierarchical model"""
        if not self.hierarchical_bn:
            return
            
        try:
            # User-level CPDs
            # User authenticity prior (based on historical data)
            cpd_user_auth_prior = TabularCPD(
                variable='user_authenticity_prior',
                variable_card=3,  # low_risk, medium_risk, high_risk
                values=[[0.7], [0.2], [0.1]]  # Most users are low risk
            )
            
            # User voice characteristics
            cpd_user_voice = TabularCPD(
                variable='user_voice_characteristics',
                variable_card=3,  # typical, distinctive, unique
                values=[[0.5], [0.3], [0.2]]
            )
            
            # User behavior pattern
            cpd_user_behavior = TabularCPD(
                variable='user_behavior_pattern',
                variable_card=3,  # consistent, variable, erratic
                values=[[0.6], [0.3], [0.1]]
            )
            
            # Session-level CPDs
            # Session authenticity given user factors
            cpd_session_auth = TabularCPD(
                variable='session_authenticity',
                variable_card=2,  # authentic, spoof
                values=[
                    # Authentic probabilities for different user risk/characteristic combinations
                    [0.95, 0.9, 0.85, 0.9, 0.85, 0.8, 0.85, 0.8, 0.75,  # low_risk users
                     0.8, 0.75, 0.7, 0.75, 0.7, 0.65, 0.7, 0.65, 0.6,   # medium_risk users  
                     0.3, 0.25, 0.2, 0.25, 0.2, 0.15, 0.2, 0.15, 0.1],  # high_risk users
                    # Spoof probabilities (complement)
                    [0.05, 0.1, 0.15, 0.1, 0.15, 0.2, 0.15, 0.2, 0.25,
                     0.2, 0.25, 0.3, 0.25, 0.3, 0.35, 0.3, 0.35, 0.4,
                     0.7, 0.75, 0.8, 0.75, 0.8, 0.85, 0.8, 0.85, 0.9]
                ],
                evidence=['user_authenticity_prior', 'user_voice_characteristics', 'user_behavior_pattern'],
                evidence_card=[3, 3, 3]
            )
            
            # Recording quality
            cpd_recording_quality = TabularCPD(
                variable='recording_quality',
                variable_card=3,  # poor, good, excellent
                values=[[0.2], [0.5], [0.3]]
            )
            
            # Device type
            cpd_device_type = TabularCPD(
                variable='device_type',
                variable_card=3,  # mobile, professional, unknown
                values=[[0.6], [0.3], [0.1]]
            )
            
            # Environmental noise
            cpd_env_noise = TabularCPD(
                variable='environmental_noise',
                variable_card=3,  # low, moderate, high
                values=[[0.4], [0.4], [0.2]]
            )
            
            # Audio-level CPDs
            # Spectral features given device and environment
            cpd_spectral = TabularCPD(
                variable='spectral_features',
                variable_card=3,  # clean, degraded, corrupted
                values=[
                    [0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.5, 0.4, 0.3,  # Different device/noise combinations
                     0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2,
                     0.6, 0.5, 0.4, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1],
                    [0.15, 0.2, 0.25, 0.08, 0.15, 0.2, 0.3, 0.35, 0.4,
                     0.2, 0.25, 0.3, 0.15, 0.2, 0.25, 0.35, 0.4, 0.45,
                     0.25, 0.3, 0.35, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5],
                    [0.05, 0.1, 0.15, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3,
                     0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.25, 0.3, 0.35,
                     0.15, 0.2, 0.25, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4]
                ],
                evidence=['user_voice_characteristics', 'device_type', 'environmental_noise'],
                evidence_card=[3, 3, 3]
            )
            
            # Physics features CPDs (simplified - these would be learned from data)
            cpd_delta_fr = TabularCPD(
                variable='delta_fr_audio',
                variable_card=3,  # low, medium, high
                values=[
                    [0.8, 0.6, 0.4],  # P(low|poor, good, excellent recording)
                    [0.15, 0.25, 0.35],
                    [0.05, 0.15, 0.25]
                ],
                evidence=['recording_quality'],
                evidence_card=[3]
            )
            
            cpd_delta_ft = TabularCPD(
                variable='delta_ft_audio',
                variable_card=3,
                values=[
                    [0.7, 0.8, 0.9],
                    [0.2, 0.15, 0.08],
                    [0.1, 0.05, 0.02]
                ],
                evidence=['recording_quality'],
                evidence_card=[3]
            )
            
            cpd_delta_fv = TabularCPD(
                variable='delta_fv_audio',
                variable_card=3,
                values=[
                    [0.75, 0.85, 0.9],
                    [0.2, 0.12, 0.08],
                    [0.05, 0.03, 0.02]
                ],
                evidence=['recording_quality'],
                evidence_card=[3]
            )
            
            # Temporal consistency
            cpd_temporal = TabularCPD(
                variable='temporal_consistency',
                variable_card=2,  # consistent, inconsistent
                values=[[0.9], [0.1]]  # Most genuine audio is temporally consistent
            )
            
            # Final audio authenticity given all evidence
            cpd_audio_auth = TabularCPD(
                variable='audio_authenticity',
                variable_card=2,  # authentic, spoof
                values=[
                    # This is a complex CPD - simplified version here
                    # In practice, this would be learned from training data
                    [0.95, 0.9, 0.85, 0.9, 0.85, 0.8, 0.8, 0.75, 0.7, 0.75, 0.7, 0.65,  # session_authentic + various feature combinations
                     0.1, 0.05, 0.02, 0.05, 0.02, 0.01, 0.02, 0.01, 0.005, 0.01, 0.005, 0.002] + [0.5]*36,  # session_spoof combinations
                    [0.05, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2, 0.25, 0.3, 0.25, 0.3, 0.35,
                     0.9, 0.95, 0.98, 0.95, 0.98, 0.99, 0.98, 0.99, 0.995, 0.99, 0.995, 0.998] + [0.5]*36
                ],
                evidence=['session_authenticity', 'delta_fr_audio', 'delta_ft_audio', 'delta_fv_audio', 'spectral_features', 'temporal_consistency'],
                evidence_card=[2, 3, 3, 3, 3, 2]
            )
            
            # Add all CPDs to the model
            cpds = [
                cpd_user_auth_prior, cpd_user_voice, cpd_user_behavior,
                cpd_session_auth, cpd_recording_quality, cpd_device_type, cpd_env_noise,
                cpd_spectral, cpd_delta_fr, cpd_delta_ft, cpd_delta_fv, cpd_temporal,
                cpd_audio_auth
            ]
            
            self.hierarchical_bn.add_cpds(*cpds)
            
            # Validate model
            if self.hierarchical_bn.check_model():
                self.logger.info("Hierarchical BN model initialized successfully")
            else:
                self.logger.warning("Hierarchical BN model validation failed")
                
        except Exception as e:
            self.logger.error(f"Failed to define hierarchical CPDs: {e}")

    async def analyze_hierarchical(self, 
                                 discretized_features: Dict[str, str],
                                 user_context: Optional[Dict] = None,
                                 audio_metadata: Optional[Dict] = None) -> Dict[str, float]:
        """
        Perform hierarchical Bayesian analysis
        
        Args:
            discretized_features: Discretized physics features
            user_context: User context information
            audio_metadata: Audio metadata
            
        Returns:
            Hierarchical analysis results
        """
        try:
            # Extract or create user/session information
            user_id = user_context.get('user_id', 'unknown') if user_context else 'unknown'
            session_id = user_context.get('session_id') if user_context else f"{user_id}_session_{int(time.time())}"
            
            # Get or create user profile
            user_profile = await self._get_or_create_user_profile(user_id, discretized_features)
            
            # Get or create session context
            session_context = await self._get_or_create_session_context(
                session_id, user_id, audio_metadata
            )
            
            # Perform hierarchical analysis
            if self.hierarchical_bn and PGMPY_AVAILABLE:
                return await self._bn_hierarchical_analysis(
                    discretized_features, user_profile, session_context
                )
            else:
                return await self._simple_hierarchical_analysis(
                    discretized_features, user_profile, session_context
                )
                
        except Exception as e:
            self.logger.error(f"Hierarchical analysis failed: {e}")
            return {
                'hierarchical_authenticity': 0.5,
                'user_level_confidence': 0.0,
                'session_level_confidence': 0.0,
                'audio_level_confidence': 0.0
            }
    
    async def _get_or_create_user_profile(self, user_id: str, features: Dict[str, str]) -> UserProfile:
        """Get existing user profile or create new one"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                voice_characteristics={},
                historical_authenticity=0.5,  # Neutral prior
                baseline_features={},
                adaptation_parameters={},
                update_count=0,
                last_updated=time.time()
            )
        
        profile = self.user_profiles[user_id]
        
        # Update profile with new features
        await self._update_user_profile(profile, features)
        
        return profile
    
    async def _update_user_profile(self, profile: UserProfile, features: Dict[str, str]):
        """Update user profile with new feature observations"""
        profile.update_count += 1
        profile.last_updated = time.time()
        
        # Convert discrete features to numeric for baseline calculation
        numeric_features = {}
        for feat_name, feat_value in features.items():
            if feat_value == 'low':
                numeric_features[feat_name] = 0.33
            elif feat_value == 'medium':
                numeric_features[feat_name] = 0.67
            else:  # high
                numeric_features[feat_name] = 1.0
        
        # Update baseline features with exponential moving average
        alpha = self.adaptation_rate
        for feat_name, feat_value in numeric_features.items():
            if feat_name in profile.baseline_features:
                profile.baseline_features[feat_name] = (
                    alpha * feat_value + (1 - alpha) * profile.baseline_features[feat_name]
                )
            else:
                profile.baseline_features[feat_name] = feat_value
        
        # Update voice characteristics
        if 'delta_fr_revised' in numeric_features:
            profile.voice_characteristics['rotational_dynamics'] = profile.baseline_features.get('delta_fr_revised', 0.5)
        if 'delta_ft_revised' in numeric_features:
            profile.voice_characteristics['translational_dynamics'] = profile.baseline_features.get('delta_ft_revised', 0.5)
        if 'delta_fv_revised' in numeric_features:
            profile.voice_characteristics['vibrational_dynamics'] = profile.baseline_features.get('delta_fv_revised', 0.5)
    
    async def _get_or_create_session_context(self, 
                                           session_id: str, 
                                           user_id: str,
                                           audio_metadata: Optional[Dict]) -> SessionContext:
        """Get existing session context or create new one"""
        if session_id not in self.session_contexts:
            recording_conditions = {}
            device_characteristics = {}
            environmental_factors = {}
            
            if audio_metadata:
                recording_conditions = {
                    'sample_rate': audio_metadata.get('sample_rate', 22050),
                    'duration': audio_metadata.get('duration', 0.0),
                    'channels': audio_metadata.get('channels', 1)
                }
                
                device_characteristics = {
                    'device_type': audio_metadata.get('device_type', 'unknown'),
                    'recording_app': audio_metadata.get('recording_app', 'unknown')
                }
                
                environmental_factors = {
                    'noise_level': audio_metadata.get('noise_level', 'unknown'),
                    'location': audio_metadata.get('location', 'unknown')
                }
            
            self.session_contexts[session_id] = SessionContext(
                session_id=session_id,
                user_id=user_id,
                recording_conditions=recording_conditions,
                device_characteristics=device_characteristics,
                environmental_factors=environmental_factors,
                session_authenticity=0.5,
                sample_count=0
            )
        
        session = self.session_contexts[session_id]
        session.sample_count += 1
        
        return session
    
    async def _bn_hierarchical_analysis(self, 
                                      features: Dict[str, str],
                                      user_profile: UserProfile,
                                      session_context: SessionContext) -> Dict[str, float]:
        """Advanced BN-based hierarchical analysis"""
        try:
            from pgmpy.inference import VariableElimination
            
            # Create inference object
            inference = VariableElimination(self.hierarchical_bn)
            
            # Prepare evidence from all hierarchical levels
            evidence = {}
            
            # User-level evidence
            user_risk = self._assess_user_risk(user_profile)
            voice_distinctiveness = self._assess_voice_distinctiveness(user_profile)
            behavior_consistency = self._assess_behavior_consistency(user_profile)
            
            evidence.update({
                'user_authenticity_prior': user_risk,
                'user_voice_characteristics': voice_distinctiveness,
                'user_behavior_pattern': behavior_consistency
            })
            
            # Session-level evidence
            recording_quality = self._assess_recording_quality(session_context)
            device_type = self._assess_device_type(session_context)
            env_noise = self._assess_environmental_noise(session_context)
            
            evidence.update({
                'recording_quality': recording_quality,
                'device_type': device_type,
                'environmental_noise': env_noise
            })
            
            # Audio-level evidence
            evidence.update({
                'delta_fr_audio': self._map_discrete_state(features.get('delta_fr_revised', 'medium')),
                'delta_ft_audio': self._map_discrete_state(features.get('delta_ft_revised', 'medium')),
                'delta_fv_audio': self._map_discrete_state(features.get('delta_fv_revised', 'medium')),
                'temporal_consistency': 0  # Assume consistent for now
            })
            
            # Perform inference at each level
            # User level
            user_result = inference.query(
                variables=['session_authenticity'],
                evidence={k: v for k, v in evidence.items() if k.startswith('user_')}
            )
            user_authenticity = user_result['session_authenticity'].values[0]
            
            # Session level  
            session_evidence = {k: v for k, v in evidence.items() 
                              if k.startswith('user_') or k in ['recording_quality', 'device_type', 'environmental_noise']}
            session_result = inference.query(
                variables=['session_authenticity'],
                evidence=session_evidence
            )
            session_authenticity = session_result['session_authenticity'].values[0]
            
            # Audio level
            audio_result = inference.query(
                variables=['audio_authenticity'],
                evidence=evidence
            )
            audio_authenticity = audio_result['audio_authenticity'].values[0]
            
            # Combine results hierarchically
            hierarchical_authenticity = (
                self.user_weight * user_authenticity +
                self.session_weight * session_authenticity +
                self.audio_weight * audio_authenticity
            )
            
            return {
                'hierarchical_authenticity': hierarchical_authenticity,
                'user_level_confidence': abs(user_authenticity - 0.5) * 2,
                'session_level_confidence': abs(session_authenticity - 0.5) * 2,
                'audio_level_confidence': abs(audio_authenticity - 0.5) * 2,
                'user_authenticity': user_authenticity,
                'session_authenticity': session_authenticity,
                'audio_authenticity': audio_authenticity
            }
            
        except Exception as e:
            self.logger.error(f"BN hierarchical analysis failed: {e}")
            return await self._simple_hierarchical_analysis(features, user_profile, session_context)
    
    async def _simple_hierarchical_analysis(self,
                                          features: Dict[str, str],
                                          user_profile: UserProfile,
                                          session_context: SessionContext) -> Dict[str, float]:
        """Simplified hierarchical analysis without BN"""
        
        # User-level analysis
        user_authenticity = user_profile.historical_authenticity
        user_confidence = min(0.8, user_profile.update_count * 0.1)  # Confidence grows with samples
        
        # Session-level analysis
        session_authenticity = 0.5  # Neutral for new sessions
        session_confidence = 0.3
        
        # Adjust based on session context
        if session_context.recording_conditions.get('sample_rate', 22050) < 16000:
            session_authenticity += 0.1  # Low quality might indicate spoofing
        
        # Audio-level analysis (based on physics features)
        audio_authenticity = 0.5
        audio_confidence = 0.0
        
        # Analyze physics features
        if 'delta_fr_revised' in features:
            if features['delta_fr_revised'] == 'high':
                audio_authenticity += 0.3  # High rotation indicates TTS
                audio_confidence += 0.4
            elif features['delta_fr_revised'] == 'low':
                audio_authenticity -= 0.1  # Low rotation suggests genuine
                audio_confidence += 0.3
        
        if 'delta_ft_revised' in features:
            if features['delta_ft_revised'] == 'high':
                audio_authenticity += 0.1
                audio_confidence += 0.2
        
        if 'delta_fv_revised' in features:
            if features['delta_fv_revised'] == 'high':
                audio_authenticity += 0.15
                audio_confidence += 0.25
        
        # Normalize probabilities
        user_authenticity = max(0.0, min(1.0, user_authenticity))
        session_authenticity = max(0.0, min(1.0, session_authenticity))
        audio_authenticity = max(0.0, min(1.0, audio_authenticity))
        
        # Combine hierarchically
        hierarchical_authenticity = (
            self.user_weight * user_authenticity +
            self.session_weight * session_authenticity +
            self.audio_weight * audio_authenticity
        )
        
        # Convert to spoof probability (invert authenticity)
        spoof_probability = 1.0 - hierarchical_authenticity
        
        return {
            'hierarchical_authenticity': spoof_probability,
            'user_level_confidence': user_confidence,
            'session_level_confidence': session_confidence,  
            'audio_level_confidence': audio_confidence,
            'user_authenticity': 1.0 - user_authenticity,
            'session_authenticity': 1.0 - session_authenticity,
            'audio_authenticity': 1.0 - audio_authenticity
        }
    
    def _assess_user_risk(self, profile: UserProfile) -> int:
        """Assess user risk level (0=low, 1=medium, 2=high)"""
        if profile.update_count < 3:
            return 1  # Medium risk for new users
        
        # Base risk on historical authenticity
        if profile.historical_authenticity > 0.8:
            return 0  # Low risk
        elif profile.historical_authenticity > 0.4:
            return 1  # Medium risk
        else:
            return 2  # High risk
    
    def _assess_voice_distinctiveness(self, profile: UserProfile) -> int:
        """Assess voice distinctiveness (0=typical, 1=distinctive, 2=unique)"""
        if not profile.voice_characteristics:
            return 0  # Typical by default
        
        # Calculate variation in voice characteristics
        char_values = list(profile.voice_characteristics.values())
        if not char_values:
            return 0
        
        variation = np.std(char_values)
        if variation > 0.3:
            return 2  # Unique
        elif variation > 0.15:
            return 1  # Distinctive
        else:
            return 0  # Typical
    
    def _assess_behavior_consistency(self, profile: UserProfile) -> int:
        """Assess behavior consistency (0=consistent, 1=variable, 2=erratic)"""
        if profile.update_count < 5:
            return 1  # Variable for new users
        
        # This would be based on more sophisticated analysis in practice
        # For now, base on update frequency
        time_since_last = time.time() - profile.last_updated
        if time_since_last < 3600:  # Recent activity
            return 0  # Consistent
        elif time_since_last < 86400:  # Within a day
            return 1  # Variable
        else:
            return 2  # Erratic
    
    def _assess_recording_quality(self, context: SessionContext) -> int:
        """Assess recording quality (0=poor, 1=good, 2=excellent)"""
        sample_rate = context.recording_conditions.get('sample_rate', 22050)
        duration = context.recording_conditions.get('duration', 0.0)
        
        if sample_rate >= 44100 and duration > 2.0:
            return 2  # Excellent
        elif sample_rate >= 22050 and duration > 1.0:
            return 1  # Good
        else:
            return 0  # Poor
    
    def _assess_device_type(self, context: SessionContext) -> int:
        """Assess device type (0=mobile, 1=professional, 2=unknown)"""
        device_type = context.device_characteristics.get('device_type', 'unknown')
        
        if device_type in ['professional', 'studio']:
            return 1
        elif device_type in ['mobile', 'phone', 'tablet']:
            return 0
        else:
            return 2
    
    def _assess_environmental_noise(self, context: SessionContext) -> int:
        """Assess environmental noise (0=low, 1=moderate, 2=high)"""
        noise_level = context.environmental_factors.get('noise_level', 'unknown')
        
        if noise_level in ['low', 'quiet']:
            return 0
        elif noise_level in ['moderate', 'normal']:
            return 1
        else:
            return 2
    
    def _map_discrete_state(self, state: str) -> int:
        """Map discrete state to integer"""
        state_map = {'low': 0, 'medium': 1, 'high': 2}
        return state_map.get(state, 1)
    
    def update_user_authenticity(self, user_id: str, authenticity_score: float):
        """Update user's historical authenticity"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            # Exponential moving average
            alpha = self.adaptation_rate
            profile.historical_authenticity = (
                alpha * authenticity_score + 
                (1 - alpha) * profile.historical_authenticity
            )
    
    def get_hierarchical_insights(self) -> Dict[str, Any]:
        """Get insights from hierarchical analysis"""
        return {
            'total_users': len(self.user_profiles),
            'total_sessions': len(self.session_contexts),
            'hierarchical_weights': {
                'user_weight': self.user_weight,
                'session_weight': self.session_weight,
                'audio_weight': self.audio_weight
            },
            'bn_available': self.hierarchical_bn is not None
        }
    
    def reset_user_data(self, user_id: Optional[str] = None):
        """Reset user data for privacy compliance"""
        if user_id:
            self.user_profiles.pop(user_id, None)
            # Remove user's sessions
            sessions_to_remove = [sid for sid, ctx in self.session_contexts.items() 
                                if ctx.user_id == user_id]
            for sid in sessions_to_remove:
                self.session_contexts.pop(sid, None)
        else:
            self.user_profiles.clear()
            self.session_contexts.clear() 