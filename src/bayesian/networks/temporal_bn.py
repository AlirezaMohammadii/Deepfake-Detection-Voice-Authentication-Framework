"""
Temporal Bayesian Network for Time-Series Analysis
Implements Dynamic Bayesian Networks for temporal consistency analysis
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
from collections import deque
import time

# Conditional imports for advanced BN libraries
try:
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import DBNInference
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False

class TemporalFeatureBuffer:
    """Buffer for maintaining temporal feature history"""
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.buffer = deque(maxlen=max_history)
        
    def add_features(self, features: Dict[str, Any], timestamp: Optional[float] = None):
        """Add features to temporal buffer"""
        entry = {
            'features': features.copy(),
            'timestamp': timestamp or time.time()
        }
        self.buffer.append(entry)
        
    def get_sequence(self, length: Optional[int] = None) -> List[Dict]:
        """Get temporal sequence of specified length"""
        if length is None:
            return list(self.buffer)
        else:
            return list(self.buffer)[-length:]
    
    def get_temporal_statistics(self) -> Dict[str, float]:
        """Calculate temporal statistics"""
        if len(self.buffer) < 2:
            return {'temporal_variance': 0.0, 'temporal_drift': 0.0}
        
        # Calculate variance and drift for numeric features
        feature_series = {}
        for entry in self.buffer:
            for feat_name, feat_value in entry['features'].items():
                if isinstance(feat_value, (int, float)):
                    if feat_name not in feature_series:
                        feature_series[feat_name] = []
                    feature_series[feat_name].append(feat_value)
        
        stats = {}
        for feat_name, values in feature_series.items():
            if len(values) >= 2:
                stats[f'{feat_name}_variance'] = np.var(values)
                stats[f'{feat_name}_drift'] = abs(values[-1] - values[0])
        
        # Overall statistics
        all_variances = [v for k, v in stats.items() if 'variance' in k]
        all_drifts = [v for k, v in stats.items() if 'drift' in k]
        
        return {
            'temporal_variance': np.mean(all_variances) if all_variances else 0.0,
            'temporal_drift': np.mean(all_drifts) if all_drifts else 0.0,
            'sequence_length': len(self.buffer)
        }
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()

class TemporalBayesianNetwork:
    """
    Dynamic Bayesian Network for temporal analysis
    Models temporal dependencies and consistency patterns
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        self.feature_buffer = TemporalFeatureBuffer(max_history=window_size * 2)
        
        # Initialize DBN if available
        if PGMPY_AVAILABLE:
            self._initialize_dbn()
        else:
            self.logger.warning("pgmpy not available. Using simplified temporal analysis.")
            self.dbn = None
            
        # Temporal state tracking
        self.temporal_patterns = {}
        self.consistency_thresholds = {
            'delta_fr_revised': 0.5,  # Rotational dynamics threshold
            'delta_ft_revised': 0.02,  # Translational dynamics threshold
            'delta_fv_revised': 0.3,   # Vibrational dynamics threshold
        }
        
    def _initialize_dbn(self):
        """Initialize Dynamic Bayesian Network structure"""
        try:
            # Define temporal variables for time t and t+1
            # Current time variables
            variables_t = [
                'delta_fr_t',    # Rotational dynamics at time t
                'delta_ft_t',    # Translational dynamics at time t
                'delta_fv_t',    # Vibrational dynamics at time t
                'authenticity_t' # Authenticity at time t
            ]
            
            # Next time variables
            variables_t1 = [
                'delta_fr_t1',
                'delta_ft_t1', 
                'delta_fv_t1',
                'authenticity_t1'
            ]
            
            # Create DBN structure
            self.dbn = DBN()
            
            # Add nodes for both time slices
            for var in variables_t + variables_t1:
                self.dbn.add_node(var)
            
            # Intra-slice edges (within time slice)
            temporal_edges = [
                ('delta_fr_t', 'authenticity_t'),
                ('delta_ft_t', 'authenticity_t'),
                ('delta_fv_t', 'authenticity_t'),
                ('delta_fr_t1', 'authenticity_t1'),
                ('delta_ft_t1', 'authenticity_t1'),
                ('delta_fv_t1', 'authenticity_t1'),
            ]
            
            # Inter-slice edges (temporal dependencies)
            temporal_edges.extend([
                ('delta_fr_t', 'delta_fr_t1'),
                ('delta_ft_t', 'delta_ft_t1'),
                ('delta_fv_t', 'delta_fv_t1'),
                ('authenticity_t', 'authenticity_t1'),
            ])
            
            self.dbn.add_edges_from(temporal_edges)
            
            # Define CPDs for the DBN
            self._define_temporal_cpds()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DBN: {e}")
            self.dbn = None
    
    def _define_temporal_cpds(self):
        """Define Conditional Probability Distributions for temporal model"""
        if not self.dbn:
            return
            
        try:
            # CPD for delta_fr_t (prior at first time step)
            cpd_delta_fr_t = TabularCPD(
                variable='delta_fr_t',
                variable_card=3,  # low, medium, high
                values=[[0.4], [0.3], [0.3]]  # Prior distribution
            )
            
            # CPD for delta_ft_t (prior)
            cpd_delta_ft_t = TabularCPD(
                variable='delta_ft_t',
                variable_card=3,
                values=[[0.5], [0.3], [0.2]]
            )
            
            # CPD for delta_fv_t (prior)
            cpd_delta_fv_t = TabularCPD(
                variable='delta_fv_t', 
                variable_card=3,
                values=[[0.6], [0.25], [0.15]]
            )
            
            # CPD for authenticity_t given physics features
            cpd_auth_t = TabularCPD(
                variable='authenticity_t',
                variable_card=2,  # authentic, spoof
                values=[
                    # Authentic probabilities
                    [0.9, 0.8, 0.7, 0.7, 0.6, 0.5, 0.6, 0.5, 0.4,  # delta_fr=low
                     0.8, 0.7, 0.6, 0.6, 0.5, 0.4, 0.5, 0.4, 0.3,  # delta_fr=medium
                     0.3, 0.2, 0.1, 0.2, 0.1, 0.05, 0.1, 0.05, 0.02], # delta_fr=high
                    # Spoof probabilities (complement)
                    [0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6,
                     0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.5, 0.6, 0.7,
                     0.7, 0.8, 0.9, 0.8, 0.9, 0.95, 0.9, 0.95, 0.98]
                ],
                evidence=['delta_fr_t', 'delta_ft_t', 'delta_fv_t'],
                evidence_card=[3, 3, 3]
            )
            
            # Temporal transition CPDs
            # delta_fr_t -> delta_fr_t1 (temporal consistency)
            cpd_delta_fr_transition = TabularCPD(
                variable='delta_fr_t1',
                variable_card=3,
                values=[
                    [0.7, 0.2, 0.1],  # P(low_t1 | low_t, medium_t, high_t)
                    [0.2, 0.6, 0.2],  # P(medium_t1 | ...)
                    [0.1, 0.2, 0.7]   # P(high_t1 | ...)
                ],
                evidence=['delta_fr_t'],
                evidence_card=[3]
            )
            
            # Similar for other features
            cpd_delta_ft_transition = TabularCPD(
                variable='delta_ft_t1',
                variable_card=3,
                values=[
                    [0.8, 0.15, 0.05],
                    [0.15, 0.7, 0.15],
                    [0.05, 0.15, 0.8]
                ],
                evidence=['delta_ft_t'],
                evidence_card=[3]
            )
            
            cpd_delta_fv_transition = TabularCPD(
                variable='delta_fv_t1',
                variable_card=3,
                values=[
                    [0.75, 0.2, 0.05],
                    [0.2, 0.6, 0.2],
                    [0.05, 0.2, 0.75]
                ],
                evidence=['delta_fv_t'],
                evidence_card=[3]
            )
            
            # Authenticity temporal transition
            cpd_auth_transition = TabularCPD(
                variable='authenticity_t1',
                variable_card=2,
                values=[
                    # Complex transition based on previous authenticity and current features
                    [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.7, 0.6, 0.5,  # Previous authentic
                     0.1, 0.05, 0.02, 0.05, 0.02, 0.01, 0.02, 0.01, 0.005], # Previous spoof
                    [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5,
                     0.9, 0.95, 0.98, 0.95, 0.98, 0.99, 0.98, 0.99, 0.995]
                ],
                evidence=['authenticity_t', 'delta_fr_t1', 'delta_ft_t1', 'delta_fv_t1'],
                evidence_card=[2, 3, 3, 3]
            )
            
            # Add CPDs to the model
            self.dbn.add_cpds(
                cpd_delta_fr_t, cpd_delta_ft_t, cpd_delta_fv_t, cpd_auth_t,
                cpd_delta_fr_transition, cpd_delta_ft_transition, 
                cpd_delta_fv_transition, cpd_auth_transition
            )
            
            # Validate the model
            if self.dbn.check_model():
                self.logger.info("Temporal DBN model initialized successfully")
            else:
                self.logger.warning("DBN model validation failed")
                
        except Exception as e:
            self.logger.error(f"Failed to define temporal CPDs: {e}")

    async def analyze_sequence(self, feature_sequence: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze temporal sequence for consistency and authenticity
        
        Args:
            feature_sequence: List of feature dictionaries in temporal order
            
        Returns:
            Dictionary with temporal analysis results
        """
        if not feature_sequence:
            return {'temporal_consistency': 0.5, 'sequence_probability': 0.5}
        
        # Add to buffer for tracking
        for features in feature_sequence:
            self.feature_buffer.add_features(features)
        
        try:
            # Advanced DBN analysis if available
            if self.dbn and PGMPY_AVAILABLE:
                return await self._dbn_sequence_analysis(feature_sequence)
            else:
                # Simplified temporal analysis
                return await self._simple_temporal_analysis(feature_sequence)
                
        except Exception as e:
            self.logger.error(f"Temporal sequence analysis failed: {e}")
            return await self._simple_temporal_analysis(feature_sequence)
    
    async def _dbn_sequence_analysis(self, feature_sequence: List[Dict[str, Any]]) -> Dict[str, float]:
        """Advanced DBN-based sequence analysis"""
        try:
            from pgmpy.inference import DBNInference
            
            # Create inference object
            dbn_inference = DBNInference(self.dbn)
            
            # Convert feature sequence to evidence format
            evidence_sequence = []
            for i, features in enumerate(feature_sequence):
                evidence = {}
                
                # Map features to DBN variables
                if i == 0:  # First time slice
                    evidence.update(self._map_features_to_dbn_vars(features, '_t'))
                else:  # Subsequent time slices
                    evidence.update(self._map_features_to_dbn_vars(features, '_t1'))
                
                evidence_sequence.append(evidence)
            
            # Perform temporal inference
            # Query for authenticity probability over the sequence
            if len(evidence_sequence) >= 2:
                # Forward filtering to get sequence probability
                results = dbn_inference.forward_inference(
                    variables=['authenticity_t1'],
                    evidence=evidence_sequence[-2:]  # Use last two time steps
                )
                
                # Extract authenticity probability
                auth_dist = results['authenticity_t1']
                sequence_probability = auth_dist.values[0]  # Probability of authentic
                
                # Calculate temporal consistency based on transitions
                consistency = self._calculate_transition_consistency(evidence_sequence)
                
                return {
                    'temporal_consistency': consistency,
                    'sequence_probability': 1.0 - sequence_probability,  # Convert to spoof probability
                    'sequence_length': len(feature_sequence)
                }
            else:
                # Single sample analysis
                evidence = self._map_features_to_dbn_vars(feature_sequence[0], '_t')
                results = dbn_inference.query(
                    variables=['authenticity_t'],
                    evidence=evidence
                )
                
                auth_prob = results['authenticity_t'].values[0]
                return {
                    'temporal_consistency': 1.0,  # Single sample, full consistency
                    'sequence_probability': 1.0 - auth_prob,
                    'sequence_length': 1
                }
                
        except Exception as e:
            self.logger.error(f"DBN sequence analysis failed: {e}")
            return await self._simple_temporal_analysis(feature_sequence)
    
    def _map_features_to_dbn_vars(self, features: Dict[str, Any], suffix: str) -> Dict[str, int]:
        """Map continuous features to discrete DBN variables"""
        evidence = {}
        
        # Map each feature to discrete states (0=low, 1=medium, 2=high)
        for feat_name, feat_value in features.items():
            if isinstance(feat_value, str):
                # Already discretized
                state_map = {'low': 0, 'medium': 1, 'high': 2}
                dbn_var = feat_name.replace('_revised', '') + suffix
                if dbn_var in ['delta_fr' + suffix, 'delta_ft' + suffix, 'delta_fv' + suffix]:
                    evidence[dbn_var] = state_map.get(feat_value, 1)
            elif isinstance(feat_value, (int, float)):
                # Convert numeric to discrete
                if 'delta_fr' in feat_name:
                    if feat_value < 6.5:
                        evidence['delta_fr' + suffix] = 0  # low
                    elif feat_value < 7.5:
                        evidence['delta_fr' + suffix] = 1  # medium
                    else:
                        evidence['delta_fr' + suffix] = 2  # high
                elif 'delta_ft' in feat_name:
                    if feat_value < 0.06:
                        evidence['delta_ft' + suffix] = 0
                    elif feat_value < 0.08:
                        evidence['delta_ft' + suffix] = 1
                    else:
                        evidence['delta_ft' + suffix] = 2
                elif 'delta_fv' in feat_name:
                    if feat_value < 1.0:
                        evidence['delta_fv' + suffix] = 0
                    elif feat_value < 1.5:
                        evidence['delta_fv' + suffix] = 1
                    else:
                        evidence['delta_fv' + suffix] = 2
        
        return evidence
    
    def _calculate_transition_consistency(self, evidence_sequence: List[Dict]) -> float:
        """Calculate consistency of temporal transitions"""
        if len(evidence_sequence) < 2:
            return 1.0
            
        total_consistency = 0.0
        transitions = 0
        
        for i in range(1, len(evidence_sequence)):
            prev_evidence = evidence_sequence[i-1]
            curr_evidence = evidence_sequence[i]
            
            # Check consistency for each feature transition
            for feature in ['delta_fr', 'delta_ft', 'delta_fv']:
                prev_key = feature + '_t'
                curr_key = feature + '_t1'
                
                if prev_key in prev_evidence and curr_key in curr_evidence:
                    prev_state = prev_evidence[prev_key]
                    curr_state = curr_evidence[curr_key]
                    
                    # Calculate transition probability from our model
                    # Higher consistency for smaller state changes
                    state_diff = abs(curr_state - prev_state)
                    if state_diff == 0:
                        consistency = 0.8  # Same state
                    elif state_diff == 1:
                        consistency = 0.6  # Adjacent state
                    else:
                        consistency = 0.2  # Distant state
                    
                    total_consistency += consistency
                    transitions += 1
        
        return total_consistency / transitions if transitions > 0 else 0.5
    
    async def _simple_temporal_analysis(self, feature_sequence: List[Dict[str, Any]]) -> Dict[str, float]:
        """Simplified temporal analysis without DBN"""
        
        if len(feature_sequence) == 1:
            return {
                'temporal_consistency': 1.0,
                'sequence_probability': 0.5,  # Neutral for single sample
                'sequence_length': 1
            }
        
        # Calculate temporal statistics
        temporal_stats = self.feature_buffer.get_temporal_statistics()
        
        # Analyze consistency based on variance and drift
        consistency_score = 1.0
        spoof_indicators = 0.0
        
        # Check each feature for temporal consistency
        for feat_name, threshold in self.consistency_thresholds.items():
            variance_key = f'{feat_name}_variance'
            drift_key = f'{feat_name}_drift'
            
            if variance_key in temporal_stats:
                variance = temporal_stats[variance_key]
                drift = temporal_stats.get(drift_key, 0.0)
                
                # High variance or drift indicates inconsistency (potential spoofing)
                if variance > threshold:
                    consistency_score *= 0.8
                    spoof_indicators += 0.2
                
                if drift > threshold * 2:
                    consistency_score *= 0.9
                    spoof_indicators += 0.1
        
        # Analyze feature value patterns
        numeric_sequences = {}
        for entry in feature_sequence:
            for feat_name, feat_value in entry.items():
                if isinstance(feat_value, (int, float)):
                    if feat_name not in numeric_sequences:
                        numeric_sequences[feat_name] = []
                    numeric_sequences[feat_name].append(feat_value)
        
        # Look for unnatural patterns (e.g., too consistent for genuine speech)
        for feat_name, values in numeric_sequences.items():
            if len(values) >= 3:
                # Check for unnatural consistency (TTS artifacts)
                value_range = max(values) - min(values)
                mean_value = np.mean(values)
                
                if 'delta_fr' in feat_name and mean_value > 7.2:
                    # High rotational dynamics indicate TTS
                    spoof_indicators += 0.3
                
                # Very low variance might indicate artificial generation
                if value_range < 0.01 and len(values) > 5:
                    spoof_indicators += 0.1  
                    consistency_score *= 0.95
        
        # Final calculations
        sequence_probability = min(1.0, spoof_indicators)
        temporal_consistency = max(0.0, consistency_score)
        
        return {
            'temporal_consistency': temporal_consistency,
            'sequence_probability': sequence_probability,
            'sequence_length': len(feature_sequence),
            'temporal_variance': temporal_stats.get('temporal_variance', 0.0),
            'temporal_drift': temporal_stats.get('temporal_drift', 0.0)
        }
    
    def reset_temporal_state(self):
        """Reset temporal state and clear buffers"""
        self.feature_buffer.clear()
        self.temporal_patterns.clear()
        
    def get_temporal_insights(self) -> Dict[str, Any]:
        """Get insights from temporal analysis"""
        stats = self.feature_buffer.get_temporal_statistics()
        
        insights = {
            'buffer_size': len(self.feature_buffer.buffer),
            'temporal_statistics': stats,
            'consistency_thresholds': self.consistency_thresholds,
            'dbn_available': self.dbn is not None
        }
        
        return insights 