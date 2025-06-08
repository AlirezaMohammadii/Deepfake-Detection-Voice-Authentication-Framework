"""
Temporal Feature Buffer for Bayesian Networks
Manages temporal sequences of features for dynamic analysis
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class TemporalSample:
    """Single temporal sample with features and metadata"""
    features: Dict[str, Any]
    timestamp: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class TemporalFeatureBuffer:
    """
    Manages temporal sequences of features for Bayesian analysis
    
    Provides functionality for:
    - Storing temporal sequences of features
    - User-specific temporal tracking
    - Sequence consistency analysis
    - Temporal pattern detection
    """
    
    def __init__(self, 
                 max_sequence_length: int = 50,
                 max_age_seconds: float = 3600.0,
                 enable_user_separation: bool = True):
        """
        Initialize temporal buffer
        
        Args:
            max_sequence_length: Maximum number of samples to keep per sequence
            max_age_seconds: Maximum age of samples before cleanup
            enable_user_separation: Whether to maintain separate buffers per user
        """
        self.max_sequence_length = max_sequence_length
        self.max_age_seconds = max_age_seconds
        self.enable_user_separation = enable_user_separation
        
        # Global buffer for all samples
        self.global_buffer = deque(maxlen=max_sequence_length)
        
        # User-specific buffers
        self.user_buffers = {}
        
        # Session tracking
        self.session_buffers = {}
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'unique_users': 0,
            'active_sessions': 0,
            'buffer_cleanups': 0
        }
        
        logger.info(f"TemporalFeatureBuffer initialized: max_length={max_sequence_length}, max_age={max_age_seconds}s")
    
    def add_sample(self, 
                   features: Dict[str, Any],
                   user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new sample to the temporal buffer
        
        Args:
            features: Feature dictionary to add
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional additional metadata
        """
        timestamp = time.time()
        
        sample = TemporalSample(
            features=features,
            timestamp=timestamp,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        # Add to global buffer
        self.global_buffer.append(sample)
        
        # Add to user-specific buffer if enabled
        if self.enable_user_separation and user_id:
            if user_id not in self.user_buffers:
                self.user_buffers[user_id] = deque(maxlen=self.max_sequence_length)
                self.stats['unique_users'] += 1
            
            self.user_buffers[user_id].append(sample)
        
        # Add to session buffer if session_id provided
        if session_id:
            if session_id not in self.session_buffers:
                self.session_buffers[session_id] = deque(maxlen=self.max_sequence_length)
                self.stats['active_sessions'] += 1
            
            self.session_buffers[session_id].append(sample)
        
        self.stats['total_samples'] += 1
        
        # Periodic cleanup
        if self.stats['total_samples'] % 100 == 0:
            self._cleanup_old_samples()
    
    def get_temporal_sequence(self, 
                            user_id: Optional[str] = None,
                            session_id: Optional[str] = None,
                            max_samples: Optional[int] = None) -> List[TemporalSample]:
        """
        Get temporal sequence of samples
        
        Args:
            user_id: Get samples for specific user
            session_id: Get samples for specific session
            max_samples: Maximum number of samples to return
            
        Returns:
            List of temporal samples in chronological order
        """
        if session_id and session_id in self.session_buffers:
            buffer = self.session_buffers[session_id]
        elif user_id and user_id in self.user_buffers:
            buffer = self.user_buffers[user_id]
        else:
            buffer = self.global_buffer
        
        # Convert deque to list and apply max_samples limit
        samples = list(buffer)
        
        if max_samples and len(samples) > max_samples:
            samples = samples[-max_samples:]  # Get most recent samples
        
        return samples
    
    def get_feature_sequence(self,
                           feature_name: str,
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None,
                           max_samples: Optional[int] = None) -> List[Any]:
        """
        Get sequence of specific feature values
        
        Args:
            feature_name: Name of feature to extract
            user_id: Get samples for specific user
            session_id: Get samples for specific session
            max_samples: Maximum number of samples to return
            
        Returns:
            List of feature values in chronological order
        """
        samples = self.get_temporal_sequence(user_id, session_id, max_samples)
        
        feature_values = []
        for sample in samples:
            if feature_name in sample.features:
                feature_values.append(sample.features[feature_name])
        
        return feature_values
    
    def analyze_temporal_consistency(self,
                                   user_id: Optional[str] = None,
                                   session_id: Optional[str] = None,
                                   feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze temporal consistency of features
        
        Args:
            user_id: Analyze for specific user
            session_id: Analyze for specific session
            feature_names: List of features to analyze
            
        Returns:
            Dictionary with consistency analysis results
        """
        samples = self.get_temporal_sequence(user_id, session_id)
        
        if len(samples) < 2:
            return {
                'consistency_score': 1.0,
                'num_samples': len(samples),
                'analysis': 'insufficient_data'
            }
        
        # Default feature names if not provided
        if feature_names is None:
            # Extract common feature names from samples
            all_features = set()
            for sample in samples:
                all_features.update(sample.features.keys())
            feature_names = list(all_features)
        
        consistency_scores = {}
        
        for feature_name in feature_names:
            values = self.get_feature_sequence(feature_name, user_id, session_id)
            
            if len(values) < 2:
                continue
            
            # Calculate consistency based on variance
            try:
                if isinstance(values[0], (int, float)):
                    # Numerical features
                    values_array = np.array(values)
                    mean_val = np.mean(values_array)
                    std_val = np.std(values_array)
                    
                    # Consistency score: higher is more consistent
                    if mean_val != 0:
                        consistency = 1.0 / (1.0 + (std_val / abs(mean_val)))
                    else:
                        consistency = 1.0 if std_val == 0 else 0.0
                    
                    consistency_scores[feature_name] = {
                        'consistency_score': consistency,
                        'mean': mean_val,
                        'std': std_val,
                        'num_values': len(values)
                    }
                
            except Exception as e:
                logger.warning(f"Could not analyze consistency for feature {feature_name}: {e}")
        
        # Overall consistency score
        if consistency_scores:
            overall_consistency = np.mean([
                score['consistency_score'] for score in consistency_scores.values()
            ])
        else:
            overall_consistency = 1.0
        
        return {
            'consistency_score': overall_consistency,
            'num_samples': len(samples),
            'feature_consistency': consistency_scores,
            'analysis': 'completed'
        }
    
    def detect_anomalies(self,
                        user_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect temporal anomalies in feature sequences
        
        Args:
            user_id: Analyze for specific user
            session_id: Analyze for specific session
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            Dictionary with anomaly detection results
        """
        samples = self.get_temporal_sequence(user_id, session_id)
        
        if len(samples) < 3:
            return {
                'anomalies_detected': 0,
                'anomaly_indices': [],
                'analysis': 'insufficient_data'
            }
        
        anomalies = []
        
        # Analyze each feature for anomalies
        all_features = set()
        for sample in samples:
            all_features.update(sample.features.keys())
        
        for feature_name in all_features:
            values = self.get_feature_sequence(feature_name, user_id, session_id)
            
            if len(values) < 3:
                continue
            
            try:
                if isinstance(values[0], (int, float)):
                    values_array = np.array(values)
                    mean_val = np.mean(values_array)
                    std_val = np.std(values_array)
                    
                    if std_val > 0:
                        z_scores = np.abs((values_array - mean_val) / std_val)
                        anomaly_indices = np.where(z_scores > threshold)[0]
                        
                        for idx in anomaly_indices:
                            anomalies.append({
                                'sample_index': idx,
                                'feature_name': feature_name,
                                'value': values[idx],
                                'z_score': z_scores[idx],
                                'timestamp': samples[idx].timestamp
                            })
            
            except Exception as e:
                logger.warning(f"Could not analyze anomalies for feature {feature_name}: {e}")
        
        return {
            'anomalies_detected': len(anomalies),
            'anomaly_details': anomalies,
            'analysis': 'completed'
        }
    
    def _cleanup_old_samples(self) -> None:
        """Remove samples older than max_age_seconds"""
        current_time = time.time()
        cutoff_time = current_time - self.max_age_seconds
        
        # Clean global buffer
        while self.global_buffer and self.global_buffer[0].timestamp < cutoff_time:
            self.global_buffer.popleft()
        
        # Clean user buffers
        for user_id in list(self.user_buffers.keys()):
            buffer = self.user_buffers[user_id]
            while buffer and buffer[0].timestamp < cutoff_time:
                buffer.popleft()
            
            # Remove empty buffers
            if not buffer:
                del self.user_buffers[user_id]
                self.stats['unique_users'] -= 1
        
        # Clean session buffers
        for session_id in list(self.session_buffers.keys()):
            buffer = self.session_buffers[session_id]
            while buffer and buffer[0].timestamp < cutoff_time:
                buffer.popleft()
            
            # Remove empty buffers
            if not buffer:
                del self.session_buffers[session_id]
                self.stats['active_sessions'] -= 1
        
        self.stats['buffer_cleanups'] += 1
        logger.debug(f"Cleaned old samples, cutoff: {cutoff_time}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            **self.stats,
            'global_buffer_size': len(self.global_buffer),
            'user_buffers_count': len(self.user_buffers),
            'session_buffers_count': len(self.session_buffers),
            'total_buffer_memory': sum(len(buf) for buf in self.user_buffers.values()) + 
                                 sum(len(buf) for buf in self.session_buffers.values()) + 
                                 len(self.global_buffer)
        }
    
    def clear_user_data(self, user_id: str) -> bool:
        """
        Clear all data for a specific user (GDPR compliance)
        
        Args:
            user_id: User ID to clear data for
            
        Returns:
            True if data was found and cleared
        """
        data_found = False
        
        # Remove from user buffer
        if user_id in self.user_buffers:
            del self.user_buffers[user_id]
            self.stats['unique_users'] -= 1
            data_found = True
        
        # Remove from global buffer
        original_size = len(self.global_buffer)
        self.global_buffer = deque(
            [sample for sample in self.global_buffer if sample.user_id != user_id],
            maxlen=self.max_sequence_length
        )
        if len(self.global_buffer) < original_size:
            data_found = True
        
        # Remove from session buffers
        for session_id in list(self.session_buffers.keys()):
            buffer = self.session_buffers[session_id]
            original_session_size = len(buffer)
            self.session_buffers[session_id] = deque(
                [sample for sample in buffer if sample.user_id != user_id],
                maxlen=self.max_sequence_length
            )
            if len(self.session_buffers[session_id]) < original_session_size:
                data_found = True
            
            # Remove empty session buffers
            if not self.session_buffers[session_id]:
                del self.session_buffers[session_id]
                self.stats['active_sessions'] -= 1
        
        if data_found:
            logger.info(f"Cleared temporal data for user: {user_id}")
        
        return data_found 