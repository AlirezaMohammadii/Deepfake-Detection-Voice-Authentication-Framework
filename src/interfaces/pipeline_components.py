"""
Compatible Pipeline Components - Drop-in Replacement with Enhancements
Maintains original interface while adding production-ready features
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any, Union, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import logging
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# Enums for standardized categorization
class AudioType(Enum):
    """Standardized audio type classification."""
    GENUINE = "genuine"
    DEEPFAKE_TTS = "deepfake_tts"
    DEEPFAKE_VC = "deepfake_vc"
    DEEPFAKE_REPLAY = "deepfake_replay"
    DEEPFAKE_OTHER = "deepfake_other"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"

class ProcessingStatus(Enum):
    """Processing status indicators."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"

# Enhanced but compatible abstract base classes
class FeatureExtractor(ABC):
    """
    Enhanced feature extractor maintaining original interface with added capabilities.
    """
    
    def __init__(self, name: str = "feature_extractor"):
        self.name = name
        self._processing_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'avg_processing_time': 0.0
        }
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    @abstractmethod
    async def extract_features(self, waveform: torch.Tensor, sr: int) -> Dict[str, Any]:
        """
        Extracts various features from an audio waveform.
        
        Args:
            waveform: Audio waveform tensor
            sr: Sample rate in Hz
            
        Returns:
            Dictionary of features, e.g., {"hubert": tensor, "physics": dict, "mel": tensor}
        """
        pass
    
    def validate_input(self, waveform: torch.Tensor, sr: int) -> Tuple[bool, Optional[str]]:
        """
        Validate input parameters (new enhancement).
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not torch.is_tensor(waveform):
                return False, "Waveform must be a torch.Tensor"
            
            if waveform.numel() == 0:
                return False, "Waveform is empty"
            
            if waveform.ndim > 2:
                return False, "Waveform must be 1D or 2D"
            
            if sr <= 0 or sr > 192000:
                return False, f"Invalid sample rate: {sr}"
            
            if torch.isnan(waveform).any() or torch.isinf(waveform).any():
                return False, "Waveform contains NaN or Inf values"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    async def extract_features_safe(self, waveform: torch.Tensor, sr: int) -> Dict[str, Any]:
        """
        Safe wrapper for feature extraction with validation and error handling.
        """
        start_time = time.time()
        self._processing_stats['total_extractions'] += 1
        
        try:
            # Validate input
            is_valid, error_msg = self.validate_input(waveform, sr)
            if not is_valid:
                self.logger.warning(f"Input validation failed: {error_msg}")
                self._processing_stats['failed_extractions'] += 1
                return {'error': error_msg, 'status': ProcessingStatus.INVALID_INPUT.value}
            
            # Extract features
            features = await self.extract_features(waveform, sr)
            
            # Update stats
            processing_time = time.time() - start_time
            self._processing_stats['successful_extractions'] += 1
            
            # Update average processing time
            total_time = (self._processing_stats['avg_processing_time'] * 
                         (self._processing_stats['successful_extractions'] - 1) + processing_time)
            self._processing_stats['avg_processing_time'] = total_time / self._processing_stats['successful_extractions']
            
            # Add metadata to features
            if isinstance(features, dict):
                features['_metadata'] = {
                    'processing_time': processing_time,
                    'status': ProcessingStatus.SUCCESS.value,
                    'extractor_name': self.name
                }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            self._processing_stats['failed_extractions'] += 1
            return {
                'error': str(e), 
                'status': ProcessingStatus.FAILED.value,
                '_metadata': {
                    'processing_time': time.time() - start_time,
                    'extractor_name': self.name
                }
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics (new enhancement)."""
        return self._processing_stats.copy()

class VerificationHead(ABC):
    """
    Enhanced verification head maintaining original interface.
    """
    
    def __init__(self, name: str = "verifier", threshold: float = 0.5):
        self.name = name
        self.threshold = threshold
        self._verification_stats = {
            'total_verifications': 0,
            'accepts': 0,
            'rejects': 0,
            'errors': 0,
            'avg_processing_time': 0.0
        }
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    @abstractmethod
    async def verify(self, features: Dict[str, Any], enrollment_template: Any) -> Dict[str, Union[float, bool, str]]:
        """
        Performs verification based on extracted features and an enrollment template.
        
        Args:
            features: Dictionary of extracted features
            enrollment_template: Enrollment template for comparison
            
        Returns:
            Dictionary with score, decision, confidence, etc.
        """
        pass
    
    async def verify_safe(self, features: Dict[str, Any], enrollment_template: Any) -> Dict[str, Union[float, bool, str]]:
        """
        Safe wrapper for verification with error handling and statistics.
        """
        start_time = time.time()
        self._verification_stats['total_verifications'] += 1
        
        try:
            # Validate inputs
            if not isinstance(features, dict):
                raise ValueError("Features must be a dictionary")
            
            if enrollment_template is None:
                raise ValueError("Enrollment template cannot be None")
            
            # Perform verification
            result = await self.verify(features, enrollment_template)
            
            # Update statistics
            processing_time = time.time() - start_time
            decision = result.get('decision', False)
            
            if isinstance(decision, bool):
                if decision:
                    self._verification_stats['accepts'] += 1
                else:
                    self._verification_stats['rejects'] += 1
            elif isinstance(decision, str):
                if decision.lower() in ['accept', 'true', 'genuine']:
                    self._verification_stats['accepts'] += 1
                else:
                    self._verification_stats['rejects'] += 1
            
            # Add metadata
            result['_metadata'] = {
                'processing_time': processing_time,
                'threshold': self.threshold,
                'verifier_name': self.name,
                'status': ProcessingStatus.SUCCESS.value
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            self._verification_stats['errors'] += 1
            return {
                'decision': False,
                'confidence': 0.0,
                'score': 0.0,
                'error': str(e),
                '_metadata': {
                    'processing_time': time.time() - start_time,
                    'verifier_name': self.name,
                    'status': ProcessingStatus.FAILED.value
                }
            }
    
    def update_threshold(self, new_threshold: float) -> bool:
        """Update verification threshold (new enhancement)."""
        try:
            if 0.0 <= new_threshold <= 1.0:
                self.threshold = new_threshold
                self.logger.info(f"Threshold updated to {new_threshold}")
                return True
            else:
                self.logger.warning(f"Invalid threshold value: {new_threshold}")
                return False
        except Exception as e:
            self.logger.error(f"Threshold update failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics (new enhancement)."""
        stats = self._verification_stats.copy()
        if stats['total_verifications'] > 0:
            stats['accept_rate'] = stats['accepts'] / stats['total_verifications']
            stats['reject_rate'] = stats['rejects'] / stats['total_verifications']
            stats['error_rate'] = stats['errors'] / stats['total_verifications']
        return stats

class SpoofDetectionHead(ABC):
    """
    Enhanced spoof detection head with physics integration.
    """
    
    def __init__(self, name: str = "spoof_detector", threshold: float = 0.5):
        self.name = name
        self.threshold = threshold
        self._detection_stats = {
            'total_detections': 0,
            'genuine_detected': 0,
            'spoofs_detected': 0,
            'errors': 0,
            'avg_processing_time': 0.0
        }
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    @abstractmethod
    async def detect_spoof(self, features: Dict[str, Any]) -> Dict[str, Union[float, bool, str]]:
        """
        Detects spoofing attempts based on extracted features.
        
        Args:
            features: Dictionary of extracted features (including physics features)
            
        Returns:
            Dictionary with spoof probability, decision, confidence, etc.
        """
        pass
    
    async def detect_spoof_safe(self, features: Dict[str, Any]) -> Dict[str, Union[float, bool, str]]:
        """
        Safe wrapper for spoof detection with error handling.
        """
        start_time = time.time()
        self._detection_stats['total_detections'] += 1
        
        try:
            # Validate input
            if not isinstance(features, dict):
                raise ValueError("Features must be a dictionary")
            
            # Perform spoof detection
            result = await self.detect_spoof(features)
            
            # Analyze physics indicators if available
            physics_indicators = {}
            if 'physics' in features:
                physics_indicators = self.analyze_physics_indicators(features['physics'])
                result['physics_indicators'] = physics_indicators
            
            # Update statistics
            processing_time = time.time() - start_time
            decision = result.get('decision', 'unknown')
            
            if isinstance(decision, bool):
                if decision:  # True means spoofed
                    self._detection_stats['spoofs_detected'] += 1
                else:  # False means genuine
                    self._detection_stats['genuine_detected'] += 1
            elif isinstance(decision, str):
                if decision.lower() in ['spoofed', 'fake', 'deepfake']:
                    self._detection_stats['spoofs_detected'] += 1
                elif decision.lower() in ['genuine', 'real', 'authentic']:
                    self._detection_stats['genuine_detected'] += 1
            
            # Add metadata
            result['_metadata'] = {
                'processing_time': processing_time,
                'threshold': self.threshold,
                'detector_name': self.name,
                'status': ProcessingStatus.SUCCESS.value
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Spoof detection failed: {e}")
            self._detection_stats['errors'] += 1
            return {
                'decision': 'error',
                'confidence': 0.0,
                'spoof_probability': 0.5,
                'error': str(e),
                '_metadata': {
                    'processing_time': time.time() - start_time,
                    'detector_name': self.name,
                    'status': ProcessingStatus.FAILED.value
                }
            }
    
    def analyze_physics_indicators(self, physics_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze physics-based indicators for spoof detection.
        Enhanced to work with our VoiceRadar physics features.
        """
        indicators = {}
        
        try:
            # Handle tensor physics features
            for key, value in physics_features.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    val = value.item()
                    
                    # Specific analysis for VoiceRadar features
                    if 'delta_ft_revised' in key:
                        indicators['translational_anomaly'] = min(abs(val), 1.0)
                    elif 'delta_fr_revised' in key:
                        indicators['rotational_anomaly'] = min(abs(val), 1.0)
                    elif 'delta_fv_revised' in key:
                        indicators['vibrational_anomaly'] = min(abs(val), 1.0)
                    elif 'delta_f_total_revised' in key:
                        indicators['total_dynamics_anomaly'] = min(abs(val), 1.0)
                    elif 'doppler_proxy_fs' in key:
                        indicators['doppler_anomaly'] = min(abs(val), 1.0)
                    elif 'embedding_mean_velocity_mag' in key:
                        indicators['velocity_anomaly'] = min(abs(val), 1.0)
                    else:
                        # Generic physics feature
                        indicators[f'{key}_anomaly'] = min(abs(val), 1.0)
            
            # Calculate overall physics anomaly score
            if indicators:
                indicators['overall_physics_anomaly'] = np.mean(list(indicators.values()))
            
        except Exception as e:
            self.logger.warning(f"Physics indicator analysis failed: {e}")
            indicators['analysis_error'] = 1.0
        
        return indicators
    
    def classify_attack_type(self, features: Dict[str, Any]) -> AudioType:
        """
        Classify the type of spoofing attack based on features.
        """
        try:
            # Simple heuristic-based classification
            # In practice, this would use a trained classifier
            
            physics_indicators = {}
            if 'physics' in features:
                physics_indicators = self.analyze_physics_indicators(features['physics'])
            
            # Example classification logic
            if physics_indicators.get('translational_anomaly', 0) > 0.7:
                return AudioType.DEEPFAKE_TTS
            elif physics_indicators.get('rotational_anomaly', 0) > 0.7:
                return AudioType.DEEPFAKE_VC
            elif physics_indicators.get('vibrational_anomaly', 0) > 0.7:
                return AudioType.DEEPFAKE_REPLAY
            else:
                return AudioType.DEEPFAKE_OTHER
                
        except Exception as e:
            self.logger.warning(f"Attack type classification failed: {e}")
            return AudioType.UNKNOWN
    
    def get_stats(self) -> Dict[str, Any]:
        """Get spoof detection statistics (new enhancement)."""
        stats = self._detection_stats.copy()
        if stats['total_detections'] > 0:
            stats['genuine_rate'] = stats['genuine_detected'] / stats['total_detections']
            stats['spoof_rate'] = stats['spoofs_detected'] / stats['total_detections']
            stats['error_rate'] = stats['errors'] / stats['total_detections']
        return stats

# Additional utility classes for enhanced functionality
class EnrollmentTemplate:
    """Simple enrollment template for verification."""
    
    def __init__(self, user_id: str, features: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        self.user_id = user_id
        self.features = features
        self.metadata = metadata or {}
        self.creation_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'features': self.features,
            'metadata': self.metadata,
            'creation_time': self.creation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnrollmentTemplate':
        """Create from dictionary."""
        template = cls(
            user_id=data['user_id'],
            features=data['features'],
            metadata=data.get('metadata', {})
        )
        template.creation_time = data.get('creation_time', time.time())
        return template

# Utility functions
def validate_features(features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate feature dictionary structure.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        if not isinstance(features, dict):
            issues.append("Features must be a dictionary")
            return False, issues
        
        if len(features) == 0:
            issues.append("Features dictionary is empty")
        
        # Check for common required features
        expected_features = ['hubert_sequence', 'physics']
        for feat in expected_features:
            if feat not in features:
                issues.append(f"Missing expected feature: {feat}")
        
        # Validate tensor features
        for name, feature in features.items():
            if torch.is_tensor(feature):
                if feature.numel() == 0:
                    issues.append(f"Empty tensor for feature: {name}")
                if torch.isnan(feature).any():
                    issues.append(f"NaN values in feature: {name}")
                if torch.isinf(feature).any():
                    issues.append(f"Inf values in feature: {name}")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        issues.append(f"Validation error: {e}")
        return False, issues

def create_feature_summary(features: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of features for analysis."""
    summary = {}
    
    try:
        for name, feature in features.items():
            if torch.is_tensor(feature):
                summary[name] = {
                    'type': 'tensor',
                    'shape': list(feature.shape),
                    'dtype': str(feature.dtype),
                    'device': str(feature.device),
                    'mean': feature.mean().item() if feature.numel() > 0 else 0.0,
                    'std': feature.std().item() if feature.numel() > 1 else 0.0
                }
            elif isinstance(feature, dict):
                summary[name] = {
                    'type': 'dict',
                    'keys': list(feature.keys())
                }
            else:
                summary[name] = {
                    'type': type(feature).__name__,
                    'value': str(feature)[:50]  # Truncate long strings
                }
    except Exception as e:
        summary['error'] = f"Summary creation failed: {e}"
    
    return summary

if __name__ == "__main__":
    print("Compatible Pipeline Components - Testing")
    print("=" * 50)
    
    # Test feature validation
    test_features = {
        'hubert_sequence': torch.randn(100, 1024),
        'physics': {
            'delta_ft_revised': torch.tensor(0.123),
            'delta_fr_revised': torch.tensor(0.456)
        },
        'mel_spectrogram': torch.randn(80, 200)
    }
    
    is_valid, issues = validate_features(test_features)
    print(f"Features valid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
    
    # Test feature summary
    summary = create_feature_summary(test_features)
    print(f"\nFeature summary:")
    for name, info in summary.items():
        print(f"  {name}: {info}")
    
    # Test enrollment template
    template = EnrollmentTemplate(
        user_id="test_user",
        features={'embedding': torch.randn(512)},
        metadata={'quality': 'high'}
    )
    
    print(f"\nEnrollment template created for: {template.user_id}")
    print(f"Template features: {list(template.features.keys())}")
    
    print("\nAll tests completed successfully!")