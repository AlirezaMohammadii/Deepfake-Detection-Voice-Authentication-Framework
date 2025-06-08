#feature_extractor.py
"""
Comprehensive Feature Extractor Module
Fixed version with proper error handling and integration
Enhanced with Bayesian Networks integration
"""

import torch
import torchaudio
import librosa
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from interfaces.pipeline_components import FeatureExtractor
from core.audio_utils import normalize_waveform, segment_audio
from core.model_loader import get_hubert_model_and_processor, DEVICE
from core.physics_features import VoiceRadarInspiredDynamics
from utils.config_loader import settings
import warnings
import time
import asyncio
import hashlib
import pickle
from pathlib import Path
import logging
from typing import Callable, Union, TypeVar
import traceback

# Bayesian Networks integration - conditional import
try:
    from bayesian.core.bayesian_engine import BayesianDeepfakeEngine, BayesianConfig
    from bayesian.utils.bayesian_config_loader import load_bayesian_config
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logging.warning("Bayesian Networks functionality not available. Install required dependencies for full functionality.")

# Suppress librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Setup logging for retry mechanism
logger = logging.getLogger(__name__)

T = TypeVar('T')

class RobustProcessor:
    """Enhanced processor with retry mechanism and exponential backoff for fault tolerance."""
    
    @staticmethod
    async def process_with_retry(
        func: Callable,
        *args,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exception_types: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """
        Process function with exponential backoff retry mechanism.
        
        Args:
            func: Async function to execute
            *args: Arguments for the function
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            exception_types: Tuple of exception types to retry on
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of successful function execution
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    # Handle sync functions in thread pool for non-blocking execution
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                    
            except exception_types as e:
                last_exception = e
                
                if attempt == max_retries:
                    logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                    raise e
                
                # Calculate delay with exponential backoff
                delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                
                await asyncio.sleep(delay)
            
            except Exception as e:
                # Non-retryable exception
                logger.error(f"Non-retryable error in {func.__name__}: {e}")
                raise e
        
        # Should never reach here, but just in case
        raise last_exception
    
    @staticmethod
    async def process_with_circuit_breaker(
        func: Callable,
        *args,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        **kwargs
    ) -> Any:
        """
        Process with circuit breaker pattern to prevent cascade failures.
        
        Args:
            func: Function to execute
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
        """
        # Simple implementation - in production would use more sophisticated circuit breaker
        try:
            return await RobustProcessor.process_with_retry(func, *args, max_retries=2, **kwargs)
        except Exception as e:
            logger.error(f"Circuit breaker triggered for {func.__name__}: {e}")
            raise
    
    @staticmethod
    def create_fallback_handler(fallback_func: Callable, fallback_result: Any = None):
        """
        Create a fallback handler for graceful degradation.
        
        Args:
            fallback_func: Function to call on failure
            fallback_result: Default result if fallback also fails
        """
        async def fallback_wrapper(func: Callable, *args, **kwargs):
            try:
                return await RobustProcessor.process_with_retry(func, *args, max_retries=2, **kwargs)
            except Exception as e:
                logger.warning(f"Primary function {func.__name__} failed, using fallback: {e}")
                try:
                    if asyncio.iscoroutinefunction(fallback_func):
                        return await fallback_func(*args, **kwargs)
                    else:
                        return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    return fallback_result
        
        return fallback_wrapper

class FeatureValidator:
    """Comprehensive feature validation for ensuring data quality."""
    
    @staticmethod
    def validate_hubert_features(features: torch.Tensor) -> Tuple[bool, List[str]]:
        """Validate HuBERT features with detailed diagnostics."""
        issues = []
        
        try:
            # Check basic tensor properties
            if not torch.is_tensor(features):
                issues.append("HuBERT features must be a tensor")
                return False, issues
            
            # Check dimensions
            if features.ndim != 2:
                issues.append(f"HuBERT features must be 2D, got {features.ndim}D")
            
            if features.shape[0] < 2:
                issues.append(f"Sequence too short: {features.shape[0]} frames (minimum 2)")
            
            if features.shape[1] < 512:
                issues.append(f"Embedding dimension too small: {features.shape[1]} (expected ≥512)")
            
            # Check for NaN/Inf values
            if torch.isnan(features).any():
                nan_count = torch.isnan(features).sum().item()
                issues.append(f"Contains {nan_count} NaN values")
            
            if torch.isinf(features).any():
                inf_count = torch.isinf(features).sum().item()
                issues.append(f"Contains {inf_count} infinite values")
            
            # Check value ranges - embeddings should be reasonable
            abs_max = features.abs().max().item()
            if abs_max > 100:
                issues.append(f"Values too large (max={abs_max:.2f}, likely unnormalized)")
            elif abs_max < 1e-6:
                issues.append(f"Values too small (max={abs_max:.2e}, likely all zeros)")
            
            # Check for constant sequences (potential issues)
            std_per_dim = features.std(dim=0)
            zero_var_dims = (std_per_dim < 1e-8).sum().item()
            if zero_var_dims > features.shape[1] * 0.5:  # More than 50% dimensions have no variance
                issues.append(f"{zero_var_dims} dimensions have zero variance")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Validation error: {e}")
            return False, issues
    
    @staticmethod
    def validate_physics_features(features: Dict[str, torch.Tensor]) -> Tuple[bool, List[str]]:
        """Validate physics features with detailed diagnostics."""
        issues = []
        
        try:
            # Expected physics features from VoiceRadar
            expected_keys = [
                'delta_ft_revised', 'delta_fr_revised', 'delta_fv_revised',
                'delta_f_total_revised', 'embedding_mean_velocity_mag', 'doppler_proxy_fs'
            ]
            
            # Check for missing keys
            for key in expected_keys:
                if key not in features:
                    issues.append(f"Missing physics feature: {key}")
            
            # Validate each physics feature
            for key, value in features.items():
                if torch.is_tensor(value):
                    # Check for scalar values (most physics features should be scalars)
                    if value.numel() != 1:
                        issues.append(f"Physics feature {key} should be scalar, got shape {value.shape}")
                    
                    # Check for NaN/Inf
                    if torch.isnan(value).any():
                        issues.append(f"Physics feature {key} contains NaN")
                    
                    if torch.isinf(value).any():
                        issues.append(f"Physics feature {key} contains infinite values")
                    
                    # Check reasonable ranges for physics features
                    val = value.item() if value.numel() == 1 else value.abs().max().item()
                    
                    if key.startswith('delta_'):
                        # Frequency deltas should be in reasonable range
                        if val < 0:
                            issues.append(f"Physics feature {key} is negative: {val}")
                        elif val > 50:  # Nyquist limit consideration
                            issues.append(f"Physics feature {key} too large: {val}")
                    
                    elif 'velocity' in key:
                        # Velocity magnitudes should be reasonable
                        if val < 0:
                            issues.append(f"Velocity magnitude {key} is negative: {val}")
                        elif val > 1000:  # Very high velocity
                            issues.append(f"Velocity magnitude {key} suspiciously high: {val}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Physics validation error: {e}")
            return False, issues
    
    @staticmethod
    def validate_mel_spectrogram(features: torch.Tensor) -> Tuple[bool, List[str]]:
        """Validate Mel spectrogram features."""
        issues = []
        
        try:
            if not torch.is_tensor(features):
                issues.append("Mel spectrogram must be a tensor")
                return False, issues
            
            if features.ndim != 2:
                issues.append(f"Mel spectrogram must be 2D, got {features.ndim}D")
            
            # Check for reasonable dimensions
            n_mels, n_frames = features.shape
            if n_mels < 40 or n_mels > 128:
                issues.append(f"Unusual number of Mel bands: {n_mels}")
            
            if n_frames < 10:
                issues.append(f"Too few time frames: {n_frames}")
            
            # Check value ranges (log Mel spectrograms are typically negative)
            if features.max() > 20:
                issues.append(f"Mel spectrogram values too high (max={features.max().item():.2f})")
            
            if features.min() < -200:
                issues.append(f"Mel spectrogram values too low (min={features.min().item():.2f})")
            
            # Check for NaN/Inf
            if torch.isnan(features).any():
                issues.append("Mel spectrogram contains NaN values")
            
            if torch.isinf(features).any():
                issues.append("Mel spectrogram contains infinite values")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Mel spectrogram validation error: {e}")
            return False, issues
    
    @staticmethod
    def validate_all_features(feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all features and return validation report."""
        report = {
            'overall_valid': True,
            'feature_validations': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Validate HuBERT features
            if 'hubert_sequence' in feature_dict:
                valid, issues = FeatureValidator.validate_hubert_features(feature_dict['hubert_sequence'])
                report['feature_validations']['hubert_sequence'] = {'valid': valid, 'issues': issues}
                if not valid:
                    report['overall_valid'] = False
                    report['errors'].extend([f"HuBERT: {issue}" for issue in issues])
            
            # Validate physics features
            if 'physics' in feature_dict:
                valid, issues = FeatureValidator.validate_physics_features(feature_dict['physics'])
                report['feature_validations']['physics'] = {'valid': valid, 'issues': issues}
                if not valid:
                    report['overall_valid'] = False
                    report['errors'].extend([f"Physics: {issue}" for issue in issues])
            
            # Validate Mel spectrogram
            if 'mel_spectrogram' in feature_dict:
                valid, issues = FeatureValidator.validate_mel_spectrogram(feature_dict['mel_spectrogram'])
                report['feature_validations']['mel_spectrogram'] = {'valid': valid, 'issues': issues}
                if not valid:
                    report['warnings'].extend([f"Mel: {issue}" for issue in issues])  # Mel issues are warnings
            
            # Validate LFCC
            if 'lfcc' in feature_dict:
                valid, issues = FeatureValidator.validate_mel_spectrogram(feature_dict['lfcc'])  # Similar validation
                report['feature_validations']['lfcc'] = {'valid': valid, 'issues': issues}
                if not valid:
                    report['warnings'].extend([f"LFCC: {issue}" for issue in issues])  # LFCC issues are warnings
            
        except Exception as e:
            report['overall_valid'] = False
            report['errors'].append(f"Validation process error: {e}")
        
        return report

class FeatureCache:
    """Feature caching system for avoiding reprocessing."""
    
    def __init__(self, cache_dir: str = "cache", version: str = "1.0"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.version = version
        self.enabled = True
        
    def get_cache_key(self, waveform: torch.Tensor, sr: int, processing_mode: str = "default") -> str:
        """Generate cache key based on waveform hash, configuration, and processing mode."""
        # Create hash from waveform and configuration
        waveform_bytes = waveform.cpu().numpy().tobytes()
        
        # Include processing mode in cache key to prevent mode conflicts
        config_str = f"{sr}_{settings.models.hubert_model_path}_{self.version}_{processing_mode}"
        combined = waveform_bytes + config_str.encode()
        
        return hashlib.md5(combined).hexdigest()
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def load_features(self, waveform: torch.Tensor, sr: int, processing_mode: str = "default") -> Dict[str, Any]:
        """Load cached features if available."""
        if not self.enabled:
            return None
            
        try:
            cache_key = self.get_cache_key(waveform, sr, processing_mode)
            cache_path = self.get_cache_path(cache_key)
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cached_features = pickle.load(f)
                    
                # Validate cache integrity and compatibility
                if isinstance(cached_features, dict) and 'hubert_sequence' in cached_features:
                    # Check if cached version is compatible with current processing mode
                    cached_mode = cached_features.get('_processing_mode', 'default')
                    if cached_mode == processing_mode:
                        print(f"Cache hit ({processing_mode}): Loading features from {cache_path.name}")
                        # Move tensors to current device
                        for key, value in cached_features.items():
                            if torch.is_tensor(value):
                                cached_features[key] = value.to(DEVICE)
                            elif isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    if torch.is_tensor(sub_value):
                                        cached_features[key][sub_key] = sub_value.to(DEVICE)
                        
                        return cached_features
                    else:
                        print(f"Cache mode mismatch: cached={cached_mode}, requested={processing_mode}")
                        return None
                    
        except Exception as e:
            print(f"Cache load error: {e}")
            
        return None
    
    def save_features(self, waveform: torch.Tensor, sr: int, features: Dict[str, Any], processing_mode: str = "default"):
        """Save features to cache with processing mode information."""
        if not self.enabled:
            return
            
        try:
            cache_key = self.get_cache_key(waveform, sr, processing_mode)
            cache_path = self.get_cache_path(cache_key)
            
            # Move tensors to CPU for serialization
            features_cpu = {}
            for key, value in features.items():
                if torch.is_tensor(value):
                    features_cpu[key] = value.cpu()
                elif isinstance(value, dict):
                    features_cpu[key] = {}
                    for sub_key, sub_value in value.items():
                        if torch.is_tensor(sub_value):
                            features_cpu[key][sub_key] = sub_value.cpu()
                        else:
                            features_cpu[key][sub_key] = sub_value
                else:
                    features_cpu[key] = value
            
            # Add processing mode metadata
            features_cpu['_processing_mode'] = processing_mode
            features_cpu['_cache_timestamp'] = time.time()
            features_cpu['_cache_version'] = self.version
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(features_cpu, f)
                
            print(f"Features cached ({processing_mode}): {cache_path.name}")
            
        except Exception as e:
            print(f"Cache save error: {e}")
    
    def clear_cache(self, processing_mode: str = None):
        """Clear cached features, optionally filtered by processing mode."""
        try:
            cleared_count = 0
            if processing_mode:
                # Clear only files matching specific processing mode
                for cache_file in self.cache_dir.glob("*.pkl"):
                    try:
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                        if cached_data.get('_processing_mode') == processing_mode:
                            cache_file.unlink()
                            cleared_count += 1
                    except:
                        continue
                print(f"Cache cleared ({processing_mode}): {cleared_count} files from {self.cache_dir}")
            else:
                # Clear all cache files
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                    cleared_count += 1
                print(f"Cache cleared (all): {cleared_count} files from {self.cache_dir}")
        except Exception as e:
            print(f"Cache clear error: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                'num_cached_files': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir)
            }
        except Exception as e:
            return {'error': str(e)}

class ComprehensiveFeatureExtractor(FeatureExtractor):
    """
    Enhanced feature extractor with robust error handling, proper integration,
    and Bayesian Networks support for advanced probabilistic analysis.
    """
    
    def __init__(self, enable_cache: bool = True, cache_dir: str = "cache",
                 _model_loader: Optional[Any] = None,
                 _config: Optional[Any] = None,
                 _physics_calculator: Optional[Any] = None,
                 _device_context: Optional[Any] = None,
                 enable_bayesian: bool = True,
                 bayesian_config_name: str = "default"):
        try:
            # Use injected dependencies or defaults
            if _model_loader is not None:
                if callable(_model_loader):
                    self.hubert_model, self.hubert_processor = _model_loader()
                else:
                    self.hubert_model, self.hubert_processor = _model_loader
            else:
                # Initialize HuBERT model and processor
                self.hubert_model, self.hubert_processor = get_hubert_model_and_processor()
            
            # Use injected device context or default
            if _device_context is not None:
                self.device_context = _device_context
                target_device = self.device_context.device
            else:
                from core.model_loader import device_context
                self.device_context = device_context
                target_device = DEVICE
            
            self.hubert_model.to(target_device)
            
            # Get embedding dimension from HuBERT config
            hubert_embedding_dim = self.hubert_model.config.hidden_size
            
            # Use injected config or default
            if _config is not None:
                self.config = _config
            else:
                from utils.config_loader import settings
                self.config = settings
            
            # Update physics config if needed
            if self.config.physics.embedding_dim_for_physics is None:
                self.config.physics.embedding_dim_for_physics = hubert_embedding_dim
            elif self.config.physics.embedding_dim_for_physics != hubert_embedding_dim:
                print(
                    f"Warning: Physics feature embedding_dim_for_physics ({self.config.physics.embedding_dim_for_physics}) "
                    f"differs from HuBERT output ({hubert_embedding_dim}). Using HuBERT dimension."
                )
                self.config.physics.embedding_dim_for_physics = hubert_embedding_dim

            # Initialize physics feature calculator with correct dimensions
            if _physics_calculator is not None:
                self.physics_feature_calculator = _physics_calculator
            else:
                from core.physics_features import VoiceRadarInspiredDynamics
                self.physics_feature_calculator = VoiceRadarInspiredDynamics(
                    embedding_dim=hubert_embedding_dim,  # Use actual HuBERT dimension
                    audio_sr=self.config.audio.sample_rate
                )
            
            # Store audio config for easy access
            self.audio_cfg = self.config.audio
            
            # Initialize feature cache
            self.cache = FeatureCache(cache_dir=cache_dir, version="1.3")  # Increment for Bayesian support
            self.cache.enabled = enable_cache
            
            # Optional feature flags (for lightweight versions)
            self._enable_physics = getattr(self, '_enable_physics', True)
            self._enable_audio_features = getattr(self, '_enable_audio_features', True)
            
            # Initialize Bayesian Networks functionality
            self.enable_bayesian = enable_bayesian and BAYESIAN_AVAILABLE
            self.bayesian_engine = None
            self.temporal_cache = {}  # Cache for temporal sequences
            
            if self.enable_bayesian:
                try:
                    # Load Bayesian configuration
                    bayesian_config = load_bayesian_config(bayesian_config_name)
                    
                    # Initialize Bayesian engine
                    self.bayesian_engine = BayesianDeepfakeEngine(bayesian_config)
                    
                    print(f"Bayesian Networks enabled with config: {bayesian_config_name}")
                except Exception as e:
                    print(f"Failed to initialize Bayesian engine: {e}")
                    self.enable_bayesian = False
            else:
                if not BAYESIAN_AVAILABLE:
                    print("Bayesian Networks not available - install required dependencies for full functionality")
                else:
                    print("Bayesian Networks disabled")
            
            print(f"FeatureExtractor initialized: HuBERT dim={hubert_embedding_dim}, Device={target_device}")
            if enable_cache:
                cache_stats = self.cache.get_cache_stats()
                print(f"Feature cache: {cache_stats}")
            
        except Exception as e:
            print(f"Error initializing FeatureExtractor: {e}")
            raise

    def enable_cache(self):
        """Enable feature caching."""
        self.cache.enabled = True
        print("Feature cache enabled")
    
    def disable_cache(self):
        """Disable feature caching."""
        self.cache.enabled = False
        print("Feature cache disabled")
    
    def clear_cache(self):
        """Clear all cached features."""
        self.cache.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_cache_stats()

    async def _extract_hubert_features(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Extract HuBERT features with proper error handling.
        """
        try:
            # Resample if necessary
            if sr != self.hubert_processor.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.hubert_processor.sampling_rate
                )
                waveform = resampler(waveform)

            # Prepare inputs for HuBERT
            waveform_np = waveform.cpu().numpy()
            inputs = self.hubert_processor(
                waveform_np,
                return_tensors="pt",
                sampling_rate=self.hubert_processor.sampling_rate,
                padding=True
            )
            
            # Move inputs to correct device
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.hubert_model(**inputs)
            
            # Return the sequence of embeddings [time_steps, hidden_size]
            embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
            
            return embeddings
            
        except Exception as e:
            print(f"Error in HuBERT feature extraction: {e}")
            # Return zero embeddings as fallback
            return torch.zeros(1, self.hubert_model.config.hidden_size, device=DEVICE)

    async def _extract_mel_spectrogram(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Extract Mel spectrogram with proper error handling.
        """
        try:
            waveform_np = waveform.cpu().numpy() if waveform.is_cuda else waveform.numpy()
            
            mel_spec = librosa.feature.melspectrogram(
                y=waveform_np,
                sr=sr,
                n_fft=self.audio_cfg.n_fft,
                hop_length=self.audio_cfg.hop_length,
                win_length=self.audio_cfg.win_length,
                n_mels=self.audio_cfg.n_mels,
                fmax=sr//2  # Ensure fmax doesn't exceed Nyquist
            )
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            return torch.from_numpy(log_mel_spec).to(DEVICE)
            
        except Exception as e:
            print(f"Error in Mel spectrogram extraction: {e}")
            # Return zero spectrogram as fallback
            return torch.zeros(self.audio_cfg.n_mels, 100, device=DEVICE)

    async def _extract_lfcc(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Extract LFCC features with proper error handling.
        """
        try:
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sr,
                n_mfcc=self.audio_cfg.n_lfcc,
                melkwargs={
                    'n_fft': self.audio_cfg.n_fft,
                    'hop_length': self.audio_cfg.hop_length,
                    'n_mels': max(self.audio_cfg.n_mels, self.audio_cfg.n_lfcc),  # Ensure n_mels >= n_mfcc
                    'win_length': self.audio_cfg.win_length,
                    'f_max': sr // 2  # Ensure f_max doesn't exceed Nyquist
                }
            ).to(DEVICE)
            
            # Add batch dimension if needed
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            
            mfcc = mfcc_transform(waveform)
            
            # Remove batch dimension and transpose to [time, features]
            return mfcc.squeeze(0).transpose(0, 1)
            
        except Exception as e:
            print(f"Error in LFCC extraction: {e}")
            # Return zero LFCC as fallback
            return torch.zeros(100, self.audio_cfg.n_lfcc, device=DEVICE)

    async def _extract_physics_features(self, hubert_sequence: torch.Tensor, sr: int) -> Dict[str, torch.Tensor]:
        """Extract physics features from HuBERT embeddings."""
        try:
            # Use the new async physics calculation method
            physics_dict = await self.physics_feature_calculator.calculate_all_dynamics_async(hubert_sequence)
            return physics_dict
        except Exception as e:
            print(f"Warning: Async physics feature calculation failed: {e}, falling back to sync")
            try:
                physics_dict = await self.physics_feature_calculator.calculate_all_dynamics(hubert_sequence)
                return physics_dict
            except Exception as e2:
                print(f"Warning: Both async and sync physics calculation failed: {e2}")
                # Provide default physics features
                return {
                    "delta_ft_revised": torch.tensor(0.0, device=DEVICE),
                    "delta_fr_revised": torch.tensor(0.0, device=DEVICE),
                    "delta_fv_revised": torch.tensor(0.0, device=DEVICE),
                    "delta_f_total_revised": torch.tensor(0.0, device=DEVICE),
                    "embedding_mean_velocity_mag": torch.tensor(0.0, device=DEVICE),
                    "doppler_proxy_fs": torch.tensor(0.0, device=DEVICE)
                }

    async def extract_features(self, waveform: torch.Tensor, sr: int, processing_mode: str = "default") -> Dict[str, Any]:
        """
        Extract comprehensive features from audio waveform using parallel processing and caching.
        
        Args:
            waveform: Audio waveform tensor [samples] or [channels, samples]
            sr: Sample rate
            processing_mode: Processing mode identifier for cache coordination
            
        Returns:
            Dictionary containing all extracted features including optional Bayesian analysis
        """
        try:
            # Ensure waveform is 1D
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            if waveform.ndim == 0:
                raise ValueError("Waveform is a scalar.")
            
            # Normalize waveform
            normalized_waveform = normalize_waveform(waveform).to(DEVICE)
            
            # Check cache first with processing mode
            cached_features = self.cache.load_features(normalized_waveform, sr, processing_mode)
            if cached_features is not None:
                # Add timing info for cached features
                cached_features['_extraction_time'] = 0.0
                cached_features['_cache_hit'] = True
                cached_features['_processing_mode'] = processing_mode
                
                # Perform Bayesian analysis if enabled (not cached)
                if self.enable_bayesian and self.bayesian_engine:
                    bayesian_result = await self._perform_bayesian_analysis(
                        cached_features.get('physics', {}), 
                        user_context=None,
                        audio_metadata={'sample_rate': sr, 'duration': len(normalized_waveform) / sr}
                    )
                    cached_features['bayesian_analysis'] = bayesian_result
                
                return cached_features
            
            # Phase 1: Run independent feature extractors in parallel
            # These can run concurrently as they don't depend on each other
            print(f"Starting parallel feature extraction (mode: {processing_mode})...")
            start_time = time.time()
            
            phase1_tasks = [
                self._extract_hubert_features(normalized_waveform, sr),
                self._extract_mel_spectrogram(normalized_waveform, sr),
                self._extract_lfcc(normalized_waveform, sr)
            ]
            
            # Execute phase 1 tasks concurrently
            hubert_sequence, mel_spec, lfcc = await asyncio.gather(*phase1_tasks, return_exceptions=True)
            
            # Handle exceptions from phase 1
            if isinstance(hubert_sequence, Exception):
                print(f"HuBERT extraction failed: {hubert_sequence}")
                raise hubert_sequence
            if isinstance(mel_spec, Exception):
                print(f"Mel-spectrogram extraction failed: {mel_spec}")
                mel_spec = None  # Continue without mel-spec
            if isinstance(lfcc, Exception):
                print(f"LFCC extraction failed: {lfcc}")
                lfcc = None  # Continue without LFCC
            
            print(f"Phase 1 completed in {time.time() - start_time:.2f}s")
            
            # Phase 2: Dependent feature extraction (depends on HuBERT)
            if hubert_sequence is None:
                raise ValueError("HuBERT extraction failed - cannot continue")
            
            phase2_start = time.time()
            physics_features = await self._extract_physics_features(hubert_sequence, sr)
            print(f"Phase 2 (physics) completed in {time.time() - phase2_start:.2f}s")
            
            # Combine all features
            all_features = {
                'hubert_sequence': hubert_sequence,
                'physics': physics_features,
                '_extraction_time': time.time() - start_time,
                '_cache_hit': False,
                '_processing_mode': processing_mode
            }
            
            # Add optional features if available
            if mel_spec is not None:
                all_features['mel_spectrogram'] = mel_spec
            if lfcc is not None:
                all_features['lfcc'] = lfcc
            
            # Phase 3: Bayesian Analysis (if enabled)
            if self.enable_bayesian and self.bayesian_engine:
                try:
                    bayesian_start = time.time()
                    print("Starting Bayesian probabilistic analysis...")
                    
                    # Get temporal context for user (if available)
                    temporal_sequence = self.temporal_cache.get('default_user', [])
                    
                    # Prepare audio metadata
                    audio_metadata = {
                        'sample_rate': sr,
                        'duration': len(normalized_waveform) / sr,
                        'channels': 1,
                        'processing_mode': processing_mode
                    }
                    
                    # Perform comprehensive Bayesian analysis
                    bayesian_result = await self.bayesian_engine.analyze_audio_probabilistic(
                        physics_features=physics_features,
                        temporal_sequence=temporal_sequence,
                        user_context={'user_id': 'default_user'},
                        audio_metadata=audio_metadata
                    )
                    
                    all_features['bayesian_analysis'] = bayesian_result
                    
                    # Update temporal cache
                    discretized_features = self.bayesian_engine._discretize_physics_features(physics_features)
                    self._update_temporal_cache('default_user', discretized_features)
                    
                    print(f"Bayesian analysis completed in {time.time() - bayesian_start:.2f}s")
                    print(f"Spoof probability: {bayesian_result.spoof_probability:.3f}, "
                          f"Confidence: {bayesian_result.confidence_score:.3f}")
                
                except Exception as e:
                    print(f"Bayesian analysis failed: {e}")
                    # Continue without Bayesian analysis
            else:
                print("Bayesian analysis disabled or unavailable")
            
            print(f"Total feature extraction completed in {time.time() - start_time:.2f}s")
            
            # Cache the results with processing mode (excluding Bayesian analysis for cache efficiency)
            cache_features = all_features.copy()
            if 'bayesian_analysis' in cache_features:
                del cache_features['bayesian_analysis']  # Don't cache Bayesian results
            self.cache.save_features(normalized_waveform, sr, cache_features, processing_mode)
            
            return all_features
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            traceback.print_exc()
            raise

    async def _perform_bayesian_analysis(self, 
                                       physics_features: Dict[str, torch.Tensor],
                                       user_context: Optional[Dict] = None,
                                       audio_metadata: Optional[Dict] = None) -> Optional[Any]:
        """
        Perform Bayesian probabilistic analysis on physics features
        
        Args:
            physics_features: Extracted physics features
            user_context: User context information
            audio_metadata: Audio metadata
            
        Returns:
            Bayesian analysis result or None if failed
        """
        if not self.enable_bayesian or not self.bayesian_engine:
            return None
        
        try:
            # Get temporal sequence for user
            user_id = user_context.get('user_id', 'default_user') if user_context else 'default_user'
            temporal_sequence = self.temporal_cache.get(user_id, [])
            
            # Perform Bayesian analysis
            result = await self.bayesian_engine.analyze_audio_probabilistic(
                physics_features=physics_features,
                temporal_sequence=temporal_sequence,
                user_context=user_context,
                audio_metadata=audio_metadata
            )
            
            # Update temporal cache
            discretized_features = self.bayesian_engine._discretize_physics_features(physics_features)
            self._update_temporal_cache(user_id, discretized_features)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Bayesian analysis failed: {e}")
            return None
    
    def _update_temporal_cache(self, user_id: str, features: Dict[str, Any]):
        """Update temporal cache for user"""
        if user_id not in self.temporal_cache:
            self.temporal_cache[user_id] = []
        
        # Add features with timestamp
        self.temporal_cache[user_id].append({
            'features': features,
            'timestamp': time.time()
        })
        
        # Keep only recent history (last 20 samples)
        if len(self.temporal_cache[user_id]) > 20:
            self.temporal_cache[user_id] = self.temporal_cache[user_id][-20:]
    
    def get_bayesian_insights(self) -> Dict[str, Any]:
        """Get insights from Bayesian analysis engine"""
        if not self.enable_bayesian or not self.bayesian_engine:
            return {'bayesian_available': False}
        
        insights = {
            'bayesian_available': True,
            'temporal_cache_size': {user_id: len(cache) for user_id, cache in self.temporal_cache.items()},
            'bayesian_config': {
                'enable_temporal_modeling': self.bayesian_engine.config.enable_temporal_modeling,
                'enable_hierarchical_modeling': self.bayesian_engine.config.enable_hierarchical_modeling,
                'enable_causal_analysis': self.bayesian_engine.config.enable_causal_analysis,
                'inference_method': self.bayesian_engine.config.inference_method
            }
        }
        
        return insights

class FeatureExtractorFactory:
    """
    Factory class for creating feature extractors with dependency injection.
    Prevents circular imports and provides flexible configuration.
    """
    
    @staticmethod
    def create(
        model_loader: Optional[Any] = None,
        config: Optional[Any] = None,
        physics_calculator: Optional[Any] = None,
        enable_cache: bool = True,
        cache_dir: str = "cache",
        device_context: Optional[Any] = None
    ) -> 'ComprehensiveFeatureExtractor':
        """
        Create feature extractor with injected dependencies.
        
        Args:
            model_loader: Optional model loader instance
            config: Optional configuration object
            physics_calculator: Optional physics calculator instance
            enable_cache: Whether to enable feature caching
            cache_dir: Directory for feature cache
            device_context: Optional device context manager
            
        Returns:
            Configured ComprehensiveFeatureExtractor instance
        """
        # Import dependencies lazily to avoid circular imports
        if model_loader is None:
            from core.model_loader import get_hubert_model_and_processor
            model_loader = get_hubert_model_and_processor
        
        if config is None:
            from utils.config_loader import settings
            config = settings
        
        if physics_calculator is None:
            from core.physics_features import VoiceRadarInspiredDynamics
            # Will be initialized with correct dimensions later
            physics_calculator = None
        
        if device_context is None:
            from core.model_loader import device_context as default_device_context
            device_context = default_device_context
        
        # Create extractor with dependency injection
        return ComprehensiveFeatureExtractor(
            enable_cache=enable_cache,
            cache_dir=cache_dir,
            _model_loader=model_loader,
            _config=config,
            _physics_calculator=physics_calculator,
            _device_context=device_context
        )
    
    @staticmethod
    def create_lightweight(
        enable_physics: bool = True,
        enable_audio_features: bool = True,
        device: Optional[str] = None
    ) -> 'ComprehensiveFeatureExtractor':
        """
        Create a lightweight version of the feature extractor.
        
        Args:
            enable_physics: Whether to enable physics feature calculation
            enable_audio_features: Whether to enable traditional audio features
            device: Device to use ('cuda', 'cpu', or None for auto)
            
        Returns:
            Lightweight ComprehensiveFeatureExtractor instance
        """
        from core.model_loader import device_context
        
        if device:
            with device_context.use_device(device):
                extractor = FeatureExtractorFactory.create(
                    enable_cache=False,  # Disable cache for lightweight version
                    device_context=device_context
                )
        else:
            extractor = FeatureExtractorFactory.create(
                enable_cache=False,
                device_context=device_context
            )
        
        # Configure lightweight settings
        extractor._enable_physics = enable_physics
        extractor._enable_audio_features = enable_audio_features
        
        return extractor
    
    @staticmethod
    def create_for_testing(
        mock_model: bool = False,
        mock_physics: bool = False
    ) -> 'ComprehensiveFeatureExtractor':
        """
        Create feature extractor for testing with optional mocks.
        
        Args:
            mock_model: Whether to use mock HuBERT model
            mock_physics: Whether to use mock physics calculator
            
        Returns:
            Feature extractor configured for testing
        """
        model_loader = None
        physics_calculator = None
        
        if mock_model:
            # Create mock model loader for testing
            async def mock_model_loader():
                import torch
                # Return mock model and processor
                class MockModel:
                    def __init__(self):
                        self.config = type('Config', (), {'hidden_size': 1024})()
                    def __call__(self, **kwargs):
                        return type('Output', (), {
                            'last_hidden_state': torch.randn(1, 100, 1024)
                        })()
                
                class MockProcessor:
                    def __init__(self):
                        self.sampling_rate = 16000
                    def __call__(self, *args, **kwargs):
                        return {'input_values': torch.randn(1, 16000)}
                
                return MockModel(), MockProcessor()
            
            model_loader = mock_model_loader
        
        if mock_physics:
            # Create mock physics calculator
            class MockPhysicsCalculator:
                async def calculate_all_dynamics_async(self, embeddings):
                    import torch
                    return {
                        "delta_ft_revised": torch.tensor(0.1),
                        "delta_fr_revised": torch.tensor(0.2),
                        "delta_fv_revised": torch.tensor(0.3),
                        "delta_f_total_revised": torch.tensor(0.6),
                        "embedding_mean_velocity_mag": torch.tensor(0.4),
                        "doppler_proxy_fs": torch.tensor(0.5)
                    }
                    
                async def calculate_all_dynamics(self, embeddings):
                    return await self.calculate_all_dynamics_async(embeddings)
            
            physics_calculator = MockPhysicsCalculator()
        
        return FeatureExtractorFactory.create(
            model_loader=model_loader,
            physics_calculator=physics_calculator,
            enable_cache=False
        )

if __name__ == '__main__':
    from core.audio_utils import load_audio

    async def main_extractor_test():
        """Test the feature extractor with comprehensive error handling."""
        print("Testing ComprehensiveFeatureExtractor...")
        
        try:
            # Create test waveform if no file available
            test_sr = 16000
            test_duration = 3.0  # 3 seconds
            test_waveform = torch.randn(int(test_sr * test_duration))
            
            # Initialize extractor
            extractor = ComprehensiveFeatureExtractor()
            print(f"Extractor initialized successfully")
            
            # Extract features
            print(f"Extracting features from waveform of shape: {test_waveform.shape}")
            features = await extractor.extract_features(test_waveform, test_sr)
            
            # Display results
            print("\nExtracted Features:")
            print("-" * 50)
            
            for name, feat in features.items():
                if isinstance(feat, torch.Tensor):
                    print(f"  {name:20}: shape={feat.shape}, device={feat.device}, dtype={feat.dtype}")
                elif isinstance(feat, dict):
                    print(f"  {name:20}: Dictionary with {len(feat)} items")
                    for sub_name, sub_feat in feat.items():
                        if torch.is_tensor(sub_feat):
                            value_str = f"{sub_feat.item():.6f}" if sub_feat.numel() == 1 else f"tensor({sub_feat.shape})"
                            print(f"    {sub_name:18}: {value_str}")
                        else:
                            print(f"    {sub_name:18}: {type(sub_feat)}")
                else:
                    print(f"  {name:20}: {type(feat)}")
            
            # Test with segments
            print(f"\nTesting segmented processing:")
            segments = segment_audio(test_waveform, 1.0, test_sr)  # 1-second segments
            print(f"Created {len(segments)} segments")
            
            for i, segment_wf in enumerate(segments[:2]):  # Test first 2 segments
                print(f"\nSegment {i+1} (shape: {segment_wf.shape}):")
                segment_features = await extractor.extract_features(segment_wf, test_sr)
                
                # Display key metrics
                physics = segment_features['physics']
                print(f"  HuBERT sequence length: {segment_features['hubert_sequence'].shape[0]}")
                print(f"  Physics delta_f_total: {physics['delta_f_total_revised'].item():.6f}")
                print(f"  Mel spectrogram shape: {segment_features['mel_spectrogram'].shape}")
                
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()

    # Run the test
    print("Starting feature extractor test...")
    asyncio.run(main_extractor_test())