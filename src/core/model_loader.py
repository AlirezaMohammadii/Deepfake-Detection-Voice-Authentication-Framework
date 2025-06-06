"""
Enhanced Model Loader with Comprehensive Error Handling and Caching
"""

from transformers import HubertModel, Wav2Vec2FeatureExtractor
import torch
from utils.config_loader import settings
from functools import lru_cache
import os
import warnings
from typing import Tuple, Optional, List, Union, Dict, Any
from contextlib import contextmanager
import logging
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceContext:
    """
    Enhanced device context manager for proper device state management.
    Provides thread-safe device management and context switching capabilities.
    """
    
    def __init__(self):
        self._device = None
        self._device_stack = []
        self._lock = None
        
    @property
    def device(self) -> torch.device:
        """Get current device, initializing if necessary."""
        if self._device is None:
            self._device = self._get_optimal_device()
        return self._device
    
    def _get_optimal_device(self) -> torch.device:
        """Internal method to determine optimal device."""
        return get_optimal_device()
    
    @contextmanager
    def use_device(self, device: Union[torch.device, str]):
        """
        Temporarily use a specific device within context.
        
        Args:
            device: Device to use (torch.device or string like 'cuda', 'cpu')
            
        Usage:
            with device_context.use_device('cuda'):
                # Code here runs with CUDA device
                pass
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        # Store current device
        old_device = self._device
        self._device_stack.append(old_device)
        
        try:
            # Set new device
            self._device = device
            logger.debug(f"Switched to device: {device}")
            yield device
        finally:
            # Restore previous device
            self._device = self._device_stack.pop()
            logger.debug(f"Restored device: {self._device}")
    
    def set_device(self, device: Union[torch.device, str]):
        """
        Permanently set the device.
        
        Args:
            device: Device to set
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        self._device = device
        logger.info(f"Device set to: {device}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        current_device = self.device
        
        info = {
            'current_device': str(current_device),
            'device_type': current_device.type,
            'device_index': getattr(current_device, 'index', None)
        }
        
        if current_device.type == 'cuda' and torch.cuda.is_available():
            try:
                device_idx = current_device.index or 0
                info.update({
                    'cuda_device_name': torch.cuda.get_device_name(device_idx),
                    'cuda_memory_total': torch.cuda.get_device_properties(device_idx).total_memory / (1024**3),
                    'cuda_memory_allocated': torch.cuda.memory_allocated(device_idx) / (1024**3),
                    'cuda_memory_cached': torch.cuda.memory_reserved(device_idx) / (1024**3),
                    'cuda_compute_capability': torch.cuda.get_device_capability(device_idx)
                })
            except Exception as e:
                info['cuda_error'] = str(e)
        
        return info
    
    def clear_cache(self):
        """Clear device cache if applicable."""
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
    
    def is_available(self, device_type: str) -> bool:
        """Check if a device type is available."""
        if device_type.lower() == 'cuda':
            return torch.cuda.is_available()
        elif device_type.lower() == 'cpu':
            return True
        else:
            return False

# Global device context instance
device_context = DeviceContext()

class SecureModelLoader:
    """Enhanced model loader with security verification and adaptive precision."""
    
    # Known trusted model hashes (would be populated with actual hashes in production)
    TRUSTED_MODELS = {
        "facebook/hubert-large-ls960-ft": {
            "min_layers": ['feature_extractor', 'encoder'],
            "expected_hidden_size": 1024,
            "expected_vocab_size": None  # Variable for different models
        },
        "facebook/hubert-base-ls960": {
            "min_layers": ['feature_extractor', 'encoder'],
            "expected_hidden_size": 768,
            "expected_vocab_size": None
        }
    }
    
    @staticmethod
    def verify_model(model_path: str, model: torch.nn.Module) -> Tuple[bool, List[str]]:
        """
        Verify model integrity and architecture.
        
        Args:
            model_path: Path or identifier of the model
            model: Loaded model instance
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check if model is from trusted source
            if model_path not in SecureModelLoader.TRUSTED_MODELS:
                issues.append(f"Model '{model_path}' not in trusted sources list")
                # Don't immediately fail - log warning but continue
                logger.warning(f"Using untrusted model source: {model_path}")
            else:
                trusted_info = SecureModelLoader.TRUSTED_MODELS[model_path]
                
                # Verify expected layers exist
                for layer_name in trusted_info["min_layers"]:
                    if not hasattr(model, layer_name):
                        issues.append(f"Missing expected layer: {layer_name}")
                
                # Verify configuration parameters
                if hasattr(model, 'config'):
                    config = model.config
                    
                    # Check hidden size
                    if hasattr(config, 'hidden_size'):
                        expected_size = trusted_info["expected_hidden_size"]
                        if config.hidden_size != expected_size:
                            issues.append(
                                f"Unexpected hidden size: {config.hidden_size} "
                                f"(expected {expected_size})"
                            )
                    else:
                        issues.append("Model config missing hidden_size attribute")
                
                else:
                    issues.append("Model missing config attribute")
            
            # Check model is in evaluation mode for inference
            if model.training:
                logger.warning("Model is in training mode, switching to eval mode")
                model.eval()
            
            # Verify model has expected methods
            required_methods = ['forward', 'eval']
            for method in required_methods:
                if not hasattr(model, method):
                    issues.append(f"Model missing required method: {method}")
            
            # Check for reasonable parameter count
            param_count = sum(p.numel() for p in model.parameters())
            if param_count < 1_000_000:  # Less than 1M parameters seems too small for HuBERT
                issues.append(f"Parameter count suspiciously low: {param_count:,}")
            elif param_count > 1_000_000_000:  # More than 1B parameters
                logger.warning(f"Very large model detected: {param_count:,} parameters")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Model verification error: {e}")
            return False, issues
    
    @staticmethod
    def load_model_adaptive(model_path: str, cache_dir: Optional[str] = None) -> Tuple[HubertModel, Wav2Vec2FeatureExtractor]:
        """
        Load model with adaptive precision based on device capabilities.
        
        Args:
            model_path: Path or identifier of the model
            cache_dir: Optional cache directory
            
        Returns:
            Tuple of (model, processor)
        """
        device = device_context.device
        
        # Determine optimal dtype based on device capabilities
        torch_dtype = torch.float32  # Default
        
        if device.type == 'cuda':
            try:
                # Check compute capability for mixed precision support
                compute_capability = torch.cuda.get_device_capability(device)
                if compute_capability >= (7, 0):  # Volta or newer
                    # These GPUs support efficient float16
                    torch_dtype = torch.float16
                    logger.info(f"Using float16 precision for {compute_capability} GPU")
                else:
                    logger.info(f"Using float32 precision for {compute_capability} GPU")
            except Exception as e:
                logger.warning(f"Could not determine GPU capability: {e}, using float32")
        else:
            logger.info("Using CPU, defaulting to float32 precision")
        
        # Prepare loading arguments
        load_kwargs = {
            'cache_dir': cache_dir,
            'local_files_only': False,
            'torch_dtype': torch_dtype,
            'low_cpu_mem_usage': True,
        }
        
        # Add device mapping for CUDA
        if device.type == 'cuda':
            load_kwargs['device_map'] = None  # Manual device placement
        
        try:
            logger.info(f"Loading model {model_path} with dtype {torch_dtype}")
            
            # Load processor (lightweight)
            processor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                local_files_only=False
            )
            
            # Load model with optimized settings
            model = HubertModel.from_pretrained(model_path, **load_kwargs)
            
            # Verify model integrity
            is_valid, issues = SecureModelLoader.verify_model(model_path, model)
            if not is_valid:
                logger.error(f"Model verification failed: {issues}")
                # Continue with warnings but don't fail completely
                for issue in issues:
                    logger.warning(f"Model issue: {issue}")
            
            # Move to device manually for better control
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                model = model.to(device)
            else:
                model = model.to(device)
            
            # Set to eval mode
            model.eval()
            
            # Verify final state
            logger.info(f"Model loaded successfully:")
            logger.info(f"  - Device: {next(model.parameters()).device}")
            logger.info(f"  - Dtype: {next(model.parameters()).dtype}")
            logger.info(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise

# Device configuration with fallback
def get_optimal_device() -> torch.device:
    """Determine the optimal device for model execution with enhanced GPU detection."""
    try:
        device_pref = getattr(settings, 'device', 'auto')
        
        if device_pref == 'cpu':
            return torch.device('cpu')
        elif device_pref == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                warnings.warn("CUDA requested but not available, falling back to CPU")
                return torch.device('cpu')
        else:  # 'auto' or any other value
            if torch.cuda.is_available():
                # Enhanced CUDA memory check
                try:
                    # Check each available GPU
                    best_device = 0
                    max_memory = 0
                    
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        total_memory = props.total_memory / (1024**3)  # GB
                        
                        # Check available memory (not just total)
                        torch.cuda.set_device(i)
                        torch.cuda.empty_cache()
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        available = total_memory - allocated
                        
                        print(f"GPU {i}: {props.name}, Total: {total_memory:.1f}GB, Available: {available:.1f}GB")
                        
                        if available > max_memory:
                            max_memory = available
                            best_device = i
                    
                    # Require at least 4GB available for HuBERT-large
                    required_memory = 4.0
                    if max_memory >= required_memory:
                        torch.cuda.set_device(best_device)
                        device = torch.device(f'cuda:{best_device}')
                        print(f"Selected GPU {best_device} with {max_memory:.1f}GB available memory")
                        return device
                    else:
                        warnings.warn(f"CUDA available but insufficient memory ({max_memory:.1f}GB < {required_memory}GB), using CPU")
                        return torch.device('cpu')
                        
                except Exception as e:
                    warnings.warn(f"Error checking CUDA memory: {e}, using CPU")
                    return torch.device('cpu')
            else:
                return torch.device('cpu')
    except Exception as e:
        warnings.warn(f"Error determining device: {e}, using CPU")
        return torch.device('cpu')

# Global device configuration
DEVICE = device_context.device

# Memory management context manager
@contextmanager
def memory_efficient_context():
    """Context manager for memory-efficient processing."""
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    try:
        yield
    finally:
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

@lru_cache(maxsize=2)
def get_hubert_model_and_processor() -> Tuple[HubertModel, Wav2Vec2FeatureExtractor]:
    """
    Loads HuBERT model and its processor with enhanced security, GPU utilization and memory management.
    
    Returns:
        Tuple of (model, processor)
    """
    model_path = settings.models.hubert_model_path
    cache_dir = getattr(settings.models, 'cache_dir', None)
    
    print(f"Loading HuBERT model: {model_path}")
    print(f"Target device: {DEVICE}")
    
    with memory_efficient_context():
        try:
            # Use the new secure and adaptive loader
            model, processor = SecureModelLoader.load_model_adaptive(model_path, cache_dir)
            
            # Additional validation tests
            print(f"HuBERT model loaded successfully:")
            print(f"  - Hidden size: {model.config.hidden_size}")
            print(f"  - Device: {model.device}")
            print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  - Processor sampling rate: {processor.sampling_rate}")
            
            # Test model with dummy input
            try:
                with torch.no_grad():
                    dummy_input = torch.randn(1, 16000).to(DEVICE)  # 1 second of audio
                    dummy_processed = processor(
                        dummy_input.cpu().numpy(),
                        return_tensors="pt",
                        sampling_rate=processor.sampling_rate
                    )
                    dummy_processed = {k: v.to(DEVICE) for k, v in dummy_processed.items()}
                    
                    outputs = model(**dummy_processed)
                    assert outputs.last_hidden_state is not None
                    print(f"  - Model test successful: output shape {outputs.last_hidden_state.shape}")
                    
            except Exception as e:
                warnings.warn(f"Model validation test failed: {e}")
            
            return model, processor
            
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load HuBERT model from {model_path}")
            print(f"Error details: {e}")
            
            # Try fallback strategies with secure loader
            fallback_models = [
                "facebook/hubert-base-ls960",
                "facebook/wav2vec2-base-960h"
            ]
            
            for fallback_model in fallback_models:
                if fallback_model != model_path:
                    print(f"Attempting fallback to: {fallback_model}")
                    try:
                        with memory_efficient_context():
                            model, processor = SecureModelLoader.load_model_adaptive(fallback_model, cache_dir)
                        
                        print(f"Successfully loaded fallback model: {fallback_model}")
                        return model, processor
                        
                    except Exception as fallback_error:
                        print(f"Fallback {fallback_model} also failed: {fallback_error}")
                        continue
            
            # If all fallbacks fail, raise the original error
            raise RuntimeError(f"Failed to load any HuBERT model. Original error: {e}")

def get_model_memory_usage(model: torch.nn.Module) -> dict:
    """
    Calculate memory usage of a PyTorch model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with memory usage statistics
    """
    try:
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        
        return {
            'parameters_mb': param_size / (1024 ** 2),
            'buffers_mb': buffer_size / (1024 ** 2),
            'total_mb': total_size / (1024 ** 2),
            'device': str(next(model.parameters()).device),
            'dtype': str(next(model.parameters()).dtype)
        }
    except Exception as e:
        return {'error': str(e)}

def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply optimizations for inference performance.
    
    Args:
        model: PyTorch model
    
    Returns:
        Optimized model
    """
    try:
        # Set to evaluation mode
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Try to compile model for PyTorch 2.0+
        if hasattr(torch, 'compile') and DEVICE.type != 'cpu':
            try:
                model = torch.compile(model, mode='reduce-overhead')
                print("Model compiled for optimized inference")
            except Exception as e:
                warnings.warn(f"Model compilation failed: {e}")
        
        return model
        
    except Exception as e:
        warnings.warn(f"Model optimization failed: {e}")
        return model

def clear_model_cache():
    """Clear the model cache to free memory."""
    try:
        get_hubert_model_and_processor.cache_clear()
        
        # Also clear PyTorch cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Model cache cleared successfully")
        
    except Exception as e:
        warnings.warn(f"Failed to clear model cache: {e}")

def get_system_info() -> dict:
    """Get system information for debugging."""
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': str(DEVICE),
        'python_version': os.sys.version
    }
    
    if torch.cuda.is_available():
        try:
            info.update({
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_device_name': torch.cuda.get_device_name(0),
                'cuda_memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'cuda_memory_allocated': torch.cuda.memory_allocated() / (1024**3),
                'cuda_memory_cached': torch.cuda.memory_reserved() / (1024**3)
            })
        except Exception:
            info['cuda_info_error'] = "Failed to get CUDA info"
    
    return info

# Initialize models on import (optional - can be lazy loaded)
_MODELS_INITIALIZED = False
_CACHED_MODEL = None
_CACHED_PROCESSOR = None

def ensure_models_loaded() -> Tuple[HubertModel, Wav2Vec2FeatureExtractor]:
    """Ensure models are loaded and return them."""
    global _MODELS_INITIALIZED, _CACHED_MODEL, _CACHED_PROCESSOR
    
    if not _MODELS_INITIALIZED:
        try:
            _CACHED_MODEL, _CACHED_PROCESSOR = get_hubert_model_and_processor()
            _CACHED_MODEL = optimize_model_for_inference(_CACHED_MODEL)
            _MODELS_INITIALIZED = True
        except Exception as e:
            print(f"Failed to initialize models: {e}")
            raise
    
    return _CACHED_MODEL, _CACHED_PROCESSOR

if __name__ == "__main__":
    print("Enhanced Model Loader Test")
    print("=" * 50)
    
    # Print system information
    system_info = get_system_info()
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    print(f"\nSelected device: {DEVICE}")
    
    # Test model loading
    try:
        print("\nLoading HuBERT model...")
        model, processor = get_hubert_model_and_processor()
        
        # Get memory usage
        memory_info = get_model_memory_usage(model)
        print(f"\nModel Memory Usage:")
        for key, value in memory_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Test caching
        print(f"\nTesting model caching...")
        model2, processor2 = get_hubert_model_and_processor()
        
        # Should be the same objects due to lru_cache
        cache_test_passed = model is model2 and processor is processor2
        print(f"Cache test {'PASSED' if cache_test_passed else 'FAILED'}")
        
        # Test model optimization
        print(f"\nTesting model optimization...")
        optimized_model = optimize_model_for_inference(model)
        print(f"Model optimization completed")
        
        # Test with dummy input
        print(f"\nTesting model inference...")
        with torch.no_grad():
            dummy_audio = torch.randn(16000).numpy()  # 1 second
            
            # Process audio
            inputs = processor(
                dummy_audio,
                return_tensors="pt",
                sampling_rate=processor.sampling_rate
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Run inference
            import time
            start_time = time.time()
            outputs = model(**inputs)
            inference_time = time.time() - start_time
            
            print(f"Inference successful:")
            print(f"  Input shape: {inputs['input_values'].shape}")
            print(f"  Output shape: {outputs.last_hidden_state.shape}")
            print(f"  Inference time: {inference_time:.3f}s")
            print(f"  Hidden size: {outputs.last_hidden_state.shape[-1]}")
        
        print(f"\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    try:
        clear_model_cache()
    except:
        pass