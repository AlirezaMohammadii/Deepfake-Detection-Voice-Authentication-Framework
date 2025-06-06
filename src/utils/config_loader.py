"""
Enhanced Configuration Loader with Validation and Error Handling
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from typing import Optional
import os
import warnings

class AudioConfig(BaseSettings):
    """Audio processing configuration with validation."""
    
    sample_rate: int = Field(16000, description="Target sample rate for audio processing", ge=8000, le=48000)
    n_mels: int = Field(80, description="Number of Mel bands for Mel spectrogram", ge=40, le=128)
    n_lfcc: int = Field(20, description="Number of LFCCs", ge=12, le=40)
    hop_length: int = Field(160, description="Hop length for STFT (10ms for 16kHz)", ge=80, le=512)
    win_length: int = Field(400, description="Window length for STFT (25ms for 16kHz)", ge=200, le=1024)
    n_fft: int = Field(512, description="FFT size", ge=256, le=2048)
    default_segment_duration_s: float = Field(2.0, description="Default segment duration in seconds", ge=0.5, le=10.0)
    
    @validator('hop_length')
    def validate_hop_length(cls, v, values):
        """Ensure hop_length is reasonable relative to sample_rate."""
        if 'sample_rate' in values:
            max_hop = values['sample_rate'] // 50  # At least 50 hops per second
            if v > max_hop:
                warnings.warn(f"hop_length {v} may be too large for sample_rate {values['sample_rate']}")
        return v
    
    @validator('win_length')
    def validate_win_length(cls, v, values):
        """Ensure win_length >= hop_length."""
        if 'hop_length' in values and v < values['hop_length']:
            warnings.warn(f"win_length {v} should be >= hop_length {values['hop_length']}")
            return values['hop_length']
        return v
    
    @validator('n_fft')
    def validate_n_fft(cls, v, values):
        """Ensure n_fft >= win_length and is power of 2."""
        if 'win_length' in values and v < values['win_length']:
            # Find next power of 2 >= win_length
            next_pow2 = 1
            while next_pow2 < values['win_length']:
                next_pow2 *= 2
            warnings.warn(f"n_fft {v} adjusted to {next_pow2} to be >= win_length {values['win_length']}")
            return next_pow2
        
        # Ensure n_fft is power of 2 for FFT efficiency
        if v & (v - 1) != 0:
            next_pow2 = 1
            while next_pow2 < v:
                next_pow2 *= 2
            warnings.warn(f"n_fft {v} adjusted to {next_pow2} (next power of 2)")
            return next_pow2
        
        return v

    @validator('n_lfcc')
    def validate_n_lfcc(cls, v, values):
        """Ensure n_lfcc <= n_mels."""
        if 'n_mels' in values and v > values['n_mels']:
            warnings.warn(f"n_lfcc {v} adjusted to n_mels {values['n_mels']}")
            return values['n_mels']
        return v

class ModelPathsConfig(BaseSettings):
    """Model paths configuration with validation."""
    
    hubert_model_path: str = Field(
        "facebook/hubert-large-ls960-ft", 
        description="Path or Hugging Face ID for HuBERT model"
    )
    cache_dir: Optional[str] = Field(
        None, 
        description="Directory to cache downloaded models"
    )
    
    @validator('hubert_model_path')
    def validate_hubert_path(cls, v):
        """Validate HuBERT model path."""
        # Check if it's a local path
        if os.path.exists(v):
            return v
        # Check if it's a valid Hugging Face model ID format
        elif '/' in v and len(v.split('/')) == 2:
            return v
        else:
            warnings.warn(f"HuBERT model path '{v}' may not be valid")
            return v

class PhysicsFeatureConfig(BaseSettings):
    """Physics-based feature configuration with validation."""
    
    embedding_dim_for_physics: Optional[int] = Field(
        None, 
        description="Dimension of embedding to use for physics features"
    )
    max_rotation_angle_rad: float = Field(
        0.1, 
        description="Max rotation angle for rotational motion simulation",
        ge=0.01, le=1.0
    )
    vibration_amplitude_scale: float = Field(
        0.05, 
        description="Scaling factor for vibration amplitude",
        ge=0.001, le=0.5
    )
    time_window_for_dynamics_ms: int = Field(
        100, 
        description="Time window in ms to analyze embedding dynamics",
        ge=20, le=500
    )
    overlap_ratio: float = Field(
        0.75,
        description="Overlap ratio for windowed analysis",
        ge=0.0, le=0.9
    )
    enable_bessel_features: bool = Field(
        True,
        description="Enable Bessel function-based micro-motion features"
    )
    max_pca_components: int = Field(
        8,
        description="Maximum number of PCA components for rotational analysis",
        ge=1, le=20
    )
    
    @validator('embedding_dim_for_physics')
    def validate_embedding_dim(cls, v):
        """Validate embedding dimension."""
        if v is not None and (v < 64 or v > 2048):
            warnings.warn(f"Embedding dimension {v} is outside typical range [64, 2048]")
        return v

class SystemConfig(BaseSettings):
    """Main system configuration."""
    
    audio: AudioConfig = AudioConfig()
    models: ModelPathsConfig = ModelPathsConfig()
    physics: PhysicsFeatureConfig = PhysicsFeatureConfig()
    
    # System-level settings
    log_level: str = Field("INFO", description="Logging level")
    device: Optional[str] = Field(None, description="Device preference: 'cpu', 'cuda', or 'auto'")
    num_workers: int = Field(4, description="Number of worker processes", ge=1, le=16)
    memory_limit_gb: Optional[float] = Field(None, description="Memory limit in GB", ge=1.0)
    
    # File processing settings
    supported_audio_formats: list = Field(
        ['.wav', '.mp3', '.flac', '.m4a'],
        description="Supported audio file formats"
    )
    max_file_size_mb: float = Field(
        100.0,
        description="Maximum audio file size in MB",
        ge=1.0, le=1000.0
    )
    
    model_config = SettingsConfigDict(
        env_nested_delimiter='__', 
        extra='ignore',
        case_sensitive=False
    )
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate logging level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            warnings.warn(f"Invalid log level '{v}', using 'INFO'")
            return 'INFO'
        return v.upper()
    
    @validator('device')
    def validate_device(cls, v):
        """Validate device setting."""
        if v is not None:
            valid_devices = ['cpu', 'cuda', 'auto']
            if v.lower() not in valid_devices:
                warnings.warn(f"Invalid device '{v}', using 'auto'")
                return 'auto'
            return v.lower()
        return v

def load_config_from_env() -> SystemConfig:
    """Load configuration from environment variables."""
    try:
        # Try to load from .env file if it exists
        env_file = '.env'
        if os.path.exists(env_file):
            return SystemConfig(_env_file=env_file, _env_file_encoding='utf-8')
        else:
            return SystemConfig()
    except Exception as e:
        warnings.warn(f"Error loading config from environment: {e}")
        return SystemConfig()

def validate_config(config: SystemConfig) -> bool:
    """Validate the entire configuration."""
    try:
        # Check if audio settings are consistent
        audio_cfg = config.audio
        
        # Validate FFT settings
        if audio_cfg.n_fft < audio_cfg.win_length:
            print(f"Warning: n_fft ({audio_cfg.n_fft}) < win_length ({audio_cfg.win_length})")
        
        # Validate physics settings
        physics_cfg = config.physics
        if physics_cfg.time_window_for_dynamics_ms < 20:
            print(f"Warning: Very short time window ({physics_cfg.time_window_for_dynamics_ms}ms)")
        
        # Check model paths
        models_cfg = config.models
        if not models_cfg.hubert_model_path:
            print("Error: HuBERT model path not specified")
            return False
        
        print("Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

# Global configuration instance
try:
    settings = load_config_from_env()
    if not validate_config(settings):
        print("Warning: Configuration validation failed, using defaults")
        settings = SystemConfig()
except Exception as e:
    print(f"Error initializing configuration: {e}")
    settings = SystemConfig()

def get_config_summary() -> dict:
    """Get a summary of current configuration."""
    return {
        'audio': {
            'sample_rate': settings.audio.sample_rate,
            'n_mels': settings.audio.n_mels,
            'n_lfcc': settings.audio.n_lfcc,
            'hop_length': settings.audio.hop_length,
            'win_length': settings.audio.win_length,
            'n_fft': settings.audio.n_fft
        },
        'physics': {
            'time_window_ms': settings.physics.time_window_for_dynamics_ms,
            'embedding_dim': settings.physics.embedding_dim_for_physics,
            'max_pca_components': settings.physics.max_pca_components
        },
        'models': {
            'hubert_model': settings.models.hubert_model_path
        },
        'system': {
            'log_level': settings.log_level,
            'device': settings.device,
            'num_workers': settings.num_workers
        }
    }

if __name__ == "__main__":
    print("Configuration Summary:")
    print("=" * 50)
    
    import json
    summary = get_config_summary()
    print(json.dumps(summary, indent=2))
    
    print("\nFull Configuration:")
    print("=" * 50)
    print(settings.model_dump_json(indent=2))