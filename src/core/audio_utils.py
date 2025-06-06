"""
Enhanced Audio Utilities with Comprehensive Error Handling
"""

import asyncio
import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
from utils.config_loader import settings
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

async def load_audio(file_path: Union[str, Path], target_sr: int = settings.audio.sample_rate, 
                    max_duration: Optional[float] = None) -> Optional[torch.Tensor]:
    """
    Asynchronously loads and preprocesses an audio file with comprehensive error handling.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        max_duration: Maximum duration in seconds (None for no limit)
    
    Returns:
        1D torch.Tensor containing the audio waveform, or None if loading failed
    """
    try:
        file_path = Path(file_path)
        
        # Check if file exists and is readable
        if not file_path.exists():
            print(f"Audio file not found: {file_path}")
            return None
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            print(f"Audio file too large: {file_size_mb:.1f}MB > {settings.max_file_size_mb}MB")
            return None
        
        # Check file extension
        if file_path.suffix.lower() not in settings.supported_audio_formats:
            print(f"Unsupported audio format: {file_path.suffix}")
            return None
        
        # Load audio file
        try:
            waveform, sr = await asyncio.to_thread(torchaudio.load, str(file_path))
        except Exception as e:
            # Fallback to librosa for problematic files
            print(f"torchaudio failed, trying librosa: {e}")
            try:
                waveform_np, sr = await asyncio.to_thread(
                    librosa.load, str(file_path), sr=None, mono=False
                )
                # Convert to torch tensor
                if waveform_np.ndim == 1:
                    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
                else:
                    waveform = torch.from_numpy(waveform_np)
            except Exception as e2:
                print(f"Both torchaudio and librosa failed: {e2}")
                return None
        
        # Validate loaded audio
        if waveform is None or waveform.numel() == 0:
            print(f"Empty or invalid audio loaded from {file_path}")
            return None
        
        # Handle stereo to mono conversion
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            # Convert stereo to mono by averaging channels
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Ensure 1D tensor
        waveform = waveform.squeeze()
        if waveform.ndim == 0:  # Scalar case
            print(f"Audio reduced to scalar, invalid: {file_path}")
            return None
        
        # Resample if necessary
        if sr != target_sr:
            try:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=target_sr
                )
                waveform = resampler(waveform)
            except Exception as e:
                print(f"Resampling failed: {e}")
                # Fallback to librosa resampling
                try:
                    waveform_np = waveform.numpy()
                    waveform_resampled = await asyncio.to_thread(
                        librosa.resample, waveform_np, orig_sr=sr, target_sr=target_sr
                    )
                    waveform = torch.from_numpy(waveform_resampled)
                except Exception as e2:
                    print(f"Librosa resampling also failed: {e2}")
                    return None
        
        # Trim to maximum duration if specified
        if max_duration is not None:
            max_samples = int(max_duration * target_sr)
            if waveform.shape[0] > max_samples:
                waveform = waveform[:max_samples]
        
        # Validate final waveform
        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            print(f"Audio contains NaN or Inf values: {file_path}")
            # Try to clean the audio
            waveform = torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Check for reasonable amplitude range
        max_amp = torch.max(torch.abs(waveform))
        if max_amp > 100:  # Likely integer audio that should be normalized
            waveform = waveform / max_amp
        elif max_amp < 1e-6:  # Very quiet audio
            print(f"Warning: Very quiet audio detected: {file_path}")
        
        return waveform.float()
        
    except Exception as e:
        print(f"Unexpected error loading audio {file_path}: {e}")
        return None

def normalize_waveform(waveform: torch.Tensor, method: str = 'zscore') -> torch.Tensor:
    """
    Normalizes waveform using various methods.
    
    Args:
        waveform: Input waveform tensor
        method: Normalization method ('zscore', 'minmax', 'rms', 'peak')
    
    Returns:
        Normalized waveform tensor
    """
    if waveform.numel() == 0:
        return waveform
    
    try:
        if method == 'zscore':
            # Z-score normalization (zero mean, unit variance)
            mean = waveform.mean()
            std = waveform.std()
            if std < 1e-8:  # Avoid division by zero
                return torch.zeros_like(waveform)
            return (waveform - mean) / std
            
        elif method == 'minmax':
            # Min-max normalization to [-1, 1]
            min_val = waveform.min()
            max_val = waveform.max()
            if abs(max_val - min_val) < 1e-8:
                return torch.zeros_like(waveform)
            return 2 * (waveform - min_val) / (max_val - min_val) - 1
            
        elif method == 'rms':
            # RMS normalization
            rms = torch.sqrt(torch.mean(waveform**2))
            if rms < 1e-8:
                return torch.zeros_like(waveform)
            return waveform / rms
            
        elif method == 'peak':
            # Peak normalization
            peak = torch.max(torch.abs(waveform))
            if peak < 1e-8:
                return torch.zeros_like(waveform)
            return waveform / peak
            
        else:
            print(f"Unknown normalization method: {method}, using zscore")
            return normalize_waveform(waveform, 'zscore')
            
    except Exception as e:
        print(f"Error in waveform normalization: {e}")
        return waveform

def segment_audio(waveform: torch.Tensor, segment_duration_s: float, 
                 sr: int, overlap: float = 0.0, 
                 pad_last: bool = True) -> List[torch.Tensor]:
    """
    Segments audio into fixed duration chunks with optional overlap.
    
    Args:
        waveform: Input waveform [samples]
        segment_duration_s: Segment duration in seconds
        sr: Sample rate
        overlap: Overlap ratio (0.0 = no overlap, 0.5 = 50% overlap)
        pad_last: Whether to pad the last segment if it's shorter
    
    Returns:
        List of audio segments
    """
    if waveform.numel() == 0:
        return []
    
    try:
        segment_length = int(segment_duration_s * sr)
        if segment_length <= 0:
            print(f"Invalid segment length: {segment_length}")
            return [waveform]
        
        # Calculate hop size based on overlap
        hop_size = int(segment_length * (1 - overlap))
        if hop_size <= 0:
            hop_size = 1
        
        segments = []
        start = 0
        
        while start < len(waveform):
            end = start + segment_length
            
            if end <= len(waveform):
                # Full segment
                segments.append(waveform[start:end])
            else:
                # Last segment (potentially shorter)
                last_segment = waveform[start:]
                
                if pad_last and len(last_segment) < segment_length:
                    # Pad with zeros or repeat the last part
                    padding_needed = segment_length - len(last_segment)
                    if len(last_segment) > padding_needed:
                        # Repeat the end of the segment
                        pad_segment = last_segment[-padding_needed:]
                    else:
                        # Zero padding
                        pad_segment = torch.zeros(padding_needed, dtype=waveform.dtype)
                    
                    last_segment = torch.cat([last_segment, pad_segment])
                
                segments.append(last_segment)
                break
            
            start += hop_size
        
        return segments
        
    except Exception as e:
        print(f"Error in audio segmentation: {e}")
        return [waveform]

def apply_audio_augmentation(waveform: torch.Tensor, sr: int, 
                           augment_type: str = 'none') -> torch.Tensor:
    """
    Apply audio augmentation techniques.
    
    Args:
        waveform: Input waveform
        sr: Sample rate
        augment_type: Type of augmentation ('none', 'noise', 'pitch', 'tempo', 'all')
    
    Returns:
        Augmented waveform
    """
    if augment_type == 'none' or waveform.numel() == 0:
        return waveform
    
    try:
        augmented = waveform.clone()
        
        if augment_type in ['noise', 'all']:
            # Add subtle noise
            noise_level = 0.005 * torch.max(torch.abs(augmented))
            noise = torch.randn_like(augmented) * noise_level
            augmented = augmented + noise
        
        if augment_type in ['pitch', 'all']:
            # Pitch shifting (requires librosa)
            try:
                augmented_np = augmented.numpy()
                # Random pitch shift of ±2 semitones
                n_steps = np.random.uniform(-2, 2)
                augmented_np = librosa.effects.pitch_shift(
                    augmented_np, sr=sr, n_steps=n_steps
                )
                augmented = torch.from_numpy(augmented_np)
            except Exception:
                pass  # Skip pitch shifting if librosa fails
        
        if augment_type in ['tempo', 'all']:
            # Time stretching
            try:
                augmented_np = augmented.numpy()
                # Random tempo change of ±10%
                rate = np.random.uniform(0.9, 1.1)
                augmented_np = librosa.effects.time_stretch(augmented_np, rate=rate)
                augmented = torch.from_numpy(augmented_np)
            except Exception:
                pass  # Skip tempo change if librosa fails
        
        return augmented.float()
        
    except Exception as e:
        print(f"Error in audio augmentation: {e}")
        return waveform

def detect_audio_properties(waveform: torch.Tensor, sr: int) -> dict:
    """
    Detect various properties of an audio signal.
    
    Args:
        waveform: Input waveform
        sr: Sample rate
    
    Returns:
        Dictionary of audio properties
    """
    if waveform.numel() == 0:
        return {}
    
    try:
        properties = {}
        
        # Basic properties
        properties['duration_s'] = len(waveform) / sr
        properties['sample_rate'] = sr
        properties['num_samples'] = len(waveform)
        
        # Amplitude properties
        properties['rms'] = torch.sqrt(torch.mean(waveform**2)).item()
        properties['peak'] = torch.max(torch.abs(waveform)).item()
        properties['dynamic_range_db'] = 20 * np.log10(properties['peak'] / (properties['rms'] + 1e-8))
        
        # Spectral properties
        try:
            # Use librosa for spectral analysis
            waveform_np = waveform.numpy()
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=waveform_np, sr=sr)
            properties['spectral_centroid_mean'] = np.mean(spectral_centroids)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(waveform_np)
            properties['zero_crossing_rate_mean'] = np.mean(zcr)
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=waveform_np, sr=sr)
            properties['spectral_rolloff_mean'] = np.mean(rolloff)
            
        except Exception:
            # Fallback to simple properties if librosa fails
            properties['spectral_centroid_mean'] = 0.0
            properties['zero_crossing_rate_mean'] = 0.0
            properties['spectral_rolloff_mean'] = 0.0
        
        # Silence detection
        silence_threshold = 0.01 * properties['peak']
        silence_mask = torch.abs(waveform) < silence_threshold
        properties['silence_ratio'] = silence_mask.float().mean().item()
        
        return properties
        
    except Exception as e:
        print(f"Error in audio property detection: {e}")
        return {}

if __name__ == '__main__':
    import asyncio
    
    async def test_audio_utils():
        """Test audio utilities with comprehensive examples."""
        print("Testing Enhanced Audio Utilities")
        print("=" * 40)
        
        # Create test audio files
        test_sr = 16000
        test_duration = 3.0
        
        # Create different types of test audio
        test_cases = {
            'sine_wave': torch.sin(2 * torch.pi * 440 * torch.linspace(0, test_duration, int(test_sr * test_duration))),
            'white_noise': torch.randn(int(test_sr * test_duration)) * 0.1,
            'chirp': torch.sin(2 * torch.pi * torch.linspace(100, 1000, int(test_sr * test_duration)) * torch.linspace(0, test_duration, int(test_sr * test_duration)))
        }
        
        for test_name, test_waveform in test_cases.items():
            print(f"\nTesting with {test_name}:")
            print(f"  Original shape: {test_waveform.shape}")
            
            # Test normalization methods
            for norm_method in ['zscore', 'minmax', 'rms', 'peak']:
                normalized = normalize_waveform(test_waveform, norm_method)
                print(f"  {norm_method} normalized - mean: {normalized.mean():.4f}, std: {normalized.std():.4f}")
            
            # Test segmentation
            segments = segment_audio(test_waveform, 1.0, test_sr, overlap=0.5)
            print(f"  Segmentation (1s, 50% overlap): {len(segments)} segments")
            
            # Test audio properties
            properties = detect_audio_properties(test_waveform, test_sr)
            print(f"  Properties: RMS={properties.get('rms', 0):.4f}, Peak={properties.get('peak', 0):.4f}")
            print(f"  Spectral centroid: {properties.get('spectral_centroid_mean', 0):.1f} Hz")
            
            # Test augmentation
            augmented = apply_audio_augmentation(test_waveform, test_sr, 'noise')
            print(f"  Augmented shape: {augmented.shape}")
        
        # Test file loading (if dummy file exists)
        try:
            # Create a dummy wav file for testing
            dummy_path = "test_audio.wav"
            torchaudio.save(dummy_path, test_cases['sine_wave'].unsqueeze(0), test_sr)
            
            loaded_audio = await load_audio(dummy_path, test_sr)
            if loaded_audio is not None:
                print(f"\nFile loading test successful: {loaded_audio.shape}")
            else:
                print(f"\nFile loading test failed")
                
            # Clean up
            import os
            if os.path.exists(dummy_path):
                os.remove(dummy_path)
                
        except Exception as e:
            print(f"\nFile loading test error: {e}")
    
    # Run tests
    asyncio.run(test_audio_utils())