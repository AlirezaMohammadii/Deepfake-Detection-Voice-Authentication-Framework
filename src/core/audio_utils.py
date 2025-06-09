"""
Audio Utilities
Basic audio processing functions for the deepfake detection system
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
import warnings

def load_audio(file_path: str, target_sr: int = 16000, normalize: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and return waveform and sample rate
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (will resample if different)
        normalize: Whether to normalize the waveform
    
    Returns:
        Tuple of (waveform_tensor, sample_rate)
    """
    try:
        import torchaudio
        
        # Load audio file
        waveform, original_sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if original_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
            waveform = resampler(waveform)
        
        # Squeeze to 1D
        waveform = waveform.squeeze(0)
        
        # Normalize if requested
        if normalize:
            waveform = normalize_waveform(waveform)
        
        return waveform, target_sr
        
    except ImportError:
        # Fallback using librosa if torchaudio not available
        try:
            import librosa
            
            waveform_np, sr = librosa.load(file_path, sr=target_sr, mono=True)
            waveform = torch.from_numpy(waveform_np).float()
            
            if normalize:
                waveform = normalize_waveform(waveform)
            
            return waveform, sr
            
        except ImportError:
            raise ImportError("Either torchaudio or librosa is required for audio loading")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {e}")

def normalize_waveform(waveform: torch.Tensor, method: str = "peak") -> torch.Tensor:
    """
    Normalize audio waveform
    
    Args:
        waveform: Input waveform tensor
        method: Normalization method ("peak", "rms", or "lufs")
    
    Returns:
        Normalized waveform tensor
    """
    if waveform.numel() == 0:
        return waveform
    
    if method == "peak":
        # Peak normalization
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            return waveform / max_val
        return waveform
    
    elif method == "rms":
        # RMS normalization
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            return waveform / rms
        return waveform

    else:
        # Default to peak normalization
        return normalize_waveform(waveform, "peak")

def segment_audio(waveform: torch.Tensor, sr: int, 
                 segment_length: float = 10.0, 
                 overlap: float = 0.0) -> List[torch.Tensor]:
    """
    Segment audio into chunks
    
    Args:
        waveform: Input waveform tensor [samples] or [channels, samples]
        sr: Sample rate
        segment_length: Length of each segment in seconds
        overlap: Overlap between segments (0.0 to 1.0)
    
    Returns:
        List of audio segments
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension
    
    channels, total_samples = waveform.shape
    segment_samples = int(segment_length * sr)
    
    if total_samples <= segment_samples:
        return [waveform.squeeze(0)]
    
    hop_samples = int(segment_samples * (1 - overlap))
    segments = []
    
    start = 0
    while start + segment_samples <= total_samples:
        segment = waveform[:, start:start + segment_samples]
        segments.append(segment.squeeze(0))
        start += hop_samples
    
    # Add final segment if there's remaining audio
    if start < total_samples:
        final_segment = waveform[:, start:]
        # Pad if necessary
        if final_segment.shape[1] < segment_samples:
            padding = segment_samples - final_segment.shape[1]
            final_segment = torch.nn.functional.pad(final_segment, (0, padding))
        segments.append(final_segment.squeeze(0))
        
    return segments

def resample_audio(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    Simple resampling using linear interpolation
    
    Args:
        waveform: Input waveform
        orig_sr: Original sample rate
        target_sr: Target sample rate
    
    Returns:
        Resampled waveform
    """
    if orig_sr == target_sr:
        return waveform
    
    # Simple linear interpolation resampling
    ratio = target_sr / orig_sr
    new_length = int(waveform.shape[-1] * ratio)
    
    # Use torch's interpolation
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
        resampled = torch.nn.functional.interpolate(
            waveform, size=new_length, mode='linear', align_corners=False
        )
        return resampled.squeeze(0).squeeze(0)
    else:
        waveform = waveform.unsqueeze(0)  # [1, channels, samples]
        resampled = torch.nn.functional.interpolate(
            waveform, size=new_length, mode='linear', align_corners=False
        )
        return resampled.squeeze(0)

def apply_preemphasis(waveform: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    """
    Apply preemphasis filter to audio
    
    Args:
        waveform: Input waveform
        coeff: Preemphasis coefficient
        
    Returns:
        Filtered waveform
    """
    if waveform.numel() <= 1:
        return waveform

    # Apply preemphasis: y[n] = x[n] - coeff * x[n-1]
    filtered = waveform.clone()
    filtered[1:] = waveform[1:] - coeff * waveform[:-1]
    
    return filtered

def remove_dc_offset(waveform: torch.Tensor) -> torch.Tensor:
    """
    Remove DC offset from audio
    
    Args:
        waveform: Input waveform
        
    Returns:
        DC-corrected waveform
    """
    return waveform - torch.mean(waveform)

def detect_silence(waveform: torch.Tensor, threshold: float = 0.01, 
                  min_duration: float = 0.1, sr: int = 16000) -> List[Tuple[int, int]]:
    """
    Detect silent regions in audio
    
    Args:
        waveform: Input waveform
        threshold: Silence threshold (relative to peak)
        min_duration: Minimum silence duration in seconds
        sr: Sample rate
    
    Returns:
        List of (start, end) sample indices for silent regions
    """
    # Calculate energy
    energy = waveform ** 2
    
    # Smooth energy with moving average
    window_size = int(0.025 * sr)  # 25ms window
    if window_size > 1:
        kernel = torch.ones(window_size) / window_size
        energy = torch.nn.functional.conv1d(
            energy.unsqueeze(0).unsqueeze(0), 
            kernel.unsqueeze(0).unsqueeze(0), 
            padding=window_size//2
        ).squeeze()
    
    # Find silent regions
    peak_energy = torch.max(energy)
    silence_mask = energy < (threshold * peak_energy)
    
    # Find contiguous silent regions
    silent_regions = []
    in_silence = False
    start_idx = 0
    min_samples = int(min_duration * sr)
    
    for i, is_silent in enumerate(silence_mask):
        if is_silent and not in_silence:
            start_idx = i
            in_silence = True
        elif not is_silent and in_silence:
            if i - start_idx >= min_samples:
                silent_regions.append((start_idx, i))
            in_silence = False
    
    # Handle case where audio ends in silence
    if in_silence and len(silence_mask) - start_idx >= min_samples:
        silent_regions.append((start_idx, len(silence_mask)))
    
    return silent_regions

def trim_silence(waveform: torch.Tensor, threshold: float = 0.01, sr: int = 16000) -> torch.Tensor:
    """
    Trim silence from beginning and end of audio
    
    Args:
        waveform: Input waveform
        threshold: Silence threshold
        sr: Sample rate
        
    Returns:
        Trimmed waveform
    """
    if waveform.numel() == 0:
        return waveform
    
    # Find non-silent regions
    energy = waveform ** 2
    peak_energy = torch.max(energy)
    non_silent_mask = energy >= (threshold * peak_energy)
    
    # Find first and last non-silent samples
    non_silent_indices = torch.where(non_silent_mask)[0]
    
    if len(non_silent_indices) == 0:
        # All silence - return a small segment
        return waveform[:min(1000, len(waveform))]
    
    start_idx = non_silent_indices[0].item()
    end_idx = non_silent_indices[-1].item() + 1
    
    return waveform[start_idx:end_idx]


    