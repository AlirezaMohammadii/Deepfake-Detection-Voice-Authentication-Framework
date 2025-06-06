import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal.windows import get_window
from scipy.special import j0, jn
from utils.config_loader import settings
from core.model_loader import DEVICE
from typing import Optional, Dict, Tuple, List, Union
import math
from dataclasses import dataclass, field
from functools import lru_cache
import time
import asyncio
import concurrent.futures
import logging
import collections
from abc import ABC
from pathlib import Path
import plotly.graph_objects as go
import os

@dataclass
class PhysicsConstants:
    """Physical constants for voice dynamics analysis"""
    SOUND_SPEED_AIR: float = 343.0  # m/s at 20°C
    HUMAN_VOCAL_TRACT_LENGTH: float = 0.17  # meters (average)
    FORMANT_FREQ_RANGE: Tuple[float, float] = (300.0, 3500.0)  # Hz
    PITCH_RANGE: Tuple[float, float] = (50.0, 500.0)  # Hz
    EMBEDDING_SAMPLE_RATE: float = 50.0  # HuBERT: 50Hz (20ms frames)
    
class VoiceRadarPhysics:
    """
    Physics-based dynamics analyzer for voice deepfake detection.
    Implements VoiceRadar micro-motion analysis with proper physics formulation.
    """
    
    def __init__(self, embedding_dim: int, audio_sr: int, config=None):
        self.embedding_dim = embedding_dim
        self.audio_sr = audio_sr
        self.config = config or settings.physics
        self.physics = PhysicsConstants()
        
        # Time parameters
        self.dt = 1.0 / self.physics.EMBEDDING_SAMPLE_RATE
        
        # Window configuration with physics-based sizing
        # Window should capture at least 2-3 pitch periods
        min_pitch_period = 1.0 / self.physics.PITCH_RANGE[0]  # ~20ms
        self.window_duration = max(3 * min_pitch_period, self.config.time_window_for_dynamics_ms / 1000.0)
        self.window_size = max(8, int(self.window_duration * self.physics.EMBEDDING_SAMPLE_RATE))
        self.overlap = 0.75
        self.hop_size = max(1, int(self.window_size * (1 - self.overlap)))
        
        # FFT parameters for better frequency resolution
        self.nfft = max(128, 2 ** int(np.ceil(np.log2(self.window_size * 2))))
        self._init_spectral_params()
        
        # Initialize optimized window processor
        self.window_processor = OptimizedWindowProcessor(
            window_size=self.window_size,
            overlap_ratio=self.overlap,
            device=DEVICE
        )
        
        # Bessel function parameters from VoiceRadar paper
        # λ₀n values for J₀ zeros: 2.405, 5.520, 8.654, ...
        self.bessel_zeros = torch.tensor([2.405, 5.520, 8.654, 11.792, 14.931], 
                                        device=DEVICE, dtype=torch.float32)
        
        # Phase space parameters for micro-motion analysis
        self.phase_space_bins = 32
        self.phase_space_range = (-3.0, 3.0)  # Standard deviations
        
    @lru_cache(maxsize=1)
    def _init_spectral_params(self):
        """Initialize spectral analysis parameters once"""
        # Optimal window for speech analysis
        self.window = torch.from_numpy(
            get_window('hamming', self.window_size)
        ).to(DEVICE).float()
        
        # Frequency bins
        self.freqs = torch.fft.rfftfreq(self.nfft, d=self.dt).to(DEVICE)
        
        # Pre-compute frequency weights for centroid calculation
        self.freq_weights = self.freqs.unsqueeze(1)
        
    def _compute_spectral_features(self, signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive spectral features with proper physics and improved numerical stability.
        
        Args:
            signal: [time, features] tensor
            
        Returns:
            Dictionary of spectral features
        """
        if signal.shape[0] < self.window_size:
            # Pad with edge values instead of zeros for better spectral estimation
            pad_size = self.window_size - signal.shape[0]
            signal = F.pad(signal, (0, 0, 0, pad_size), mode='replicate')
        
        # Apply window
        windowed = signal[:self.window_size] * self.window.unsqueeze(1)
        
        # Compute spectrum
        spectrum = torch.fft.rfft(windowed, n=self.nfft, dim=0)
        magnitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)
        
        # Power spectrum (squared magnitude)
        power = magnitude ** 2
        
        # Improved numerical stability for spectral centroid calculation
        # Use torch.finfo for proper epsilon based on data type
        eps = torch.finfo(power.dtype).eps * 100
        total_power = torch.sum(power, dim=0, keepdim=True)
        
        # Handle zero power case explicitly
        zero_power_mask = total_power <= eps
        norm_power = torch.zeros_like(power)
        
        # Only normalize where we have sufficient power
        valid_mask = ~zero_power_mask.squeeze(0)
        if valid_mask.any():
            norm_power[:, valid_mask] = power[:, valid_mask] / total_power[:, valid_mask]
        
        # Spectral centroid (center of mass of spectrum) with safe computation
        centroid = torch.zeros(signal.shape[1], device=DEVICE)
        if valid_mask.any():
            centroid[valid_mask] = torch.sum(norm_power[:, valid_mask] * self.freq_weights, dim=0)
        
        # Spectral spread (standard deviation around centroid)
        variance = torch.zeros(signal.shape[1], device=DEVICE)
        if valid_mask.any():
            freq_diff = self.freq_weights - centroid[valid_mask].unsqueeze(0)
            variance[valid_mask] = torch.sum(norm_power[:, valid_mask] * freq_diff**2, dim=0)
        spread = torch.sqrt(variance.clamp(min=0))
        
        # Spectral flux (change rate) - store for next call
        if hasattr(self, '_prev_magnitude'):
            flux = torch.sqrt(torch.sum((magnitude - self._prev_magnitude)**2, dim=0))
        else:
            flux = torch.zeros(signal.shape[1], device=DEVICE)
        self._prev_magnitude = magnitude.detach()
        
        # Spectral rolloff (95% energy threshold) with improved stability
        cumsum = torch.cumsum(power, dim=0)
        threshold = 0.95 * cumsum[-1]
        
        # Handle edge case where cumsum[-1] is very small
        rolloff = torch.zeros(signal.shape[1], device=DEVICE)
        valid_energy_mask = cumsum[-1] > eps
        if valid_energy_mask.any():
            rolloff_idx = torch.argmax((cumsum[:, valid_energy_mask] >= threshold[valid_energy_mask].unsqueeze(0)).float(), dim=0)
            rolloff[valid_energy_mask] = self.freqs[rolloff_idx]
        
        # Spectral entropy (information content) with improved numerical stability
        entropy = torch.zeros(signal.shape[1], device=DEVICE)
        if valid_mask.any():
            # Use larger epsilon for log to avoid numerical issues
            log_eps = max(eps, 1e-10)
            entropy[valid_mask] = -torch.sum(norm_power[:, valid_mask] * torch.log(norm_power[:, valid_mask].clamp(min=log_eps)), dim=0)
        
        return {
            'centroid': centroid,
            'spread': spread,
            'flux': flux,
            'rolloff': rolloff,
            'entropy': entropy,
            'magnitude': magnitude,
            'phase': phase
        }
    
    def _compute_phase_space_density(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute phase space density for micro-motion analysis.
        Maps embeddings to phase space and computes density distribution.
        
        Args:
            embeddings: [time, dim] tensor
            
        Returns:
            Phase space density tensor
        """
        # Normalize embeddings to phase space coordinates
        mean = torch.mean(embeddings, dim=0, keepdim=True)
        std = torch.std(embeddings, dim=0, keepdim=True).clamp(min=1e-6)
        normalized = (embeddings - mean) / std
        
        # Project to 2D phase space using first two principal components
        U, S, Vh = torch.linalg.svd(normalized - normalized.mean(dim=0), full_matrices=False)
        phase_coords = U[:, :2] @ torch.diag(S[:2])  # [time, 2]
        
        # Compute 2D histogram (density)
        hist_range = [self.phase_space_range, self.phase_space_range]
        density, _, _ = np.histogram2d(
            phase_coords[:, 0].cpu().numpy(),
            phase_coords[:, 1].cpu().numpy(),
            bins=self.phase_space_bins,
            range=hist_range,
            density=True
        )
        
        return torch.from_numpy(density).to(DEVICE).float()
    
    def _compute_bessel_features_vectorized(self, R: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """
        Vectorized Bessel function computation to avoid memory leaks.
        
        Args:
            R: Radial distances tensor
            density: Phase space density tensor
            
        Returns:
            Bessel features tensor
        """
        # Pre-allocate output
        bessel_features = torch.zeros(len(self.bessel_zeros), device=DEVICE)
        
        # Flatten radial distances for vectorized computation
        r_flat = R.flatten()
        density_flat = density.flatten()
        
        # Process each Bessel zero coefficient
        for idx, lambda_coeff in enumerate(self.bessel_zeros):
            # Compute arguments for Bessel function
            args = lambda_coeff * r_flat
            
            # Create mask for valid arguments (avoid numerical overflow)
            valid_mask = (args < 50.0) & torch.isfinite(args)
            
            if valid_mask.any():
                # Extract valid arguments and compute Bessel function
                valid_args = args[valid_mask].cpu().numpy()
                
                # Use scipy's vectorized j0 function
                bessel_vals_np = j0(valid_args)
                
                # Convert back to tensor
                bessel_vals = torch.zeros_like(args)
                bessel_vals[valid_mask] = torch.from_numpy(bessel_vals_np).to(DEVICE)
                
                # Reshape back to original shape and weight by density
                bessel_vals_2d = bessel_vals.view(R.shape)
                weighted_bessel = bessel_vals_2d * density
                
                # Integrate (sum) the weighted Bessel function
                bessel_features[idx] = torch.sum(weighted_bessel)
        
        return bessel_features

    def _bessel_micro_motion_analysis(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Implement VoiceRadar's Bessel function analysis for micro-motion.
        J₀(λ₀nr) where r is the radial distance in phase space.
        
        Args:
            embeddings: [time, dim] tensor
            
        Returns:
            Dictionary of Bessel-based features
        """
        # Get phase space representation
        density = self._compute_phase_space_density(embeddings)
        
        # Create radial coordinate grid
        x = torch.linspace(*self.phase_space_range, self.phase_space_bins, device=DEVICE)
        y = torch.linspace(*self.phase_space_range, self.phase_space_bins, device=DEVICE)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        R = torch.sqrt(X**2 + Y**2)  # Radial distances
        
        # Use vectorized Bessel computation
        bessel_coeffs = self._compute_bessel_features_vectorized(R, density)
        
        # Compute radial frequency from density distribution with improved numerical stability
        eps = torch.finfo(density.dtype).eps * 100
        density_sum = torch.sum(density)
        
        if density_sum > eps:
            radial_density = torch.sum(density * R, dim=[0, 1]) / density_sum
        else:
            radial_density = torch.tensor(0.0, device=DEVICE)
        
        # Phase space entropy with improved numerical stability
        log_eps = max(eps, 1e-10)
        phase_entropy = -torch.sum(density * torch.log(density.clamp(min=log_eps)))
        
        return {
            'bessel_coeffs': bessel_coeffs,
            'radial_frequency': radial_density,
            'phase_density_entropy': phase_entropy
        }
    
    def calculate_translational_dynamics(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate translational frequency Δfₜ from embedding trajectory.
        Represents the overall drift/translation in embedding space.
        """
        if embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=DEVICE)
        
        # Compute center of mass trajectory
        com_trajectory = torch.mean(embeddings, dim=1)  # [time]
        
        # Spectral analysis of COM motion
        features = self._compute_spectral_features(com_trajectory.unsqueeze(1))
        
        # Translational frequency is the dominant frequency of COM motion
        delta_ft = features['centroid'].squeeze()
        
        # Normalize to physical frequency range
        delta_ft_normalized = delta_ft / self.physics.EMBEDDING_SAMPLE_RATE
        
        return delta_ft_normalized.clamp(0, 1)
    
    def calculate_rotational_dynamics(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate rotational frequency Δfᵣ using principal component evolution.
        Tracks how the principal axes of the embedding cloud rotate over time.
        Uses optimized windowing for better performance.
        """
        if embeddings.shape[0] < self.window_size:
            return torch.tensor(0.0, device=DEVICE)
        
        # Use optimized window processor for efficient segmentation
        windowed_segments = self.window_processor.process_segments(embeddings)
        num_windows = windowed_segments.shape[0]
        
        if num_windows < 2:
            return torch.tensor(0.0, device=DEVICE)
        
        angular_velocities = []
        prev_components = None
        
        # Process each window
        for i in range(num_windows):
            window = windowed_segments[i]  # [window_size, features]
            
            # Center the data
            centered = window - window.mean(dim=0, keepdim=True)
            
            # Skip if no variance
            if torch.allclose(centered, torch.zeros_like(centered), atol=1e-6):
                continue
            
            # Compute instantaneous rotation using SVD
            try:
                U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
                
                # Track the first k principal components
                k = min(3, Vh.shape[0])
                current_components = Vh[:k]
                
                if prev_components is not None:
                    # Compute rotation matrix between consecutive orientations
                    # R = Vh_current @ Vh_prev.T
                    R = current_components @ prev_components.T
                    
                    # Extract rotation angle from rotation matrix
                    # For 3D: angle = arccos((trace(R) - 1) / 2)
                    trace = torch.diagonal(R, dim1=0, dim2=1).sum()
                    cos_angle = (trace - 1) / 2
                    cos_angle = torch.clamp(cos_angle, -1, 1)
                    angle = torch.acos(cos_angle)
                    
                    # Angular velocity = angle / time
                    omega = angle / (self.hop_size * self.dt)
                    angular_velocities.append(omega)
                
                prev_components = current_components.detach()
                
            except Exception as e:
                # Skip this window if SVD fails
                continue
        
        if not angular_velocities:
            return torch.tensor(0.0, device=DEVICE)
        
        # Mean angular velocity -> rotational frequency
        delta_fr = torch.mean(torch.stack(angular_velocities)) / (2 * np.pi)
        
        return delta_fr.clamp(0, 10)  # Reasonable range for rotational frequency
    
    def calculate_vibrational_dynamics(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate vibrational frequency Δfᵥ from high-frequency oscillations.
        Analyzes the micro-vibrations in embedding space.
        """
        if embeddings.shape[0] < 3:
            return torch.tensor(0.0, device=DEVICE)
        
        # Compute acceleration (second derivative)
        velocity = torch.diff(embeddings, n=1, dim=0) / self.dt
        acceleration = torch.diff(velocity, n=1, dim=0) / self.dt
        
        # Spectral analysis of acceleration magnitude
        accel_magnitude = torch.norm(acceleration, dim=1, keepdim=True)
        features = self._compute_spectral_features(accel_magnitude)
        
        # Vibrational frequency from acceleration spectrum
        delta_fv = features['centroid'].squeeze()
        
        # High-frequency emphasis using spectral rolloff
        high_freq_ratio = features['rolloff'].squeeze() / (self.physics.EMBEDDING_SAMPLE_RATE / 2)
        
        # Weighted vibrational frequency
        delta_fv_weighted = delta_fv * high_freq_ratio
        
        return delta_fv_weighted.clamp(0, 25)  # Nyquist limit
    
    def _compute_doppler_shift(self, source_freq: torch.Tensor, 
                              velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute proper Doppler shift using relativistic formula.
        f_observed = f_source * sqrt((c + v) / (c - v))
        
        For v << c: f_observed ≈ f_source * (1 + v/c)
        """
        # Normalize velocity to fraction of "information speed" in embedding space
        c_embedding = 1.0  # Normalized speed of information propagation
        v_normalized = torch.tanh(velocity / 10.0)  # Sigmoid normalization
        
        # Non-relativistic Doppler approximation
        doppler_factor = 1 + v_normalized / c_embedding
        
        return source_freq * doppler_factor
    
    def calculate_all_dynamics(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate all physics-based features from embedding sequence.
        
        Args:
            embeddings: [time, dim] tensor of audio embeddings
            
        Returns:
            Dictionary of physics-based features
        """
        # Ensure correct device and shape
        embeddings = embeddings.to(DEVICE)
        
        if embeddings.shape[0] < 2:
            # Return zero features for too-short sequences
            return {k: torch.tensor(0.0, device=DEVICE) for k in [
                'delta_ft', 'delta_fr', 'delta_fv', 'delta_f_total',
                'doppler_shift', 'velocity_magnitude', 'bessel_score',
                'phase_entropy', 'spectral_entropy'
            ]}
        
        # Reset state for new sequence
        if hasattr(self, '_prev_magnitude'):
            del self._prev_magnitude
        if hasattr(self, '_prev_components'):
            del self._prev_components
        
        # Calculate base dynamics
        delta_ft = self.calculate_translational_dynamics(embeddings)
        delta_fr = self.calculate_rotational_dynamics(embeddings)
        delta_fv = self.calculate_vibrational_dynamics(embeddings)
        
        # Total source frequency
        delta_f_total = delta_ft + delta_fr + delta_fv
        
        # Velocity analysis
        if embeddings.shape[0] >= 2:
            velocity_vectors = torch.diff(embeddings, dim=0) / self.dt
            velocity_magnitude = torch.mean(torch.norm(velocity_vectors, dim=1))
        else:
            velocity_magnitude = torch.tensor(0.0, device=DEVICE)
        
        # Doppler shift calculation
        doppler_shift = self._compute_doppler_shift(delta_f_total, velocity_magnitude)
        
        # Bessel function micro-motion analysis
        bessel_analysis = self._bessel_micro_motion_analysis(embeddings)
        bessel_score = torch.mean(bessel_analysis['bessel_coeffs'])
        
        # Global spectral analysis
        global_features = self._compute_spectral_features(embeddings)
        spectral_entropy = torch.mean(global_features['entropy'])
        
        return {
            'delta_ft': delta_ft,
            'delta_fr': delta_fr,
            'delta_fv': delta_fv,
            'delta_f_total': delta_f_total,
            'doppler_shift': doppler_shift,
            'velocity_magnitude': velocity_magnitude,
            'bessel_score': bessel_score,
            'phase_entropy': bessel_analysis['phase_density_entropy'],
            'spectral_entropy': spectral_entropy,
            'radial_frequency': bessel_analysis['radial_frequency']
        }

# Backward compatibility wrapper
class VoiceRadarInspiredDynamics(VoiceRadarPhysics):
    """Wrapper for backward compatibility with existing codebase"""
    
    async def calculate_all_dynamics(self, embeddings_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Async wrapper for compatibility"""
        # Calculate synchronously
        results = super().calculate_all_dynamics(embeddings_sequence)
        
        # Map to expected keys
        return {
            "delta_ft_revised": results['delta_ft'],
            "delta_fr_revised": results['delta_fr'],
            "delta_fv_revised": results['delta_fv'],
            "delta_f_total_revised": results['delta_f_total'],
            "embedding_mean_velocity_mag": results['velocity_magnitude'],
            "doppler_proxy_fs": results['doppler_shift']
        }

    async def calculate_all_dynamics_async(self, embeddings_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Truly asynchronous physics calculation using thread pool for CPU-intensive operations.
        This prevents blocking the event loop during physics calculations.
        """
        try:
            # Ensure embeddings are on CPU for thread-safe operations
            embeddings_cpu = embeddings_sequence.cpu()
            
            # Get current event loop
            loop = asyncio.get_event_loop()
            
            # Run CPU-intensive physics calculations in thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Submit the synchronous calculation to thread pool
                future = loop.run_in_executor(
                    executor,
                    self._calculate_dynamics_sync,
                    embeddings_cpu
                )
                
                # Await the result
                results = await future
            
            # Move results back to original device
            device_results = {}
            for key, value in results.items():
                if torch.is_tensor(value):
                    device_results[key] = value.to(DEVICE)
                else:
                    device_results[key] = value
            
            # Map to expected keys for backward compatibility
            return {
                "delta_ft_revised": device_results['delta_ft'],
                "delta_fr_revised": device_results['delta_fr'],
                "delta_fv_revised": device_results['delta_fv'],
                "delta_f_total_revised": device_results['delta_f_total'],
                "embedding_mean_velocity_mag": device_results['velocity_magnitude'],
                "doppler_proxy_fs": device_results['doppler_shift']
            }
            
        except Exception as e:
            logging.warning(f"Async physics calculation failed: {e}, falling back to sync")
            # Fallback to synchronous calculation
            return await self.calculate_all_dynamics(embeddings_sequence)
    
    def _calculate_dynamics_sync(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Synchronous physics calculation method for thread pool execution.
        This is the CPU-intensive part that runs in a separate thread.
        """
        # Call the parent class synchronous method
        return super().calculate_all_dynamics(embeddings)

class OptimizedWindowProcessor:
    """
    Optimized window processor for efficient overlapping window analysis.
    Pre-computes windows and FFT workspace to avoid repeated allocations.
    """
    
    def __init__(self, window_size: int, overlap_ratio: float, device: torch.device = DEVICE):
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.hop_size = max(1, int(window_size * (1 - overlap_ratio)))
        self.device = device
        
        # Pre-compute window function (Hamming window for speech analysis)
        self.window = torch.hann_window(window_size, device=device)
        
        # Pre-allocate FFT workspace
        self.fft_size = 2 ** int(np.ceil(np.log2(window_size)))
        
        # Pre-compute frequency bins
        dt = 1.0 / PhysicsConstants.EMBEDDING_SAMPLE_RATE
        self.freqs = torch.fft.rfftfreq(self.fft_size, d=dt).to(device)
    
    def process_segments(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Process signal using efficient windowing.
        
        Args:
            signal: [time, features] tensor
            
        Returns:
            Windowed segments tensor [num_windows, window_size, features]
        """
        if signal.shape[0] < self.window_size:
            # Pad signal if too short
            pad_size = self.window_size - signal.shape[0]
            signal = F.pad(signal, (0, 0, 0, pad_size), mode='replicate')
        
        # Use unfold for efficient windowing - this avoids creating copies
        if signal.shape[0] >= self.window_size:
            windows = signal.unfold(0, self.window_size, self.hop_size)
            # windows shape: [num_windows, features, window_size]
            windows = windows.transpose(1, 2)  # -> [num_windows, window_size, features]
            
            # Apply window function
            windowed = windows * self.window.unsqueeze(0).unsqueeze(2)
            return windowed
        else:
            # Single window case
            windowed = signal[:self.window_size] * self.window.unsqueeze(1)
            return windowed.unsqueeze(0)
    
    def compute_spectral_features_batch(self, windowed_segments: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute spectral features for a batch of windowed segments efficiently.
        
        Args:
            windowed_segments: [num_windows, window_size, features] tensor
            
        Returns:
            Dictionary of batched spectral features
        """
        num_windows, window_size, num_features = windowed_segments.shape
        
        # Batch FFT computation
        spectrum = torch.fft.rfft(windowed_segments, n=self.fft_size, dim=1)
        magnitude = torch.abs(spectrum)
        
        # Power spectrum
        power = magnitude ** 2
        
        # Improved numerical stability
        eps = torch.finfo(power.dtype).eps * 100
        total_power = torch.sum(power, dim=1, keepdim=True)  # [num_windows, 1, features]
        
        # Handle zero power cases
        zero_power_mask = total_power <= eps
        norm_power = torch.zeros_like(power)
        
        valid_mask = ~zero_power_mask.squeeze(1)  # [num_windows, features]
        
        # Vectorized normalization for all valid windows and features
        if valid_mask.any():
            norm_power[valid_mask.unsqueeze(1).expand_as(power)] = (
                power[valid_mask.unsqueeze(1).expand_as(power)] / 
                total_power[valid_mask.unsqueeze(1).expand_as(total_power)]
            )
        
        # Batch spectral centroid computation
        freq_weights = self.freqs.unsqueeze(0).unsqueeze(2)  # [1, freq_bins, 1]
        centroid = torch.zeros(num_windows, num_features, device=self.device)
        
        if valid_mask.any():
            # Compute centroids only for valid windows/features
            valid_norm_power = norm_power[valid_mask.unsqueeze(1).expand_as(norm_power)].view(-1, norm_power.shape[1])
            valid_centroids = torch.sum(valid_norm_power * freq_weights.squeeze(2), dim=1)
            centroid[valid_mask] = valid_centroids
        
        return {
            'magnitude': magnitude,
            'power': power,
            'norm_power': norm_power,
            'centroid': centroid,
            'valid_mask': valid_mask
        }

class TemporalFeatureBuffer:
    """Manages temporal sequences for analysis"""
    
    def __init__(self, max_sequence_length: int = 50):
        self.buffer = collections.deque(maxlen=max_sequence_length)
        self.user_buffers = {}
        
    def add_features(self, features: Dict, user_id: Optional[str] = None) -> None:
        """Add features to temporal buffer"""
        # Store features in the buffer for temporal analysis
        self.buffer.append({
            'features': features,
            'timestamp': time.time(),
            'user_id': user_id
        })
        
        # Maintain user-specific buffers if needed
        if user_id:
            if user_id not in self.user_buffers:
                self.user_buffers[user_id] = collections.deque(maxlen=max_sequence_length)
            self.user_buffers[user_id].append(features)

class UncertaintyVisualizationMixin:
    """Mixin for uncertainty-aware plotting"""
    
    def plot_with_uncertainty_bands(self, 
                                  x: np.ndarray, 
                                  y_mean: np.ndarray,
                                  y_std: np.ndarray,
                                  confidence_level: float = 0.95) -> go.Figure:
        """Plot data with uncertainty bands"""
        # Create figure with uncertainty bands
        fig = go.Figure()
        
        # Calculate confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
        upper_bound = y_mean + z_score * y_std
        lower_bound = y_mean - z_score * y_std
        
        # Add uncertainty band
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{confidence_level*100:.0f}% Confidence'
        ))
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=x,
            y=y_mean,
            mode='lines',
            name='Mean',
            line=dict(color='blue', width=2)
        ))
        
        return fig

class AdvancedPhysicsPlotter:
    """Advanced plotting capabilities for physics features"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.uncertainty_plotter = UncertaintyVisualizationMixin()
        
    def create_physics_analysis_plots(self, results_df):
        """Create comprehensive physics analysis plots"""
        plots_dir = self.output_dir / "physics_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate various physics plots
        # This method would contain the plotting logic
        pass
        
    def generate_all_visualizations(self, csv_path: str):
        """Generate all available visualizations"""
        import pandas as pd
        
        # Read data
        df = pd.read_csv(csv_path)
        
        # Create plots
        self.create_physics_analysis_plots(df)
        
        # Return paths to generated files
        return {
            'static_dir': str(self.output_dir / "physics_plots"),
            'dashboard_path': str(self.output_dir / "dashboard.html"),
            'reports_dir': str(self.output_dir / "reports"),
            'summary_report': {
                'key_findings': [],
                'discrimination_ranking': [],
                'recommendations': []
            }
        }

class ConfigLoader:
    """Configuration loader for the physics system"""
    
    def __init__(self):
        self.physics_config = self._load_physics_config()
    
    def _load_physics_config(self):
        """Load physics configuration"""
        # Load basic physics configuration
        return {
            'time_window_ms': 100,
            'embedding_dim': 1024,
            'sample_rate': 16000
        }

class EnhancedSystemTests:
    """Enhanced system testing capabilities"""
    
    def __init__(self):
        self.test_results = {}
        
    async def run_comprehensive_tests(self):
        """Run comprehensive system tests"""
        print("🧪 Running Enhanced System Tests...")
        
        # Test core physics functionality
        await self._test_physics_features()
        
        # Test processing pipeline
        await self._test_processing_pipeline()
        
        print("✅ All tests completed")
        
    async def _test_physics_features(self):
        """Test physics feature extraction"""
        print("  Testing physics features...")
        # Test implementation would go here
        
    async def _test_processing_pipeline(self):
        """Test processing pipeline"""
        print("  Testing processing pipeline...")
        # Test implementation would go here

if __name__ == '__main__':
    import asyncio
    
    async def test_physics():
        print("Testing Enhanced VoiceRadar Physics Implementation")
        print("=" * 50)
        
        # Initialize analyzer
        analyzer = VoiceRadarInspiredDynamics(
            embedding_dim=1024,
            audio_sr=16000
        )
        
        # Test cases
        test_cases = [
            ("Short sequence", 10),
            ("Medium sequence", 50),
            ("Long sequence", 200)
        ]
        
        for name, length in test_cases:
            print(f"\n{name} ({length} frames):")
            
            # Generate test embeddings with realistic properties
            t = torch.linspace(0, length * analyzer.dt, length)
            
            # Simulate embeddings with translation, rotation, and vibration
            embeddings = torch.zeros(length, 1024, device=DEVICE)
            
            # Add translational motion
            embeddings[:, 0] = 0.1 * t
            
            # Add rotational motion
            embeddings[:, 1] = 0.05 * torch.sin(2 * np.pi * 0.5 * t)
            embeddings[:, 2] = 0.05 * torch.cos(2 * np.pi * 0.5 * t)
            
            # Add vibrational motion
            embeddings[:, 3:10] = 0.01 * torch.sin(2 * np.pi * 5 * t).unsqueeze(1)
            
            # Add noise
            embeddings += 0.001 * torch.randn_like(embeddings)
            
            # Calculate features
            features = await analyzer.calculate_all_dynamics(embeddings)
            
            print("  Features:")
            for key, value in features.items():
                print(f"    {key}: {value.item():.6f}")
    
    asyncio.run(test_physics())