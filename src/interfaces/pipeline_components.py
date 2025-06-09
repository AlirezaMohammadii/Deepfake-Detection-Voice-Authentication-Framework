"""
Pipeline Components Interface
Defines base classes for the deepfake detection pipeline
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors in the deepfake detection pipeline
    """
    
    @abstractmethod
    async def extract_features(self, waveform: torch.Tensor, sr: int, 
                              processing_mode: str = "default") -> Dict[str, Any]:
        """
        Extract features from audio waveform
        
        Args:
            waveform: Audio waveform tensor [samples] or [channels, samples]
            sr: Sample rate
            processing_mode: Processing mode identifier
            
        Returns:
            Dictionary containing extracted features
        """
        pass
    
    def validate_input(self, waveform: torch.Tensor, sr: int) -> Tuple[bool, str]:
        """
        Validate input waveform and sample rate
        
        Args:
            waveform: Input waveform tensor
            sr: Sample rate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not torch.is_tensor(waveform):
            return False, "Waveform must be a tensor"
            
        if waveform.numel() == 0:
            return False, "Waveform is empty"
        
        if sr <= 0:
            return False, "Sample rate must be positive"
        
        return True, ""
    
    def get_supported_sample_rates(self) -> list:
        """Get list of supported sample rates"""
        return [16000, 22050, 44100, 48000]
    
    def get_processing_modes(self) -> list:
        """Get list of available processing modes"""
        return ["default", "lightweight", "high_accuracy"] 