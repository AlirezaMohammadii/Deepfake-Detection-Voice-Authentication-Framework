"""
Processing Pipeline Module
Creates different processing pipelines for various use cases
"""

from typing import Any, Dict, Optional
from core.feature_extractor import ComprehensiveFeatureExtractor, FeatureExtractorFactory

def create_standard_pipeline(enable_cache: bool = True, 
                           cache_dir: str = "cache",
                           enable_bayesian: bool = True) -> ComprehensiveFeatureExtractor:
    """
    Create a standard processing pipeline with full features
    
    Args:
        enable_cache: Whether to enable feature caching
        cache_dir: Directory for cache files
        enable_bayesian: Whether to enable Bayesian analysis
        
    Returns:
        Configured ComprehensiveFeatureExtractor
    """
    return FeatureExtractorFactory.create(
        enable_cache=enable_cache,
        cache_dir=cache_dir
    )

def create_lightweight_pipeline(enable_physics: bool = True,
                               enable_audio_features: bool = True,
                               device: Optional[str] = None) -> ComprehensiveFeatureExtractor:
    """
    Create a lightweight pipeline for faster processing
    
    Args:
        enable_physics: Whether to enable physics feature calculation
        enable_audio_features: Whether to enable traditional audio features
        device: Device to use ('cuda', 'cpu', or None for auto)
        
    Returns:
        Lightweight ComprehensiveFeatureExtractor
    """
    return FeatureExtractorFactory.create_lightweight(
        enable_physics=enable_physics,
        enable_audio_features=enable_audio_features,
        device=device
    )

def create_testing_pipeline(mock_model: bool = False,
                          mock_physics: bool = False) -> ComprehensiveFeatureExtractor:
    """
    Create a pipeline for testing with optional mocks
    
    Args:
        mock_model: Whether to use mock HuBERT model
        mock_physics: Whether to use mock physics calculator
        
    Returns:
        Testing-configured ComprehensiveFeatureExtractor
    """
    return FeatureExtractorFactory.create_for_testing(
        mock_model=mock_model,
        mock_physics=mock_physics
    )

class PipelineManager:
    """Manages different processing pipelines"""
    
    def __init__(self):
        self.pipelines = {}
        
    def register_pipeline(self, name: str, pipeline: ComprehensiveFeatureExtractor):
        """Register a named pipeline"""
        self.pipelines[name] = pipeline
        
    def get_pipeline(self, name: str) -> Optional[ComprehensiveFeatureExtractor]:
        """Get a registered pipeline by name"""
        return self.pipelines.get(name)
        
    def list_pipelines(self) -> list:
        """List all registered pipeline names"""
        return list(self.pipelines.keys()) 