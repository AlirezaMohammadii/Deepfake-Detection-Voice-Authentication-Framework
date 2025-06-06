"""
Enhanced Processing Pipeline with Stage-based Architecture
Implements efficient data flow with error handling and early exit capabilities.
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import torch
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    """Processing status for pipeline stages."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class ProcessingResult:
    """Result container for pipeline processing."""
    data: Any
    status: ProcessingStatus
    stage_name: str
    processing_time: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        return self.status == ProcessingStatus.SUCCESS
    
    @property
    def is_error(self) -> bool:
        return self.status in [ProcessingStatus.FAILED, ProcessingStatus.ERROR]

class ProcessingStage(ABC):
    """Abstract base class for pipeline processing stages."""
    
    def __init__(self, name: str, retries: int = 2, timeout: Optional[float] = None):
        self.name = name
        self.retries = retries
        self.timeout = timeout
        self.statistics = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
    
    @abstractmethod
    async def process(self, data: Any) -> ProcessingResult:
        """Process data through this stage."""
        pass
    
    async def process_with_retry(self, data: Any) -> ProcessingResult:
        """Process with retry mechanism."""
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.retries + 1):
            try:
                if self.timeout:
                    result = await asyncio.wait_for(
                        self.process(data), 
                        timeout=self.timeout
                    )
                else:
                    result = await self.process(data)
                
                # Update statistics
                processing_time = time.time() - start_time
                self._update_statistics(True, processing_time)
                
                return result
                
            except asyncio.TimeoutError as e:
                last_error = f"Stage {self.name} timed out after {self.timeout}s"
                logger.warning(f"Attempt {attempt + 1} timed out: {last_error}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                
                if attempt < self.retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # All attempts failed
        processing_time = time.time() - start_time
        self._update_statistics(False, processing_time)
        
        return ProcessingResult(
            data=None,
            status=ProcessingStatus.FAILED,
            stage_name=self.name,
            processing_time=processing_time,
            metadata={'attempts': self.retries + 1},
            error_message=last_error
        )
    
    def _update_statistics(self, success: bool, processing_time: float):
        """Update stage statistics."""
        self.statistics['total_processed'] += 1
        self.statistics['total_time'] += processing_time
        
        if success:
            self.statistics['successful'] += 1
        else:
            self.statistics['failed'] += 1
        
        self.statistics['avg_time'] = (
            self.statistics['total_time'] / self.statistics['total_processed']
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get stage processing statistics."""
        return self.statistics.copy()

class AudioLoadingStage(ProcessingStage):
    """Stage for loading and preprocessing audio files."""
    
    def __init__(self, target_sr: int = 16000, max_duration: Optional[float] = None):
        super().__init__("AudioLoading", retries=2, timeout=30.0)
        self.target_sr = target_sr
        self.max_duration = max_duration
    
    async def process(self, data: Any) -> ProcessingResult:
        """Load audio file."""
        start_time = time.time()
        
        try:
            # Import audio utils lazily
            from .audio_utils import load_audio
            
            file_path = data if isinstance(data, (str, Path)) else data.get('filepath')
            if not file_path:
                raise ValueError("No file path provided")
            
            # Load audio
            waveform = await load_audio(file_path, self.target_sr, self.max_duration)
            
            if waveform is None:
                raise ValueError(f"Failed to load audio from {file_path}")
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                data={
                    'waveform': waveform,
                    'sample_rate': self.target_sr,
                    'filepath': file_path,
                    'duration': waveform.shape[0] / self.target_sr
                },
                status=ProcessingStatus.SUCCESS,
                stage_name=self.name,
                processing_time=processing_time,
                metadata={
                    'waveform_shape': waveform.shape,
                    'sample_rate': self.target_sr
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                data=None,
                status=ProcessingStatus.FAILED,
                stage_name=self.name,
                processing_time=processing_time,
                metadata={},
                error_message=str(e)
            )

class PreprocessingStage(ProcessingStage):
    """Stage for audio preprocessing and normalization."""
    
    def __init__(self, normalization_method: str = 'zscore'):
        super().__init__("Preprocessing", retries=1, timeout=10.0)
        self.normalization_method = normalization_method
    
    async def process(self, data: Any) -> ProcessingResult:
        """Preprocess audio data."""
        start_time = time.time()
        
        try:
            # Import audio utils lazily
            from .audio_utils import normalize_waveform
            
            if not isinstance(data, dict) or 'waveform' not in data:
                raise ValueError("Expected data with 'waveform' key")
            
            waveform = data['waveform']
            
            # Normalize waveform
            normalized_waveform = normalize_waveform(waveform, self.normalization_method)
            
            # Update data
            processed_data = data.copy()
            processed_data['waveform'] = normalized_waveform
            processed_data['normalized'] = True
            processed_data['normalization_method'] = self.normalization_method
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                data=processed_data,
                status=ProcessingStatus.SUCCESS,
                stage_name=self.name,
                processing_time=processing_time,
                metadata={
                    'normalization_method': self.normalization_method,
                    'waveform_stats': {
                        'mean': normalized_waveform.mean().item(),
                        'std': normalized_waveform.std().item(),
                        'min': normalized_waveform.min().item(),
                        'max': normalized_waveform.max().item()
                    }
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                data=data,  # Pass through original data
                status=ProcessingStatus.FAILED,
                stage_name=self.name,
                processing_time=processing_time,
                metadata={},
                error_message=str(e)
            )

class FeatureExtractionStage(ProcessingStage):
    """Stage for comprehensive feature extraction."""
    
    def __init__(self, feature_extractor, retries: int = 2, timeout: float = 300.0):
        super().__init__("FeatureExtraction", retries, timeout)
        self.feature_extractor = feature_extractor
        self.processing_mode = "pipeline"  # Default mode
    
    def set_processing_mode(self, mode: str):
        """Set the processing mode for cache coordination."""
        self.processing_mode = mode
    
    async def process(self, data: Any) -> ProcessingResult:
        """Extract features from audio with processing mode coordination."""
        start_time = time.time()
        
        try:
            # Extract file path and waveform information
            if isinstance(data, dict):
                waveform = data.get('waveform')
                sample_rate = data.get('sample_rate')
                filepath = data.get('filepath', 'unknown')
            else:
                raise ValueError("Expected dictionary with waveform and sample_rate")
            
            if waveform is None or sample_rate is None:
                raise ValueError("Missing waveform or sample_rate in data")
            
            # Extract features with processing mode
            features = await self.feature_extractor.extract_features(
                waveform, sample_rate, self.processing_mode
            )
            
            # Prepare result data
            result_data = {
                **data,  # Include original data
                'features': features,
                'processing_mode': self.processing_mode
            }
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                data=result_data,
                status=ProcessingStatus.SUCCESS,
                stage_name=self.name,
                processing_time=processing_time,
                metadata={
                    'feature_keys': list(features.keys()),
                    'extraction_time': features.get('_extraction_time', 0),
                    'cache_hit': features.get('_cache_hit', False),
                    'processing_mode': self.processing_mode,
                    'validation_status': features.get('_validation', {}).get('overall_valid', True)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                data=data,  # Pass through original data
                status=ProcessingStatus.FAILED,
                stage_name=self.name,
                processing_time=processing_time,
                metadata={'processing_mode': self.processing_mode},
                error_message=str(e)
            )

class ValidationStage(ProcessingStage):
    """Stage for validating extracted features."""
    
    def __init__(self, strict_validation: bool = False):
        super().__init__("Validation", retries=0, timeout=5.0)
        self.strict_validation = strict_validation
    
    async def process(self, data: Any) -> ProcessingResult:
        """Validate extracted features."""
        start_time = time.time()
        
        try:
            if not isinstance(data, dict) or 'features' not in data:
                raise ValueError("Expected data with 'features' key")
            
            features = data['features']
            
            # Import validator lazily
            from .feature_extractor import FeatureValidator
            
            # Validate features
            validation_report = FeatureValidator.validate_all_features(features)
            
            # Determine if validation passed
            if self.strict_validation:
                # In strict mode, any error fails validation
                validation_passed = validation_report['overall_valid']
            else:
                # In non-strict mode, only critical errors fail validation
                critical_errors = [
                    error for error in validation_report.get('errors', [])
                    if 'HuBERT' in error or 'Physics' in error
                ]
                validation_passed = len(critical_errors) == 0
            
            # Update data with validation results
            result_data = data.copy()
            result_data['validation'] = validation_report
            result_data['validation_passed'] = validation_passed
            
            processing_time = time.time() - start_time
            
            status = ProcessingStatus.SUCCESS if validation_passed else ProcessingStatus.FAILED
            
            return ProcessingResult(
                data=result_data,
                status=status,
                stage_name=self.name,
                processing_time=processing_time,
                metadata={
                    'validation_passed': validation_passed,
                    'num_errors': len(validation_report.get('errors', [])),
                    'num_warnings': len(validation_report.get('warnings', [])),
                    'strict_mode': self.strict_validation
                },
                error_message=None if validation_passed else f"Validation failed with {len(validation_report.get('errors', []))} errors"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                data=data,  # Pass through original data
                status=ProcessingStatus.FAILED,
                stage_name=self.name,
                processing_time=processing_time,
                metadata={},
                error_message=str(e)
            )

class ResultAggregationStage(ProcessingStage):
    """Stage for aggregating and formatting final results."""
    
    def __init__(self, output_format: str = 'dict'):
        super().__init__("ResultAggregation", retries=0, timeout=5.0)
        self.output_format = output_format
    
    async def process(self, data: Any) -> ProcessingResult:
        """Aggregate results into final format."""
        start_time = time.time()
        
        try:
            if not isinstance(data, dict):
                raise ValueError("Expected dictionary data")
            
            # Extract key information
            features = data.get('features', {})
            validation = data.get('validation', {})
            filepath = data.get('filepath', 'unknown')
            
            # Create aggregated result
            aggregated_result = {
                'filepath': filepath,
                'processing_successful': True,
                'audio_duration': data.get('duration', 0),
                'sample_rate': data.get('sample_rate', 16000),
                'features': {},
                'validation_summary': {
                    'overall_valid': validation.get('overall_valid', True),
                    'num_errors': len(validation.get('errors', [])),
                    'num_warnings': len(validation.get('warnings', []))
                },
                'processing_metadata': {
                    'normalization_method': data.get('normalization_method', 'unknown'),
                    'extraction_time': features.get('_extraction_time', 0),
                    'cache_hit': features.get('_cache_hit', False)
                }
            }
            
            # Extract specific feature types
            if 'physics' in features:
                physics = features['physics']
                aggregated_result['features']['physics'] = {}
                for key, value in physics.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        aggregated_result['features']['physics'][key] = value.item()
                    else:
                        aggregated_result['features']['physics'][key] = str(value)
            
            # Add HuBERT info
            if 'hubert_sequence' in features:
                hubert_seq = features['hubert_sequence']
                aggregated_result['features']['hubert'] = {
                    'sequence_length': hubert_seq.shape[0],
                    'embedding_dim': hubert_seq.shape[1]
                }
            
            # Add audio feature info
            for feature_name in ['mel_spectrogram', 'lfcc']:
                if feature_name in features:
                    feature_tensor = features[feature_name]
                    aggregated_result['features'][feature_name] = {
                        'shape': list(feature_tensor.shape),
                        'mean': feature_tensor.mean().item(),
                        'std': feature_tensor.std().item()
                    }
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                data=aggregated_result,
                status=ProcessingStatus.SUCCESS,
                stage_name=self.name,
                processing_time=processing_time,
                metadata={
                    'output_format': self.output_format,
                    'num_feature_types': len(aggregated_result['features'])
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                data=data,  # Pass through original data
                status=ProcessingStatus.FAILED,
                stage_name=self.name,
                processing_time=processing_time,
                metadata={},
                error_message=str(e)
            )

class ProcessingPipeline:
    """
    Main processing pipeline for efficient audio feature extraction.
    Implements stage-based processing with error handling and early exit.
    """
    
    def __init__(self, stages: List[ProcessingStage], 
                 early_exit_on_error: bool = False,
                 collect_statistics: bool = True):
        self.stages = stages
        self.early_exit_on_error = early_exit_on_error
        self.collect_statistics = collect_statistics
        self.pipeline_statistics = {
            'total_processed': 0,
            'fully_successful': 0,
            'partially_successful': 0,
            'failed': 0,
            'avg_total_time': 0.0
        }
    
    async def process(self, input_data: Any, processing_mode: str = "pipeline") -> Dict[str, Any]:
        """
        Process input through the entire pipeline.
        
        Args:
            input_data: Input data (file path, audio data, etc.)
            processing_mode: Processing mode identifier for cache coordination
            
        Returns:
            Dictionary containing processing results and metadata
        """
        start_time = time.time()
        pipeline_context = {
            'start_time': start_time,
            'input_data': input_data,
            'processing_mode': processing_mode,
            'stage_results': {},
            'current_data': input_data
        }
        
        successful_stages = 0
        total_stages = len(self.stages)
        
        print(f"🔄 Starting pipeline processing ({processing_mode}) with {total_stages} stages...")
        
        try:
            for i, stage in enumerate(self.stages):
                stage_start = time.time()
                print(f"  Stage {i+1}/{total_stages}: {stage.name}")
                
                try:
                    # Pass processing mode to stages that support it
                    if hasattr(stage, 'set_processing_mode'):
                        stage.set_processing_mode(processing_mode)
                    
                    result = await stage.process(pipeline_context['current_data'])
                    
                    if result.status == ProcessingStatus.SUCCESS:
                        pipeline_context['current_data'] = result.data
                        pipeline_context['stage_results'][stage.name] = result
                        successful_stages += 1
                        
                        stage_time = time.time() - stage_start
                        print(f"    ✅ {stage.name} completed in {stage_time:.2f}s")
                        
                    elif result.status == ProcessingStatus.FAILED:
                        if self.early_exit_on_error:
                            print(f"    ❌ {stage.name} failed: {result.error_message}")
                            break
                        else:
                            print(f"    ⚠️  {stage.name} failed, continuing: {result.error_message}")
                            pipeline_context['stage_results'][stage.name] = result
                    
                    elif result.status == ProcessingStatus.SKIPPED:
                        print(f"    ⏭️  {stage.name} skipped")
                        pipeline_context['stage_results'][stage.name] = result
                        successful_stages += 1
                
                except Exception as e:
                    error_msg = f"Stage {stage.name} encountered unexpected error: {e}"
                    print(f"    💥 {error_msg}")
                    
                    error_result = ProcessingResult(
                        data=pipeline_context['current_data'],
                        status=ProcessingStatus.FAILED,
                        stage_name=stage.name,
                        processing_time=time.time() - stage_start,
                        error_message=error_msg
                    )
                    pipeline_context['stage_results'][stage.name] = error_result
                    
                    if self.early_exit_on_error:
                        break
        
        except Exception as e:
            print(f"🚨 Pipeline execution failed: {e}")
            return {
                'overall_status': 'pipeline_error',
                'error_message': str(e),
                'successful_stages': successful_stages,
                'total_stages': total_stages,
                'total_processing_time': time.time() - start_time,
                'processing_mode': processing_mode
            }
        
        # Determine overall status
        if successful_stages == total_stages:
            overall_status = 'fully_successful'
        elif successful_stages > 0:
            overall_status = 'partially_successful'
        else:
            overall_status = 'failed'
        
        total_time = time.time() - start_time
        print(f"✅ Pipeline processing ({processing_mode}) completed: {successful_stages}/{total_stages} stages successful in {total_time:.2f}s")
        
        return {
            'overall_status': overall_status,
            'successful_stages': successful_stages,
            'total_stages': total_stages,
            'total_processing_time': total_time,
            'processing_mode': processing_mode,
            'final_data': pipeline_context['current_data'],
            'stage_results': pipeline_context['stage_results'],
            'input_data': input_data
        }
    
    def _update_pipeline_statistics(self, status: str, total_time: float):
        """Update pipeline-level statistics."""
        self.pipeline_statistics['total_processed'] += 1
        self.pipeline_statistics[status] += 1
        
        # Update average time
        total_processed = self.pipeline_statistics['total_processed']
        current_avg = self.pipeline_statistics['avg_total_time']
        self.pipeline_statistics['avg_total_time'] = (
            (current_avg * (total_processed - 1) + total_time) / total_processed
        )
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline-level statistics."""
        stats = self.pipeline_statistics.copy()
        
        # Add stage statistics
        stats['stage_statistics'] = {}
        for stage in self.stages:
            stats['stage_statistics'][stage.name] = stage.get_statistics()
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.pipeline_statistics = {
            'total_processed': 0,
            'fully_successful': 0,
            'partially_successful': 0,
            'failed': 0,
            'avg_total_time': 0.0
        }
        
        for stage in self.stages:
            stage.statistics = {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'total_time': 0.0,
                'avg_time': 0.0
            }

# Factory function for creating standard pipelines
def create_standard_pipeline(
    feature_extractor=None,
    strict_validation: bool = False,
    early_exit_on_error: bool = False
) -> ProcessingPipeline:
    """
    Create a standard processing pipeline for audio feature extraction.
    
    Args:
        feature_extractor: Optional pre-initialized feature extractor
        strict_validation: Whether to use strict validation
        early_exit_on_error: Whether to exit early on stage errors
        
    Returns:
        Configured ProcessingPipeline instance
    """
    stages = [
        AudioLoadingStage(target_sr=16000, max_duration=None),
        PreprocessingStage(normalization_method='zscore'),
        FeatureExtractionStage(feature_extractor=feature_extractor, retries=2, timeout=300.0),
        ValidationStage(strict_validation=strict_validation),
        ResultAggregationStage(output_format='dict')
    ]
    
    return ProcessingPipeline(
        stages=stages,
        early_exit_on_error=early_exit_on_error,
        collect_statistics=True
    )

# Factory function for creating lightweight pipelines
def create_lightweight_pipeline(
    enable_physics: bool = True,
    enable_audio_features: bool = True
) -> ProcessingPipeline:
    """
    Create a lightweight pipeline for faster processing.
    
    Args:
        enable_physics: Whether to enable physics features
        enable_audio_features: Whether to enable audio features
        
    Returns:
        Lightweight ProcessingPipeline instance
    """
    # Create lightweight feature extractor
    from .feature_extractor import FeatureExtractorFactory
    feature_extractor = FeatureExtractorFactory.create_lightweight(
        enable_physics=enable_physics,
        enable_audio_features=enable_audio_features
    )
    
    stages = [
        AudioLoadingStage(target_sr=16000, max_duration=10.0),  # Limit duration
        PreprocessingStage(normalization_method='peak'),  # Faster normalization
        FeatureExtractionStage(feature_extractor=feature_extractor, retries=2, timeout=300.0),
        ResultAggregationStage(output_format='dict')
    ]
    
    return ProcessingPipeline(
        stages=stages,
        early_exit_on_error=True,  # Exit early for speed
        collect_statistics=False  # Disable stats for speed
    )

if __name__ == "__main__":
    import asyncio
    
    async def test_pipeline():
        """Test the processing pipeline."""
        print("Testing Processing Pipeline")
        print("=" * 50)
        
        # Create test pipeline
        pipeline = create_standard_pipeline(strict_validation=False)
        
        # Test with dummy data
        test_file_path = "test_audio.wav"
        
        print(f"Processing test file: {test_file_path}")
        result = await pipeline.process(test_file_path)
        
        print(f"\nPipeline Results:")
        print(f"Overall Status: {result['overall_status']}")
        print(f"Total Time: {result['total_processing_time']:.3f}s")
        print(f"Successful Stages: {result['successful_stages']}/{result['total_stages']}")
        
        # Print stage results
        print(f"\nStage Results:")
        for stage_result in result['stage_results']:
            status_icon = "✓" if stage_result['status'] == 'success' else "✗"
            print(f"  {status_icon} {stage_result['stage_name']}: {stage_result['status']} ({stage_result['processing_time']:.3f}s)")
            if stage_result['error_message']:
                print(f"    Error: {stage_result['error_message']}")
        
        # Print statistics
        stats = result['pipeline_statistics']
        print(f"\nPipeline Statistics:")
        print(f"Total Processed: {stats['total_processed']}")
        print(f"Average Time: {stats['avg_total_time']:.3f}s")
    
    # Run test
    asyncio.run(test_pipeline()) 