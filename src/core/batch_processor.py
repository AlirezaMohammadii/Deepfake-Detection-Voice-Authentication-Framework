"""
Batch Processing Module
Implements efficient batch processing for audio feature extraction
"""

import asyncio
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import time
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from .audio_utils import load_audio, normalize_waveform
from .feature_extractor import ComprehensiveFeatureExtractor
from utils.security_validator import ResourceLimiter, InputValidator

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int = 32
    max_concurrent_batches: int = 4
    enable_padding: bool = True
    max_sequence_length: Optional[int] = None
    min_sequence_length: int = 1
    enable_length_bucketing: bool = True
    bucket_tolerance: float = 0.1
    memory_efficient_mode: bool = True
    
class BatchProcessor:
    """Advanced batch processor with memory optimization and length bucketing"""
    
    def __init__(self, 
                 feature_extractor: Optional[ComprehensiveFeatureExtractor] = None,
                 config: Optional[BatchConfig] = None,
                 resource_limiter: Optional[ResourceLimiter] = None):
        
        self.config = config or BatchConfig()
        self.feature_extractor = feature_extractor
        self.resource_limiter = resource_limiter
        
        # Performance tracking
        self.batch_stats = {
            'total_batches': 0,
            'total_files': 0,
            'total_processing_time': 0.0,
            'avg_batch_time': 0.0,
            'memory_peak_gb': 0.0
        }
        
        # Length bucketing for efficient processing
        self.length_buckets = defaultdict(list)
        
    async def process_files_batch(self, file_paths: List[Path], metadata: List[Dict] = None, 
                                processing_mode: str = "batch") -> List[Dict[str, Any]]:
        """
        Process a batch of audio files efficiently with mode coordination.
        
        Args:
            file_paths: List of file paths to process
            metadata: Optional metadata for each file
            processing_mode: Processing mode identifier for cache coordination
            
        Returns:
            List of processing results
        """
        if not file_paths:
            return []
        
        if metadata is None:
            metadata = [{}] * len(file_paths)
        
        print(f"🔄 Starting batch processing ({processing_mode}) of {len(file_paths)} files...")
        
        # Organize files by length for bucketing (if enabled)
        if self.config.enable_length_bucketing:
            batch_groups = self._organize_by_length(file_paths, metadata)
        else:
            # Single group with all files
            batch_groups = [(file_paths, metadata)]
        
        all_results = []
        
        # Process each length bucket
        for group_idx, (group_paths, group_metadata) in enumerate(batch_groups):
            print(f"📦 Processing group {group_idx + 1}/{len(batch_groups)} with {len(group_paths)} files")
            
            # Create batches from this group
            group_batches = []
            for i in range(0, len(group_paths), self.config.batch_size):
                batch_paths = group_paths[i:i + self.config.batch_size]
                batch_metadata = group_metadata[i:i + self.config.batch_size]
                
                # Prepare batch info with processing mode
                batch_info = [
                    (idx, path, {**meta, 'processing_mode': processing_mode}) 
                    for idx, (path, meta) in enumerate(zip(batch_paths, batch_metadata), start=i)
                ]
                group_batches.append(batch_info)
            
            # Process batches with controlled concurrency
            semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
            
            batch_tasks = [
                self._process_single_batch(batch_info, semaphore) 
                for batch_info in group_batches
            ]
            
            group_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Flatten results and handle exceptions
            for batch_result in group_results:
                if isinstance(batch_result, Exception):
                    # Create error results for this batch
                    print(f"❌ Batch processing failed: {batch_result}")
                    error_results = [
                        {
                            'filepath': str(group_paths[i]),
                            'status': 'batch_error',
                            'processing_mode': processing_mode,
                            'error': str(batch_result),
                            **group_metadata[i]
                        }
                        for i in range(len(group_paths))
                    ]
                    all_results.extend(error_results)
                else:
                    all_results.extend(batch_result)
        
        # Update batch statistics
        self.batch_stats['total_batches'] += len(batch_groups)
        self.batch_stats['total_files'] += len(file_paths)
        
        print(f"✅ Batch processing ({processing_mode}) completed: {len(all_results)} results")
        return all_results
    
    async def _analyze_files_for_bucketing(self, file_paths: List[Union[str, Path]], 
                                         metadata: List[Dict]) -> List[Tuple[int, Path, Dict, int]]:
        """Analyze files to determine their audio length for bucketing"""
        file_info = []
        
        async def get_file_info(idx: int, filepath: Union[str, Path], meta: Dict) -> Tuple[int, Path, Dict, int]:
            try:
                # Quick analysis to get approximate length without full loading
                import torchaudio
                filepath = Path(filepath)
                
                # Get file info without loading full audio
                info = torchaudio.info(str(filepath))
                duration_frames = info.num_frames
                
                return (idx, filepath, meta, duration_frames)
                
            except Exception as e:
                logger.warning(f"Could not analyze file {filepath}: {e}")
                # Default to medium length for bucketing
                return (idx, Path(filepath), meta, 50000)  # ~3 seconds at 16kHz
        
        # Analyze files concurrently
        tasks = [
            get_file_info(idx, filepath, meta) 
            for idx, (filepath, meta) in enumerate(zip(file_paths, metadata))
        ]
        
        file_info = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_info = []
        for item in file_info:
            if isinstance(item, Exception):
                logger.warning(f"File analysis failed: {item}")
            else:
                processed_info.append(item)
        
        return processed_info
    
    def _create_length_bucketed_batches(self, file_info: List[Tuple[int, Path, Dict, int]]) -> List[List[Tuple[int, Path, Dict]]]:
        """Create batches based on audio length for efficient padding"""
        # Sort by audio length
        file_info.sort(key=lambda x: x[3])  # Sort by duration_frames
        
        batches = []
        current_batch = []
        current_length = None
        
        for idx, filepath, meta, duration in file_info:
            # Determine if this file should start a new batch
            if current_length is None:
                current_length = duration
            
            length_diff = abs(duration - current_length) / max(current_length, duration)
            
            if (len(current_batch) >= self.config.batch_size or 
                (len(current_batch) > 0 and length_diff > self.config.bucket_tolerance)):
                
                # Start new batch
                batches.append(current_batch)
                current_batch = [(idx, filepath, meta)]
                current_length = duration
            else:
                current_batch.append((idx, filepath, meta))
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _create_simple_batches(self, file_info: List[Tuple[int, Path, Dict]]) -> List[List[Tuple[int, Path, Dict]]]:
        """Create simple batches without length bucketing"""
        batches = []
        for i in range(0, len(file_info), self.config.batch_size):
            batch = file_info[i:i + self.config.batch_size]
            batches.append(batch)
        return batches
    
    async def _process_single_batch(self, batch_info: List[Tuple[int, Path, Dict]], 
                                  semaphore: asyncio.Semaphore) -> List[Dict[str, Any]]:
        """Process a single batch of files"""
        async with semaphore:
            try:
                batch_start_time = time.time()
                
                # Load all audio files in the batch
                audio_load_tasks = [
                    self._load_audio_safe(filepath, meta) 
                    for _, filepath, meta in batch_info
                ]
                
                audio_data = await asyncio.gather(*audio_load_tasks, return_exceptions=True)
                
                # Filter successful loads and handle failures
                valid_audio = []
                results = []
                
                for i, (audio_result, (_, filepath, meta)) in enumerate(zip(audio_data, batch_info)):
                    if isinstance(audio_result, Exception):
                        results.append({
                            'filepath': str(filepath),
                            'status': 'load_error',
                            'error': str(audio_result),
                            **meta
                        })
                    else:
                        valid_audio.append((i, audio_result, filepath, meta))
                        results.append(None)  # Placeholder for successful processing
                
                # Process valid audio files
                if valid_audio and self.feature_extractor:
                    try:
                        # Extract features for valid audio
                        feature_tasks = [
                            self._extract_features_safe(waveform, sr, filepath, meta)
                            for _, (waveform, sr), filepath, meta in valid_audio
                        ]
                        
                        feature_results = await asyncio.gather(*feature_tasks, return_exceptions=True)
                        
                        # Place feature results in correct positions
                        for (result_idx, _, _, _), feature_result in zip(valid_audio, feature_results):
                            if isinstance(feature_result, Exception):
                                results[result_idx] = {
                                    'filepath': str(batch_info[result_idx][1]),
                                    'status': 'feature_error',
                                    'error': str(feature_result),
                                    **batch_info[result_idx][2]
                                }
                            else:
                                results[result_idx] = feature_result
                        
                    except Exception as e:
                        logger.error(f"Batch feature extraction failed: {e}")
                        # Mark all valid audio as failed
                        for result_idx, _, filepath, meta in valid_audio:
                            results[result_idx] = {
                                'filepath': str(filepath),
                                'status': 'feature_error',
                                'error': str(e),
                                **meta
                            }
                
                # Ensure no None results remain
                for i, result in enumerate(results):
                    if result is None:
                        _, filepath, meta = batch_info[i]
                        results[i] = {
                            'filepath': str(filepath),
                            'status': 'error',
                            'error': 'Unprocessed result',
                            **meta
                        }
                
                batch_time = time.time() - batch_start_time
                logger.debug(f"Batch of {len(batch_info)} files processed in {batch_time:.2f}s")
                
                return results
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Return error for all files in batch
                return [
                    {
                        'filepath': str(filepath),
                        'status': 'batch_error',
                        'error': str(e),
                        **meta
                    }
                    for _, filepath, meta in batch_info
                ]
    
    async def _load_audio_safe(self, filepath: Path, metadata: Dict) -> Tuple[torch.Tensor, int]:
        """Safely load audio with error handling"""
        try:
            # Load audio using existing audio utils (load_audio is not async)
            waveform, sr = load_audio(str(filepath))
            
            if waveform is None:
                raise ValueError("Failed to load audio")
            
            # Validate audio
            InputValidator.validate_tensor(waveform, f"audio_{filepath.name}")
            
            return waveform, sr
            
        except Exception as e:
            logger.error(f"Audio loading failed for {filepath}: {e}")
            raise
    
    async def _extract_features_safe(self, waveform: torch.Tensor, sr: int, 
                                   filepath: Path, metadata: Dict) -> Dict[str, Any]:
        """Safely extract features with error handling"""
        try:
            if self.feature_extractor is None:
                raise ValueError("No feature extractor available")
            
            # Get processing mode from metadata
            processing_mode = metadata.get('processing_mode', 'batch')
            
            # Extract features with processing mode
            features = await self.feature_extractor.extract_features(waveform, sr, processing_mode)
            
            # Prepare result
            result = {
                'filepath': str(filepath),
                'status': 'success',
                'processing_mode': processing_mode,
                'audio_duration_s': len(waveform) / sr,
                **{k: v for k, v in metadata.items() if k != 'processing_mode'}
            }
            
            # Add physics features
            if 'physics' in features:
                physics = features['physics']
                for key, value in physics.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        result[f'physics_{key}'] = value.item()
                    else:
                        result[f'physics_{key}'] = str(value)
            
            # Add HuBERT info
            if 'hubert_sequence' in features:
                hubert = features['hubert_sequence']
                result['hubert_seq_len_frames'] = hubert.shape[0]
                result['hubert_embedding_dim'] = hubert.shape[1]
            
            # Add processing metadata
            result['extraction_time'] = features.get('_extraction_time', 0.0)
            result['cache_hit'] = features.get('_cache_hit', False)
            
            return result
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {filepath}: {e}")
            return {
                'filepath': str(filepath),
                'status': 'feature_error',
                'processing_mode': metadata.get('processing_mode', 'batch'),
                'error': str(e),
                **{k: v for k, v in metadata.items() if k != 'processing_mode'}
            }
    
    def _update_batch_stats(self, num_batches: int, num_files: int, processing_time: float):
        """Update batch processing statistics"""
        self.batch_stats['total_batches'] += num_batches
        self.batch_stats['total_files'] += num_files
        self.batch_stats['total_processing_time'] += processing_time
        
        if self.batch_stats['total_batches'] > 0:
            self.batch_stats['avg_batch_time'] = (
                self.batch_stats['total_processing_time'] / self.batch_stats['total_batches']
            )
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        return self.batch_stats.copy()
    
    async def process_streaming_batch(self, audio_stream_generator, 
                                    chunk_size: int = 16000) -> List[Dict[str, Any]]:
        """Process streaming audio data in batches"""
        results = []
        chunk_buffer = []
        
        async for audio_chunk in audio_stream_generator:
            chunk_buffer.append(audio_chunk)
            
            # Process when buffer reaches batch size
            if len(chunk_buffer) >= self.config.batch_size:
                # Create temporary file paths for chunks
                temp_metadata = [
                    {'chunk_id': i, 'timestamp': time.time()}
                    for i in range(len(chunk_buffer))
                ]
                
                # Process the buffered chunks
                # Note: This would need adaptation for actual streaming data
                chunk_results = await self._process_audio_chunks(chunk_buffer, temp_metadata)
                results.extend(chunk_results)
                
                # Clear buffer
                chunk_buffer = []
        
        # Process remaining chunks
        if chunk_buffer:
            temp_metadata = [
                {'chunk_id': i, 'timestamp': time.time()}
                for i in range(len(chunk_buffer))
            ]
            chunk_results = await self._process_audio_chunks(chunk_buffer, temp_metadata)
            results.extend(chunk_results)
        
        return results
    
    async def _process_audio_chunks(self, chunks: List[torch.Tensor], 
                                  metadata: List[Dict]) -> List[Dict[str, Any]]:
        """Process a list of audio chunks"""
        if not self.feature_extractor:
            raise ValueError("No feature extractor available for chunk processing")
        
        results = []
        
        for chunk, meta in zip(chunks, metadata):
            try:
                # Extract features from chunk
                from utils.config_loader import settings
                sr = settings.audio.sample_rate
                
                features = await self.feature_extractor.extract_features(chunk, sr)
                
                result = {
                    'chunk_id': meta.get('chunk_id', 0),
                    'timestamp': meta.get('timestamp', time.time()),
                    'status': 'success',
                    'audio_duration_s': len(chunk) / sr
                }
                
                # Add physics features
                if 'physics' in features:
                    physics = features['physics']
                    for key, value in physics.items():
                        if torch.is_tensor(value) and value.numel() == 1:
                            result[f'physics_{key}'] = value.item()
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                results.append({
                    'chunk_id': meta.get('chunk_id', 0),
                    'timestamp': meta.get('timestamp', time.time()),
                    'status': 'error',
                    'error': str(e)
                })
        
        return results

    def _organize_by_length(self, file_paths: List[Path], metadata: List[Dict]) -> List[Tuple[List[Path], List[Dict]]]:
        """Organize files by length for efficient batching."""
        # For now, return a single group (length bucketing can be implemented later)
        # In a full implementation, this would group files by similar audio duration
        return [(file_paths, metadata)]

class StreamingProcessor:
    """Real-time streaming audio processor"""
    
    def __init__(self, 
                 chunk_size: int = 16000,  # 1 second at 16kHz
                 overlap_ratio: float = 0.25,
                 feature_extractor: Optional[ComprehensiveFeatureExtractor] = None):
        
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap_ratio)
        self.feature_extractor = feature_extractor
        self.buffer = torch.tensor([])
        self.chunk_counter = 0
        
    async def process_stream(self, audio_stream) -> Dict[str, Any]:
        """
        Process audio stream in real-time chunks
        
        Args:
            audio_stream: Async iterator yielding audio data
            
        Yields:
            Feature extraction results for each chunk
        """
        async for audio_data in audio_stream:
            # Convert to tensor if needed
            if isinstance(audio_data, (list, np.ndarray)):
                audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
            else:
                audio_tensor = audio_data
            
            # Add to buffer
            self.buffer = torch.cat([self.buffer, audio_tensor])
            
            # Process complete chunks
            while len(self.buffer) >= self.chunk_size:
                # Extract chunk with overlap
                chunk = self.buffer[:self.chunk_size]
                
                # Process chunk
                try:
                    chunk_result = await self._process_chunk(chunk)
                    yield chunk_result
                except Exception as e:
                    logger.error(f"Streaming chunk processing failed: {e}")
                    yield {
                        'chunk_id': self.chunk_counter,
                        'status': 'error',
                        'error': str(e),
                        'timestamp': time.time()
                    }
                
                # Move buffer forward (with overlap)
                advance_size = self.chunk_size - self.overlap_size
                self.buffer = self.buffer[advance_size:]
                self.chunk_counter += 1
    
    async def _process_chunk(self, chunk: torch.Tensor) -> Dict[str, Any]:
        """Process a single audio chunk"""
        if self.feature_extractor is None:
            raise ValueError("No feature extractor available")
        
        # Extract features
        from utils.config_loader import settings
        sr = settings.audio.sample_rate
        
        features = await self.feature_extractor.extract_features(chunk, sr)
        
        # Prepare result
        result = {
            'chunk_id': self.chunk_counter,
            'status': 'success',
            'timestamp': time.time(),
            'audio_duration_s': len(chunk) / sr,
            'chunk_size': len(chunk)
        }
        
        # Add key features
        if 'physics' in features:
            physics = features['physics']
            for key, value in physics.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    result[f'physics_{key}'] = value.item()
        
        if 'hubert_sequence' in features:
            hubert = features['hubert_sequence']
            result['hubert_seq_len'] = hubert.shape[0]
        
        return result
    
    def reset_buffer(self):
        """Reset the internal buffer"""
        self.buffer = torch.tensor([])
        self.chunk_counter = 0

if __name__ == "__main__":
    # Test batch processor
    print("Testing Batch Processing Components")
    print("=" * 50)
    
    async def test_batch_processor():
        try:
            # Create test configuration
            config = BatchConfig(batch_size=4, enable_length_bucketing=True)
            
            # Initialize batch processor (without feature extractor for testing)
            processor = BatchProcessor(config=config)
            
            print("✓ Batch processor initialized")
            
            # Test statistics
            stats = processor.get_batch_stats()
            print(f"✓ Initial stats: {stats}")
            
            print("Batch processing components ready!")
            
        except Exception as e:
            print(f"✗ Batch processor test failed: {e}")
    
    # Run test
    asyncio.run(test_batch_processor()) 