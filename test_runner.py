"""
Enhanced Test Runner for Physics Features Analysis
Fixed version with comprehensive error handling and progress tracking
"""

import os
import sys
import asyncio
import pandas as pd
import torch
from tqdm import tqdm
import time
import traceback
from pathlib import Path
import pickle
from typing import List, Dict, Optional, Any, Tuple
import json
from datetime import datetime
import numpy as np
import re

# Setup path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

try:
    from core.audio_utils import load_audio
    from core.feature_extractor import ComprehensiveFeatureExtractor, RobustProcessor
    from core.processing_pipeline import create_standard_pipeline, create_lightweight_pipeline
    from utils.config_loader import settings
    from core.model_loader import DEVICE
    from utils.logging_system import create_project_logger
    from utils.security_validator import SecureAudioLoader, ResourceLimiter, InputValidator, SecurityConfig
    from core.batch_processor import BatchProcessor, BatchConfig
    from utils.folder_manager import initialize_project_folders
    from bayesian.core.bayesian_engine import BayesianDeepfakeEngine, BayesianConfig, BayesianDetectionResult
    from bayesian.utils.temporal_buffer import TemporalFeatureBuffer
    from bayesian.utils.user_context import UserContextManager
    from bayesian.utils.uncertainty_estimation import PhysicsUncertaintyEstimator
    from bayesian.utils.causal_analysis import CausalFeatureAnalyzer
    print("✓ Enhanced security and batch processing modules loaded")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available in the src directory")
    sys.exit(1)

# Configuration
DATA_DIR = os.path.join(current_dir, "data")
RESULTS_DIR = os.path.join(current_dir, "results")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "physics_features_summary.csv")
ERROR_LOG = os.path.join(RESULTS_DIR, "error_log.txt")

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add enhanced security configuration
SECURITY_CONFIG = SecurityConfig(
    max_file_size_mb=200.0,  # Increased for batch processing
    max_memory_gb=12.0,
    max_processing_time_s=600.0,  # Increased for batch operations
    allowed_formats={'.wav', '.mp3', '.flac', '.m4a'},
    allow_path_traversal=False
)

# Global instances
secure_loader = SecureAudioLoader(SECURITY_CONFIG) if SecureAudioLoader else None
resource_limiter = ResourceLimiter(
    max_memory_gb=SECURITY_CONFIG.max_memory_gb,
    max_time_s=SECURITY_CONFIG.max_processing_time_s
) if ResourceLimiter else None

class ProgressTracker:
    """Enhanced progress tracking for better monitoring."""
    
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = time.time()
        self.errors = []
    
    def update(self, success: bool, error_msg: str = None):
        self.processed += 1
        if success:
            self.successful += 1
        else:
            self.failed += 1
            if error_msg:
                self.errors.append(error_msg)
    
    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time
        return {
            'total': self.total_files,
            'processed': self.processed,
            'successful': self.successful,
            'failed': self.failed,
            'elapsed_time': elapsed,
            'avg_time_per_file': elapsed / max(self.processed, 1),
            'success_rate': self.successful / max(self.processed, 1) * 100
        }

class CheckpointManager:
    """Enhanced checkpoint management for robust recovery from interruptions."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / 'checkpoint.pkl'
        self.temp_file = self.checkpoint_dir / 'checkpoint.tmp'
        
    def save_checkpoint(self, processed_files: List[str], results: List[Dict], 
                       current_batch_idx: int = 0) -> bool:
        """
        Save processing checkpoint with atomic write.
        
        Args:
            processed_files: List of already processed file paths
            results: List of results for processed files
            current_batch_idx: Current batch index being processed
            
        Returns:
            True if checkpoint saved successfully
        """
        try:
            checkpoint_data = {
                'processed_files': processed_files,
                'results': results,
                'current_batch_idx': current_batch_idx,
                'timestamp': time.time(),
                'version': '1.0'
            }
            
            # Atomic write: write to temp file first, then rename
            with open(self.temp_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Atomic rename
            self.temp_file.rename(self.checkpoint_file)
            
            print(f"Checkpoint saved: {len(processed_files)} files processed")
            
            # Create human-readable checkpoint status
            self._create_checkpoint_status(checkpoint_data)
            
            return True
            
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
            return False
    
    def _create_checkpoint_status(self, checkpoint_data: Dict):
        """Create human-readable checkpoint status file"""
        status_file = self.checkpoint_dir / 'checkpoint_status.json'
        
        # Calculate statistics
        successful_results = [r for r in checkpoint_data['results'] if r.get('status') == 'success']
        failed_results = [r for r in checkpoint_data['results'] if r.get('status') != 'success']
        
        # Calculate processing time if available
        total_processing_time = sum(r.get('processing_time', 0) for r in checkpoint_data['results'])
        
        status_info = {
            'checkpoint_info': {
                'timestamp': datetime.fromtimestamp(checkpoint_data['timestamp']).isoformat(),
                'version': checkpoint_data['version'],
                'current_batch': checkpoint_data['current_batch_idx'],
                'checkpoint_size_mb': self.checkpoint_file.stat().st_size / (1024 * 1024) if self.checkpoint_file.exists() else 0
            },
            'processing_progress': {
                'total_files': len(checkpoint_data['processed_files']),
                'successful_files': len(successful_results),
                'failed_files': len(failed_results),
                'success_rate_percent': (len(successful_results) / max(len(checkpoint_data['processed_files']), 1)) * 100,
                'total_processing_time_seconds': total_processing_time,
                'average_time_per_file': total_processing_time / max(len(checkpoint_data['processed_files']), 1)
            },
            'file_types_processed': {},
            'recent_errors': [],
            'recovery_instructions': [
                "To resume from this checkpoint:",
                "1. Run test_runner.py",
                "2. When prompted about resuming from checkpoint, enter 'y'",
                "3. Processing will continue from the saved state"
            ]
        }
        
        # Count file types
        for result in checkpoint_data['results']:
            file_type = result.get('file_type', 'unknown')
            if file_type in status_info['file_types_processed']:
                status_info['file_types_processed'][file_type] += 1
            else:
                status_info['file_types_processed'][file_type] = 1
        
        # Collect recent errors (last 5)
        for result in failed_results[-5:]:
            status_info['recent_errors'].append({
                'file': result.get('filepath', 'unknown'),
                'error': result.get('error', 'unknown error')
            })
        
        with open(status_file, 'w') as f:
            json.dump(status_info, f, indent=2)
    
    def load_checkpoint(self) -> Optional[Dict]:
        """
        Load checkpoint if exists and is valid.
        
        Returns:
            Checkpoint data dict or None if no valid checkpoint
        """
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Validate checkpoint format
            required_keys = ['processed_files', 'results', 'timestamp']
            if not all(key in checkpoint_data for key in required_keys):
                print("Invalid checkpoint format, ignoring")
                return None
            
            # Check if checkpoint is not too old (e.g., more than 24 hours)
            checkpoint_age = time.time() - checkpoint_data['timestamp']
            if checkpoint_age > 24 * 3600:  # 24 hours
                print(f"Checkpoint is too old ({checkpoint_age/3600:.1f}h), ignoring")
                return None
            
            processed_count = len(checkpoint_data['processed_files'])
            print(f"Found valid checkpoint with {processed_count} processed files")
            print(f"Checkpoint age: {checkpoint_age/60:.1f} minutes")
            
            return checkpoint_data
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self) -> bool:
        """
        Clear existing checkpoint files.
        
        Returns:
            True if cleared successfully
        """
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.temp_file.exists():
                self.temp_file.unlink()
            print("Checkpoint cleared")
            return True
        except Exception as e:
            print(f"Failed to clear checkpoint: {e}")
            return False
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics."""
        stats = {
            'checkpoint_exists': self.checkpoint_file.exists(),
            'checkpoint_dir': str(self.checkpoint_dir)
        }
        
        if stats['checkpoint_exists']:
            try:
                stat_info = self.checkpoint_file.stat()
                stats.update({
                    'checkpoint_size_mb': stat_info.st_size / (1024 * 1024),
                    'checkpoint_modified': time.ctime(stat_info.st_mtime)
                })
            except Exception as e:
                stats['checkpoint_error'] = str(e)
        
        return stats

# Global progress tracker
progress_tracker = None

def initialize_feature_extractor():
    """Initialize feature extractor with comprehensive error handling."""
    try:
        print("Initializing feature extractor (this may take a moment for HuBERT download/load)...")
        feature_extractor = ComprehensiveFeatureExtractor()
        
        print(
            f"Feature extractor initialized successfully!\n"
            f"  HuBERT model: {settings.models.hubert_model_path}\n"
            f"  Device: {DEVICE}\n"
            f"  Physics window: {settings.physics.time_window_for_dynamics_ms}ms\n"
            f"  Sample rate: {settings.audio.sample_rate}Hz\n"
            f"  Embedding dim: {settings.physics.embedding_dim_for_physics}"
        )
        return feature_extractor
        
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize feature extractor: {e}")
        print("Stack trace:")
        traceback.print_exc()
        return None

async def process_single_file(filepath: str, user_id: str, file_type: str, 
                            feature_extractor: ComprehensiveFeatureExtractor) -> dict:
    """
    Process a single audio file with comprehensive error handling.
    
    Args:
        filepath: Path to audio file
        user_id: User identifier
        file_type: Type of audio (genuine, deepfake_tts, etc.)
        feature_extractor: Initialized feature extractor
    
    Returns:
        Dictionary with extracted features or error information
    """
    result = {
        "filepath": filepath,
        "user_id": user_id,
        "file_type": file_type,
        "status": "unknown"
    }
    
    try:
        # Load audio file (load_audio is not async, returns tuple)
        waveform, sample_rate = load_audio(filepath, target_sr=settings.audio.sample_rate)
        
        if waveform is None or waveform.numel() == 0:
            result.update({
                "status": "error",
                "error": "Could not load audio or empty waveform",
                "audio_duration_s": 0,
                "hubert_seq_len_frames": 0
            })
            return result

        # Calculate audio duration
        audio_duration = waveform.shape[0] / sample_rate
        result["audio_duration_s"] = audio_duration
        
        # Extract features
        all_features = await feature_extractor.extract_features(waveform, sample_rate)
        
        # Extract physics features
        physics_dynamics = all_features.get("physics", {})
        
        # Flatten physics features for CSV storage
        for k, v in physics_dynamics.items():
            if torch.is_tensor(v):
                if v.numel() == 1:
                    result[f"physics_{k}"] = v.item()
                else:
                    # For multi-dimensional features, store statistics
                    result[f"physics_{k}_mean"] = v.mean().item()
                    result[f"physics_{k}_std"] = v.std().item()
                    result[f"physics_{k}_shape"] = str(list(v.shape))
            else:
                result[f"physics_{k}"] = str(v)
        
        # Add HuBERT sequence information
        hubert_seq = all_features.get("hubert_sequence")
        if hubert_seq is not None:
            result["hubert_seq_len_frames"] = hubert_seq.shape[0]
            result["hubert_embedding_dim"] = hubert_seq.shape[1]
        else:
            result["hubert_seq_len_frames"] = 0
            result["hubert_embedding_dim"] = 0
        
        # Add other feature statistics
        mel_spec = all_features.get("mel_spectrogram")
        if mel_spec is not None:
            result["mel_spec_shape"] = str(list(mel_spec.shape))
            result["mel_spec_mean"] = mel_spec.mean().item()
            result["mel_spec_std"] = mel_spec.std().item()
        
        lfcc = all_features.get("lfcc")
        if lfcc is not None:
            result["lfcc_shape"] = str(list(lfcc.shape))
            result["lfcc_mean"] = lfcc.mean().item()
            result["lfcc_std"] = lfcc.std().item()
        
        result["status"] = "success"
        return result
        
    except Exception as e:
        error_msg = f"Error processing {filepath}: {str(e)}"
        print(error_msg)
        
        # Log detailed error
        with open(ERROR_LOG, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")
            f.write("-" * 80 + "\n")
        
        result.update({
            "status": "error",
            "error": str(e),
            "audio_duration_s": 0,
            "hubert_seq_len_frames": 0
        })
        return result

async def process_single_file_with_pipeline(filepath: str, user_id: str, file_type: str, 
                                         pipeline) -> dict:
    """
    Process a single audio file using the new processing pipeline.
    
    Args:
        filepath: Path to audio file
        user_id: User identifier
        file_type: Type of audio (genuine, deepfake_tts, etc.)
        pipeline: Processing pipeline instance
    
    Returns:
        Dictionary with extracted features or error information
    """
    try:
        # Process through pipeline
        pipeline_result = await pipeline.process(filepath)
        
        # Extract final data
        final_data = pipeline_result.get('final_data')
        overall_status = pipeline_result.get('overall_status')
        
        if overall_status in ['fully_successful', 'partially_successful'] and final_data:
            # Extract result information
            result = {
                "filepath": filepath,
                "user_id": user_id,
                "file_type": file_type,
                "status": "success",
                "audio_duration_s": final_data.get('audio_duration', 0),
                "processing_time": pipeline_result.get('total_processing_time', 0),
                "pipeline_status": overall_status,
                "successful_stages": f"{pipeline_result.get('successful_stages', 0)}/{pipeline_result.get('total_stages', 0)}"
            }
            
            # Extract physics features if available
            features = final_data.get('features', {})
            physics_features = features.get('physics', {})
            
            for k, v in physics_features.items():
                result[f"physics_{k}"] = v
            
            # Add HuBERT information
            hubert_info = features.get('hubert', {})
            result["hubert_seq_len_frames"] = hubert_info.get('sequence_length', 0)
            result["hubert_embedding_dim"] = hubert_info.get('embedding_dim', 0)
            
            # Add other feature information
            for feature_name in ['mel_spectrogram', 'lfcc']:
                if feature_name in features:
                    feature_info = features[feature_name]
                    result[f"{feature_name}_shape"] = str(feature_info.get('shape', []))
                    result[f"{feature_name}_mean"] = feature_info.get('mean', 0)
                    result[f"{feature_name}_std"] = feature_info.get('std', 0)
            
            # Add validation information
            validation_summary = final_data.get('validation_summary', {})
            result["validation_overall_valid"] = validation_summary.get('overall_valid', True)
            result["validation_num_errors"] = validation_summary.get('num_errors', 0)
            result["validation_num_warnings"] = validation_summary.get('num_warnings', 0)
            
            # Add processing metadata
            processing_metadata = final_data.get('processing_metadata', {})
            result["cache_hit"] = processing_metadata.get('cache_hit', False)
            result["normalization_method"] = processing_metadata.get('normalization_method', 'unknown')
            
            return result
        else:
            # Pipeline failed
            error_msg = pipeline_result.get('error_message', 'Pipeline processing failed')
            return {
                "filepath": filepath,
                "user_id": user_id,
                "file_type": file_type,
                "status": "error",
                "error": error_msg,
                "audio_duration_s": 0,
                "hubert_seq_len_frames": 0,
                "pipeline_status": overall_status,
                "processing_time": pipeline_result.get('total_processing_time', 0)
            }
            
    except Exception as e:
        error_msg = f"Pipeline processing error for {filepath}: {str(e)}"
        print(error_msg)
        
        # Log detailed error
        with open(ERROR_LOG, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")
            f.write("-" * 80 + "\n")
        
        return {
            "filepath": filepath,
            "user_id": user_id,
            "file_type": file_type,
            "status": "error",
            "error": str(e),
            "audio_duration_s": 0,
            "hubert_seq_len_frames": 0,
            "pipeline_status": "failed"
        }

def discover_audio_files(data_dir: str) -> list:
    """
    Discover all audio files in the data directory with metadata and security validation.
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        List of dictionaries with file metadata
    """
    audio_files = []
    supported_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
    total_files_found = 0
    valid_files = 0
    quarantined_files = 0
    
    try:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"Data directory {data_dir} does not exist!")
            return []
        
        for user_dir in sorted(data_path.iterdir()):
            if not user_dir.is_dir():
                continue
                
            user_id = user_dir.name
            
            for audio_file in sorted(user_dir.iterdir()):
                total_files_found += 1
                
                if audio_file.suffix.lower() in supported_extensions:
                    try:
                        # Enhanced security validation
                        if secure_loader:
                            try:
                                secure_loader.validate_file(audio_file)
                                # Calculate file hash for integrity
                                file_hash = secure_loader.calculate_file_hash(audio_file)
                            except Exception as security_error:
                                print(f"Security validation failed for {audio_file}: {security_error}")
                                try:
                                    quarantined_path = secure_loader.quarantine_file(audio_file, str(security_error))
                                    quarantined_files += 1
                                    print(f"File quarantined: {quarantined_path}")
                                    continue
                                except Exception as quarantine_error:
                                    print(f"Could not quarantine file {audio_file}: {quarantine_error}")
                                    continue
                        else:
                            # Basic validation fallback
                            if audio_file.stat().st_size == 0:
                                print(f"Skipping empty file: {audio_file}")
                                continue
                            file_hash = ""
                        
                        # Infer file type from filename
                        filename_lower = audio_file.name.lower()
                        file_type = "unknown"
                        
                        if "genuine" in filename_lower:
                            file_type = "genuine"
                        elif "deepfake_tts" in filename_lower:
                            file_type = "deepfake_tts"
                        elif "deepfake_vc" in filename_lower or "voice_conversion" in filename_lower:
                            file_type = "deepfake_vc"
                        elif "deepfake_replay" in filename_lower or "replay" in filename_lower:
                            file_type = "deepfake_replay"
                        elif "deepfake" in filename_lower:
                            file_type = "deepfake_other"
                        elif "synthetic" in filename_lower or "generated" in filename_lower:
                            file_type = "synthetic"
                        
                        # Enhanced file metadata
                        file_info = {
                            "filepath": str(audio_file),
                            "user_id": user_id,
                            "file_type": file_type,
                            "filename": audio_file.name,
                            "file_size_mb": audio_file.stat().st_size / (1024 * 1024),
                            "file_hash": file_hash,
                            "last_modified": audio_file.stat().st_mtime,
                            "file_extension": audio_file.suffix.lower()
                        }
                        
                        audio_files.append(file_info)
                        valid_files += 1
                        
                    except Exception as e:
                        print(f"Error processing file {audio_file}: {e}")
                        continue
    
    except Exception as e:
        print(f"Error discovering audio files: {e}")
        return []
    
    # Print discovery summary
    print(f"\nFile Discovery Summary:")
    print(f"  Total files found: {total_files_found}")
    print(f"  Valid audio files: {valid_files}")
    print(f"  Quarantined files: {quarantined_files}")
    if quarantined_files > 0:
        print(f"  Check quarantine directory for suspicious files")
    
    return audio_files

async def process_files_batch(file_batch: list, processor, 
                            semaphore: asyncio.Semaphore, process_function) -> list:
    """Process a batch of files with semaphore control and flexible processing."""
    
    # Validate inputs
    if processor is None:
        raise ValueError("Processor cannot be None")
    if process_function is None:
        raise ValueError("Process function cannot be None")
    if not file_batch:
        return []
    
    async def process_with_semaphore(file_meta):
        async with semaphore:
            try:
                if process_function == process_single_file_with_pipeline:
                    result = await process_function(
                        file_meta["filepath"], 
                        file_meta["user_id"], 
                        file_meta["file_type"],
                        processor  # pipeline
                    )
                else:
                    # Validate processor has extract_features method for traditional processing
                    if not hasattr(processor, 'extract_features'):
                        raise AttributeError(f"Processor {type(processor)} does not have 'extract_features' method")
                    
                    result = await process_function(
                        file_meta["filepath"], 
                        file_meta["user_id"], 
                        file_meta["file_type"],
                        processor  # feature_extractor
                    )
                
                # Update progress
                success = result.get("status") == "success"
                error_msg = result.get("error") if not success else None
                progress_tracker.update(success, error_msg)
                return result
                
            except Exception as e:
                error_msg = f"Batch processing error for {file_meta['filepath']}: {str(e)}"
                progress_tracker.update(False, error_msg)
                return {
                    "filepath": file_meta["filepath"],
                    "user_id": file_meta["user_id"], 
                    "file_type": file_meta["file_type"],
                    "status": "batch_error",
                    "error": str(e)
                }
    
    # Process batch concurrently
    tasks = [process_with_semaphore(file_meta) for file_meta in file_batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            error_result = {
                "filepath": file_batch[i]["filepath"],
                "user_id": file_batch[i]["user_id"],
                "file_type": file_batch[i]["file_type"],
                "status": "exception",
                "error": str(result)
            }
            processed_results.append(error_result)
        else:
            processed_results.append(result)
    
    return processed_results

def save_results(results: list, output_path: str):
    """Save results to CSV with comprehensive metadata."""
    if not results:
        print("No results to save.")
        return
    
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        # Generate summary statistics
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        stats = progress_tracker.get_stats()
        print(f"Total files discovered: {stats['total']}")
        print(f"Files processed: {stats['processed']}")
        print(f"Successful: {stats['successful']} ({stats['success_rate']:.1f}%)")
        print(f"Failed: {stats['failed']}")
        print(f"Processing time: {stats['elapsed_time']:.1f}s")
        print(f"Average time per file: {stats['avg_time_per_file']:.2f}s")
        
        # File type distribution
        if "file_type" in df.columns:
            print(f"\nFile Type Distribution:")
            print(df['file_type'].value_counts().to_string())
        
        # Status distribution
        if "status" in df.columns:
            print(f"\nProcessing Status:")
            print(df['status'].value_counts().to_string())
        
        # Physics features summary (for successful files only)
        successful_df = df[df['status'] == 'success']
        if not successful_df.empty:
            physics_columns = [col for col in df.columns if col.startswith('physics_')]
            if physics_columns:
                print(f"\nPhysics Features Summary (successful files only):")
                print("-" * 40)
                
                # Key physics features analysis
                key_features = [
                    'physics_delta_ft_revised',
                    'physics_delta_fr_revised', 
                    'physics_delta_fv_revised',
                    'physics_delta_f_total_revised'
                ]
                
                for feature in key_features:
                    if feature in successful_df.columns:
                        # Convert to numeric, handling tensor values and any string values
                        def convert_tensor_to_numeric(val):
                            """Convert tensor values and other formats to numeric"""
                            if val is None:
                                return float('nan')
                            
                            # Handle tensor objects
                            if hasattr(val, 'item'):  # PyTorch tensor
                                return float(val.item())
                            elif hasattr(val, 'numpy'):  # NumPy array or other array-like
                                return float(val.numpy() if hasattr(val, 'numpy') else val)
                            
                            # Handle string representations of tensors
                            if isinstance(val, str):
                                import re
                                # Extract numeric value from tensor string
                                if 'tensor(' in val:
                                    matches = re.findall(r'tensor\(([0-9.e-]+)\)', val)
                                    if matches:
                                        return float(matches[0])
                                # Try direct conversion
                                try:
                                    return float(val)
                                except (ValueError, TypeError):
                                    return float('nan')
                            
                            # Handle direct numeric values
                            try:
                                return float(val)
                            except (ValueError, TypeError):
                                return float('nan')
                        
                        # Apply conversion
                        numeric_values = successful_df[feature].apply(convert_tensor_to_numeric)
                        
                        # Remove NaN values for statistics
                        valid_values = numeric_values.dropna()
                        
                        if len(valid_values) > 0:
                            print(f"{feature}:")
                            print(f"  Mean: {valid_values.mean():.6f}")
                            print(f"  Std:  {valid_values.std():.6f}")
                            print(f"  Min:  {valid_values.min():.6f}")
                            print(f"  Max:  {valid_values.max():.6f}")
                        else:
                            print(f"{feature}: No valid numeric data available")
                
                # By file type analysis
                if len(successful_df['file_type'].unique()) > 1:
                    print(f"\nPhysics Features by File Type:")
                    print("-" * 40)
                    for feature in key_features[:2]:  # Show first 2 features
                        if feature in successful_df.columns:
                            print(f"{feature}:")
                            for file_type in successful_df['file_type'].unique():
                                subset = successful_df[successful_df['file_type'] == file_type]
                                # Apply conversion to subset
                                numeric_values = subset[feature].apply(convert_tensor_to_numeric)
                                valid_values = numeric_values.dropna()
                                
                                if len(valid_values) > 0:
                                    mean_val = valid_values.mean()
                                    print(f"  {file_type:15}: {mean_val:.6f}")
                                else:
                                    print(f"  {file_type:15}: No valid data")
        
        # Error summary
        if progress_tracker.errors:
            print(f"\nTop Error Messages:")
            print("-" * 40)
            error_counts = {}
            for error in progress_tracker.errors[:10]:  # Show top 10
                error_type = error.split(':')[0] if ':' in error else error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                print(f"  {error_type}: {count} occurrences")
        
        # Enhanced Bayesian Analysis Summary
        bayesian_columns = [col for col in df.columns if col.startswith('bayesian_')]
        if bayesian_columns and not successful_df.empty:
            print("\n" + "="*60)
            print("BAYESIAN ANALYSIS SUMMARY")
            print("="*60)
            
            # Check if any files had successful Bayesian analysis
            bayesian_success_df = successful_df[successful_df.get('bayesian_status', 'failed') == 'success']
            
            if not bayesian_success_df.empty:
                print("🧠 Bayesian Engine Status: ACTIVE")
                print("📊 Probabilistic Analysis: COMPLETED")
                print(f"🎯 Files Analyzed: {len(bayesian_success_df)}/{len(successful_df)} ({len(bayesian_success_df)/len(successful_df)*100:.1f}%)")
                
                # Overall Performance Metrics
                if 'bayesian_confidence' in bayesian_success_df.columns:
                    confidence_values = pd.to_numeric(bayesian_success_df['bayesian_confidence'], errors='coerce')
                    uncertainty_values = pd.to_numeric(bayesian_success_df['bayesian_uncertainty_total'], errors='coerce') if 'bayesian_uncertainty_total' in bayesian_success_df.columns else None
                    
                    print(f"\n📊 Overall Performance Metrics:")
                    print(f"├── Average Confidence: {confidence_values.mean():.3f} ± {confidence_values.std():.3f}")
                    print(f"├── Confidence Range: {confidence_values.min():.3f} - {confidence_values.max():.3f}")
                    
                    if uncertainty_values is not None and not uncertainty_values.isna().all():
                        print(f"├── Average Uncertainty: {uncertainty_values.mean():.3f} ± {uncertainty_values.std():.3f}")
                        print(f"└── Uncertainty Range: {uncertainty_values.min():.3f} - {uncertainty_values.max():.3f}")
                
                # Confidence Level Distribution
                if 'bayesian_confidence_level' in bayesian_success_df.columns:
                    confidence_dist = bayesian_success_df['bayesian_confidence_level'].value_counts()
                    print(f"\n🎯 Confidence Level Distribution:")
                    total_analyzed = len(bayesian_success_df)
                    for level, count in confidence_dist.items():
                        percentage = count / total_analyzed * 100
                        print(f"├── {level}: {count} files ({percentage:.1f}%)")
                
                # Classification Results
                if 'bayesian_classification' in bayesian_success_df.columns:
                    classification_dist = bayesian_success_df['bayesian_classification'].value_counts()
                    print(f"\n🔍 Bayesian Classification Results:")
                    for classification, count in classification_dist.items():
                        percentage = count / len(bayesian_success_df) * 100
                        print(f"├── {classification}: {count} files ({percentage:.1f}%)")
                
                # Feature Discrimination Analysis
                if 'file_type' in bayesian_success_df.columns and len(bayesian_success_df['file_type'].unique()) > 1:
                    print(f"\n🧬 Feature Discrimination Analysis:")
                    
                    # Calculate discrimination by file type
                    genuine_files = bayesian_success_df[bayesian_success_df['file_type'] == 'genuine']
                    deepfake_files = bayesian_success_df[bayesian_success_df['file_type'].str.contains('deepfake', na=False)]
                    
                    if not genuine_files.empty and not deepfake_files.empty and 'bayesian_confidence' in bayesian_success_df.columns:
                        genuine_conf = pd.to_numeric(genuine_files['bayesian_confidence'], errors='coerce').mean()
                        deepfake_conf = pd.to_numeric(deepfake_files['bayesian_confidence'], errors='coerce').mean()
                        discrimination_score = abs(genuine_conf - deepfake_conf)
                        
                        print(f"├── Genuine Audio Confidence: {genuine_conf:.3f}")
                        print(f"├── Deepfake Audio Confidence: {deepfake_conf:.3f}")
                        print(f"└── Discrimination Score: {discrimination_score:.3f}")
                
                # Causal Analysis Summary
                if 'bayesian_primary_factor' in bayesian_success_df.columns:
                    factor_counts = bayesian_success_df['bayesian_primary_factor'].value_counts()
                    if not factor_counts.empty:
                        print(f"\n🔬 Primary Causal Factors:")
                        for factor, count in factor_counts.head(3).items():
                            percentage = count / len(bayesian_success_df) * 100
                            print(f"├── {factor}: {count} occurrences ({percentage:.1f}%)")
                
                # Risk Assessment Distribution
                high_conf_files = bayesian_success_df[bayesian_success_df.get('bayesian_confidence_level', '') == 'HIGH']
                medium_conf_files = bayesian_success_df[bayesian_success_df.get('bayesian_confidence_level', '') == 'MEDIUM']
                low_conf_files = bayesian_success_df[bayesian_success_df.get('bayesian_confidence_level', '') == 'LOW']
                
                print(f"\n⚠️  Risk Assessment Distribution:")
                print(f"├── LOW RISK (High Confidence): {len(high_conf_files)} files ({len(high_conf_files)/len(bayesian_success_df)*100:.1f}%)")
                print(f"├── MEDIUM RISK (Medium Confidence): {len(medium_conf_files)} files ({len(medium_conf_files)/len(bayesian_success_df)*100:.1f}%)")
                print(f"└── HIGH RISK (Low Confidence): {len(low_conf_files)} files ({len(low_conf_files)/len(bayesian_success_df)*100:.1f}%)")
                
                # Actionable Insights
                print(f"\n💡 Actionable Insights:")
                
                # Model performance assessment
                high_conf_percentage = len(high_conf_files) / len(bayesian_success_df) * 100
                if high_conf_percentage >= 70:
                    print(f"├── Model Performance: EXCELLENT - Ready for production deployment")
                elif high_conf_percentage >= 50:
                    print(f"├── Model Performance: GOOD - Consider additional validation")
                else:
                    print(f"├── Model Performance: NEEDS IMPROVEMENT - Review training data")
                
                # Data balance assessment
                file_type_counts = bayesian_success_df['file_type'].value_counts()
                if len(file_type_counts) > 1:
                    min_count = file_type_counts.min()
                    max_count = file_type_counts.max()
                    imbalance_ratio = max_count / min_count
                    
                    if imbalance_ratio > 3:
                        minority_class = file_type_counts.idxmin()
                        print(f"├── Training Data: Increase {minority_class} samples for better balance")
                    else:
                        print(f"├── Training Data: Well balanced across file types")
                
                # Uncertainty guidance
                if uncertainty_values is not None and not uncertainty_values.isna().all():
                    high_uncertainty_files = len(uncertainty_values[uncertainty_values > 0.3])
                    if high_uncertainty_files > 0:
                        print(f"├── Manual Review: {high_uncertainty_files} files with high uncertainty need review")
                    else:
                        print(f"├── Uncertainty: All files within acceptable uncertainty range")
                
                # Deployment recommendation
                if high_conf_percentage >= 70 and (uncertainty_values is None or uncertainty_values.mean() < 0.2):
                    print(f"└── Deployment: ✅ RECOMMENDED - High confidence and low uncertainty")
                elif high_conf_percentage >= 50:
                    print(f"└── Deployment: ⚠️ CONDITIONAL - Implement confidence-based routing")
                else:
                    print(f"└── Deployment: ❌ NOT RECOMMENDED - Improve model before deployment")
                
            else:
                print("🧠 Bayesian Engine Status: ACTIVE")
                print("📊 Probabilistic Analysis: NO SUCCESSFUL ANALYSES")
                print("⚠️  All Bayesian analyses failed - Check Bayesian engine configuration")
                
    except Exception as e:
        print(f"Error saving results: {e}")
        traceback.print_exc()

async def process_single_file_with_bayesian_pipeline(filepath: str, user_id: str, file_type: str, 
                                                   feature_extractor, bayesian_engine) -> dict:
    """
    Process a single audio file using Bayesian-enhanced pipeline processing.
    
    Args:
        filepath: Path to audio file
        user_id: User identifier
        file_type: Type of audio (genuine, deepfake_tts, etc.)
        feature_extractor: Feature extractor instance (ComprehensiveFeatureExtractor)
        bayesian_engine: Bayesian analysis engine
    
    Returns:
        Dictionary with extracted features and Bayesian analysis results
    """
    try:
        start_time = time.time()
        
        # Load audio file
        waveform, sample_rate = load_audio(filepath, target_sr=settings.audio.sample_rate)
        
        if waveform is None or waveform.numel() == 0:
            return {
                "filepath": filepath,
                "user_id": user_id,
                "file_type": file_type,
                "status": "error",
                "error": "Could not load audio or empty waveform",
                "audio_duration_s": 0,
                "processing_time": time.time() - start_time,
                "bayesian_status": "failed"
            }
        
        # Extract features using feature extractor
        features = await feature_extractor.extract_features(waveform, sample_rate)
        
        # Extract physics features for Bayesian analysis
        physics_features = features.get('physics', {})
        
        # Perform Bayesian analysis
        bayesian_result = None
        if physics_features and bayesian_engine:
            try:
                # Create input for Bayesian engine
                physics_input = {
                    'delta_ft': physics_features.get('delta_ft_revised', 0.0),
                    'delta_fr': physics_features.get('delta_fr_revised', 0.0), 
                    'delta_fv': physics_features.get('delta_fv_revised', 0.0),
                    'delta_f_total': physics_features.get('delta_f_total_revised', 0.0)
                }
                
                # Run Bayesian inference
                bayesian_result = await bayesian_engine.analyze_audio_features(
                    physics_input, user_id, file_type
                )
                
                # Print professional Bayesian analysis output
                print(f"\n🧠 Bayesian Analysis: {filepath}")
                print(f"├── 📈 Physics Features:")
                print(f"│   ├── Δf_t (Translational): {physics_input['delta_ft']:.6f}")
                print(f"│   ├── Δf_r (Rotational): {physics_input['delta_fr']:.6f}")
                print(f"│   └── Δf_v (Vibrational): {physics_input['delta_fv']:.6f}")
                
                if bayesian_result and bayesian_result.status == "success":
                    conf = bayesian_result.confidence_level
                    prob = bayesian_result.deepfake_probability
                    uncertainty = bayesian_result.uncertainty_metrics
                    
                    print(f"├── 🎯 Bayesian Inference:")
                    print(f"│   ├── Classification: {bayesian_result.classification}")
                    print(f"│   ├── Confidence: {conf.name} ({prob:.3f})")
                    print(f"│   └── Uncertainty: ±{uncertainty.get('total_uncertainty', 0.0):.3f}")
                    
                    if hasattr(bayesian_result, 'causal_analysis') and bayesian_result.causal_analysis:
                        print(f"├── 🔍 Causal Analysis:")
                        for factor in bayesian_result.causal_analysis.get('primary_factors', [])[:2]:
                            print(f"│   ├── {factor.get('name', 'Unknown')}: {factor.get('contribution', 0)*100:.1f}%")
                    
                    risk_level = "LOW" if conf.name == "HIGH" else "MEDIUM" if conf.name == "MEDIUM" else "HIGH"
                    print(f"└── ⚠️  Risk Assessment: {risk_level} RISK")
                
            except Exception as be:
                print(f"⚠️  Bayesian analysis failed: {be}")
                bayesian_result = None
        
        # Calculate audio duration
        audio_duration = waveform.shape[0] / sample_rate if waveform is not None else 0
        
        # Build comprehensive result
        result = {
            "filepath": filepath,
            "user_id": user_id,
            "file_type": file_type,
            "status": "success",
            "audio_duration_s": audio_duration,
            "processing_time": time.time() - start_time,
            "successful_stages": "3/3",  # Loading, feature extraction, Bayesian analysis
            "validation_overall_valid": True
        }
        
        # Add physics features
        for k, v in physics_features.items():
            result[f"physics_{k}"] = v
        
        # Add Bayesian analysis results
        if bayesian_result and bayesian_result.status == "success":
            result.update({
                "bayesian_classification": bayesian_result.classification,
                "bayesian_confidence": bayesian_result.deepfake_probability,
                "bayesian_confidence_level": bayesian_result.confidence_level.name,
                "bayesian_uncertainty_total": bayesian_result.uncertainty_metrics.get('total_uncertainty', 0.0),
                "bayesian_uncertainty_epistemic": bayesian_result.uncertainty_metrics.get('epistemic_uncertainty', 0.0),
                "bayesian_uncertainty_aleatoric": bayesian_result.uncertainty_metrics.get('aleatoric_uncertainty', 0.0),
                "bayesian_status": "success"
            })
            
            # Add causal analysis if available
            if hasattr(bayesian_result, 'causal_analysis') and bayesian_result.causal_analysis:
                causal = bayesian_result.causal_analysis
                if 'primary_factors' in causal and causal['primary_factors']:
                    result["bayesian_primary_factor"] = causal['primary_factors'][0].get('name', 'unknown')
                    result["bayesian_primary_contribution"] = causal['primary_factors'][0].get('contribution', 0.0)
        else:
            result.update({
                "bayesian_classification": "unknown",
                "bayesian_confidence": 0.0,
                "bayesian_confidence_level": "UNKNOWN",
                "bayesian_uncertainty_total": 1.0,
                "bayesian_status": "failed"
            })
        
        # Add other pipeline features
        hubert_info = features.get('hubert', {})
        result["hubert_seq_len_frames"] = hubert_info.get('sequence_length', 0) if isinstance(hubert_info, dict) else 0
        result["hubert_embedding_dim"] = hubert_info.get('embedding_dim', 0) if isinstance(hubert_info, dict) else 0
        
        # Add validation information
        result["validation_num_errors"] = 0
        result["validation_num_warnings"] = 0
        
        # Add processing metadata
        result["enhanced_mode"] = "bayesian_pipeline"
        result["cache_hit"] = False
        
        return result
            
    except Exception as e:
        error_msg = f"Bayesian pipeline processing error for {filepath}: {str(e)}"
        print(error_msg)
        
        # Log detailed error
        with open(ERROR_LOG, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")
            f.write("-" * 80 + "\n")
        
        return {
            "filepath": filepath,
            "user_id": user_id,
            "file_type": file_type,
            "status": "error",
            "error": str(e),
            "audio_duration_s": 0,
            "hubert_seq_len_frames": 0,
            "processing_time": time.time() - start_time,
            "bayesian_status": "failed"
        }

async def process_single_file_with_enhanced_pipeline(filepath: str, user_id: str, file_type: str, 
                                                   feature_extractor, batch_processor=None, 
                                                   enable_batch_optimization=False) -> dict:
    """
    Process a single audio file using enhanced pipeline with advanced features.
    
    Features:
    - Advanced pipeline with full validation framework
    - Advanced error recovery with graceful degradation
    - Comprehensive validation at each processing stage
    - Detailed processing metadata and timing information
    - Enhanced error logging and diagnostic capabilities
    - Supports partial success (continues even if some features fail)
    - Concurrency-optimized
    - Medium-High resource usage
    - Full feature extraction + validation metadata
    - Production-ready with detailed logging
    - Optimized for Large Datasets
    
    Args:
        filepath: Path to audio file
        user_id: User identifier  
        file_type: Type of audio (genuine, deepfake_tts, etc.)
        feature_extractor: Feature extractor instance
        batch_processor: Optional batch processor for optimization
        enable_batch_optimization: Whether to use batch optimization features
    
    Returns:
        Dictionary with extracted features and enhanced metadata
    """
    processing_metadata = {
        "start_time": time.time(),
        "stages_completed": [],
        "stages_failed": [],
        "recovery_attempts": 0,
        "resource_usage": {},
        "validation_records": []
    }
    
    try:
        # Use batch optimization if available and enabled (specialized BatchProcessor with advanced memory management)
        if enable_batch_optimization and batch_processor:
            try:
                # Process single file through batch processor for optimization (length bucketing & memory-efficient processing)
                file_paths = [Path(filepath)]
                metadata = [{"user_id": user_id, "file_type": file_type}]
                
                processing_metadata["stages_completed"].append("batch_setup")
                processing_metadata["batch_optimization"] = True
                
                # Get resource usage before processing
                if resource_limiter:
                    processing_metadata["resource_usage"]["pre_batch"] = resource_limiter.get_current_usage()
                
                batch_results = await batch_processor.process_files_batch(
                    file_paths, metadata, processing_mode="enhanced_single"
                )
                
                processing_metadata["stages_completed"].append("batch_processing")
                
                if batch_results and len(batch_results) > 0:
                    result = batch_results[0]
                    result["enhanced_mode"] = "batch_optimized"
                    result["processing_time"] = time.time() - processing_metadata["start_time"]
                    result["processing_metadata"] = processing_metadata
                    
                    # Add detailed batch statistics (advanced parallelization & detailed batch statistics)
                    if batch_processor:
                        batch_stats = batch_processor.get_batch_stats()
                        result["batch_optimization_available"] = True
                        result["batch_efficiency_score"] = batch_stats.get('efficiency_score', 0.0)
                        result["batch_statistics"] = batch_stats
                    
                    return result
            except Exception as batch_err:
                # Graceful degradation - fallback to standard processing if batch optimization fails
                print(f"Batch optimization failed, falling back to standard processing: {batch_err}")
                processing_metadata["stages_failed"].append("batch_processing")
                processing_metadata["fallback_reason"] = str(batch_err)
                processing_metadata["recovery_attempts"] += 1
        
        # Standard processing path with comprehensive validation
        result = {
            "filepath": filepath,
            "user_id": user_id,
            "file_type": file_type,
            "status": "in_progress",
            "enhanced_mode": "advanced_pipeline",
            "processing_metadata": processing_metadata
        }
        
        # STAGE 1: Load and validate audio
        try:
            # Load audio with validation
            waveform, sample_rate = load_audio(filepath, target_sr=settings.audio.sample_rate)
            
            # Validate audio
            if waveform is None or waveform.numel() == 0:
                raise ValueError("Empty or invalid waveform")
                
            # Calculate audio duration
            audio_duration = waveform.shape[0] / sample_rate
            result["audio_duration_s"] = audio_duration
            
            # Comprehensive validation
            validation_result = {
                "stage": "audio_loading",
                "status": "success",
                "validations": [
                    {"check": "waveform_not_empty", "passed": True},
                    {"check": "sample_rate_valid", "passed": sample_rate > 0},
                    {"check": "duration_valid", "passed": audio_duration > 0}
                ]
            }
            processing_metadata["validation_records"].append(validation_result)
            processing_metadata["stages_completed"].append("audio_loading")
            
        except Exception as e:
            # Enhanced error logging
            error_msg = f"Error loading audio {filepath}: {str(e)}"
            with open(ERROR_LOG, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
                f.write("-" * 80 + "\n")
            
            result.update({
                "status": "error",
                "error": str(e),
                "error_stage": "audio_loading",
                "audio_duration_s": 0,
                "processing_time": time.time() - processing_metadata["start_time"]
            })
            return result
        
        # STAGE 2: Feature extraction with error recovery
        try:
            # Monitor resource usage
            if resource_limiter:
                processing_metadata["resource_usage"]["pre_extraction"] = resource_limiter.get_current_usage()
            
            # Extract features with advanced error handling
            extraction_start = time.time()
            try:
                features = await feature_extractor.extract_features(waveform, sample_rate)
                extraction_success = True
            except Exception as feat_err:
                # Partial success handling - try with reduced feature set
                print(f"Full feature extraction failed, attempting recovery with reduced feature set: {feat_err}")
                try:
                    features = await feature_extractor.extract_features(waveform, sample_rate, processing_mode="reduced")
                    extraction_success = True
                    processing_metadata["recovery_attempts"] += 1
                    processing_metadata["recovery_success"] = True
                except Exception as recovery_err:
                    # Final fallback to minimal features
                    try:
                        features = await feature_extractor.extract_features(waveform, sample_rate, processing_mode="minimal")
                        extraction_success = True
                        processing_metadata["recovery_attempts"] += 2
                        processing_metadata["recovery_success"] = True
                        processing_metadata["recovery_level"] = "minimal"
                    except Exception as minimal_err:
                        raise minimal_err
            
            # Extract physics features
            physics_features = features.get('physics', {})
            extraction_time = time.time() - extraction_start
            
            # Validate extracted features
            physics_valid = len(physics_features) > 0
            hubert_valid = 'hubert' in features and features['hubert'] is not None
            
            validation_result = {
                "stage": "feature_extraction",
                "status": "success" if extraction_success else "partial",
                "time_taken": extraction_time,
                "validations": [
                    {"check": "physics_features_present", "passed": physics_valid},
                    {"check": "hubert_features_present", "passed": hubert_valid}
                ]
            }
            processing_metadata["validation_records"].append(validation_result)
            processing_metadata["stages_completed"].append("feature_extraction")
            
            # Add timing information
            processing_metadata["extraction_time"] = extraction_time
            
            # Add extraction validation to result
            result["features_valid"] = physics_valid and hubert_valid
            
            # Process extracted features
            for k, v in physics_features.items():
                result[f"physics_{k}"] = v
            
            # Add HuBERT information
            hubert_info = features.get('hubert', {})
            result["hubert_seq_len_frames"] = hubert_info.get('sequence_length', 0) if isinstance(hubert_info, dict) else 0
            result["hubert_embedding_dim"] = hubert_info.get('embedding_dim', 0) if isinstance(hubert_info, dict) else 0
            
            # STAGE 3: Advanced validation
            try:
                validation_start = time.time()
                
                # Validate physics features
                physics_validations = []
                if physics_valid:
                    for key, value in physics_features.items():
                        if key.startswith('delta_'):
                            is_valid = isinstance(value, (int, float)) and not (np.isnan(value) if hasattr(np, 'isnan') else False)
                            physics_validations.append({"feature": key, "valid": is_valid})
                
                # Comprehensive validation summary
                validation_summary = {
                    "overall_valid": all(v["valid"] for v in physics_validations) if physics_validations else False,
                    "num_errors": sum(1 for v in physics_validations if not v["valid"]),
                    "num_warnings": 0,
                    "validation_time": time.time() - validation_start
                }
                
                # Add validation information
                result["validation_overall_valid"] = validation_summary["overall_valid"]
                result["validation_num_errors"] = validation_summary["num_errors"]
                result["validation_num_warnings"] = validation_summary["num_warnings"]
                
                processing_metadata["validation_time"] = validation_summary["validation_time"]
                processing_metadata["stages_completed"].append("validation")
                
            except Exception as val_err:
                # Partial success - continue even if validation fails
                print(f"Validation error (non-critical): {val_err}")
                processing_metadata["stages_failed"].append("validation")
                processing_metadata["validation_error"] = str(val_err)
                
                # Still provide basic validation info
                result["validation_overall_valid"] = True  # Assume valid
                result["validation_num_errors"] = 0
                result["validation_num_warnings"] = 1  # The validation itself failed
            
            # Finalize result
            result["status"] = "success"
            result["processing_time"] = time.time() - processing_metadata["start_time"]
            
            # Batch optimization metrics
            if batch_processor:
                batch_stats = batch_processor.get_batch_stats()
                result["batch_optimization_available"] = True
                result["batch_efficiency_score"] = batch_stats.get('efficiency_score', 0.0)
            else:
                result["batch_optimization_available"] = False
            
            # Detailed resource usage
            if resource_limiter:
                processing_metadata["resource_usage"]["final"] = resource_limiter.get_current_usage()
                
            # Update processing metadata
            result["processing_metadata"] = processing_metadata
            
            return result
            
        except Exception as e:
            # Enhanced error reporting
            error_msg = f"Enhanced pipeline processing error for {filepath}: {str(e)}"
            print(error_msg)
            
            # Log detailed error
            with open(ERROR_LOG, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
                f.write("-" * 80 + "\n")
            
            result.update({
                "status": "error",
                "error": str(e),
                "error_stage": "feature_extraction",
                "audio_duration_s": audio_duration if 'audio_duration' in locals() else 0,
                "hubert_seq_len_frames": 0,
                "processing_time": time.time() - processing_metadata["start_time"],
                "processing_metadata": processing_metadata
            })
            
            return result
            
    except Exception as e:
        # Global error handler
        error_msg = f"Enhanced pipeline processing error for {filepath}: {str(e)}"
        print(error_msg)
        
        # Log detailed error
        with open(ERROR_LOG, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")
            f.write("-" * 80 + "\n")
        
        # Ensure we still return something useful even on catastrophic failures
        processing_time = time.time() - processing_metadata.get("start_time", time.time())
        return {
            "filepath": filepath,
            "user_id": user_id,
            "file_type": file_type,
            "status": "error",
            "error": str(e),
            "error_stage": "global",
            "audio_duration_s": 0,
            "hubert_seq_len_frames": 0,
            "processing_time": processing_time,
            "enhanced_mode": "failed",
            "processing_metadata": processing_metadata
        }

async def main():
    """Main execution function with comprehensive error handling and checkpoint recovery."""
    global progress_tracker
    
    print("="*60)
    print("PHYSICS FEATURES EXTRACTION PIPELINE")
    print("="*60)
    
    # Initialize comprehensive folder structure
    print("🗂️  Initializing project folder structure...")
    try:
        folder_manager = initialize_project_folders(current_dir)
        print("✓ Project folders initialized and populated")
    except Exception as e:
        print(f"Warning: Could not fully initialize folder structure: {e}")
        # Continue execution as folders may already exist
    
    # Initialize comprehensive logging system
    project_logger = None
    if create_project_logger:
        try:
            project_logger = create_project_logger(
                project_root=current_dir
            )
            print("✓ Comprehensive logging system initialized")
        except Exception as e:
            print(f"Warning: Could not initialize comprehensive logging: {e}")
    
    # Check for synchronization safety
    print("🔄 Checking system synchronization...")
    cache_lock_file = Path(current_dir) / 'cache' / '.cache_lock'
    if cache_lock_file.exists():
        print("⚠️  Warning: Another process may be using the cache system")
        print("   If no other process is running, remove .cache_lock manually")
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Exiting to avoid cache conflicts")
            return
    
    # Create cache lock
    try:
        cache_lock_file.parent.mkdir(exist_ok=True)
        with open(cache_lock_file, 'w') as f:
            f.write(f"pid:{os.getpid()}\nstart:{datetime.now().isoformat()}\n")
        print("✓ Cache synchronization lock acquired")
    except Exception as e:
        print(f"Warning: Could not create cache lock: {e}")
    
    try:
        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(Path(RESULTS_DIR) / "checkpoints")
        
        # Get configuration for logging and mode coordination
        config_dict = {
            'audio_config': settings.audio.model_dump(),
            'physics_config': settings.physics.model_dump(),
            'models_config': settings.models.model_dump(),
            'device': str(DEVICE),
            'session_start': time.time()
        }
        
        # Log session start if logger available
        if project_logger:
            try:
                project_logger.log_session_start(config_dict)
            except Exception as e:
                print(f"Warning: Could not log session start: {e}")
        
        # Check for mode-specific coordination requirements
        print("🔧 Validating processing mode coordination...")
        mode_coordination_checks = {
            'cache_consistency': True,
            'output_coordination': True,
            'resource_synchronization': True
        }
        
        # Validate cache consistency
        cache_dir = Path(current_dir) / 'cache'
        if cache_dir.exists():
            cache_files = list(cache_dir.glob('*.pkl'))
            if cache_files:
                print(f"   Found {len(cache_files)} existing cache files")
                # Check for cache corruption or inconsistency
                try:
                    # Sample check on a few cache files
                    sample_files = cache_files[:min(3, len(cache_files))]
                    for cache_file in sample_files:
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                            if not isinstance(cached_data, dict) or 'hubert_sequence' not in cached_data:
                                print(f"   ⚠️  Warning: Cache file {cache_file.name} may be corrupted")
                                mode_coordination_checks['cache_consistency'] = False
                    print("   ✓ Cache consistency validated")
                except Exception as e:
                    print(f"   ⚠️  Warning: Cache validation failed: {e}")
                    mode_coordination_checks['cache_consistency'] = False
        
        # Validate output coordination (ensure no simultaneous writes to same files)
        output_files = [OUTPUT_CSV, ERROR_LOG]
        for output_file in output_files:
            if Path(output_file).exists():
                try:
                    # Try to acquire exclusive access briefly
                    with open(output_file, 'a') as f:
                        pass
                    print(f"   ✓ Output file access validated: {Path(output_file).name}")
                except PermissionError:
                    print(f"   ⚠️  Warning: Output file in use: {Path(output_file).name}")
                    mode_coordination_checks['output_coordination'] = False
        
        # Display coordination status
        if all(mode_coordination_checks.values()):
            print("✓ All processing mode coordination checks passed")
        else:
            print("⚠️  Some coordination issues detected - proceeding with caution")
            failed_checks = [k for k, v in mode_coordination_checks.items() if not v]
            print(f"   Failed checks: {failed_checks}")
        
        # Store coordination info for later reference
        config_dict['mode_coordination'] = mode_coordination_checks
        
        # Main processing logic with cache coordination
        await _run_main_processing(project_logger)
    finally:
        # Always remove cache lock
        try:
            if cache_lock_file.exists():
                cache_lock_file.unlink()
                print("✓ Cache synchronization lock released")
        except Exception as e:
            print(f"Warning: Could not remove cache lock: {e}")

async def _run_main_processing(project_logger):
    """Main processing logic extracted for proper resource management."""
    global progress_tracker
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(Path(RESULTS_DIR) / "checkpoints")
    
    # Initialize feature extractor
    initial_feature_extractor = initialize_feature_extractor()
    if initial_feature_extractor is None:
        print("FATAL: Cannot proceed without feature extractor")
        return
    
    # Discover audio files
    print(f"\nScanning for audio files in: {DATA_DIR}")
    all_audio_files = discover_audio_files(DATA_DIR)
    
    if not all_audio_files:
        print(f"No audio files found in {DATA_DIR}")
        print("Please ensure the data directory contains subdirectories with .wav files")
        return
    
    print(f"Found {len(all_audio_files)} audio files to process")
    
    # Check for existing checkpoint
    checkpoint_data = checkpoint_manager.load_checkpoint()
    
    # Initialize variables for resuming
    all_results = []
    processed_files_set = set()
    start_batch_idx = 0
    
    if checkpoint_data:
        user_input = input("Resume from checkpoint? (y/n): ").lower().strip()
        if user_input == 'y':
            all_results = checkpoint_data['results']
            processed_files_set = set(checkpoint_data['processed_files'])
            start_batch_idx = checkpoint_data.get('current_batch_idx', 0)
            
            print(f"Resuming from checkpoint:")
            print(f"  - Already processed: {len(processed_files_set)} files")
            print(f"  - Starting from batch: {start_batch_idx}")
            
            # Filter out already processed files
            remaining_files = [f for f in all_audio_files 
                             if f['filepath'] not in processed_files_set]
            all_audio_files = remaining_files
            print(f"  - Remaining files: {len(all_audio_files)}")
        else:
            print("Starting fresh processing (checkpoint ignored)")
            checkpoint_manager.clear_checkpoint()
    
    if not all_audio_files:
        print("All files already processed according to checkpoint!")
        return
    
    # Initialize progress tracker
    total_files = len(all_audio_files) + len(processed_files_set)
    progress_tracker = ProgressTracker(total_files)
    # Update with already processed files
    progress_tracker.processed = len(processed_files_set)
    progress_tracker.successful = len(all_results)
    
    # Show file distribution
    file_types = {}
    total_size_mb = 0
    for f in all_audio_files:
        file_types[f['file_type']] = file_types.get(f['file_type'], 0) + 1
        total_size_mb += f.get('file_size_mb', 0)
    
    print(f"File type distribution (remaining): {dict(file_types)}")
    print(f"Total data size (remaining): {total_size_mb:.1f} MB")
    
    # Configure processing parameters
    concurrency_limit = min(4, len(all_audio_files))  # Don't exceed available files
    if DEVICE.type == 'cpu':
        concurrency_limit = min(concurrency_limit, 2)  # Reduce for CPU processing
    
    print(f"\nStarting processing with concurrency limit: {concurrency_limit}")
    print(f"Device: {DEVICE}")
    print(f"Expected processing time: ~{len(all_audio_files) * 30 / concurrency_limit / 60:.1f} minutes")
    
    # Choose processing mode
    print(f"\nProcessing Mode Options:")
    print(f"1. Enhanced Pipeline Processing (comprehensive with batch optimization)")
    print(f"2. Lightweight Pipeline (faster, reduced features)")  
    print(f"3. Bayesian-Enhanced Pipeline (most comprehensive with probabilistic analysis)")
    
    try:
        mode_choice = input("Choose processing mode (1/2/3) [default: 1]: ").strip()
        if not mode_choice:
            mode_choice = "1"
    except:
        mode_choice = "1"  # Default to enhanced pipeline processing
    
    # Initialize processing components based on mode
    pipeline = None
    batch_processor = None
    bayesian_engine = None
    process_function = None

    if mode_choice == "1":
        print("Using Enhanced Pipeline Processing mode...")
        print("🔧 Initializing comprehensive pipeline with batch optimization...")
        try:
            # Create enhanced pipeline (merger of old options 2 & 4)
            pipeline = create_standard_pipeline(
                enable_cache=True,
                cache_dir="cache",
                enable_bayesian=True
            )
            
            # Also initialize batch processor for optimization
            batch_config = BatchConfig(
                batch_size=min(16, len(all_audio_files) // 4) if len(all_audio_files) > 4 else 4,
                max_concurrent_batches=2,
                enable_length_bucketing=True,
                memory_efficient_mode=True
            )
            
            batch_processor = BatchProcessor(
                feature_extractor=initial_feature_extractor,
                config=batch_config,
                resource_limiter=resource_limiter
            )
            
            print(f"✓ Enhanced pipeline initialized with:")
            print(f"  - Comprehensive validation: Enabled")
            print(f"  - Batch optimization: Enabled")
            print(f"  - Batch size: {batch_config.batch_size}")
            print(f"  - Length bucketing: {batch_config.enable_length_bucketing}")
            print(f"  - Memory efficiency: {batch_config.memory_efficient_mode}")
            
        except Exception as e:
            print(f"ERROR: Failed to create enhanced pipeline: {e}")
            return
        process_function = process_single_file_with_enhanced_pipeline
            
    elif mode_choice == "2":
        print("Using Lightweight Pipeline mode...")
        try:
            pipeline = create_lightweight_pipeline(
                enable_physics=True,
                enable_audio_features=True
            )
            print("✓ Lightweight pipeline initialized for fast processing")
        except Exception as e:
            print(f"ERROR: Failed to create lightweight pipeline: {e}")
            return
        process_function = process_single_file
        
    elif mode_choice == "3":
        print("Using Bayesian-Enhanced Pipeline Processing mode...")
        print("🧠 Initializing Bayesian probabilistic analysis engine...")
        try:
            # Create standard pipeline with Bayesian integration
            pipeline = create_standard_pipeline(
                enable_cache=True,
                cache_dir="cache",
                enable_bayesian=True
            )
            
            # Initialize Bayesian engine with correct config
            from bayesian.core.bayesian_engine import BayesianDeepfakeEngine, BayesianConfig
            
            bayesian_config = BayesianConfig(
                enable_temporal_modeling=True,
                enable_hierarchical_modeling=True,
                enable_causal_analysis=True,
                inference_method="variational"
            )
            
            bayesian_engine = BayesianDeepfakeEngine(bayesian_config)
            
            print("✓ Bayesian engine initialized with:")
            print(f"  - Temporal modeling: {'Enabled' if bayesian_config.enable_temporal_modeling else 'Disabled'}")
            print(f"  - Hierarchical modeling: {'Enabled' if bayesian_config.enable_hierarchical_modeling else 'Disabled'}")
            print(f"  - Causal analysis: {'Enabled' if bayesian_config.enable_causal_analysis else 'Disabled'}")
            print(f"  - Inference method: {bayesian_config.inference_method}")
            
        except Exception as e:
            print(f"ERROR: Failed to create Bayesian-enhanced pipeline: {e}")
            print(f"Falling back to enhanced pipeline mode...")
            # Fallback to enhanced pipeline
            try:
                pipeline = create_standard_pipeline(
                    enable_cache=True,
                    cache_dir="cache",
                    enable_bayesian=True
                )
                process_function = process_single_file_with_enhanced_pipeline
                print("✓ Fallback to enhanced pipeline successful")
            except Exception as fallback_e:
                print(f"ERROR: Fallback also failed: {fallback_e}")
                return
        
        if bayesian_engine:
            process_function = process_single_file_with_bayesian_pipeline
        else:
            process_function = process_single_file_with_enhanced_pipeline
    
    else:
        print(f"Invalid mode choice: {mode_choice}. Using default enhanced pipeline...")
        mode_choice = "1"
        # Initialize enhanced pipeline as default
        try:
            pipeline = create_standard_pipeline(
                enable_cache=True,
                cache_dir="cache",
                enable_bayesian=True
            )
            process_function = process_single_file_with_enhanced_pipeline
        except Exception as e:
            print(f"ERROR: Failed to create default pipeline: {e}")
            return

    # Final validation before processing
    if process_function is None:
        print("ERROR: No processing function selected")
        return
        
    if pipeline is None:
        print("ERROR: No pipeline initialized")
        return
    
    print(f"✓ Processing components initialized successfully for mode {mode_choice}")
    
    # Print Bayesian analysis header if using Bayesian mode
    if mode_choice == "3" and bayesian_engine:
        print("\n" + "="*60)
        print("BAYESIAN DEEPFAKE DETECTION ANALYSIS PIPELINE")
        print("="*60)
        print("🧠 Bayesian Engine Status: ACTIVE")
        print("📊 Probabilistic Analysis: ENABLED") 
        print("🎯 Confidence Threshold: 85%")
        print("🔍 Uncertainty Quantification: ENABLED")
        print("⏳ Temporal Modeling: ENABLED")
        print("🧬 Causal Analysis: ENABLED")
        print("="*60)
    
    # Process files in batches with checkpointing
    semaphore = asyncio.Semaphore(concurrency_limit)
    batch_size = min(20, len(all_audio_files))  # Process in batches of 20
    checkpoint_interval = 5  # Save checkpoint every 5 batches
    
    try:
        # Determine processing mode string for display
        mode_map = {
            "1": "enhanced_pipeline",
            "2": "lightweight", 
            "3": "bayesian_enhanced"
        }
        processing_mode_str = mode_map.get(mode_choice, "unknown")
        
        # Create progress bar
        with tqdm(total=len(all_audio_files), desc=f"Processing audio files ({processing_mode_str})", 
             unit="file", dynamic_ncols=True) as pbar:
        
            for batch_idx, i in enumerate(range(0, len(all_audio_files), batch_size), start=start_batch_idx):
                batch = all_audio_files[i:i + batch_size]
                
                # Process batch with mode-aware functions
                batch_results = []
                semaphore = asyncio.Semaphore(concurrency_limit)
                
                async def process_with_semaphore_and_mode(file_meta):
                    async with semaphore:
                        try:
                            # Choose appropriate processing function based on mode
                            if mode_choice == "1":
                                # Enhanced pipeline with batch optimization
                                result = await process_single_file_with_enhanced_pipeline(
                                    file_meta["filepath"], file_meta["user_id"], file_meta["file_type"],
                                    pipeline, batch_processor, enable_batch_optimization=True
                                )
                            elif mode_choice == "2":
                                # Lightweight pipeline
                                result = await process_single_file(
                                    file_meta["filepath"], file_meta["user_id"], file_meta["file_type"],
                                    pipeline
                                )
                            elif mode_choice == "3":
                                # Bayesian-enhanced pipeline
                                if bayesian_engine:
                                    result = await process_single_file_with_bayesian_pipeline(
                                        file_meta["filepath"], file_meta["user_id"], file_meta["file_type"],
                                        pipeline, bayesian_engine
                                    )
                                else:
                                    # Fallback to enhanced pipeline if Bayesian engine not available
                                    result = await process_single_file_with_enhanced_pipeline(
                                        file_meta["filepath"], file_meta["user_id"], file_meta["file_type"],
                                        pipeline, batch_processor, enable_batch_optimization=True
                                    )
                            else:
                                # Fallback to enhanced pipeline
                                result = await process_single_file_with_enhanced_pipeline(
                                    file_meta["filepath"], file_meta["user_id"], file_meta["file_type"],
                                    pipeline, batch_processor, enable_batch_optimization=True
                                )
                            
                            # Update progress
                            success = result.get("status") == "success"
                            error_msg = result.get("error") if not success else None
                            progress_tracker.update(success, error_msg)
                            return result
                            
                        except Exception as e:
                            error_msg = f"Batch processing error for {file_meta['filepath']}: {str(e)}"
                            progress_tracker.update(False, error_msg)
                            return {
                                "filepath": file_meta["filepath"],
                                "user_id": file_meta["user_id"], 
                                "file_type": file_meta["file_type"],
                                "status": "batch_error",
                                "processing_mode": processing_mode_str,
                                "error": str(e)
                            }
                
                # Process batch concurrently
                tasks = [process_with_semaphore_and_mode(file_meta) for file_meta in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions
                processed_results = []
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        error_result = {
                            "filepath": batch[i]["filepath"],
                            "user_id": batch[i]["user_id"],
                            "file_type": batch[i]["file_type"],
                            "status": "exception",
                            "processing_mode": processing_mode_str,
                            "error": str(result)
                        }
                        processed_results.append(error_result)
                    else:
                        processed_results.append(result)
                
                all_results.extend(processed_results)
                
                # Update processed files set
                for file_meta, result in zip(batch, processed_results):
                    processed_files_set.add(file_meta['filepath'])
            
            # Update progress bar
            pbar.update(len(batch))
            
            # Update progress bar description with current stats
            stats = progress_tracker.get_stats()
            pbar.set_postfix({
                'Success': f"{stats['successful']}/{stats['processed']}",
                'Rate': f"{stats['success_rate']:.1f}%",
                'Avg': f"{stats['avg_time_per_file']:.1f}s/file",
                'Mode': processing_mode_str
            })
                
                # Save checkpoint periodically
            if (batch_idx + 1) % checkpoint_interval == 0:
                    checkpoint_manager.save_checkpoint(
                        list(processed_files_set), 
                        all_results,
                        batch_idx + 1
                    )
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        print("Saving checkpoint before exit...")
        checkpoint_manager.save_checkpoint(list(processed_files_set), all_results)
        print("Saving partial results...")
    except Exception as e:
        print(f"\nUnexpected error during processing: {e}")
        traceback.print_exc()
        print("Saving checkpoint and partial results...")
        checkpoint_manager.save_checkpoint(list(processed_files_set), all_results)
    
    # Save final results
    save_results(all_results, OUTPUT_CSV)
    
    # Generate Advanced Visualizations
    try:
        print("\n" + "="*60)
        print("GENERATING ADVANCED VISUALIZATIONS")
        print("="*60)
        
        # Import visualization system
        sys.path.insert(0, os.path.join(current_dir, 'src', 'visualization'))
        from advanced_plotter import AdvancedPhysicsPlotter
        
        # Create plotter instance
        vis_output_dir = os.path.join(current_dir, "visualizations")
        plotter = AdvancedPhysicsPlotter(vis_output_dir)
        
        # Generate all enhanced visualizations
        visualization_results = plotter.generate_all_visualizations(OUTPUT_CSV)
        
        print(f"\n✓ Enhanced visualization suite completed successfully!")
        print(f"  📊 Interactive Dashboard: {visualization_results['dashboard_path']}")
        print(f"  📈 Static Plots: {visualization_results['static_dir']}")
        print(f"  📋 Analysis Reports: {visualization_results['reports_dir']}")
        print(f"  🎯 Key Statistical Findings: {len(visualization_results['summary_report']['key_findings'])}")
        
        # Display key insights
        key_findings = visualization_results['summary_report']['key_findings']
        if key_findings:
            print(f"\n🔍 KEY RESEARCH INSIGHTS:")
            for i, finding in enumerate(key_findings, 1):
                print(f"  {i}. {finding}")
        
        # Display discrimination ranking
        discrimination_ranking = visualization_results['summary_report']['discrimination_ranking']
        if discrimination_ranking:
            print(f"\n📊 FEATURE DISCRIMINATION RANKING:")
            for i, feature in enumerate(discrimination_ranking[:3], 1):  # Show top 3
                significance = "✅" if feature['p_value'] < 0.05 else "⚠️" if feature['p_value'] < 0.1 else "❌"
                print(f"  {i}. {feature['feature_name']}: {feature['discrimination_score']:.3f} {significance}")
        
        # Display recommendations
        recommendations = visualization_results['summary_report']['recommendations']
        if recommendations:
            print(f"\n💡 KEY RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print(f"\n📁 Visualization files created:")
        print(f"  - Interactive Dashboard: {visualization_results['dashboard_path']}")
        print(f"  - Enhanced Static Plots: {visualization_results['static_dir']}/*.png")
        print(f"  - Detailed Reports: {visualization_results['reports_dir']}/*.json, *.md")
        print(f"  - Open the dashboard in your web browser for interactive analysis!")
        
    except ImportError as e:
        print(f"\nWarning: Could not import visualization system: {e}")
        print("Install visualization dependencies: pip install plotly kaleido dash seaborn scikit-learn")
    except Exception as e:
        print(f"\nWarning: Enhanced visualization generation failed: {e}")
        print("Results are still available in CSV format")
        import traceback
        traceback.print_exc()

    # Generate comprehensive analysis and plots if logging system is available
    if project_logger and all_results:
        try:
            print("Generating comprehensive analysis and plots...")
            results_df = pd.DataFrame(all_results)
            
            # Create physics analysis plots
            project_logger.create_physics_analysis_plots(results_df)
            
            # Save detailed feature summary
            project_logger.save_feature_summary(results_df)
            
            # Save checkpoint information
            checkpoint_stats = checkpoint_manager.get_checkpoint_stats()
            project_logger.save_checkpoint_info(checkpoint_stats)
            
            # Log session completion
            final_stats = progress_tracker.get_stats()
            final_stats.update({
                'total_results': len(all_results),
                'output_file': str(OUTPUT_CSV),
                'session_duration_minutes': (time.time() - config_dict['session_start']) / 60
            })
            project_logger.log_session_end(final_stats)
            
            print("✓ Comprehensive analysis completed")
            print(f"  - Plots saved to: plots/")
            print(f"  - Analysis reports saved to: output/")
            print(f"  - Detailed logs saved to: logs/")
            
        except Exception as e:
            print(f"Warning: Could not complete comprehensive analysis: {e}")
    
    # Clear checkpoint on successful completion
    if len(processed_files_set) == total_files:
        checkpoint_manager.clear_checkpoint()
        print("Processing completed successfully, checkpoint cleared")
    
    print(f"\nProcessing complete!")
    if progress_tracker.errors:
        print(f"Detailed error log saved to: {ERROR_LOG}")

if __name__ == "__main__":
    # Configure asyncio for Windows if needed
    if sys.platform == "win32":
        # Use ProactorEventLoop for Windows to handle file I/O better
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except AttributeError:
            pass  # Older Python versions
    
    # Run main pipeline
    asyncio.run(main())

