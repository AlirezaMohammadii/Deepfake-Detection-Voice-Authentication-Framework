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
from typing import List, Dict, Optional, Any
import json
from datetime import datetime

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
        # Load audio file
        waveform = await load_audio(filepath, target_sr=settings.audio.sample_rate)
        
        if waveform is None or waveform.numel() == 0:
            result.update({
                "status": "error",
                "error": "Could not load audio or empty waveform",
                "audio_duration_s": 0,
                "hubert_seq_len_frames": 0
            })
            return result

        # Calculate audio duration
        audio_duration = waveform.shape[0] / settings.audio.sample_rate
        result["audio_duration_s"] = audio_duration
        
        # Extract features
        all_features = await feature_extractor.extract_features(waveform, settings.audio.sample_rate)
        
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

async def process_single_file_with_retry(filepath: str, user_id: str, file_type: str, 
                                       feature_extractor: ComprehensiveFeatureExtractor) -> dict:
    """
    Enhanced version of process_single_file with retry mechanism.
    
    Args:
        filepath: Path to audio file
        user_id: User identifier
        file_type: Type of audio (genuine, deepfake_tts, etc.)
        feature_extractor: Initialized feature extractor
    
    Returns:
        Dictionary with extracted features or error information
    """
    # Use the RobustProcessor for retry mechanism
    return await RobustProcessor.process_with_retry(
        process_single_file,
        filepath, user_id, file_type, feature_extractor,
        max_retries=2,
        exception_types=(Exception,)
    )

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
                        # Convert to numeric, handling any string values
                        numeric_values = pd.to_numeric(successful_df[feature], errors='coerce')
                        if not numeric_values.isna().all():
                            print(f"{feature}:")
                            print(f"  Mean: {numeric_values.mean():.6f}")
                            print(f"  Std:  {numeric_values.std():.6f}")
                            print(f"  Min:  {numeric_values.min():.6f}")
                            print(f"  Max:  {numeric_values.max():.6f}")
                
                # By file type analysis
                if len(successful_df['file_type'].unique()) > 1:
                    print(f"\nPhysics Features by File Type:")
                    print("-" * 40)
                    for feature in key_features[:2]:  # Show first 2 features
                        if feature in successful_df.columns:
                            numeric_values = pd.to_numeric(successful_df[feature], errors='coerce')
                            grouped = successful_df.groupby('file_type')[feature].apply(
                                lambda x: pd.to_numeric(x, errors='coerce').mean()
                            )
                            print(f"{feature}:")
                            for file_type, mean_val in grouped.items():
                                if pd.notna(mean_val):
                                    print(f"  {file_type:15}: {mean_val:.6f}")
        
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
                
    except Exception as e:
        print(f"Error saving results: {e}")
        traceback.print_exc()

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
                logs_dir=os.path.join(current_dir, "logs"),
                session_name="physics_features_extraction"
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
    print(f"1. Traditional processing (original method)")
    print(f"2. Pipeline processing (enhanced with retry and validation)")
    print(f"3. Lightweight pipeline (faster, reduced features)")
    print(f"4. Batch processing (optimized for large datasets)")
    
    try:
        mode_choice = input("Choose processing mode (1/2/3/4) [default: 2]: ").strip()
        if not mode_choice:
            mode_choice = "2"
    except:
        mode_choice = "2"  # Default to pipeline processing
    
    # Initialize processing components based on mode
    selected_feature_extractor = None
    pipeline = None
    batch_processor = None
    process_function = None

    if mode_choice == "1":
        print("Using traditional processing mode...")
        selected_feature_extractor = initial_feature_extractor  # Use the initialized extractor
        process_function = process_single_file_with_retry
        
        # Validate feature extractor is properly initialized
        if selected_feature_extractor is None:
            print("ERROR: Feature extractor is None for traditional processing mode")
            return
            
    elif mode_choice == "3":
        print("Using lightweight pipeline mode...")
        try:
            pipeline = create_lightweight_pipeline(
                enable_physics=True,
                enable_audio_features=True
            )
        except Exception as e:
            print(f"ERROR: Failed to create lightweight pipeline: {e}")
            return
        process_function = process_single_file_with_pipeline
        
    elif mode_choice == "4":
        print("Using batch processing mode...")
        try:
            # Configure batch processing
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
            
            print(f"  Batch size: {batch_config.batch_size}")
            print(f"  Length bucketing: {batch_config.enable_length_bucketing}")
            print(f"  Max concurrent batches: {batch_config.max_concurrent_batches}")
            
        except Exception as e:
            print(f"ERROR: Failed to create batch processor: {e}")
            return
        
    else:  # Default to mode 2
        print("Using enhanced pipeline processing mode...")
        try:
            pipeline = create_standard_pipeline(
                feature_extractor=initial_feature_extractor,  # Use the initialized extractor
                strict_validation=False,
                early_exit_on_error=False
            )
        except Exception as e:
            print(f"ERROR: Failed to create standard pipeline: {e}")
            return
        process_function = process_single_file_with_pipeline

    # Final validation before processing
    if mode_choice != "4" and process_function is None:
        print("ERROR: No processing function selected")
        return
        
    if mode_choice == "1" and selected_feature_extractor is None:
        print("ERROR: Traditional mode requires a valid feature extractor")
        return
        
    if mode_choice in ["2", "3"] and pipeline is None:
        print("ERROR: Pipeline mode requires a valid pipeline")
        return
        
    if mode_choice == "4" and batch_processor is None:
        print("ERROR: Batch mode requires a valid batch processor")
        return
    
    print(f"✓ Processing components initialized successfully for mode {mode_choice}")
    
    # Process files in batches with checkpointing
    semaphore = asyncio.Semaphore(concurrency_limit)
    batch_size = min(20, len(all_audio_files))  # Process in batches of 20
    checkpoint_interval = 5  # Save checkpoint every 5 batches
    
    try:
        # Handle batch processing mode separately
        if mode_choice == "4" and batch_processor:
            print(f"\nStarting batch processing with resource monitoring...")
            
            # Use resource limiter for the entire batch operation
            if resource_limiter:
                with resource_limiter.limit_resources("full_batch_processing"):
                    # Extract file paths and metadata
                    file_paths = [Path(f['filepath']) for f in all_audio_files]
                    metadata = [{k: v for k, v in f.items() if k != 'filepath'} for f in all_audio_files]
                    
                    # Process all files using batch processor with mode information
                    all_results = await batch_processor.process_files_batch(file_paths, metadata, processing_mode="batch")
                    
                    # Update progress tracker with batch results
                    for result in all_results:
                        success = result.get("status") == "success"
                        error_msg = result.get("error") if not success else None
                        progress_tracker.update(success, error_msg)
            else:
                # Fallback without resource limiter
                file_paths = [Path(f['filepath']) for f in all_audio_files]
                metadata = [{k: v for k, v in f.items() if k != 'filepath'} for f in all_audio_files]
                all_results = await batch_processor.process_files_batch(file_paths, metadata, processing_mode="batch")
                
                for result in all_results:
                    success = result.get("status") == "success"
                    error_msg = result.get("error") if not success else None
                    progress_tracker.update(success, error_msg)
            
            # Display batch processing statistics
            batch_stats = batch_processor.get_batch_stats()
            print(f"\nBatch Processing Statistics:")
            print(f"  Total batches: {batch_stats['total_batches']}")
            print(f"  Total files: {batch_stats['total_files']}")
            print(f"  Avg batch time: {batch_stats['avg_batch_time']:.2f}s")
            print(f"  Total processing time: {batch_stats['total_processing_time']:.2f}s")
            
        else:
            # Original processing logic for other modes with mode coordination
            # Determine processing mode string
            mode_map = {
                "1": "traditional",
                "2": "pipeline", 
                "3": "lightweight"
            }
            processing_mode_str = mode_map.get(mode_choice, "unknown")
            
        # Create progress bar
            with tqdm(total=len(all_audio_files), desc=f"Processing audio files ({processing_mode_str})", 
                 unit="file", dynamic_ncols=True) as pbar:
            
                for batch_idx, i in enumerate(range(0, len(all_audio_files), batch_size), start=start_batch_idx):
                    batch = all_audio_files[i:i + batch_size]
                    
                    # Use the appropriate processor and pass processing mode
                    processor = pipeline if pipeline is not None else selected_feature_extractor
                    
                    # Determine the appropriate process function and mode
                    if mode_choice == "1":
                        process_func = lambda file_meta: process_single_file_with_retry(
                            file_meta["filepath"], file_meta["user_id"], file_meta["file_type"],
                            processor
                        )
                    elif mode_choice == "2":
                        process_func = lambda file_meta: process_single_file_with_pipeline(
                            file_meta["filepath"], file_meta["user_id"], file_meta["file_type"],
                            processor
                        )
                    elif mode_choice == "3":
                        process_func = lambda file_meta: process_single_file_with_pipeline(
                            file_meta["filepath"], file_meta["user_id"], file_meta["file_type"],
                            processor
                        )
                    else:
                        # Default fallback
                        process_func = lambda file_meta: process_single_file_with_retry(
                            file_meta["filepath"], file_meta["user_id"], file_meta["file_type"],
                            processor
                        )
                    
                    # Process batch with mode-aware functions
                    batch_results = []
                    semaphore = asyncio.Semaphore(concurrency_limit)
                    
                    async def process_with_semaphore_and_mode(file_meta):
                        async with semaphore:
                            try:
                                result = await process_func(file_meta)
                                
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


