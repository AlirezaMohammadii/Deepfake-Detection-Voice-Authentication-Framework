"""
Security Validator Module
Implements comprehensive security measures for audio processing
"""

import os
import sys
import time
import json
from datetime import datetime
try:
    import resource  # Unix-only module
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False
    # Windows fallback - we'll use psutil for resource monitoring
    
import psutil
import hashlib
from pathlib import Path
from typing import Union, Dict, Any, List, Optional
from contextlib import contextmanager
import logging
import mimetypes
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security configuration parameters"""
    max_file_size_mb: float = 100.0
    max_memory_gb: float = 8.0
    max_processing_time_s: float = 300.0
    allowed_formats: set = None
    max_path_length: int = 260
    max_filename_length: int = 255
    allow_path_traversal: bool = False
    
    def __post_init__(self):
        if self.allowed_formats is None:
            self.allowed_formats = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}

class SecureAudioLoader:
    """Enhanced secure audio loader with comprehensive validation"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.quarantine_dir = Path("quarantine")
        self.quarantine_dir.mkdir(exist_ok=True)
        
    def validate_file(self, filepath: Union[str, Path]) -> bool:
        """
        Comprehensive file validation with detailed security checks
        
        Args:
            filepath: Path to the audio file
            
        Returns:
            True if file passes all security checks
            
        Raises:
            Various security exceptions for different validation failures
        """
        filepath = Path(filepath)
        
        # Check file existence
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Check if it's actually a file (not a directory or symlink)
        if not filepath.is_file():
            raise ValueError(f"Path is not a regular file: {filepath}")
        
        # Check for symbolic links (potential security risk)
        if filepath.is_symlink():
            logger.warning(f"Symbolic link detected: {filepath}")
            # Resolve and validate the target
            try:
                real_path = filepath.resolve()
                if not real_path.exists():
                    raise ValueError(f"Symbolic link target does not exist: {real_path}")
            except (OSError, RuntimeError) as e:
                raise ValueError(f"Invalid symbolic link: {e}")
        
        # Path traversal protection
        if not self.config.allow_path_traversal:
            if ".." in str(filepath) or str(filepath).startswith("/"):
                raise ValueError(f"Path traversal detected in: {filepath}")
        
        # Path length validation
        if len(str(filepath)) > self.config.max_path_length:
            raise ValueError(f"Path too long: {len(str(filepath))} > {self.config.max_path_length}")
        
        if len(filepath.name) > self.config.max_filename_length:
            raise ValueError(f"Filename too long: {len(filepath.name)} > {self.config.max_filename_length}")
        
        # File size validation
        file_size_bytes = filepath.stat().st_size
        max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
        
        if file_size_bytes > max_size_bytes:
            raise ValueError(
                f"File too large: {file_size_bytes / (1024*1024):.1f}MB > "
                f"{self.config.max_file_size_mb}MB"
            )
        
        if file_size_bytes == 0:
            raise ValueError(f"Empty file: {filepath}")
        
        # Format validation
        file_extension = filepath.suffix.lower()
        if file_extension not in self.config.allowed_formats:
            raise ValueError(
                f"Unsupported format: {file_extension}. "
                f"Allowed: {', '.join(sorted(self.config.allowed_formats))}"
            )
        
        # MIME type validation (additional security layer)
        try:
            mime_type, _ = mimetypes.guess_type(str(filepath))
            if mime_type and not mime_type.startswith('audio/'):
                logger.warning(f"MIME type mismatch for {filepath}: {mime_type}")
        except Exception as e:
            logger.warning(f"MIME type detection failed for {filepath}: {e}")
        
        # File permissions check
        if not os.access(filepath, os.R_OK):
            raise PermissionError(f"File not readable: {filepath}")
        
        # Basic file header validation for common formats
        self._validate_file_header(filepath)
        
        return True
    
    def _validate_file_header(self, filepath: Path):
        """Validate file header to ensure it matches the expected format"""
        try:
            with open(filepath, 'rb') as f:
                header = f.read(16)  # Read first 16 bytes
            
            file_extension = filepath.suffix.lower()
            
            # WAV file validation
            if file_extension == '.wav':
                if not header.startswith(b'RIFF') or b'WAVE' not in header:
                    raise ValueError(f"Invalid WAV file header: {filepath}")
            
            # MP3 file validation
            elif file_extension == '.mp3':
                # MP3 files can start with ID3 tag or direct audio data
                if not (header.startswith(b'ID3') or 
                       header.startswith(b'\xff\xfb') or 
                       header.startswith(b'\xff\xfa')):
                    logger.warning(f"Suspicious MP3 file header: {filepath}")
            
            # FLAC file validation
            elif file_extension == '.flac':
                if not header.startswith(b'fLaC'):
                    raise ValueError(f"Invalid FLAC file header: {filepath}")
            
        except Exception as e:
            logger.warning(f"File header validation failed for {filepath}: {e}")
    
    def calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA-256 hash of the file for integrity checking"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {filepath}: {e}")
            return ""
    
    def quarantine_file(self, filepath: Path, reason: str) -> Path:
        """Move suspicious file to quarantine directory"""
        try:
            quarantine_path = self.quarantine_dir / f"{time.time()}_{filepath.name}"
            filepath.rename(quarantine_path)
            
            # Log quarantine action
            logger.warning(f"File quarantined: {filepath} -> {quarantine_path}. Reason: {reason}")
            
            # Create quarantine log
            log_file = quarantine_path.with_suffix('.log')
            with open(log_file, 'w') as f:
                f.write(f"QUARANTINE RECORD\n")
                f.write("=" * 30 + "\n")
                f.write(f"Original path: {filepath}\n")
                f.write(f"Quarantine time: {time.ctime()}\n")
                f.write(f"Quarantine timestamp: {time.time()}\n")
                f.write(f"Reason: {reason}\n")
                f.write(f"File size: {filepath.stat().st_size if filepath.exists() else 'unknown'} bytes\n")
                f.write(f"File extension: {filepath.suffix}\n")
                f.write("=" * 30 + "\n")
            
            # Update quarantine summary
            self._update_quarantine_summary(filepath, reason, quarantine_path)
            
            return quarantine_path
            
        except Exception as e:
            logger.error(f"Failed to quarantine file {filepath}: {e}")
            raise
    
    def _update_quarantine_summary(self, original_path: Path, reason: str, quarantine_path: Path):
        """Update quarantine summary file"""
        summary_file = self.quarantine_dir / 'logs' / 'quarantine_summary.json'
        
        # Load existing summary or create new one
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {
                'total_quarantined': 0,
                'quarantine_log': [],
                'reasons_summary': {},
                'first_quarantine': None,
                'last_quarantine': None
            }
        
        # Update summary
        timestamp = datetime.now().isoformat()
        summary['total_quarantined'] += 1
        summary['quarantine_log'].append({
            'timestamp': timestamp,
            'original_path': str(original_path),
            'quarantine_path': str(quarantine_path),
            'reason': reason,
            'file_size': original_path.stat().st_size if original_path.exists() else 0
        })
        
        # Update reasons summary
        if reason in summary['reasons_summary']:
            summary['reasons_summary'][reason] += 1
        else:
            summary['reasons_summary'][reason] = 1
        
        # Update timestamps
        if summary['first_quarantine'] is None:
            summary['first_quarantine'] = timestamp
        summary['last_quarantine'] = timestamp
        
        # Save updated summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

class ResourceLimiter:
    """Advanced resource limiting with monitoring"""
    
    def __init__(self, 
                 max_memory_gb: float = 8.0, 
                 max_time_s: float = 300.0,
                 max_cpu_percent: float = 80.0,
                 monitoring_interval: float = 1.0):
        self.max_memory_gb = max_memory_gb
        self.max_time_s = max_time_s
        self.max_cpu_percent = max_cpu_percent
        self.monitoring_interval = monitoring_interval
        
        # Get current process for monitoring
        self.process = psutil.Process()
        
    @contextmanager
    def limit_resources(self, operation_name: str = "unknown"):
        """
        Context manager to limit resources during operation.
        Works on both Windows and Unix systems.
        
        Args:
            operation_name: Name of the operation for logging
        """
        logger.info(f"Starting resource-limited operation: {operation_name}")
        
        # Store original limits (Unix only)
        original_limits = {}
        
        # Set resource limits on Unix systems
        if HAS_RESOURCE:
            try:
                # Memory limit (Unix only)
                if self.max_memory_gb:
                    memory_limit_bytes = int(self.max_memory_gb * 1024 * 1024 * 1024)
                    original_limits['memory'] = resource.getrlimit(resource.RLIMIT_AS)
                    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
                
                # CPU time limit (Unix only)
                if self.max_time_s:
                    original_limits['cpu_time'] = resource.getrlimit(resource.RLIMIT_CPU)
                    resource.setrlimit(resource.RLIMIT_CPU, (int(self.max_time_s), int(self.max_time_s)))
                
                logger.info(f"Resource limits set for Unix system")
                
            except Exception as e:
                logger.warning(f"Failed to set Unix resource limits: {e}")
        else:
            logger.info(f"Running on Windows - using psutil monitoring instead of hard limits")
        
        # Start monitoring thread for all systems
        monitor_active = True
        start_time = time.time()
        
        def monitor_resources():
            """Monitor resources and warn/terminate if limits exceeded"""
            while monitor_active:
                try:
                    # Check elapsed time
                    if time.time() - start_time > self.max_time_s:
                        logger.error(f"Operation {operation_name} exceeded time limit ({self.max_time_s}s)")
                        raise TimeoutError(f"Operation exceeded time limit of {self.max_time_s}s")
                    
                    # Check memory usage
                    memory_info = psutil.virtual_memory()
                    if memory_info.percent > 90:  # Warning at 90% system memory
                        logger.warning(f"High system memory usage: {memory_info.percent:.1f}%")
                    
                    # Check CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    if cpu_percent > self.max_cpu_percent:
                        logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                    
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    break
        
        # Start monitoring in background
        import threading
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        
        try:
            yield
        finally:
            # Stop monitoring
            monitor_active = False
            
            # Restore original limits on Unix systems
            if HAS_RESOURCE and original_limits:
                try:
                    if 'memory' in original_limits:
                        resource.setrlimit(resource.RLIMIT_AS, original_limits['memory'])
                    if 'cpu_time' in original_limits:
                        resource.setrlimit(resource.RLIMIT_CPU, original_limits['cpu_time'])
                    logger.info(f"Resource limits restored")
                except Exception as e:
                    logger.warning(f"Failed to restore resource limits: {e}")
            
            logger.info(f"Resource-limited operation completed: {operation_name}")
    
    def check_system_resources(self) -> Dict[str, Any]:
        """
        Check current system resource usage.
        
        Returns:
            Dictionary with resource usage information
        """
        try:
            # Memory information
            memory = psutil.virtual_memory()
            
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Disk information
            disk = psutil.disk_usage('/')
            
            # Process information
            process_memory = self.process.memory_info().rss
            
            return {
                'memory': {
                    'total_gb': memory.total / 1024**3,
                    'used_gb': memory.used / 1024**3,
                    'available_gb': memory.available / 1024**3,
                    'percent_used': memory.percent,
                    'can_allocate_target': memory.available > self.max_memory_gb * 1024**3
                },
                'cpu': {
                    'count': cpu_count,
                    'percent_used': cpu_percent,
                    'load_avg': getattr(psutil, 'getloadavg', lambda: [0, 0, 0])() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
                },
                'disk': {
                    'total_gb': disk.total / 1024**3,
                    'used_gb': disk.used / 1024**3,
                    'free_gb': disk.free / 1024**3,
                    'percent_used': (disk.used / disk.total) * 100
                },
                'process': {
                    'memory_mb': process_memory / 1024**2,
                    'memory_percent': (process_memory / memory.total) * 100
                },
                'limits': {
                    'max_memory_gb': self.max_memory_gb,
                    'max_time_s': self.max_time_s,
                    'max_cpu_percent': self.max_cpu_percent,
                    'has_resource_module': HAS_RESOURCE
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return {'error': str(e)}

class InputValidator:
    """Comprehensive input validation for all data types"""
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor, 
                       name: str,
                       expected_shape: Optional[tuple] = None,
                       min_val: Optional[float] = None,
                       max_val: Optional[float] = None) -> bool:
        """Validate tensor input with comprehensive checks"""
        try:
            # Basic type check
            if not torch.is_tensor(tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
            
            # Check for empty tensor
            if tensor.numel() == 0:
                raise ValueError(f"{name} is empty")
            
            # Shape validation
            if expected_shape and tensor.shape != expected_shape:
                raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}")
            
            # Check for NaN/Inf values
            if torch.isnan(tensor).any():
                raise ValueError(f"{name} contains NaN values")
            
            if torch.isinf(tensor).any():
                raise ValueError(f"{name} contains infinite values")
            
            # Value range validation
            if min_val is not None and tensor.min().item() < min_val:
                raise ValueError(f"{name} contains values below minimum: {tensor.min().item()} < {min_val}")
            
            if max_val is not None and tensor.max().item() > max_val:
                raise ValueError(f"{name} contains values above maximum: {tensor.max().item()} > {max_val}")
            
            return True
            
        except Exception as e:
            logger.error(f"Tensor validation failed for {name}: {e}")
            raise
    
    @staticmethod
    def validate_audio_params(sample_rate: int, 
                            duration: Optional[float] = None,
                            channels: Optional[int] = None) -> bool:
        """Validate audio parameters"""
        # Sample rate validation
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        
        if sample_rate < 8000 or sample_rate > 192000:
            raise ValueError(f"Sample rate out of reasonable range: {sample_rate}")
        
        # Duration validation
        if duration is not None:
            if not isinstance(duration, (int, float)) or duration <= 0:
                raise ValueError(f"Invalid duration: {duration}")
            
            if duration > 3600:  # More than 1 hour
                raise ValueError(f"Duration too long: {duration}s")
        
        # Channels validation
        if channels is not None:
            if not isinstance(channels, int) or channels <= 0:
                raise ValueError(f"Invalid channel count: {channels}")
            
            if channels > 8:  # Reasonable limit
                raise ValueError(f"Too many channels: {channels}")
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent security issues"""
        import string
        
        # Remove path separators and dangerous characters
        dangerous_chars = '<>:"|?*\\'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Limit length
        max_length = 255
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            available_length = max_length - len(ext) - 3  # Reserve for '...'
            filename = name[:available_length] + '...' + ext
        
        # Ensure it's not empty
        if not filename:
            filename = 'unnamed_file'
        
        return filename

# Global security instances for easy access
default_security_config = SecurityConfig()
secure_loader = SecureAudioLoader(default_security_config)
resource_limiter = ResourceLimiter(
    max_memory_gb=default_security_config.max_memory_gb,
    max_time_s=default_security_config.max_processing_time_s
)

if __name__ == "__main__":
    # Test security components
    print("Testing Security Validation Components")
    print("=" * 50)
    
    # Test resource limiter
    try:
        with resource_limiter.limit_resources("test_operation"):
            print("✓ Resource limiter test passed")
            time.sleep(0.1)
    except Exception as e:
        print(f"✗ Resource limiter test failed: {e}")
    
    # Test input validator
    try:
        test_tensor = torch.randn(100, 50)
        InputValidator.validate_tensor(test_tensor, "test_tensor")
        print("✓ Tensor validation test passed")
    except Exception as e:
        print(f"✗ Tensor validation test failed: {e}")
    
    # Test system resource check
    try:
        resources = resource_limiter.check_system_resources()
        print(f"✓ System resources: {resources['memory']['available_gb']:.1f}GB available")
    except Exception as e:
        print(f"✗ System resource check failed: {e}")
    
    print("\nSecurity validation components ready!") 