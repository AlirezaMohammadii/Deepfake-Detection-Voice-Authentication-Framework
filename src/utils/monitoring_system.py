"""
Comprehensive Monitoring System for Physics Features Project
Implements metrics collection, performance tracking, and health monitoring
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json
import csv
from pathlib import Path
import torch
import asyncio
from datetime import datetime, timedelta
import pickle
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MetricEntry:
    """Individual metric entry with timestamp and metadata"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = "count"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags,
            'unit': self.unit
        }

@dataclass
class HealthCheckResult:
    """Health check result with status and details"""
    name: str
    status: bool
    message: str
    timestamp: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Thread-safe metrics collector with aggregation capabilities"""
    
    def __init__(self, max_metrics_per_type: int = 10000):
        self.metrics = defaultdict(lambda: deque(maxlen=max_metrics_per_type))
        self.lock = threading.RLock()
        self.start_time = time.time()
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = "count"):
        """
        Record a metric with optional tags
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
            unit: Unit of measurement
        """
        metric = MetricEntry(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            unit=unit
        )
        
        with self.lock:
            self.metrics[name].append(metric)
    
    def record_counter(self, name: str, increment: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Record a counter metric"""
        self.record_metric(name, increment, tags, "count")
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = "value"):
        """Record a gauge metric"""
        self.record_metric(name, value, tags, unit)
    
    def record_timer(self, name: str, duration_seconds: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        self.record_metric(name, duration_seconds, tags, "seconds")
    
    def get_metric_summary(self, name: str, time_window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Get summary statistics for a metric
        
        Args:
            name: Metric name
            time_window_seconds: Optional time window to consider (default: all)
            
        Returns:
            Summary statistics dictionary
        """
        with self.lock:
            if name not in self.metrics:
                return {'error': f'Metric {name} not found'}
            
            metric_entries = list(self.metrics[name])
            
            # Filter by time window if specified
            if time_window_seconds:
                cutoff_time = time.time() - time_window_seconds
                metric_entries = [m for m in metric_entries if m.timestamp >= cutoff_time]
            
            if not metric_entries:
                return {'error': 'No metrics in specified time window'}
            
            values = [m.value for m in metric_entries]
            
            return {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'sum': np.sum(values),
                'p50': np.percentile(values, 50),
                'p90': np.percentile(values, 90),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99),
                'first_timestamp': metric_entries[0].timestamp,
                'last_timestamp': metric_entries[-1].timestamp,
                'rate_per_second': len(values) / max(metric_entries[-1].timestamp - metric_entries[0].timestamp, 1.0)
            }
    
    def get_all_metrics_summary(self, time_window_seconds: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """Get summary for all metrics"""
        with self.lock:
            return {name: self.get_metric_summary(name, time_window_seconds) 
                   for name in self.metrics.keys()}
    
    def export_metrics_csv(self, filepath: Path, time_window_seconds: Optional[float] = None):
        """Export metrics to CSV file"""
        with self.lock:
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['name', 'value', 'timestamp', 'tags', 'unit']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                cutoff_time = time.time() - time_window_seconds if time_window_seconds else 0
                
                for name, metric_list in self.metrics.items():
                    for metric in metric_list:
                        if metric.timestamp >= cutoff_time:
                            row = metric.to_dict()
                            row['tags'] = json.dumps(row['tags'])
                            writer.writerow(row)
    
    def clear_metrics(self, metric_name: Optional[str] = None):
        """Clear metrics (specific metric or all)"""
        with self.lock:
            if metric_name:
                if metric_name in self.metrics:
                    self.metrics[metric_name].clear()
            else:
                self.metrics.clear()

class PerformanceMonitor:
    """Performance monitoring with automatic system metrics collection"""
    
    def __init__(self, metrics_collector: MetricsCollector, collection_interval: float = 1.0):
        self.metrics = metrics_collector
        self.collection_interval = collection_interval
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start automatic system metrics collection"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop automatic monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            self.metrics.record_gauge("system.cpu.percent", cpu_percent, unit="percent")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.record_gauge("system.memory.used_gb", memory.used / (1024**3), unit="gb")
            self.metrics.record_gauge("system.memory.available_gb", memory.available / (1024**3), unit="gb")
            self.metrics.record_gauge("system.memory.percent", memory.percent, unit="percent")
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics.record_gauge("system.disk.used_gb", disk.used / (1024**3), unit="gb")
            self.metrics.record_gauge("system.disk.free_gb", disk.free / (1024**3), unit="gb")
            self.metrics.record_gauge("system.disk.percent", (disk.used / disk.total) * 100, unit="percent")
            
            # GPU metrics (if available)
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        
                        self.metrics.record_gauge(
                            "system.gpu.memory_allocated_gb", 
                            memory_allocated, 
                            tags={"device": f"cuda:{i}"}, 
                            unit="gb"
                        )
                        self.metrics.record_gauge(
                            "system.gpu.memory_reserved_gb", 
                            memory_reserved, 
                            tags={"device": f"cuda:{i}"}, 
                            unit="gb"
                        )
                    except Exception as gpu_error:
                        logger.debug(f"GPU metrics collection failed for device {i}: {gpu_error}")
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")

class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.health_checks = {}
        self.check_history = deque(maxlen=1000)
        
    def register_health_check(self, name: str, check_function: Callable[[], bool], 
                            timeout_seconds: float = 5.0):
        """
        Register a health check function
        
        Args:
            name: Name of the health check
            check_function: Function that returns True if healthy
            timeout_seconds: Timeout for the check
        """
        self.health_checks[name] = {
            'function': check_function,
            'timeout': timeout_seconds
        }
    
    async def run_health_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if name not in self.health_checks:
            return HealthCheckResult(
                name=name,
                status=False,
                message=f"Health check '{name}' not found",
                timestamp=time.time(),
                duration_ms=0
            )
        
        check_config = self.health_checks[name]
        start_time = time.time()
        
        try:
            # Run check with timeout
            check_function = check_config['function']
            timeout = check_config['timeout']
            
            if asyncio.iscoroutinefunction(check_function):
                result = await asyncio.wait_for(check_function(), timeout=timeout)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, check_function),
                    timeout=timeout
                )
            
            duration_ms = (time.time() - start_time) * 1000
            
            health_result = HealthCheckResult(
                name=name,
                status=bool(result),
                message="OK" if result else "Check failed",
                timestamp=time.time(),
                duration_ms=duration_ms
            )
            
        except asyncio.TimeoutError:
            health_result = HealthCheckResult(
                name=name,
                status=False,
                message=f"Health check timed out after {timeout}s",
                timestamp=time.time(),
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            health_result = HealthCheckResult(
                name=name,
                status=False,
                message=f"Health check error: {str(e)}",
                timestamp=time.time(),
                duration_ms=(time.time() - start_time) * 1000
            )
        
        self.check_history.append(health_result)
        return health_result
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        # Run all checks concurrently
        tasks = {
            name: self.run_health_check(name) 
            for name in self.health_checks.keys()
        }
        
        completed_results = await asyncio.gather(
            *tasks.values(), 
            return_exceptions=True
        )
        
        for name, result in zip(tasks.keys(), completed_results):
            if isinstance(result, Exception):
                results[name] = HealthCheckResult(
                    name=name,
                    status=False,
                    message=f"Health check exception: {str(result)}",
                    timestamp=time.time(),
                    duration_ms=0
                )
            else:
                results[name] = result
        
        return results
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        recent_checks = [check for check in self.check_history 
                        if check.timestamp > time.time() - 300]  # Last 5 minutes
        
        if not recent_checks:
            return {'status': 'unknown', 'message': 'No recent health checks'}
        
        # Group by check name
        by_name = defaultdict(list)
        for check in recent_checks:
            by_name[check.name].append(check)
        
        summary = {
            'overall_status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }
        
        for name, checks in by_name.items():
            latest_check = max(checks, key=lambda x: x.timestamp)
            success_rate = sum(1 for check in checks if check.status) / len(checks)
            
            summary['checks'][name] = {
                'status': latest_check.status,
                'message': latest_check.message,
                'success_rate': success_rate,
                'avg_duration_ms': np.mean([check.duration_ms for check in checks]),
                'last_check': latest_check.timestamp
            }
            
            # Update overall status
            if not latest_check.status or success_rate < 0.8:
                summary['overall_status'] = 'unhealthy'
        
        return summary

class MonitoringSystem:
    """Complete monitoring system combining metrics, performance, and health"""
    
    def __init__(self, output_dir: str = "monitoring"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.metrics = MetricsCollector()
        self.performance_monitor = PerformanceMonitor(self.metrics)
        self.health_checker = HealthChecker()
        
        # Register default health checks
        self._register_default_health_checks()
        
        # Monitoring state
        self.monitoring_active = False
        
    def start(self):
        """Start the complete monitoring system"""
        self.performance_monitor.start_monitoring()
        self.monitoring_active = True
        logger.info("Monitoring system started")
    
    def stop(self):
        """Stop the monitoring system"""
        self.performance_monitor.stop_monitoring()
        self.monitoring_active = False
        logger.info("Monitoring system stopped")
    
    def _register_default_health_checks(self):
        """Register default health checks"""
        
        def check_cpu_usage():
            """Check if CPU usage is reasonable"""
            return psutil.cpu_percent(interval=1) < 90.0
        
        def check_memory_usage():
            """Check if memory usage is reasonable"""
            return psutil.virtual_memory().percent < 90.0
        
        def check_disk_space():
            """Check if disk space is sufficient"""
            return psutil.disk_usage('/').percent < 90.0
        
        def check_torch_availability():
            """Check if PyTorch is working"""
            try:
                return torch.tensor([1.0]).sum().item() == 1.0
            except Exception:
                return False
        
        # Register checks
        self.health_checker.register_health_check("cpu_usage", check_cpu_usage)
        self.health_checker.register_health_check("memory_usage", check_memory_usage)
        self.health_checker.register_health_check("disk_space", check_disk_space)
        self.health_checker.register_health_check("torch", check_torch_availability)
        
        # GPU check if available
        if torch.cuda.is_available():
            def check_gpu():
                try:
                    return torch.cuda.is_available() and torch.randn(10, device='cuda').sum().item() != 0
                except Exception:
                    return False
            
            self.health_checker.register_health_check("gpu", check_gpu)
    
    def record_processing_metrics(self, 
                                processing_time: float,
                                file_count: int,
                                success_count: int,
                                error_count: int,
                                file_type: str = "unknown"):
        """Record metrics for a processing session"""
        tags = {"file_type": file_type}
        
        self.metrics.record_timer("processing.total_time", processing_time, tags)
        self.metrics.record_counter("processing.files_total", file_count, tags)
        self.metrics.record_counter("processing.files_success", success_count, tags)
        self.metrics.record_counter("processing.files_error", error_count, tags)
        
        if file_count > 0:
            self.metrics.record_gauge("processing.success_rate", success_count / file_count, tags, "ratio")
            self.metrics.record_gauge("processing.avg_time_per_file", processing_time / file_count, tags, "seconds")
    
    def record_feature_extraction_metrics(self, 
                                        extraction_time: float,
                                        cache_hit: bool,
                                        feature_types: List[str]):
        """Record feature extraction specific metrics"""
        self.metrics.record_timer("feature_extraction.time", extraction_time)
        self.metrics.record_counter("feature_extraction.cache_hits" if cache_hit else "feature_extraction.cache_misses", 1)
        
        for feature_type in feature_types:
            self.metrics.record_counter("feature_extraction.features", 1, {"type": feature_type})
    
    async def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report"""
        # Get health status
        health_results = await self.health_checker.run_all_health_checks()
        health_summary = self.health_checker.get_health_summary()
        
        # Get metrics summary
        metrics_summary = self.metrics.get_all_metrics_summary(time_window_seconds=3600)  # Last hour
        
        # System information
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'uptime_seconds': time.time() - self.metrics.start_time
        }
        
        return {
            'timestamp': time.time(),
            'health': health_summary,
            'health_checks': {name: result.__dict__ for name, result in health_results.items()},
            'metrics': metrics_summary,
            'system': system_info,
            'monitoring_active': self.monitoring_active
        }
    
    def save_monitoring_report(self, report: Optional[Dict[str, Any]] = None):
        """Save monitoring report to file"""
        if report is None:
            # Must be called from async context
            raise ValueError("Report must be provided for sync save")
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"monitoring_report_{timestamp_str}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Monitoring report saved to {report_file}")
        
        # Also export metrics to CSV
        metrics_file = self.output_dir / f"metrics_{timestamp_str}.csv"
        self.metrics.export_metrics_csv(metrics_file, time_window_seconds=3600)
        
        logger.info(f"Metrics exported to {metrics_file}")

# Context manager for easy monitoring
class MonitoringContext:
    """Context manager for automatic monitoring during operations"""
    
    def __init__(self, monitoring_system: MonitoringSystem, operation_name: str):
        self.monitoring = monitoring_system
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.monitoring.metrics.record_counter(f"operations.{self.operation_name}.started", 1)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.monitoring.metrics.record_timer(f"operations.{self.operation_name}.duration", duration)
        
        if exc_type is None:
            self.monitoring.metrics.record_counter(f"operations.{self.operation_name}.success", 1)
        else:
            self.monitoring.metrics.record_counter(f"operations.{self.operation_name}.error", 1)
            self.monitoring.metrics.record_counter(f"operations.{self.operation_name}.errors.{exc_type.__name__}", 1)

# Global monitoring instance
global_monitoring = MonitoringSystem()

def get_monitoring_system() -> MonitoringSystem:
    """Get the global monitoring system instance"""
    return global_monitoring

if __name__ == "__main__":
    # Test monitoring system
    async def test_monitoring():
        print("Testing Monitoring System")
        print("=" * 50)
        
        # Initialize monitoring
        monitoring = MonitoringSystem()
        monitoring.start()
        
        # Record some test metrics
        monitoring.metrics.record_counter("test.counter", 5)
        monitoring.metrics.record_gauge("test.gauge", 42.5, unit="units")
        monitoring.metrics.record_timer("test.timer", 1.234)
        
        # Wait for system metrics collection
        await asyncio.sleep(2)
        
        # Get monitoring report
        report = await monitoring.get_monitoring_report()
        
        print(f"Health Status: {report['health']['overall_status']}")
        print(f"Active Checks: {len(report['health_checks'])}")
        print(f"Metrics Collected: {len(report['metrics'])}")
        print(f"System Uptime: {report['system']['uptime_seconds']:.1f}s")
        
        # Test context manager
        with MonitoringContext(monitoring, "test_operation"):
            await asyncio.sleep(0.1)
        
        monitoring.stop()
        print("✓ Monitoring system test completed")
    
    # Run test
    asyncio.run(test_monitoring()) 