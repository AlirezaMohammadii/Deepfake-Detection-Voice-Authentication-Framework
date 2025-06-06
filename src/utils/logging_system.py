"""
Comprehensive Logging and Output System for Physics Features Project
Populates output, log, checkpoints, and plots folders with relevant information
"""

import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pickle
import warnings

# Configure matplotlib backend
plt.switch_backend('Agg')  # Non-interactive backend for server environments

class ProjectLogger:
    """Comprehensive logging system for the physics features project."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.setup_directories()
        self.setup_logging()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def setup_directories(self):
        """Create and setup all logging directories."""
        self.dirs = {
            'output': self.project_root / 'output',
            'logs': self.project_root / 'logs', 
            'checkpoints': self.project_root / 'checkpoints',
            'plots': self.project_root / 'plots',
            'results': self.project_root / 'results',
        }
        
        # Create directories if they don't exist
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
            
        # Create subdirectories for organized storage
        (self.dirs['plots'] / 'physics_analysis').mkdir(exist_ok=True)
        (self.dirs['plots'] / 'performance').mkdir(exist_ok=True)
        (self.dirs['plots'] / 'validation').mkdir(exist_ok=True)
        (self.dirs['logs'] / 'feature_extraction').mkdir(exist_ok=True)
        (self.dirs['logs'] / 'model_performance').mkdir(exist_ok=True)
        (self.dirs['output'] / 'feature_summaries').mkdir(exist_ok=True)
        (self.dirs['output'] / 'analysis_reports').mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Setup comprehensive logging configuration."""
        # Main application logger
        self.logger = logging.getLogger('physics_features')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for detailed logs
        detailed_log_file = self.dirs['logs'] / f'detailed_{self.session_id}.log'
        file_handler = logging.FileHandler(detailed_log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # Separate handlers for different components
        self.setup_component_loggers()
        
    def setup_component_loggers(self):
        """Setup specialized loggers for different components."""
        components = {
            'feature_extraction': logging.DEBUG,
            'physics_calculation': logging.DEBUG,
            'model_loading': logging.INFO,
            'validation': logging.INFO,
            'performance': logging.INFO
        }
        
        self.component_loggers = {}
        for component, level in components.items():
            logger = logging.getLogger(f'physics_features.{component}')
            logger.setLevel(level)
            
            # Component-specific log file
            log_file = self.dirs['logs'] / component / f'{component}_{self.session_id}.log'
            log_file.parent.mkdir(exist_ok=True)
            
            handler = logging.FileHandler(log_file)
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            ))
            logger.addHandler(handler)
            
            self.component_loggers[component] = logger
    
    def log_session_start(self, config: Dict[str, Any]):
        """Log session start with configuration."""
        self.logger.info("="*60)
        self.logger.info(f"PHYSICS FEATURES SESSION STARTED - {self.session_id}")
        self.logger.info("="*60)
        
        # Save session configuration
        config_file = self.dirs['output'] / f'session_config_{self.session_id}.json'
        with open(config_file, 'w') as f:
            # Convert torch tensors and other non-serializable objects
            serializable_config = self._make_serializable(config)
            json.dump(serializable_config, f, indent=2)
        
        self.logger.info(f"Session configuration saved to: {config_file}")
        
        # Create detailed session log in sessions subdirectory
        session_log_file = self.dirs['logs'] / 'sessions' / f'session_{self.session_id}.log'
        with open(session_log_file, 'w') as f:
            f.write(f"PHYSICS FEATURES ANALYSIS SESSION LOG\n")
            f.write("=" * 50 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start Time: {datetime.now().isoformat()}\n")
            f.write(f"System Configuration:\n")
            for key, value in serializable_config.items():
                f.write(f"  {key}: {value}\n")
            f.write("=" * 50 + "\n\n")
        
        # Log system information
        self.logger.info(f"Python executable: {os.sys.executable}")
        self.logger.info(f"Working directory: {os.getcwd()}")
        self.logger.info(f"Device configuration: {config.get('device', 'unknown')}")
        if 'audio_config' in config:
            self.logger.info(f"Audio sample rate: {config['audio_config'].get('sample_rate', 'unknown')}")
        if 'physics_config' in config:
            self.logger.info(f"Physics window: {config['physics_config'].get('time_window_for_dynamics_ms', 'unknown')}ms")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif torch.is_tensor(obj):
            return f"Tensor{list(obj.shape)}"
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def log_feature_extraction(self, filepath: str, features: Dict[str, Any], 
                             processing_time: float, success: bool):
        """Log detailed feature extraction information."""
        fe_logger = self.component_loggers['feature_extraction']
        
        if success:
            fe_logger.info(f"SUCCESS: {filepath} processed in {processing_time:.2f}s")
            
            # Log feature statistics
            if 'physics' in features:
                physics = features['physics']
                fe_logger.debug(f"Physics features: {list(physics.keys())}")
                for key, value in physics.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        fe_logger.debug(f"  {key}: {value.item():.6f}")
            
            if 'hubert_sequence' in features:
                hubert = features['hubert_sequence']
                fe_logger.debug(f"HuBERT: shape={hubert.shape}, mean={hubert.mean().item():.4f}")
                
        else:
            error_msg = features.get('error', 'Unknown error')
            fe_logger.error(f"FAILED: {filepath} - {error_msg}")
    
    def create_physics_analysis_plots(self, results_df: pd.DataFrame):
        """Create comprehensive physics feature analysis plots."""
        if results_df.empty:
            self.logger.warning("No data available for plotting")
            return
            
        self.logger.info("Creating physics analysis plots...")
        
        # Physics features columns
        physics_cols = [col for col in results_df.columns if col.startswith('physics_')]
        
        if not physics_cols:
            self.logger.warning("No physics features found for plotting")
            return
        
        # 1. Distribution plots for each physics feature
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Physics Features Distribution Analysis', fontsize=16)
        
        key_features = [
            'physics_delta_ft_revised',
            'physics_delta_fr_revised', 
            'physics_delta_fv_revised',
            'physics_delta_f_total_revised'
        ]
        
        for i, feature in enumerate(key_features):
            if feature in results_df.columns and i < 4:
                ax = axes[i//2, i%2]
                
                # Convert to numeric, handling any string values
                numeric_data = pd.to_numeric(results_df[feature], errors='coerce')
                numeric_data = numeric_data.dropna()
                
                if len(numeric_data) > 0:
                    # Distribution plot
                    sns.histplot(data=numeric_data, ax=ax, kde=True)
                    ax.set_title(f'{feature.replace("physics_", "").replace("_", " ").title()}')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    
                    # Add statistics text
                    stats_text = f'Mean: {numeric_data.mean():.4f}\nStd: {numeric_data.std():.4f}\nMin: {numeric_data.min():.4f}\nMax: {numeric_data.max():.4f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, ha='center')
                    ax.set_title(f'{feature} (No Data)')
        
        plt.tight_layout()
        plot_file = self.dirs['plots'] / 'physics_analysis' / f'physics_distributions_{self.session_id}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Comparison by file type
        if 'file_type' in results_df.columns and len(results_df['file_type'].unique()) > 1:
            self.create_filetype_comparison_plots(results_df, physics_cols)
        
        # 3. Correlation matrix
        self.create_correlation_plot(results_df, physics_cols)
        
        # 4. Performance metrics
        self.create_performance_plots(results_df)
        
        self.logger.info(f"Physics analysis plots saved to: {self.dirs['plots'] / 'physics_analysis'}")
    
    def create_filetype_comparison_plots(self, results_df: pd.DataFrame, physics_cols: List[str]):
        """Create plots comparing physics features across file types."""
        file_types = results_df['file_type'].unique()
        
        # Select key physics features for comparison
        key_features = [col for col in physics_cols if any(key in col for key in ['delta_ft', 'delta_fr', 'delta_fv'])][:4]
        
        if not key_features:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Physics Features by File Type', fontsize=16)
        
        for i, feature in enumerate(key_features):
            if i < 4:
                ax = axes[i//2, i%2]
                
                # Prepare data for plotting
                plot_data = []
                labels = []
                
                for file_type in file_types:
                    type_data = results_df[results_df['file_type'] == file_type][feature]
                    numeric_data = pd.to_numeric(type_data, errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        plot_data.append(numeric_data.values)
                        labels.append(f'{file_type}\n(n={len(numeric_data)})')
                
                if plot_data:
                    ax.boxplot(plot_data, labels=labels)
                    ax.set_title(feature.replace('physics_', '').replace('_', ' ').title())
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, ha='center')
        
        plt.tight_layout()
        plot_file = self.dirs['plots'] / 'physics_analysis' / f'filetype_comparison_{self.session_id}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_correlation_plot(self, results_df: pd.DataFrame, physics_cols: List[str]):
        """Create correlation matrix plot for physics features."""
        # Select numeric physics columns
        numeric_cols = []
        for col in physics_cols[:8]:  # Limit to first 8 for readability
            if col in results_df.columns:
                numeric_data = pd.to_numeric(results_df[col], errors='coerce')
                if not numeric_data.isna().all():
                    numeric_cols.append(col)
        
        if len(numeric_cols) < 2:
            return
            
        # Calculate correlation matrix
        correlation_data = results_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        correlation_matrix = correlation_data.corr()
        
        # Create plot
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .5})
        plt.title('Physics Features Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plot_file = self.dirs['plots'] / 'physics_analysis' / f'correlation_matrix_{self.session_id}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_plots(self, results_df: pd.DataFrame):
        """Create performance analysis plots."""
        performance_metrics = ['processing_time', 'audio_duration_s', 'hubert_seq_len_frames']
        available_metrics = [col for col in performance_metrics if col in results_df.columns]
        
        if not available_metrics:
            return
            
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 5))
        if len(available_metrics) == 1:
            axes = [axes]
            
        for i, metric in enumerate(available_metrics):
            numeric_data = pd.to_numeric(results_df[metric], errors='coerce').dropna()
            
            if len(numeric_data) > 0:
                axes[i].hist(numeric_data, bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
                axes[i].set_xlabel(metric.replace("_", " ").title())
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics
                stats_text = f'Mean: {numeric_data.mean():.3f}\nMedian: {numeric_data.median():.3f}'
                axes[i].text(0.98, 0.98, stats_text, transform=axes[i].transAxes, 
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plot_file = self.dirs['plots'] / 'performance' / f'performance_metrics_{self.session_id}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_feature_summary(self, results_df: pd.DataFrame):
        """Save comprehensive feature analysis summary."""
        if results_df.empty:
            return
            
        self.logger.info("Creating feature analysis summary...")
        
        summary = {
            'session_info': {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'total_files': len(results_df),
                'successful_files': len(results_df[results_df['status'] == 'success'])
            },
            'file_type_distribution': results_df['file_type'].value_counts().to_dict() if 'file_type' in results_df.columns else {},
            'processing_stats': {},
            'physics_feature_stats': {},
            'audio_characteristics': {}
        }
        
        # Processing statistics
        if 'processing_time' in results_df.columns:
            proc_times = pd.to_numeric(results_df['processing_time'], errors='coerce').dropna()
            if len(proc_times) > 0:
                summary['processing_stats'] = {
                    'mean_processing_time': float(proc_times.mean()),
                    'median_processing_time': float(proc_times.median()),
                    'total_processing_time': float(proc_times.sum()),
                    'min_processing_time': float(proc_times.min()),
                    'max_processing_time': float(proc_times.max())
                }
        
        # Physics feature statistics
        physics_cols = [col for col in results_df.columns if col.startswith('physics_')]
        for col in physics_cols:
            numeric_data = pd.to_numeric(results_df[col], errors='coerce').dropna()
            if len(numeric_data) > 0:
                summary['physics_feature_stats'][col] = {
                    'mean': float(numeric_data.mean()),
                    'std': float(numeric_data.std()),
                    'min': float(numeric_data.min()),
                    'max': float(numeric_data.max()),
                    'count': len(numeric_data)
                }
        
        # Audio characteristics
        audio_cols = ['audio_duration_s', 'hubert_seq_len_frames', 'hubert_embedding_dim']
        for col in audio_cols:
            if col in results_df.columns:
                numeric_data = pd.to_numeric(results_df[col], errors='coerce').dropna()
                if len(numeric_data) > 0:
                    summary['audio_characteristics'][col] = {
                        'mean': float(numeric_data.mean()),
                        'median': float(numeric_data.median()),
                        'range': [float(numeric_data.min()), float(numeric_data.max())]
                    }
        
        # Save summary
        summary_file = self.dirs['output'] / 'feature_summaries' / f'feature_summary_{self.session_id}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save as readable text report
        self.create_text_report(summary)
        
        self.logger.info(f"Feature summary saved to: {summary_file}")
    
    def create_text_report(self, summary: Dict[str, Any]):
        """Create human-readable text report."""
        report_lines = [
            "PHYSICS FEATURES ANALYSIS REPORT",
            "=" * 50,
            f"Generated: {summary['session_info']['timestamp']}",
            f"Session ID: {summary['session_info']['session_id']}",
            "",
            "PROCESSING SUMMARY",
            "-" * 20,
            f"Total files processed: {summary['session_info']['total_files']}",
            f"Successful extractions: {summary['session_info']['successful_files']}",
            f"Success rate: {summary['session_info']['successful_files'] / summary['session_info']['total_files'] * 100:.1f}%",
            ""
        ]
        
        # File type distribution
        if summary['file_type_distribution']:
            report_lines.extend([
                "FILE TYPE DISTRIBUTION",
                "-" * 20
            ])
            for file_type, count in summary['file_type_distribution'].items():
                report_lines.append(f"{file_type}: {count} files")
            report_lines.append("")
        
        # Processing statistics
        if summary['processing_stats']:
            report_lines.extend([
                "PERFORMANCE METRICS",
                "-" * 20,
                f"Average processing time: {summary['processing_stats']['mean_processing_time']:.2f}s",
                f"Total processing time: {summary['processing_stats']['total_processing_time']:.1f}s",
                f"Fastest file: {summary['processing_stats']['min_processing_time']:.2f}s",
                f"Slowest file: {summary['processing_stats']['max_processing_time']:.2f}s",
                ""
            ])
        
        # Physics features summary
        if summary['physics_feature_stats']:
            report_lines.extend([
                "PHYSICS FEATURES ANALYSIS",
                "-" * 30
            ])
            for feature, stats in summary['physics_feature_stats'].items():
                clean_name = feature.replace('physics_', '').replace('_', ' ').title()
                report_lines.extend([
                    f"{clean_name}:",
                    f"  Mean ± Std: {stats['mean']:.6f} ± {stats['std']:.6f}",
                    f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]",
                    f"  Valid samples: {stats['count']}",
                    ""
                ])
        
        # Save report
        report_file = self.dirs['output'] / 'analysis_reports' / f'analysis_report_{self.session_id}.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
    
    def save_checkpoint_info(self, checkpoint_stats: Dict[str, Any]):
        """Save checkpoint information for recovery."""
        checkpoint_info = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_stats': checkpoint_stats
        }
        
        info_file = self.dirs['checkpoints'] / f'checkpoint_info_{self.session_id}.json'
        with open(info_file, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
    
    def log_session_end(self, final_stats: Dict[str, Any]):
        """Log session completion with final statistics."""
        self.logger.info("="*60)
        self.logger.info("PHYSICS FEATURES SESSION COMPLETED")
        self.logger.info("="*60)
        
        for key, value in final_stats.items():
            self.logger.info(f"{key}: {value}")
        
        # Save final session stats
        stats_file = self.dirs['output'] / f'session_stats_{self.session_id}.json'
        with open(stats_file, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'completion_time': datetime.now().isoformat(),
                'final_stats': final_stats
            }, f, indent=2)
        
        self.logger.info(f"Session completed - ID: {self.session_id}")

# Factory function for easy integration
def create_project_logger(project_root: str = ".") -> ProjectLogger:
    """Create and initialize project logger."""
    return ProjectLogger(project_root)

if __name__ == "__main__":
    # Test the logging system
    logger = create_project_logger()
    
    # Simulate some data
    test_data = {
        'filepath': ['test1.wav', 'test2.wav'],
        'file_type': ['genuine', 'deepfake_tts'],
        'status': ['success', 'success'],
        'processing_time': [1.2, 1.5],
        'physics_delta_ft_revised': [0.1, 0.12],
        'physics_delta_fr_revised': [7.2, 7.3],
        'audio_duration_s': [3.0, 2.8]
    }
    
    df = pd.DataFrame(test_data)
    
    logger.log_session_start({'test': True})
    logger.create_physics_analysis_plots(df)
    logger.save_feature_summary(df)
    logger.log_session_end({'total_files': 2, 'success_rate': 100.0})
    
    print("Logging system test completed!") 