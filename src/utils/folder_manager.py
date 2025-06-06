"""
Comprehensive Folder Management System for Physics Features Project
Ensures all directories are properly created and populated with useful information
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import shutil


class FolderManager:
    """
    Centralized folder management system for the physics features project.
    Ensures all directories are created and populated with relevant information.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define all project folders with their purposes
        self.folders = {
            'results': {
                'path': self.project_root / 'results',
                'description': 'Main analysis results and CSV outputs',
                'subdirs': ['checkpoints'],
                'auto_populate': True
            },
            'checkpoints': {
                'path': self.project_root / 'checkpoints',
                'description': 'Processing checkpoints for recovery and resumption',
                'subdirs': [],
                'auto_populate': True
            },
            'logs': {
                'path': self.project_root / 'logs',
                'description': 'Comprehensive logging for all system components',
                'subdirs': ['feature_extraction', 'model_performance', 'system', 'sessions'],
                'auto_populate': True
            },
            'output': {
                'path': self.project_root / 'output',
                'description': 'Analysis reports, summaries, and documentation',
                'subdirs': ['feature_summaries', 'analysis_reports', 'session_reports'],
                'auto_populate': True
            },
            'plots': {
                'path': self.project_root / 'plots',
                'description': 'Matplotlib plots and statistical visualizations',
                'subdirs': ['physics_analysis', 'performance', 'validation', 'comparisons'],
                'auto_populate': True
            },
            'quarantine': {
                'path': self.project_root / 'quarantine',
                'description': 'Quarantined files that failed security validation',
                'subdirs': ['logs'],
                'auto_populate': True
            },
            'cache': {
                'path': self.project_root / 'cache',
                'description': 'Cached model outputs and intermediate results',
                'subdirs': ['hubert', 'features', 'processing'],
                'auto_populate': True
            },
            'visualizations': {
                'path': self.project_root / 'visualizations',
                'description': 'Enhanced interactive and static visualizations',
                'subdirs': ['static', 'interactive', 'reports'],
                'auto_populate': False
            }
        }
    
    def setup_all_folders(self) -> Dict[str, Any]:
        """
        Create and initialize all project folders with documentation.
        
        Returns:
            Dictionary with setup results and statistics
        """
        setup_stats = {
            'folders_created': 0,
            'subdirs_created': 0,
            'files_created': 0,
            'session_id': self.session_id,
            'setup_time': datetime.now().isoformat()
        }
        
        print("🗂️  Setting up project folder structure...")
        
        for folder_name, config in self.folders.items():
            folder_path = config['path']
            
            # Create main folder
            if not folder_path.exists():
                folder_path.mkdir(exist_ok=True)
                setup_stats['folders_created'] += 1
                print(f"   ✅ Created: {folder_name}/")
            
            # Create subdirectories
            for subdir in config['subdirs']:
                subdir_path = folder_path / subdir
                if not subdir_path.exists():
                    subdir_path.mkdir(exist_ok=True)
                    setup_stats['subdirs_created'] += 1
                    print(f"   📁 Created: {folder_name}/{subdir}/")
            
            # Create README file for folder documentation
            readme_path = folder_path / 'README.md'
            if not readme_path.exists():
                self._create_folder_readme(folder_name, config, readme_path)
                setup_stats['files_created'] += 1
            
            # Auto-populate with sample/informational content if needed
            if config['auto_populate']:
                populated_files = self._populate_folder(folder_name, config)
                setup_stats['files_created'] += populated_files
        
        # Create master folder index
        self._create_folder_index(setup_stats)
        setup_stats['files_created'] += 1
        
        print(f"✅ Folder setup complete!")
        print(f"   📁 Folders: {setup_stats['folders_created']} created")
        print(f"   📂 Subdirs: {setup_stats['subdirs_created']} created") 
        print(f"   📄 Files: {setup_stats['files_created']} created")
        
        return setup_stats
    
    def _create_folder_readme(self, folder_name: str, config: Dict[str, Any], readme_path: Path):
        """Create informative README file for each folder."""
        content = f"""# {folder_name.title()} Directory

## Purpose
{config['description']}

## Structure
```
{folder_name}/
├── README.md (this file)
"""
        
        # Add subdirectories to structure
        for subdir in config['subdirs']:
            content += f"├── {subdir}/\n"
        
        content += "```\n\n"
        
        # Add specific information based on folder type
        if folder_name == 'results':
            content += """## Contents
- `physics_features_summary.csv` - Main analysis results
- `error_log.txt` - Processing error logs
- `checkpoints/` - Processing checkpoint files

## Usage
This directory contains the primary outputs from audio analysis runs.
Check the CSV file for feature extraction results and statistics.
"""

        elif folder_name == 'checkpoints':
            content += """## Contents
Checkpoint files are created during processing to enable recovery:
- `checkpoint.pkl` - Current processing state
- `checkpoint.tmp` - Temporary checkpoint during saving

## Usage
Checkpoints allow resuming interrupted processing sessions.
Files are automatically created and cleaned up by the system.
"""

        elif folder_name == 'logs':
            content += """## Contents
- `feature_extraction/` - Feature extraction process logs
- `model_performance/` - Model loading and performance logs
- `system/` - System-level operation logs
- `sessions/` - Complete session logs

## Log Levels
- DEBUG: Detailed diagnostic information
- INFO: General information about operations
- WARNING: Warning messages for potential issues
- ERROR: Error messages for failures
"""

        elif folder_name == 'output':
            content += """## Contents
- `feature_summaries/` - Statistical summaries of extracted features
- `analysis_reports/` - Comprehensive analysis reports
- `session_reports/` - Session-specific outputs

## File Types
- JSON: Machine-readable analysis results
- TXT: Human-readable summaries
- CSV: Tabular statistical data
"""

        elif folder_name == 'plots':
            content += """## Contents
- `physics_analysis/` - Physics feature distribution and correlation plots
- `performance/` - System performance and timing analysis
- `validation/` - Data validation and quality checks
- `comparisons/` - Feature comparison between file types

## Plot Types
- PNG: High-resolution static plots (300 DPI)
- PDF: Vector graphics for publications
- Interactive plots saved in visualizations/ directory
"""

        elif folder_name == 'quarantine':
            content += """## Contents
Files that fail security validation are moved here:
- Suspicious or corrupted audio files
- Files with invalid headers or formats
- Oversized files exceeding limits

## Security Logs
Each quarantined file has an associated .log file explaining the reason.
"""

        elif folder_name == 'cache':
            content += """## Contents
This directory stores cached model outputs and intermediate processing results:
- `hubert/` - HuBERT model embedding cache (reserved for future use)
- `features/` - Extracted feature cache (reserved for future use)  
- `processing/` - Processing state cache (reserved for future use)
- `*.pkl` files - Current active cache files (MD5-hashed filenames)

## Cache Structure
Currently, the system stores cache files directly in the root cache directory using MD5 hashes of the input audio for fast lookup. The subdirectories are reserved for future organizational improvements where different types of cached data may be separated.

## Cache Management
- Cache files are automatically created during feature extraction
- Each file is ~721KB containing HuBERT embeddings and computed features
- Cache key is based on audio content and configuration for consistency
- Old cache files can be safely removed to free disk space

## Future Development
The subdirectories (hubert/, features/, processing/) are reserved for:
- `hubert/`: Raw HuBERT model outputs only
- `features/`: Computed physics and audio features only
- `processing/`: Temporary processing state and recovery data
"""

        content += f"""
## Last Updated
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Session ID
{self.session_id}
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _populate_folder(self, folder_name: str, config: Dict[str, Any]) -> int:
        """Populate folder with useful initial content."""
        files_created = 0
        folder_path = config['path']
        
        if folder_name == 'checkpoints':
            # Create checkpoint status file
            status_file = folder_path / 'checkpoint_status.json'
            if not status_file.exists():
                status_data = {
                    'checkpoint_system': 'active',
                    'last_session': self.session_id,
                    'checkpoint_directory': str(folder_path),
                    'retention_policy': 'Checkpoints cleared on successful completion',
                    'recovery_info': 'Use test_runner.py to resume from checkpoints',
                    'created': datetime.now().isoformat()
                }
                with open(status_file, 'w') as f:
                    json.dump(status_data, f, indent=2)
                files_created += 1
        
        elif folder_name == 'logs':
            # Create initial system log
            system_log_dir = folder_path / 'system'
            system_log_dir.mkdir(exist_ok=True)
            
            log_file = system_log_dir / f'system_init_{self.session_id}.log'
            if not log_file.exists():
                with open(log_file, 'w') as f:
                    f.write(f"{datetime.now().isoformat()} - INFO - Logging system initialized\n")
                    f.write(f"{datetime.now().isoformat()} - INFO - Session ID: {self.session_id}\n")
                    f.write(f"{datetime.now().isoformat()} - INFO - Log directory structure created\n")
                files_created += 1
            
            # Create logging configuration
            log_config_file = folder_path / 'logging_config.json'
            if not log_config_file.exists():
                log_config = {
                    'logging_system': 'comprehensive',
                    'log_levels': {
                        'feature_extraction': 'DEBUG',
                        'model_performance': 'INFO',
                        'system': 'INFO',
                        'sessions': 'INFO'
                    },
                    'retention_days': 30,
                    'max_log_size_mb': 100,
                    'session_id': self.session_id,
                    'created': datetime.now().isoformat()
                }
                with open(log_config_file, 'w') as f:
                    json.dump(log_config, f, indent=2)
                files_created += 1
        
        elif folder_name == 'output':
            # Create session summary template
            session_dir = folder_path / 'session_reports'
            session_dir.mkdir(exist_ok=True)
            
            session_template = session_dir / f'session_template_{self.session_id}.json'
            if not session_template.exists():
                template_data = {
                    'session_info': {
                        'session_id': self.session_id,
                        'start_time': datetime.now().isoformat(),
                        'system': 'physics-based deepfake detection',
                        'version': '2.0'
                    },
                    'analysis_results': {
                        'total_files_processed': 0,
                        'successful_extractions': 0,
                        'failed_extractions': 0,
                        'physics_features_extracted': [],
                        'key_findings': []
                    },
                    'performance_metrics': {
                        'processing_time_total': 0,
                        'average_time_per_file': 0,
                        'memory_usage_peak': 0,
                        'success_rate_percent': 0
                    }
                }
                with open(session_template, 'w') as f:
                    json.dump(template_data, f, indent=2)
                files_created += 1
        
        elif folder_name == 'plots':
            # Create plot configuration
            plot_config_file = folder_path / 'plot_config.json'
            if not plot_config_file.exists():
                plot_config = {
                    'matplotlib_settings': {
                        'dpi': 300,
                        'format': 'png',
                        'style': 'seaborn-v0_8',
                        'figure_size': [12, 8],
                        'font_family': 'serif'
                    },
                    'physics_plot_config': {
                        'feature_colors': {
                            'delta_ft_revised': '#1f77b4',
                            'delta_fr_revised': '#ff7f0e', 
                            'delta_fv_revised': '#2ca02c',
                            'delta_f_total_revised': '#d62728'
                        },
                        'significance_threshold': 0.05
                    },
                    'session_id': self.session_id,
                    'created': datetime.now().isoformat()
                }
                with open(plot_config_file, 'w') as f:
                    json.dump(plot_config, f, indent=2)
                files_created += 1
        
        elif folder_name == 'quarantine':
            # Create quarantine log
            quarantine_log_dir = folder_path / 'logs'
            quarantine_log_dir.mkdir(exist_ok=True)
            
            quarantine_log = quarantine_log_dir / f'quarantine_log_{self.session_id}.txt'
            if not quarantine_log.exists():
                with open(quarantine_log, 'w') as f:
                    f.write(f"QUARANTINE LOG - Session {self.session_id}\n")
                    f.write("="*50 + "\n")
                    f.write(f"Created: {datetime.now().isoformat()}\n")
                    f.write("Purpose: Log of quarantined files and security events\n\n")
                    f.write("Format: [TIMESTAMP] - [SEVERITY] - [FILE] - [REASON]\n")
                    f.write("="*50 + "\n\n")
                files_created += 1
        
        elif folder_name == 'cache':
            # Create cache organization documentation for subdirectories
            subdirs_info = {
                'hubert': {
                    'purpose': 'Reserved for raw HuBERT model embeddings',
                    'status': 'Future development - currently unused',
                    'file_types': 'HuBERT tensor outputs (.pt files)'
                },
                'features': {
                    'purpose': 'Reserved for computed physics and audio features',
                    'status': 'Future development - currently unused', 
                    'file_types': 'Processed feature vectors (.pkl files)'
                },
                'processing': {
                    'purpose': 'Reserved for temporary processing state',
                    'status': 'Future development - currently unused',
                    'file_types': 'Processing checkpoints and intermediate data'
                }
            }
            
            # Create README files for each cache subdirectory
            for subdir_name, info in subdirs_info.items():
                subdir_path = folder_path / subdir_name
                readme_file = subdir_path / 'README.md'
                if not readme_file.exists():
                    readme_content = f"""# {subdir_name.title()} Cache Directory

## Purpose
{info['purpose']}

## Current Status
{info['status']}

## Intended File Types
{info['file_types']}

## Implementation Notes
This directory is part of the planned cache organization system. Currently, all cache files are stored in the parent cache/ directory using MD5-hashed filenames for efficient lookup.

Future versions may implement:
- Hierarchical cache organization
- Type-specific cache management
- Automated cache migration
- Enhanced cache metadata tracking

## Session Info
Created: {datetime.now().isoformat()}
Session ID: {self.session_id}
"""
                    with open(readme_file, 'w') as f:
                        f.write(readme_content)
                    files_created += 1
            
            # Create cache status file
            cache_status_file = folder_path / 'cache_status.json'
            if not cache_status_file.exists():
                cache_status = {
                    'cache_system': 'active',
                    'organization': 'flat_structure',
                    'subdirectories': {
                        'hubert': 'reserved_unused',
                        'features': 'reserved_unused',
                        'processing': 'reserved_unused'
                    },
                    'cache_files_location': 'root_cache_directory',
                    'cache_key_method': 'md5_hash',
                    'average_file_size_kb': 721,
                    'session_id': self.session_id,
                    'created': datetime.now().isoformat()
                }
                with open(cache_status_file, 'w') as f:
                    json.dump(cache_status, f, indent=2)
                files_created += 1
        
        return files_created
    
    def _create_folder_index(self, setup_stats: Dict[str, Any]):
        """Create master index of all folders and their purposes."""
        index_file = self.project_root / 'FOLDER_INDEX.md'
        
        content = f"""# Project Folder Structure Index

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Session ID:** {self.session_id}

## Overview
This document provides a comprehensive overview of the project folder structure and the purpose of each directory.

## Folder Structure

```
physics_feature_test_project/
├── 📁 data/                    # Input audio files organized by user
├── 📁 src/                     # Source code modules
├── 📁 results/                 # Analysis results and outputs
├── 📁 checkpoints/             # Processing checkpoints for recovery
├── 📁 logs/                    # Comprehensive system logging
├── 📁 output/                  # Analysis reports and summaries
├── 📁 plots/                   # Statistical plots and visualizations
├── 📁 quarantine/              # Quarantined suspicious files
├── 📁 visualizations/          # Enhanced interactive visualizations
├── 📁 cache/                   # Cached model outputs
├── 📁 venv/                    # Python virtual environment
├── 📄 test_runner.py           # Main execution script
└── 📄 requirements.txt         # Python dependencies
```

## Detailed Folder Descriptions

"""
        
        for folder_name, config in self.folders.items():
            content += f"### 📁 {folder_name}/\n"
            content += f"**Purpose:** {config['description']}\n\n"
            
            if config['subdirs']:
                content += "**Subdirectories:**\n"
                for subdir in config['subdirs']:
                    content += f"- `{subdir}/`\n"
                content += "\n"
            
            content += f"**Auto-populated:** {'Yes' if config['auto_populate'] else 'No'}\n\n"
        
        content += f"""## Usage Guidelines

### For Researchers
- Check `results/` for CSV analysis outputs
- Review `plots/` for statistical visualizations  
- Examine `output/` for comprehensive reports

### For Developers
- Monitor `logs/` for debugging information
- Use `checkpoints/` for recovery from interruptions
- Check `quarantine/` for security validation issues

### For System Administrators
- Review `logs/system/` for operational status
- Monitor `cache/` for storage usage
- Check `quarantine/logs/` for security events

## Setup Statistics
- Folders created: {setup_stats['folders_created']}
- Subdirectories created: {setup_stats['subdirs_created']}
- Documentation files created: {setup_stats['files_created']}

## Maintenance
- Logs are retained for 30 days
- Checkpoints are cleared on successful completion
- Cache files are managed automatically
- Quarantine files require manual review

## Last Updated
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} (Session: {self.session_id})
"""
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def cleanup_empty_folders(self) -> Dict[str, List[str]]:
        """
        Clean up truly empty folders but preserve structure.
        
        Returns:
            Dictionary of cleanup actions taken
        """
        cleanup_actions = {
            'removed_empty_dirs': [],
            'preserved_structure': [],
            'warnings': []
        }
        
        for folder_name, config in self.folders.items():
            folder_path = config['path']
            
            if folder_path.exists():
                # Check if folder is truly empty (no files, no useful subdirs)
                contents = list(folder_path.iterdir())
                useful_contents = [
                    item for item in contents 
                    if not item.name.startswith('.') and 
                    (item.is_file() or any(item.iterdir()) if item.is_dir() else True)
                ]
                
                if not useful_contents:
                    # Don't remove, just note
                    cleanup_actions['preserved_structure'].append(folder_name)
                else:
                    cleanup_actions['preserved_structure'].append(folder_name)
        
        return cleanup_actions
    
    def get_folder_status(self) -> Dict[str, Any]:
        """Get current status of all project folders."""
        status = {
            'session_id': self.session_id,
            'scan_time': datetime.now().isoformat(),
            'folders': {}
        }
        
        for folder_name, config in self.folders.items():
            folder_path = config['path']
            
            if folder_path.exists():
                # Count contents
                files = sum(1 for item in folder_path.rglob('*') if item.is_file())
                dirs = sum(1 for item in folder_path.rglob('*') if item.is_dir())
                
                # Calculate size
                total_size = sum(f.stat().st_size for f in folder_path.rglob('*') if f.is_file())
                
                status['folders'][folder_name] = {
                    'exists': True,
                    'files_count': files,
                    'dirs_count': dirs,
                    'total_size_mb': round(total_size / (1024 * 1024), 2),
                    'last_modified': datetime.fromtimestamp(folder_path.stat().st_mtime).isoformat(),
                    'description': config['description']
                }
            else:
                status['folders'][folder_name] = {
                    'exists': False,
                    'description': config['description']
                }
        
        return status


def initialize_project_folders(project_root: str = ".") -> FolderManager:
    """
    Convenience function to initialize all project folders.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        FolderManager instance for further operations
    """
    manager = FolderManager(project_root)
    setup_stats = manager.setup_all_folders()
    
    # Save setup statistics
    stats_file = Path(project_root) / 'output' / f'folder_setup_stats_{manager.session_id}.json'
    with open(stats_file, 'w') as f:
        json.dump(setup_stats, f, indent=2)
    
    return manager


if __name__ == "__main__":
    # Test the folder management system
    manager = initialize_project_folders()
    status = manager.get_folder_status()
    
    print("\n📊 Folder Status Summary:")
    for folder_name, info in status['folders'].items():
        if info['exists']:
            print(f"  {folder_name}: {info['files_count']} files, {info['total_size_mb']} MB")
        else:
            print(f"  {folder_name}: Not found") 