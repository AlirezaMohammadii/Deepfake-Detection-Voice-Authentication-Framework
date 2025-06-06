#!/usr/bin/env python3
"""
Comprehensive Project Management Utility
Combines folder management, cleanup, and project reset functionality
"""

import os
import sys
import shutil
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

try:
    from utils.folder_manager import initialize_project_folders, FolderManager
    FOLDER_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Folder manager not available: {e}")
    FOLDER_MANAGER_AVAILABLE = False


class ProjectManager:
    """
    Comprehensive project management utility combining folder setup, 
    cleanup, and reset functionality
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or current_dir)
        self.parent_dir = self.project_root.parent
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cleanup_stats = {
            'files_removed': 0,
            'dirs_removed': 0,
            'bytes_freed': 0
        }
        
        # Define project directories created by test_runner.py
        self.project_directories = [
            'results', 'checkpoints', 'logs', 'output', 'plots', 
            'quarantine', 'cache', 'visualizations'
        ]
        
        # Define cache directories (including external ones)
        self.cache_directories = [
            self.project_root / 'cache',
            Path.home() / '.cache' / 'voice_models',
            Path.home() / '.cache' / 'huggingface',
            Path.home() / '.cache' / 'transformers',
            self.parent_dir / 'voice_models_cache',
            self.parent_dir / 'model_cache'
        ]
    
    def test_folder_management(self) -> bool:
        """Test the comprehensive folder management system"""
        
        print("🗂️  FOLDER MANAGEMENT SYSTEM TEST")
        print("=" * 50)
        
        if not FOLDER_MANAGER_AVAILABLE:
            print("❌ Folder management module not available")
            return False
        
        # Test 1: Initialize folder manager
        print("\n1️⃣  Testing folder manager initialization...")
        try:
            manager = FolderManager(str(self.project_root))
            print(f"   ✅ Manager created with session ID: {manager.session_id}")
        except Exception as e:
            print(f"   ❌ Failed to create manager: {e}")
            return False
        
        # Test 2: Setup all folders
        print("\n2️⃣  Testing folder structure setup...")
        try:
            setup_stats = manager.setup_all_folders()
            print(f"   ✅ Setup completed successfully!")
            print(f"      📁 Folders created: {setup_stats['folders_created']}")
            print(f"      📂 Subdirs created: {setup_stats['subdirs_created']}")
            print(f"      📄 Files created: {setup_stats['files_created']}")
        except Exception as e:
            print(f"   ❌ Failed to setup folders: {e}")
            return False
        
        # Test 3: Get folder status
        print("\n3️⃣  Testing folder status reporting...")
        try:
            status = manager.get_folder_status()
            print(f"   ✅ Status report generated successfully!")
            print(f"   📊 Folder Status Summary:")
            
            for folder_name, info in status['folders'].items():
                if info['exists']:
                    print(f"      {folder_name:15}: {info['files_count']:3} files, {info['total_size_mb']:6.2f} MB")
                else:
                    print(f"      {folder_name:15}: ❌ Not found")
        except Exception as e:
            print(f"   ❌ Failed to get status: {e}")
            return False
        
        # Test 4: Check specific folder contents
        print("\n4️⃣  Testing folder content verification...")
        try:
            folders_to_check = {
                'logs': ['README.md', 'logging_config.json'],
                'checkpoints': ['README.md', 'checkpoint_status.json'],  
                'output': ['README.md'],
                'plots': ['README.md', 'plot_config.json'],
                'quarantine': ['README.md']
            }
            
            all_good = True
            for folder_name, expected_files in folders_to_check.items():
                folder_path = self.project_root / folder_name
                if folder_path.exists():
                    missing_files = []
                    for expected_file in expected_files:
                        if not (folder_path / expected_file).exists():
                            missing_files.append(expected_file)
                    
                    if missing_files:
                        print(f"      {folder_name:15}: ⚠️  Missing {missing_files}")
                        all_good = False
                    else:
                        print(f"      {folder_name:15}: ✅ All expected files present")
                else:
                    print(f"      {folder_name:15}: ❌ Folder not found")
                    all_good = False
            
            if all_good:
                print("   ✅ All folders contain expected files!")
            else:
                print("   ⚠️  Some files are missing (this may be normal)")
        except Exception as e:
            print(f"   ❌ Failed content verification: {e}")
            return False
        
        # Test 5: Check documentation files
        print("\n5️⃣  Testing documentation generation...")
        try:
            master_index = self.project_root / 'FOLDER_INDEX.md'
            if master_index.exists():
                print(f"   ✅ Master folder index created: {master_index}")
                size_kb = master_index.stat().st_size / 1024
                print(f"      📄 File size: {size_kb:.1f} KB")
                
                if size_kb > 1:
                    print("   ✅ Documentation appears comprehensive")
                else:
                    print("   ⚠️  Documentation file seems small")
            else:
                print("   ❌ Master folder index not found")
                return False
        except Exception as e:
            print(f"   ❌ Failed documentation check: {e}")
            return False
        
        print("\n✅ FOLDER MANAGEMENT SYSTEM TEST COMPLETED SUCCESSFULLY!")
        return True
    
    def show_folder_structure(self):
        """Display the current folder structure"""
        print("\n📁 CURRENT PROJECT STRUCTURE:")
        print("=" * 30)
        
        # List main directories
        for item in sorted(self.project_root.iterdir()):
            if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
                print(f"📁 {item.name}/")
                
                try:
                    files = [f for f in item.iterdir() if f.is_file() and not f.name.startswith('.')]
                    if files:
                        for file in sorted(files)[:3]:
                            size_kb = file.stat().st_size / 1024
                            print(f"   📄 {file.name} ({size_kb:.1f} KB)")
                        if len(files) > 3:
                            print(f"   ... and {len(files) - 3} more files")
                    
                    subdirs = [d for d in item.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    for subdir in sorted(subdirs)[:2]:
                        print(f"   📁 {subdir.name}/")
                    if len(subdirs) > 2:
                        print(f"   ... and {len(subdirs) - 2} more subdirectories")
                        
                except (PermissionError, Exception):
                    print("   (Error reading directory)")
        
        # Show key files in root
        print(f"\n📄 Key files in root:")
        key_files = ['test_runner.py', 'requirements.txt', 'README.md', 'FOLDER_INDEX.md']
        for filename in key_files:
            filepath = self.project_root / filename
            if filepath.exists():
                size_kb = filepath.stat().st_size / 1024
                print(f"   📄 {filename} ({size_kb:.1f} KB)")
    
    def identify_unnecessary_files(self) -> Dict[str, List[str]]:
        """Identify files and directories that can be safely removed"""
        unnecessary_items = {
            'cache_files': [],
            'temp_files': [],
            'log_files': [],
            'python_cache': [],
            'empty_directories': [],
            'large_redundant_files': [],
            'backup_files': []
        }
        
        print("🔍 Scanning project for unnecessary files...")
        
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            
            # Skip virtual environment directory
            if 'venv' in root_path.parts or '.venv' in root_path.parts:
                continue
            
            # Check for Python cache directories
            if '__pycache__' in dirs:
                cache_dir = root_path / '__pycache__'
                unnecessary_items['python_cache'].append(str(cache_dir))
            
            # Check each file
            for file in files:
                file_path = root_path / file
                try:
                    file_size = file_path.stat().st_size
                    
                    # Python cache files
                    if file.endswith('.pyc') or file.endswith('.pyo'):
                        unnecessary_items['cache_files'].append(str(file_path))
                    
                    # Temporary files
                    elif file.endswith('.tmp') or file.endswith('.temp') or file.startswith('~'):
                        unnecessary_items['temp_files'].append(str(file_path))
                    
                    # Log files (keep recent ones)
                    elif file.endswith('.log') and self._is_old_log_file(file_path):
                        unnecessary_items['log_files'].append(str(file_path))
                    
                    # Backup files
                    elif file.endswith('.bak') or file.endswith('.backup') or file.endswith('~'):
                        unnecessary_items['backup_files'].append(str(file_path))
                    
                    # Large cache pickle files (keep recent ones)
                    elif file.endswith('.pkl') and 'cache' in str(root_path).lower():
                        if file_size > 500 * 1024 and self._is_old_cache_file(file_path):
                            unnecessary_items['large_redundant_files'].append(str(file_path))
                
                except (PermissionError, FileNotFoundError):
                    continue
        
        # Check for empty directories
        for root, dirs, files in os.walk(self.project_root, topdown=False):
            root_path = Path(root)
            
            # Skip essential directories
            if any(essential in root_path.name.lower() for essential in ['src', 'core', 'venv', '.git']):
                continue
            
            try:
                if not any(root_path.iterdir()):
                    # Don't delete certain important empty directories
                    if root_path.name not in ['quarantine', 'checkpoints', 'logs', 'output']:
                        unnecessary_items['empty_directories'].append(str(root_path))
            except (PermissionError, FileNotFoundError):
                continue
        
        return unnecessary_items
    
    def _is_old_log_file(self, file_path: Path) -> bool:
        """Check if log file is older than 7 days"""
        try:
            file_age = time.time() - file_path.stat().st_mtime
            return file_age > 7 * 24 * 3600  # 7 days
        except:
            return False
    
    def _is_old_cache_file(self, file_path: Path) -> bool:
        """Check if cache file is older than 1 day"""
        try:
            file_age = time.time() - file_path.stat().st_mtime
            return file_age > 24 * 3600  # 1 day
        except:
            return False
    
    def calculate_space_savings(self, unnecessary_items: Dict[str, List[str]]) -> int:
        """Calculate total space that would be freed"""
        total_size = 0
        
        for category, items in unnecessary_items.items():
            for item_path in items:
                try:
                    path = Path(item_path)
                    if path.is_file():
                        total_size += path.stat().st_size
                    elif path.is_dir():
                        total_size += sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                except:
                    continue
        
        return total_size
    
    def get_project_reset_analysis(self) -> Dict[str, any]:
        """Analyze what would be removed in a complete project reset"""
        reset_analysis = {
            'project_directories': {},
            'cache_directories': {},
            'generated_files': [],
            'total_size_mb': 0,
            'total_files': 0,
            'total_dirs': 0
        }
        
        # Analyze project directories
        for dir_name in self.project_directories:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                try:
                    files = list(dir_path.rglob('*'))
                    file_count = sum(1 for f in files if f.is_file())
                    dir_count = sum(1 for f in files if f.is_dir())
                    total_size = sum(f.stat().st_size for f in files if f.is_file())
                    
                    reset_analysis['project_directories'][dir_name] = {
                        'files': file_count,
                        'dirs': dir_count,
                        'size_mb': total_size / (1024 * 1024),
                        'exists': True
                    }
                    reset_analysis['total_files'] += file_count
                    reset_analysis['total_dirs'] += dir_count + 1  # +1 for the directory itself
                    reset_analysis['total_size_mb'] += total_size / (1024 * 1024)
                    
                except (PermissionError, FileNotFoundError):
                    reset_analysis['project_directories'][dir_name] = {
                        'files': 0, 'dirs': 0, 'size_mb': 0, 'exists': True, 'error': 'Access denied'
                    }
            else:
                reset_analysis['project_directories'][dir_name] = {
                    'files': 0, 'dirs': 0, 'size_mb': 0, 'exists': False
                }
        
        # Analyze cache directories
        for cache_dir in self.cache_directories:
            if cache_dir.exists():
                try:
                    files = list(cache_dir.rglob('*'))
                    file_count = sum(1 for f in files if f.is_file())
                    dir_count = sum(1 for f in files if f.is_dir())
                    total_size = sum(f.stat().st_size for f in files if f.is_file())
                    
                    reset_analysis['cache_directories'][str(cache_dir)] = {
                        'files': file_count,
                        'dirs': dir_count,
                        'size_mb': total_size / (1024 * 1024),
                        'exists': True
                    }
                    reset_analysis['total_files'] += file_count
                    reset_analysis['total_dirs'] += dir_count + 1
                    reset_analysis['total_size_mb'] += total_size / (1024 * 1024)
                    
                except (PermissionError, FileNotFoundError):
                    reset_analysis['cache_directories'][str(cache_dir)] = {
                        'files': 0, 'dirs': 0, 'size_mb': 0, 'exists': True, 'error': 'Access denied'
                    }
        
        # Analyze generated files in root
        generated_files = [
            'FOLDER_INDEX.md', 'FOLDER_CLEANUP_SUMMARY.md', 
            'enhanced_visualization_*.html', 'comprehensive_dashboard.html'
        ]
        
        for pattern in generated_files:
            if '*' in pattern:
                matching_files = list(self.project_root.glob(pattern))
                for file_path in matching_files:
                    if file_path.is_file():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        reset_analysis['generated_files'].append({
                            'file': str(file_path.name),
                            'size_mb': size_mb
                        })
                        reset_analysis['total_files'] += 1
                        reset_analysis['total_size_mb'] += size_mb
            else:
                file_path = self.project_root / pattern
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    reset_analysis['generated_files'].append({
                        'file': pattern,
                        'size_mb': size_mb
                    })
                    reset_analysis['total_files'] += 1
                    reset_analysis['total_size_mb'] += size_mb
        
        return reset_analysis
    
    def display_cleanup_report(self, unnecessary_items: Dict[str, List[str]]):
        """Display detailed cleanup report"""
        print("\n" + "="*60)
        print("PROJECT CLEANUP ANALYSIS REPORT")
        print("="*60)
        
        total_files = sum(len(items) for items in unnecessary_items.values())
        total_size = self.calculate_space_savings(unnecessary_items)
        
        print(f"📊 Summary:")
        print(f"  Total unnecessary items: {total_files}")
        print(f"  Potential space savings: {total_size / (1024*1024):.1f} MB")
        
        print(f"\n📂 Breakdown by category:")
        
        for category, items in unnecessary_items.items():
            if items:
                print(f"\n  {category.replace('_', ' ').title()}:")
                print(f"    Count: {len(items)}")
                
                for i, item in enumerate(items[:3]):
                    rel_path = os.path.relpath(item, self.project_root)
                    try:
                        size = Path(item).stat().st_size / 1024
                        print(f"    - {rel_path} ({size:.1f} KB)")
                    except:
                        print(f"    - {rel_path}")
                
                if len(items) > 3:
                    print(f"    ... and {len(items) - 3} more items")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if unnecessary_items['python_cache']:
            print(f"  ✓ Safe to remove: Python cache files (__pycache__ directories)")
        if unnecessary_items['temp_files']:
            print(f"  ✓ Safe to remove: Temporary files")
        if unnecessary_items['backup_files']:
            print(f"  ⚠️ Review first: Backup files (may contain important data)")
        if unnecessary_items['large_redundant_files']:
            print(f"  ⚠️ Review first: Large cache files (can be regenerated but takes time)")
        if unnecessary_items['log_files']:
            print(f"  ✓ Safe to remove: Old log files (older than 7 days)")
        
        return total_files, total_size
    
    def display_reset_analysis(self, reset_analysis: Dict[str, any]):
        """Display detailed project reset analysis"""
        print("\n" + "="*70)
        print("PROJECT RESET ANALYSIS")
        print("="*70)
        
        print(f"📊 Reset Summary:")
        print(f"  Total files to remove: {reset_analysis['total_files']}")
        print(f"  Total directories to remove: {reset_analysis['total_dirs']}")
        print(f"  Total space to free: {reset_analysis['total_size_mb']:.1f} MB")
        
        print(f"\n📁 Project Directories to Reset:")
        for dir_name, info in reset_analysis['project_directories'].items():
            if info['exists']:
                if 'error' in info:
                    print(f"  {dir_name:15}: ❌ {info['error']}")
                else:
                    print(f"  {dir_name:15}: {info['files']:4} files, {info['dirs']:3} dirs, {info['size_mb']:8.1f} MB")
            else:
                print(f"  {dir_name:15}: ➖ Not exists")
        
        print(f"\n💾 Cache Directories to Clear:")
        for cache_path, info in reset_analysis['cache_directories'].items():
            cache_name = Path(cache_path).name
            if info['exists']:
                if 'error' in info:
                    print(f"  {cache_name:20}: ❌ {info['error']}")
                else:
                    print(f"  {cache_name:20}: {info['files']:4} files, {info['size_mb']:8.1f} MB")
        
        if reset_analysis['generated_files']:
            print(f"\n📄 Generated Files to Remove:")
            for file_info in reset_analysis['generated_files']:
                print(f"  {file_info['file']:30}: {file_info['size_mb']:8.1f} MB")
        
        print(f"\n⚠️  WARNING: This will completely reset the project to pre-test_runner.py state!")
        print(f"   - All analysis results will be lost")
        print(f"   - All visualizations and reports will be deleted")
        print(f"   - All cached models and processing results will be removed")
        print(f"   - You will need to re-run test_runner.py to regenerate everything")
    
    def perform_cleanup(self, unnecessary_items: Dict[str, List[str]], categories_to_clean: List[str] = None):
        """Perform the actual cleanup"""
        if categories_to_clean is None:
            categories_to_clean = ['python_cache', 'temp_files', 'log_files']
        
        # Reset cleanup stats for this operation
        self.cleanup_stats = {
            'files_removed': 0,
            'dirs_removed': 0,
            'bytes_freed': 0
        }
        
        print(f"\n🧹 Starting cleanup of categories: {categories_to_clean}")
        
        # Validate that categories exist in unnecessary_items
        valid_categories = []
        for category in categories_to_clean:
            if category in unnecessary_items and unnecessary_items[category]:
                valid_categories.append(category)
                print(f"  ✓ {category}: {len(unnecessary_items[category])} items")
            else:
                print(f"  ⚠️  {category}: No items found or category doesn't exist")
        
        if not valid_categories:
            print("❌ No valid categories with items to clean!")
            return
        
        # Confirm before proceeding
        total_items = sum(len(unnecessary_items[cat]) for cat in valid_categories)
        print(f"\nTotal items to remove: {total_items}")
        
        # Actually perform cleanup
        for category in valid_categories:
            items = unnecessary_items[category]
            print(f"\n  🗑️  Cleaning {category.replace('_', ' ')}...")
            
            for item_path in items:
                try:
                    path = Path(item_path)
                    
                    # Verify path exists before attempting removal
                    if not path.exists():
                        print(f"    ⚠️  Skipped (not found): {path.name}")
                        continue
                    
                    if path.is_file():
                        size = path.stat().st_size
                        path.unlink()
                        self.cleanup_stats['files_removed'] += 1
                        self.cleanup_stats['bytes_freed'] += size
                        print(f"    ✓ Removed file: {path.name} ({size/1024:.1f} KB)")
                    
                    elif path.is_dir():
                        # Calculate directory size before removal
                        try:
                            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                            file_count = sum(1 for f in path.rglob('*') if f.is_file())
                        except:
                            size = 0
                            file_count = 0
                        
                        shutil.rmtree(path)
                        self.cleanup_stats['dirs_removed'] += 1
                        self.cleanup_stats['bytes_freed'] += size
                        print(f"    ✓ Removed directory: {path.name} ({file_count} files, {size/(1024*1024):.1f} MB)")
                    
                    else:
                        print(f"    ⚠️  Skipped (unknown type): {path.name}")
                
                except PermissionError as e:
                    print(f"    ❌ Permission denied: {path.name}")
                except FileNotFoundError:
                    print(f"    ⚠️  Already removed: {path.name}")
                except Exception as e:
                    print(f"    ❌ Failed to remove {item_path}: {e}")
        
        print(f"\n✅ Cleanup completed!")
        print(f"  📁 Files removed: {self.cleanup_stats['files_removed']}")
        print(f"  📂 Directories removed: {self.cleanup_stats['dirs_removed']}")
        print(f"  💾 Space freed: {self.cleanup_stats['bytes_freed'] / (1024*1024):.2f} MB")
        
        # Save cleanup log
        if self.cleanup_stats['files_removed'] > 0 or self.cleanup_stats['dirs_removed'] > 0:
            cleanup_log = {
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id,
                'categories_cleaned': valid_categories,
                'stats': self.cleanup_stats,
                'items_processed': total_items
            }
            
            try:
                log_file = self.project_root / 'logs' / f'cleanup_log_{self.session_id}.json'
                log_file.parent.mkdir(exist_ok=True)
                with open(log_file, 'w') as f:
                    json.dump(cleanup_log, f, indent=2)
                print(f"  📄 Cleanup log saved: {log_file.name}")
            except Exception as e:
                print(f"  ⚠️  Could not save cleanup log: {e}")
        
        return self.cleanup_stats
    
    def perform_project_reset(self, reset_analysis: Dict[str, any], confirmed: bool = False):
        """Perform complete project reset"""
        if not confirmed:
            print("❌ Project reset not confirmed. Aborting.")
            return False
        
        print(f"\n🔄 Starting COMPLETE PROJECT RESET...")
        print(f"⚠️  This action cannot be undone!")
        
        reset_stats = {
            'files_removed': 0,
            'dirs_removed': 0,
            'bytes_freed': 0,
            'errors': []
        }
        
        # 1. Remove project directories
        print(f"\n1️⃣  Removing project directories...")
        for dir_name in self.project_directories:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                try:
                    # Calculate size before removal
                    size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    file_count = sum(1 for f in dir_path.rglob('*') if f.is_file())
                    
                    shutil.rmtree(dir_path)
                    reset_stats['dirs_removed'] += 1
                    reset_stats['files_removed'] += file_count
                    reset_stats['bytes_freed'] += size
                    print(f"    ✓ Removed: {dir_name}/ ({file_count} files, {size/(1024*1024):.1f} MB)")
                    
                except Exception as e:
                    error_msg = f"Failed to remove {dir_name}: {e}"
                    reset_stats['errors'].append(error_msg)
                    print(f"    ✗ {error_msg}")
        
        # 2. Remove cache directories
        print(f"\n2️⃣  Clearing cache directories...")
        for cache_dir in self.cache_directories:
            if cache_dir.exists():
                try:
                    size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                    file_count = sum(1 for f in cache_dir.rglob('*') if f.is_file())
                    
                    shutil.rmtree(cache_dir)
                    reset_stats['dirs_removed'] += 1
                    reset_stats['files_removed'] += file_count
                    reset_stats['bytes_freed'] += size
                    print(f"    ✓ Cleared: {cache_dir.name} ({file_count} files, {size/(1024*1024):.1f} MB)")
                    
                except Exception as e:
                    error_msg = f"Failed to clear cache {cache_dir}: {e}"
                    reset_stats['errors'].append(error_msg)
                    print(f"    ✗ {error_msg}")
        
        # 3. Remove generated files in root
        print(f"\n3️⃣  Removing generated files...")
        generated_patterns = [
            'FOLDER_INDEX.md', 'FOLDER_CLEANUP_SUMMARY.md',
            'enhanced_visualization_*.html', 'comprehensive_dashboard.html',
            '*.png', '*.jpg', '*.jpeg'  # Generated plots that might be in root
        ]
        
        for pattern in generated_patterns:
            if '*' in pattern:
                matching_files = list(self.project_root.glob(pattern))
                for file_path in matching_files:
                    if file_path.is_file():
                        try:
                            size = file_path.stat().st_size
                            file_path.unlink()
                            reset_stats['files_removed'] += 1
                            reset_stats['bytes_freed'] += size
                            print(f"    ✓ Removed: {file_path.name} ({size/1024:.1f} KB)")
                        except Exception as e:
                            error_msg = f"Failed to remove {file_path.name}: {e}"
                            reset_stats['errors'].append(error_msg)
                            print(f"    ✗ {error_msg}")
            else:
                file_path = self.project_root / pattern
                if file_path.exists():
                    try:
                        size = file_path.stat().st_size
                        file_path.unlink()
                        reset_stats['files_removed'] += 1
                        reset_stats['bytes_freed'] += size
                        print(f"    ✓ Removed: {pattern} ({size/1024:.1f} KB)")
                    except Exception as e:
                        error_msg = f"Failed to remove {pattern}: {e}"
                        reset_stats['errors'].append(error_msg)
                        print(f"    ✗ {error_msg}")
        
        # 4. Clean up Python cache
        print(f"\n4️⃣  Cleaning Python cache...")
        for root, dirs, files in os.walk(self.project_root):
            if '__pycache__' in dirs:
                cache_dir = Path(root) / '__pycache__'
                try:
                    size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                    file_count = sum(1 for f in cache_dir.rglob('*') if f.is_file())
                    
                    shutil.rmtree(cache_dir)
                    reset_stats['dirs_removed'] += 1
                    reset_stats['files_removed'] += file_count
                    reset_stats['bytes_freed'] += size
                    print(f"    ✓ Removed: {cache_dir.relative_to(self.project_root)}")
                except Exception as e:
                    error_msg = f"Failed to remove Python cache {cache_dir}: {e}"
                    reset_stats['errors'].append(error_msg)
                    print(f"    ✗ {error_msg}")
        
        # Final report
        print(f"\n🎯 PROJECT RESET COMPLETED!")
        print(f"  Total files removed: {reset_stats['files_removed']}")
        print(f"  Total directories removed: {reset_stats['dirs_removed']}")
        print(f"  Total space freed: {reset_stats['bytes_freed'] / (1024*1024):.1f} MB")
        
        if reset_stats['errors']:
            print(f"\n⚠️  Errors encountered ({len(reset_stats['errors'])}):")
            for error in reset_stats['errors'][:5]:  # Show first 5 errors
                print(f"    - {error}")
            if len(reset_stats['errors']) > 5:
                print(f"    ... and {len(reset_stats['errors']) - 5} more errors")
        
        # Save reset log
        reset_log = {
            'reset_timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'stats': reset_stats,
            'reset_analysis': reset_analysis
        }
        
        log_file = self.project_root / f'project_reset_log_{self.session_id}.json'
        try:
            with open(log_file, 'w') as f:
                json.dump(reset_log, f, indent=2)
            print(f"\n📄 Reset log saved: {log_file.name}")
        except Exception as e:
            print(f"\n⚠️  Could not save reset log: {e}")
        
        print(f"\n✨ PROJECT IS NOW IN CLEAN STATE!")
        print(f"🔄 Ready to run test_runner.py again to regenerate all functionality")
        
        return True


def main():
    """Main execution function with interactive menu"""
    print("🛠️  PHYSICS FEATURES PROJECT MANAGER")
    print("=" * 60)
    print("Comprehensive project management, cleanup, and reset utility")
    
    # Initialize project manager
    manager = ProjectManager()
    
    while True:
        print(f"\n📋 PROJECT MANAGER MENU:")
        print(f"  1. Test and setup folder management system")
        print(f"  2. Show current project structure")
        print(f"  3. Analyze unnecessary files (safe cleanup)")
        print(f"  4. Perform safe cleanup")
        print(f"  5. Analyze complete project reset")
        print(f"  6. Perform COMPLETE PROJECT RESET (⚠️ DANGEROUS)")
        print(f"  7. Custom cleanup options")
        print(f"  8. Exit")
        
        try:
            choice = input("\nEnter your choice (1-8): ").strip()
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        
        if choice == '1':
            # Test folder management
            print(f"\n{'='*60}")
            print("TESTING FOLDER MANAGEMENT SYSTEM")
            print(f"{'='*60}")
            
            success = manager.test_folder_management()
            if success:
                print(f"\n🎉 SUCCESS: Folder management system is working correctly!")
                print(f"💡 All folders are properly set up and documented")
            else:
                print(f"\n⚠️  WARNING: Issues detected with folder management system")
        
        elif choice == '2':
            # Show project structure
            manager.show_folder_structure()
        
        elif choice == '3':
            # Analyze unnecessary files
            print(f"\n{'='*60}")
            print("ANALYZING UNNECESSARY FILES")
            print(f"{'='*60}")
            
            unnecessary_items = manager.identify_unnecessary_files()
            total_files, total_size = manager.display_cleanup_report(unnecessary_items)
            
            if total_files == 0:
                print("\n✨ Project is clean! No unnecessary files found.")
            else:
                print(f"\n💡 Found {total_files} unnecessary items totaling {total_size/(1024*1024):.1f} MB")
        
        elif choice == '4':
            # Perform safe cleanup
            print(f"\n{'='*60}")
            print("PERFORMING SAFE CLEANUP")
            print(f"{'='*60}")
            
            unnecessary_items = manager.identify_unnecessary_files()
            total_files, total_size = manager.display_cleanup_report(unnecessary_items)
            
            if total_files == 0:
                print("\n✨ No unnecessary files to clean!")
                continue
            
            print(f"\n🤔 Safe cleanup will remove:")
            print(f"  - Python cache files (__pycache__)")
            print(f"  - Temporary files (.tmp, .temp)")
            print(f"  - Old log files (>7 days)")
            
            confirm = input(f"\nProceed with safe cleanup? (y/n): ").lower().strip()
            if confirm in ['y', 'yes']:
                manager.perform_cleanup(unnecessary_items, ['python_cache', 'temp_files', 'log_files'])
            else:
                print("Safe cleanup cancelled.")
        
        elif choice == '5':
            # Analyze project reset
            print(f"\n{'='*60}")
            print("ANALYZING COMPLETE PROJECT RESET")
            print(f"{'='*60}")
            
            reset_analysis = manager.get_project_reset_analysis()
            manager.display_reset_analysis(reset_analysis)
            
            print(f"\n💡 This analysis shows what would be removed in a complete reset.")
            print(f"📊 Use option 6 to actually perform the reset (if needed).")
        
        elif choice == '6':
            # Perform complete project reset
            print(f"\n{'='*60}")
            print("⚠️  COMPLETE PROJECT RESET")
            print(f"{'='*60}")
            
            reset_analysis = manager.get_project_reset_analysis()
            manager.display_reset_analysis(reset_analysis)
            
            print(f"\n🚨 FINAL WARNING:")
            print(f"   This will PERMANENTLY DELETE all:")
            print(f"   - Analysis results and reports")
            print(f"   - Visualizations and plots")
            print(f"   - Cached models and processing results")
            print(f"   - All generated project content")
            print(f"   - Project will be reset to pre-test_runner.py state")
            
            print(f"\n❓ Are you absolutely sure you want to proceed?")
            print(f"   Type 'YES I WANT TO RESET THE PROJECT' to confirm:")
            
            confirmation = input().strip()
            if confirmation == 'YES I WANT TO RESET THE PROJECT':
                print(f"\n⏳ Performing complete project reset...")
                success = manager.perform_project_reset(reset_analysis, confirmed=True)
                if success:
                    print(f"\n🎯 Project reset completed successfully!")
                    print(f"🔄 Project is now ready for fresh test_runner.py execution")
                else:
                    print(f"\n❌ Project reset failed or was incomplete")
            else:
                print(f"\n✅ Project reset cancelled - smart choice!")
                print(f"🔒 Your project data is safe")
        
        elif choice == '7':
            # Custom cleanup options
            print(f"\n{'='*60}")
            print("CUSTOM CLEANUP OPTIONS")
            print(f"{'='*60}")
            
            unnecessary_items = manager.identify_unnecessary_files()
            
            print(f"\nAvailable cleanup categories:")
            available_categories = [(i+1, cat, items) for i, (cat, items) in enumerate(unnecessary_items.items()) if items]
            
            if not available_categories:
                print("✨ No unnecessary files found!")
                continue
            
            for i, cat, items in available_categories:
                print(f"  {i}. {cat.replace('_', ' ').title()} ({len(items)} items)")
            
            try:
                selections = input("Enter category numbers (comma-separated): ").strip()
                if not selections:
                    print("No categories selected.")
                    continue
                
                # Parse user input and validate
                selected_indices = []
                for x in selections.split(','):
                    x = x.strip()
                    if x.isdigit():
                        idx = int(x)
                        if 1 <= idx <= len(available_categories):
                            selected_indices.append(idx)
                        else:
                            print(f"Warning: Category {idx} is out of range (1-{len(available_categories)})")
                    elif x:  # Non-empty but not a digit
                        print(f"Warning: '{x}' is not a valid number")
                
                if not selected_indices:
                    print("No valid categories selected.")
                    continue
                
                # Map indices to category names
                selected_categories = [available_categories[i-1][1] for i in selected_indices]
                
                # Display selected categories and their contents
                print(f"\nSelected categories:")
                total_items = 0
                total_size = 0
                for idx in selected_indices:
                    cat_name = available_categories[idx-1][1]
                    items = available_categories[idx-1][2]
                    total_items += len(items)
                    
                    # Calculate size for this category
                    category_size = 0
                    for item_path in items:
                        try:
                            path = Path(item_path)
                            if path.is_file():
                                category_size += path.stat().st_size
                            elif path.is_dir():
                                category_size += sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        except:
                            continue
                    total_size += category_size
                    
                    print(f"  {idx}. {cat_name.replace('_', ' ').title()}: {len(items)} items ({category_size/(1024*1024):.1f} MB)")
                
                print(f"\nTotal: {total_items} items, {total_size/(1024*1024):.1f} MB")
                
                # Confirm cleanup
                confirm = input("\nProceed with cleanup? (y/n): ").lower().strip()
                
                if confirm in ['y', 'yes']:
                    print(f"\nStarting cleanup of selected categories...")
                    manager.perform_cleanup(unnecessary_items, selected_categories)
                else:
                    print("Custom cleanup cancelled.")
                    
            except ValueError as e:
                print(f"Invalid selection format: {e}")
            except Exception as e:
                print(f"Error during custom cleanup: {e}")
        
        elif choice == '8':
            print("\n👋 Exiting Project Manager. Have a great day!")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-8.")


if __name__ == "__main__":
    main() 