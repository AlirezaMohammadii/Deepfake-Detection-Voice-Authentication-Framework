# Physics-Based Deepfake Detection System - How to Run

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Guide](#installation-guide)
3. [Project Structure Overview](#project-structure-overview)
4. [Data Preparation](#data-preparation)
5. [Running the System](#running-the-system)
6. [Understanding the Output](#understanding-the-output)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)
10. [Maintenance](#maintenance)
11. [Folder Management System](#folder-management-system)
12. [Quick Start Guide](#quick-start-guide)

---

## 🖥️ System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 10GB+ free space for models and cache
- **GPU**: Optional but recommended (CUDA-compatible for faster processing)

### Software Requirements
- **Python**: Version 3.8 or higher (3.10+ recommended)
- **Operating System**: Windows 10+, macOS 10.15+, or Linux Ubuntu 18.04+
- **Git**: For cloning the repository
- **Web Browser**: Modern browser for viewing interactive visualizations

---

## 🛠️ Installation Guide

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd physics_feature_test_project
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print('HuggingFace Transformers installed successfully')"
```

### Step 4: Verify GPU Support (Optional)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🗂️ Project Structure Overview

```
physics_feature_test_project/
├── 📁 src/                          # Core source code
│   ├── 📁 core/                     # Main processing modules
│   ├── 📁 utils/                    # Utility functions
│   ├── 📁 security/                 # Security validation
│   ├── 📁 streaming/                # Streaming processing
│   └── 📁 visualization/            # Advanced plotting
├── 📁 data/                         # Audio data directory
├── 📁 results/                      # Processing results
├── 📁 visualizations/               # Generated plots and reports
├── 📁 cache/                        # Feature extraction cache
├── 📁 logs/                         # System logs
├── 📁 checkpoints/                  # Processing checkpoints
├── 📁 tests/                        # Unit tests
├── 🐍 test_runner.py               # Main execution script
├── 🐍 cleanup_unnecessary_files.py # Project cleanup utility
├── 📄 requirements.txt             # Python dependencies
└── 📄 README.md                    # Project overview
```

---

## 📂 Data Preparation

### Step 1: Data Directory Structure
Create the following directory structure in the `data/` folder:

```
data/
├── user_001/
│   ├── genuine_audio_001.wav
│   ├── deepfake_tts_001.wav
│   └── ...
├── user_002/
│   ├── genuine_audio_002.wav
│   ├── deepfake_tts_002.wav
│   └── ...
└── ...
```

### Step 2: Audio File Requirements
- **Format**: WAV, MP3, FLAC, or M4A
- **Sample Rate**: Any (will be normalized to 16kHz)
- **Duration**: Minimum 1 second, no maximum limit
- **File Size**: Maximum 200MB per file (configurable)
- **Naming Convention**: Include descriptive keywords:
  - `genuine_*` for authentic audio
  - `deepfake_tts_*` for text-to-speech deepfakes
  - `deepfake_vc_*` for voice conversion deepfakes

### Step 3: File Validation
Before running, validate your data:
```bash
python -c "
import os
data_dir = 'data'
total_files = sum([len(files) for r, d, files in os.walk(data_dir)])
print(f'Total audio files found: {total_files}')
"
```

---

## 🚀 Running the System

### Method 1: Interactive Mode (Recommended)
```bash
# Activate virtual environment first
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run the main script
python test_runner.py
```

The system will prompt you to choose from 4 processing modes:

#### Mode 1: Traditional Processing
- **Best for**: Small datasets (< 50 files)
- **Features**: Basic feature extraction with retry mechanisms
- **Speed**: Moderate
- **Resource Usage**: Low

#### Mode 2: Enhanced Pipeline Processing (Recommended)
- **Best for**: Medium datasets (50-200 files)
- **Features**: Advanced validation, error recovery, comprehensive logging
- **Speed**: Fast
- **Resource Usage**: Medium

#### Mode 3: Lightweight Pipeline
- **Best for**: Quick testing or resource-constrained environments
- **Features**: Essential features only, faster processing
- **Speed**: Very Fast
- **Resource Usage**: Low

#### Mode 4: Batch Processing (For Large Datasets)
- **Best for**: Large datasets (200+ files)
- **Features**: Optimized memory management, parallel processing, advanced resource monitoring
- **Speed**: Very Fast
- **Resource Usage**: High but controlled

### Method 2: Command Line Mode
```bash
# Set processing mode via environment variable
export PROCESSING_MODE=2  # or 1, 3, 4

# Run with specific configuration
python test_runner.py --mode 2 --concurrency 4
```

### Method 3: Programmatic Usage
```python
import asyncio
from test_runner import main

# Run the pipeline programmatically
asyncio.run(main())
```

---

## 📊 Understanding the Output

### 1. Console Output
During processing, you'll see:
```
============================================================
PHYSICS FEATURES EXTRACTION PIPELINE
============================================================
✓ Enhanced security and batch processing modules loaded
Initializing feature extractor...
✓ Feature extractor initialized successfully!

Found 40 audio files to process
Processing Mode: 2 (Enhanced Pipeline Processing)
Starting processing with concurrency limit: 4

Processing audio files: 100%|████████████| 40/40 [01:39<00:00, 2.49s/file]

PROCESSING SUMMARY
===========================================
Total files discovered: 40
Files processed: 40
Successful: 40 (100.0%)
Failed: 0
Processing time: 99.7s
Average time per file: 2.49s
```

### 2. Results Files

#### A. CSV Results (`results/physics_features_summary.csv`)
Contains detailed features for each audio file:
- **File metadata**: Path, type, size, duration
- **Physics features**: All 4 VoiceRadar dynamics measurements
- **Traditional features**: Mel-spectrogram, LFCC statistics
- **Processing metadata**: Times, validation status, cache hits

#### B. Interactive Dashboard (`visualizations/interactive/comprehensive_dashboard.html`)
- **Box plots**: Feature distributions by file type
- **Correlation heatmap**: Inter-feature relationships
- **Scatter plots**: Feature evolution patterns
- **Statistical tests**: Significance analysis
- **PCA visualization**: 2D feature space projection

#### C. Static Plots (`visualizations/static/`)
- `physics_distributions.png`: Feature histograms
- `physics_correlation_analysis.png`: Correlation matrix and key feature pairs
- `statistical_comparison.png`: Genuine vs deepfake comparisons
- `performance_analysis.png`: System performance metrics

#### D. Analysis Reports (`visualizations/reports/`)
- `summary_report.md`: Human-readable analysis summary
- `analysis_summary.json`: Machine-readable detailed results
- `statistical_analysis.csv`: Statistical test results

### 3. Physics Features Explanation

#### Translational Dynamics (Δf_t)
- **Range**: Typically 0.01-0.15 Hz
- **Interpretation**: Overall drift in embedding space
- **Discrimination**: Minimal difference between genuine and TTS

#### Rotational Dynamics (Δf_r)
- **Range**: Typically 5-10 Hz
- **Interpretation**: Principal component rotation rate
- **Discrimination**: **Strongest discriminator** - TTS shows ~2.8% higher values

#### Vibrational Dynamics (Δf_v)
- **Range**: Typically 0.5-2.5 Hz
- **Interpretation**: High-frequency oscillations
- **Discrimination**: High variability, moderate discrimination

#### Total Dynamics (Δf_total)
- **Range**: Typically 6-12 Hz
- **Interpretation**: Combined dynamic signature
- **Discrimination**: Composite measure with good overall discrimination

---

## 🎛️ Advanced Usage

### Configuration Customization

#### Security Settings
Edit security parameters in `test_runner.py`:
```python
SECURITY_CONFIG = SecurityConfig(
    max_file_size_mb=200.0,     # Maximum file size
    max_memory_gb=12.0,         # Memory limit
    max_processing_time_s=600.0, # Processing timeout
    allowed_formats={'.wav', '.mp3', '.flac', '.m4a'}
)
```

#### Physics Feature Parameters
Modify `src/utils/config_loader.py`:
```python
physics:
  time_window_for_dynamics_ms: 50    # Analysis window
  embedding_dim_for_physics: 768     # HuBERT dimension
  dynamics_calculation_method: "revised"
```

#### Batch Processing Optimization
Adjust batch parameters for your system:
```python
batch_config = BatchConfig(
    batch_size=16,              # Files per batch
    max_concurrent_batches=2,   # Parallel batches
    enable_length_bucketing=True, # Group similar lengths
    memory_efficient_mode=True   # Memory optimization
)
```

### Resume from Checkpoint
If processing is interrupted:
```bash
python test_runner.py
# Choose your processing mode
# When prompted: "Resume from checkpoint? (y/n): y"
```

### Custom Visualization
Generate specific plots:
```python
from src.visualization.advanced_plotter import AdvancedPhysicsPlotter

plotter = AdvancedPhysicsPlotter("custom_visualizations")
results = plotter.generate_all_visualizations("results/physics_features_summary.csv")
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. "No module named 'resource'" Error (Windows)
**Solution**: This is expected on Windows. The system automatically falls back to psutil for resource monitoring.

#### 2. CUDA Out of Memory
**Solutions**:
- Reduce batch size: `batch_size=8`
- Use CPU mode: Set `CUDA_VISIBLE_DEVICES=""`
- Enable memory efficient mode in batch processing

#### 3. HuBERT Model Download Fails
**Solutions**:
- Check internet connection
- Verify disk space (>2GB needed)
- Clear HuggingFace cache: `rm -rf ~/.cache/huggingface/`
- Manual download:
```python
from transformers import HubertModel
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
```

#### 4. Audio Loading Errors
**Solutions**:
- Verify file format compatibility
- Check file permissions
- Ensure files aren't corrupted:
```bash
# Test with librosa
python -c "import librosa; librosa.load('path/to/file.wav')"
```

#### 5. Memory Issues with Large Datasets
**Solutions**:
- Use Mode 4 (Batch Processing)
- Enable `memory_efficient_mode=True`
- Reduce `concurrency_limit`
- Clear cache: `rm -rf cache/*`

#### 6. Visualization Generation Fails
**Solutions**:
- Install visualization dependencies:
```bash
pip install plotly kaleido dash dash-bootstrap-components
```
- Check file permissions in visualizations folder
- Verify CSV results file exists

### Debugging Mode
Enable detailed logging:
```bash
export DEBUG_MODE=1
python test_runner.py
```

### Performance Monitoring
Monitor system resources during processing:
```python
# Built-in monitoring (automatic in Mode 4)
# Or use external tools:
# Windows: Task Manager, Resource Monitor
# macOS: Activity Monitor
# Linux: htop, nvidia-smi (for GPU)
```

---

## ⚡ Performance Optimization

### Hardware Optimization

#### CPU Optimization
```python
# Adjust based on your CPU cores
concurrency_limit = min(os.cpu_count() - 1, 6)
```

#### GPU Optimization
```python
# For multiple GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use specific GPUs
```

#### Memory Optimization
```python
# Conservative memory usage
batch_config = BatchConfig(
    batch_size=8,               # Smaller batches
    memory_efficient_mode=True,
    enable_length_bucketing=False  # Disable if memory critical
)
```

### Software Optimization

#### Cache Management
- **Keep cache** for repeated processing of same files
- **Clear cache** if switching between different audio types
- **Monitor cache size**: Can grow to several GB

#### Processing Strategy
- **Small datasets (< 50 files)**: Mode 1 or 2
- **Medium datasets (50-200 files)**: Mode 2
- **Large datasets (200+ files)**: Mode 4
- **Testing/Development**: Mode 3

---

## 🧹 Maintenance

### Regular Cleanup
Run the cleanup utility periodically:
```bash
python cleanup_unnecessary_files.py
# Choose option 1 for safe cleanup
```

### Cache Management
```bash
# View cache size
du -sh cache/

# Clear old cache (>24 hours)
find cache/ -name "*.pkl" -mtime +1 -delete

# Clear all cache (regenerates on next run)
rm -rf cache/*
```

### Log Management
```bash
# Archive old logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
rm -rf logs/*
```

### Dependency Updates
```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Check for outdated packages
pip list --outdated

# Update specific critical packages
pip install --upgrade torch transformers librosa
```

### System Health Check
```bash
# Quick system verification
python -c "
import torch, transformers, librosa, pandas, plotly
print('✓ All critical dependencies available')
print(f'PyTorch CUDA: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU\"}')
"
```

---

## 📞 Support and Contact

For issues, questions, or contributions:

1. **Check this guide** for common solutions
2. **Review error logs** in `logs/` directory
3. **Run diagnostics**:
   ```bash
   python test_enhanced_system.py  # Run comprehensive tests
   ```
4. **Create issue report** with:
   - Operating system and Python version
   - Complete error message
   - Sample data characteristics
   - Hardware specifications

---

## 🎯 Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Audio data prepared in `data/` directory
- [ ] At least 8GB RAM available
- [ ] 10GB+ free disk space
- [ ] Run `python test_runner.py`
- [ ] Choose processing mode (2 recommended for first run)
- [ ] View results in `visualizations/interactive/comprehensive_dashboard.html`

**Estimated time for first run**: 10-30 minutes depending on dataset size and hardware.

---

## 🗂️ Folder Management System

The project includes a comprehensive project management utility that handles folder setup, cleanup, and complete project reset functionality:

### Unified Project Manager
All project management tasks are now handled by a single, powerful utility:

```bash
# Activate virtual environment
venv\Scripts\activate

# Run the comprehensive project manager
python project_manager.py
```

### Project Manager Features

#### 📁 Folder Management
- **Automatic Setup**: Creates and populates all necessary project directories
- **Status Monitoring**: Real-time folder status and content analysis
- **Documentation**: Auto-generates README files and configuration for each folder
- **Validation**: Ensures all folders contain expected files and structure

#### 🧹 Cleanup Options
- **Safe Cleanup**: Removes Python cache, temporary files, and old logs
- **Custom Cleanup**: Choose specific categories of files to remove
- **Analysis**: Preview what will be removed before taking action
- **Statistics**: Detailed reports on space savings and file counts

#### 🔄 Complete Project Reset
- **Full Analysis**: Comprehensive preview of what will be reset
- **Cache Clearing**: Removes all cached models and processing results
- **Voice Models Cache**: Clears external voice model caches in parent directory
- **Generated Content**: Removes all visualizations, reports, and analysis outputs
- **Safety Confirmation**: Requires explicit confirmation to prevent accidental resets

### Project Manager Menu Options

When you run `python project_manager.py`, you'll see an interactive menu:

```
📋 PROJECT MANAGER MENU:
  1. Test and setup folder management system
  2. Show current project structure
  3. Analyze unnecessary files (safe cleanup)
  4. Perform safe cleanup
  5. Analyze complete project reset
  6. Perform COMPLETE PROJECT RESET (⚠️ DANGEROUS)
  7. Custom cleanup options
  8. Exit
```

### Quick Actions

#### Setup Project Folders
```bash
python project_manager.py
# Choose option 1 to test and setup folder management
```

#### Clean Unnecessary Files
```bash
python project_manager.py
# Choose option 4 for safe cleanup of cache and temporary files
```

#### Reset Project to Clean State
```bash
python project_manager.py
# Choose option 6 for complete project reset
# ⚠️ This removes ALL generated content and caches!
```

### Safety Features

- **Preview Mode**: All operations show what will be affected before taking action
- **Confirmation Required**: Destructive operations require explicit confirmation
- **Error Handling**: Graceful handling of permission errors and missing files
- **Logging**: All reset operations are logged with detailed statistics
- **Recovery Information**: Clear instructions on how to regenerate content

### What Gets Reset

A complete project reset will remove:

**📁 Project Directories:**
- `results/` - All analysis results and CSV files
- `checkpoints/` - Processing checkpoints and recovery files
- `logs/` - System logs and session records
- `output/` - Analysis reports and summaries
- `plots/` - Statistical visualizations and plots
- `quarantine/` - Security quarantined files
- `cache/` - Local model outputs and cached results
- `visualizations/` - Interactive dashboards and enhanced plots

**💾 External Caches:**
- `~/.cache/voice_models` - Voice model cache
- `~/.cache/huggingface` - HuggingFace model cache
- `~/.cache/transformers` - Transformer model cache
- Parent directory model caches

**📄 Generated Files:**
- `FOLDER_INDEX.md` - Project folder documentation
- `FOLDER_CLEANUP_SUMMARY.md` - Cleanup operation summaries
- HTML visualization files
- Generated plot images

### After Reset

After a complete project reset:
1. ✅ Project is returned to clean, pre-test_runner.py state
2. ✅ All source code and configuration files remain intact
3. ✅ Virtual environment and dependencies remain unchanged
4. ✅ Ready to run `test_runner.py` again to regenerate all functionality
5. ✅ Models will be re-downloaded as needed (or loaded from system cache)

## 🚨 Troubleshooting

### Issue: Project manager not responding
**Solution:**
```bash
# Check if folder manager dependencies are available
python -c "from src.utils.folder_manager import FolderManager; print('✓ Available')"

# If not available, the project manager will still work with limited functionality
```

### Issue: Permission errors during cleanup
**Solution:**
```bash
# Run as administrator on Windows, or with sudo on Linux/Mac
# Or check file permissions and ownership
```

### Issue: Cache directories not found
**Solution:**
The project manager gracefully handles missing cache directories and will show accurate analysis of what actually exists on your system.

### Issue: Need to recover after accidental reset
**Solution:**
1. Check if a reset log was created: `project_reset_log_YYYYMMDD_HHMMSS.json`
2. Re-run `test_runner.py` to regenerate all analysis and visualizations
3. Models will be automatically re-downloaded as needed

## 🔍 Advanced Usage

### Non-Interactive Mode
For automated scripts or CI/CD, you can extend the project manager to support command-line arguments:

```bash
# These would be potential future enhancements:
python project_manager.py --analyze-only    # Show analysis without prompting
python project_manager.py --safe-cleanup    # Perform safe cleanup automatically
python project_manager.py --setup-folders   # Setup folders without interaction
```

### Integration with Test Runner
The project manager is designed to work seamlessly with `test_runner.py`:

1. **Before First Run**: Use project manager to setup folder structure
2. **Regular Maintenance**: Use project manager for periodic cleanup
3. **Fresh Start**: Use project manager to reset everything and start over
4. **Debugging**: Use project manager to check folder status and content

## 🎯 Quick Start Guide

*This guide covers comprehensive usage of the Physics-Based Deepfake Detection System. For technical details about the algorithms and research methodology, refer to `PROJECT_OVERVIEW.md`.* 