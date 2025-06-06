# Physics-Based Deepfake Detection System - User Guide

**Version:** 3.0  
**Last Updated:** December 2024  
**Complete User Manual for Installation, Usage, and Troubleshooting**

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Guide](#installation-guide)
3. [Data Preparation](#data-preparation)
4. [Running the System](#running-the-system)
5. [Understanding Results](#understanding-results)
6. [Project Management](#project-management)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)
10. [Best Practices](#best-practices)

---

## 🖥️ System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 10GB+ free space for models and cache
- **GPU**: Optional but recommended (CUDA-compatible for faster processing)

### Software Requirements
- **Python**: Version 3.8+ (3.10+ recommended)
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

### Step 4: Verify Installation
```bash
# Run quick test to verify setup
python quick_test.py

# Run import test to check all dependencies
python test_imports.py
```

### Step 5: Verify GPU Support (Optional)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📂 Data Preparation

### Directory Structure
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

### Audio File Requirements
- **Format**: WAV, MP3, FLAC, or M4A
- **Sample Rate**: Any (will be normalized to 16kHz)
- **Duration**: Minimum 1 second, no maximum limit
- **File Size**: Maximum 200MB per file (configurable)

### Naming Convention
Include descriptive keywords in filenames:
- `genuine_*` for authentic audio
- `deepfake_tts_*` for text-to-speech deepfakes
- `deepfake_vc_*` for voice conversion deepfakes

### File Validation
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

### Quick Start
```bash
# Activate virtual environment
venv\Scripts\activate

# Run quick test
python quick_test.py

# Run full analysis (recommended for first time)
echo "4" | python test_runner.py
```

### Interactive Mode (Recommended)
```bash
python test_runner.py
```

The system will prompt you to choose from 4 processing modes:

#### Mode 1: Traditional Processing
- **Best for**: Small datasets (< 50 files)
- **Features**: Basic feature extraction with retry mechanisms
- **Speed**: Moderate
- **Resource Usage**: Low

#### Mode 2: Enhanced Pipeline Processing ⭐ **Recommended**
- **Best for**: Medium datasets (50-200 files)
- **Features**: Advanced validation, error recovery, comprehensive logging
- **Speed**: Fast
- **Resource Usage**: Medium

#### Mode 3: Lightweight Pipeline
- **Best for**: Quick testing or resource-constrained environments
- **Features**: Essential features only, faster processing
- **Speed**: Very Fast
- **Resource Usage**: Low

#### Mode 4: Batch Processing
- **Best for**: Large datasets (200+ files)
- **Features**: Optimized memory management, parallel processing, advanced resource monitoring
- **Speed**: Very Fast
- **Resource Usage**: High but controlled

### Command Line Mode
```bash
# Set processing mode via environment variable
export PROCESSING_MODE=2  # or 1, 3, 4

# Run with specific configuration
python test_runner.py --mode 2 --concurrency 4
```

### Resuming Interrupted Processing
If processing is interrupted:
```bash
python test_runner.py
# Choose your processing mode
# When prompted: "Resume from checkpoint? (y/n): y"
```

---

## 📊 Understanding Results

### Console Output
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

### Results Files

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

### Physics Features Explanation

#### Translational Dynamics (Δf_t)
- **Range**: Typically 0.01-0.15 Hz
- **Interpretation**: Overall drift in embedding space
- **Discrimination**: Minimal difference between genuine and TTS

#### Rotational Dynamics (Δf_r) ⭐ **Most Important**
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

## 🔧 Project Management

### Unified Project Manager
All project management tasks are handled by `project_manager.py`:

```bash
# Activate virtual environment
venv\Scripts\activate

# Run project manager
python project_manager.py
```

### Menu Options

1. **Test and setup folder management system**
   - Creates all necessary directories
   - Generates documentation and configuration files
   - Validates project structure

2. **Show current project structure**
   - Displays complete folder organization
   - Shows file counts and sizes
   - Provides status overview

3. **Analyze unnecessary files (safe cleanup)**
   - Scans for Python cache, temporary files, old logs
   - Shows what can be safely removed
   - Provides size estimates

4. **Perform safe cleanup**
   - Removes Python cache files
   - Clears temporary processing files
   - Removes old log files
   - Preserves all important data

5. **Analyze complete project reset**
   - Shows what would be removed in a full reset
   - Previews external cache clearing
   - Provides detailed breakdown

6. **Perform COMPLETE PROJECT RESET**
   - Returns project to pre-test_runner.py state
   - Removes all generated files and caches
   - Clears external voice model caches
   - **⚠️ Requires explicit confirmation**

7. **Custom cleanup options**
   - Choose specific categories to clean
   - Preview changes before execution
   - Detailed feedback and statistics

### Complete Project Reset
To return the project to a clean state:
```bash
python project_manager.py
# Choose option 6
# Type exact confirmation phrase: "YES I WANT TO RESET THE PROJECT"
```

**What gets reset:**
- All project directories (`results/`, `cache/`, `logs/`, etc.)
- External caches (HuggingFace, voice models)
- Generated files (visualizations, reports, plots)
- Preserves source code and configuration

---

## ⚡ Performance Optimization

### Dataset Size Guidelines
- **Small (< 50 files)**: Mode 1 or 2
- **Medium (50-200 files)**: Mode 2 (Enhanced Pipeline)
- **Large (200+ files)**: Mode 4 (Batch Processing)

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
- Use Mode 4 (Batch Processing) for large datasets
- Enable `memory_efficient_mode=True`
- Reduce `concurrency_limit` if needed
- Clear cache if memory is critical

### Cache Management
- **Keep cache** for repeated processing of same files
- **Clear cache** if switching between different audio types
- **Monitor cache size**: Can grow to several GB

```bash
# View cache size
du -sh cache/

# Clear cache safely
python project_manager.py  # Option 4 - Safe cleanup
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
**Symptom**: `ModuleNotFoundError` or import failures
**Solution**:
```bash
# Test all imports
python test_imports.py

# If issues found, reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 2. "No module named 'resource'" Error (Windows)
**Solution**: This is expected on Windows. The system automatically falls back to psutil for resource monitoring.

#### 3. CUDA Out of Memory
**Solutions**:
- Use CPU mode: `echo "3" | python test_runner.py`
- Clear GPU memory and restart
- Use smaller batch sizes

#### 4. HuBERT Model Download Fails
**Solutions**:
- Check internet connection
- Verify disk space (>2GB needed)
- Clear HuggingFace cache:
```bash
python project_manager.py  # Option 4 - Safe cleanup
```

#### 5. Audio Loading Errors
**Solutions**:
- Verify file format compatibility
- Check file permissions
- Test with librosa:
```bash
python -c "import librosa; librosa.load('path/to/file.wav')"
```

#### 6. Memory Issues with Large Datasets
**Solutions**:
- Use Mode 4 (Batch Processing)
- Enable memory efficient mode
- Reduce concurrency limit
- Clear cache: `rm -rf cache/*`

#### 7. Processing Failures
**Solutions**:
- Check logs in `logs/` directory
- Run diagnostics: `python quick_test.py`
- Use project manager for system health check

#### 8. Visualization Generation Fails
**Solutions**:
- Install visualization dependencies:
```bash
pip install plotly kaleido dash dash-bootstrap-components
```
- Check file permissions in visualizations folder
- Verify CSV results file exists

#### 9. Cache Lock Persists
**Symptom**: "Cache lock exists" warning
**Solution**:
```bash
rm cache/.cache_lock
python test_runner.py
```

#### 10. Custom Cleanup Not Working
**Solution**: Use the enhanced system (fixed in v3.0):
```bash
python project_manager.py
# Choose option 7 - All categories now work correctly
```

### Debugging Mode
Enable detailed logging:
```bash
export DEBUG_MODE=1
python test_runner.py
```

### Performance Monitoring
Monitor system resources during processing:
```bash
# Built-in monitoring (automatic in Mode 4)
# Or use external tools:
# Windows: Task Manager, Resource Monitor
# macOS: Activity Monitor
# Linux: htop, nvidia-smi (for GPU)
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
```python
batch_config = BatchConfig(
    batch_size=16,              # Files per batch
    max_concurrent_batches=2,   # Parallel batches
    enable_length_bucketing=True, # Group similar lengths
    memory_efficient_mode=True   # Memory optimization
)
```

### Custom Visualization
Generate specific plots:
```python
from src.visualization.advanced_plotter import AdvancedPhysicsPlotter

plotter = AdvancedPhysicsPlotter("custom_visualizations")
results = plotter.generate_all_visualizations("results/physics_features_summary.csv")
```

### Environment Variables
```bash
# Force offline mode
export TRANSFORMERS_OFFLINE=1

# Disable interactive prompts
export AUTO_DOWNLOAD=1

# Use specific cache directory
export HF_CACHE_DIR="/path/to/custom/cache"

# Enable debug mode
export DEBUG_MODE=1
```

### Programmatic Usage
```python
import asyncio
from test_runner import main

# Run the pipeline programmatically
asyncio.run(main())
```

---

## 📋 Best Practices

### Data Organization
1. **Consistent naming**: Use descriptive filenames with clear labels
2. **Balanced datasets**: Include similar amounts of genuine and synthetic audio
3. **Quality control**: Validate audio files before processing
4. **Backup important data**: Keep original files safe

### Processing Strategy
1. **Start small**: Test with a few files first
2. **Use appropriate mode**: Match processing mode to dataset size
3. **Monitor resources**: Keep an eye on memory and CPU usage
4. **Regular maintenance**: Use project manager for cleanup

### Performance Tips
1. **Cache management**: Keep cache for repeated analysis
2. **Hardware optimization**: Use GPU if available
3. **Batch processing**: For large datasets, use Mode 4
4. **Storage**: Use SSD for better performance

### Security Practices
1. **File validation**: Let the system validate input files
2. **Resource limits**: Use built-in resource management
3. **Quarantine system**: Review quarantined files
4. **Regular updates**: Keep dependencies updated

### Results Management
1. **Save important results**: Export CSV files to safe location
2. **Document findings**: Use generated reports and visualizations
3. **Version control**: Track different analysis runs
4. **Share responsibly**: Follow ethical guidelines for deepfake detection

---

## 📞 Support and Contact

### Getting Help
1. **Check this guide** for common solutions
2. **Review error logs** in `logs/` directory
3. **Run diagnostics**:
   ```bash
   python quick_test.py
   python test_imports.py
   ```
4. **Use project manager** for system health checks

### Before Reporting Issues
Include the following information:
- Operating system and Python version
- Complete error message from logs
- Sample data characteristics (size, format, duration)
- Hardware specifications (RAM, CPU, GPU)
- Steps to reproduce the problem

### Self-Help Checklist
- [ ] Virtual environment activated
- [ ] Dependencies installed correctly
- [ ] Audio files in proper format and location
- [ ] Sufficient disk space and memory
- [ ] Error logs checked for specific issues
- [ ] Quick test and import test passed

---

## 🎯 Quick Reference

### Essential Commands
```bash
# Setup
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt

# Quick test
python quick_test.py

# Main analysis
echo "4" | python test_runner.py

# Project management
python project_manager.py

# Import validation
python test_imports.py

# Clean cache
python project_manager.py  # Option 4

# Complete reset
python project_manager.py  # Option 6
```

### File Locations
- **Results**: `results/physics_features_summary.csv`
- **Dashboard**: `visualizations/interactive/comprehensive_dashboard.html`
- **Logs**: `logs/feature_extraction_YYYYMMDD_HHMMSS.log`
- **Cache**: `cache/*.pkl`
- **Reports**: `visualizations/reports/summary_report.md`

### Processing Modes
1. **Traditional** (< 50 files)
2. **Enhanced Pipeline** (50-200 files) ⭐ **Recommended**
3. **Lightweight** (quick testing)
4. **Batch** (200+ files)

---

**This guide covers all aspects of using the Physics-Based Deepfake Detection System. For technical implementation details, see [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md).** 