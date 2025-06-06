# Physics Feature Test Project

This project implements a deep learning model for audio feature extraction using the HuBERT model with enhanced **smart model loading**, **intelligent caching**, and **clean logging**.

## 🚀 Key Improvements

- ✅ **Clean Logging**: No more HTTP debug spam
- ✅ **Smart Model Caching**: Efficient downloading and storage
- ✅ **Offline Mode**: Works without internet after first download
- ✅ **Intelligent Fallbacks**: Automatic retry with smaller models if needed
- ✅ **Network-Aware**: Detects connectivity and adjusts behavior
- ✅ **Interactive Setup**: User-guided model downloading and caching
- ✅ **One-Time Downloads**: Models downloaded once, used forever

## 🔧 Interactive Setup System

The project now includes an **interactive setup wizard** that:

- ✅ **Checks existing cached models** and their status
- ✅ **Asks user permission** before downloading large models
- ✅ **Provides offline alternatives** when internet is unavailable
- ✅ **Manages cache expiry** and updates intelligently
- ✅ **Verifies project dependencies** and structure

### Running the Setup Wizard

**Option 1: Standalone Setup (Recommended)**
```bash
python setup_check.py
```

**Option 2: Integrated with Test Runner**
```bash
python test_runner.py  # Includes setup check
```

**Option 3: Quick Test with Setup**
```bash
python quick_test.py  # Minimal setup check
```

### Setup Wizard Features

#### 📦 Model Status Check
The wizard shows detailed information about your cached models:

```
📦 MODEL STATUS:
  ✅ hubert-large-ls960-ft [1.2GB] (cached 3 days ago)
  ✅ hubert-base-ls960 [360MB] (cached 7 days ago)
  ❌ wav2vec2-base-960h - Not downloaded
```

#### ❓ Interactive User Choices

**When models are missing:**
```
❓ MISSING MODELS (1):
  - wav2vec2-base-960h

❓ Download missing models now?
  1. Yes, download now (default)
  2. No, run in offline mode (may fail)
  3. Exit setup

Enter choice (1-3) [default: 1]:
```

**When models are outdated:**
```
❓ OUTDATED MODELS (1):
  - hubert-base-ls960 (cached 35 days ago)

❓ Check for model updates?
  1. Yes, check for updates
  2. No, use cached models (default)
  3. Clear cache and re-download

Enter choice (1-3) [default: 2]:
```

#### 🔄 Cache Management

The system intelligently manages model cache:

- **Fresh models** (< 7 days): Used immediately
- **Recent models** (7-30 days): Used with notification
- **Old models** (30+ days): User prompted for update
- **Corrupted cache**: Automatically cleared and re-downloaded

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Internet connection (for first model download only)

## Installation

1. **Clone and navigate to the project:**
```bash
git clone <repository-url>
cd physics_feature_test_project
```

2. **Create and activate a virtual environment (recommended):**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the setup wizard:**
```bash
python setup_check.py
```

The setup wizard will guide you through:
- ✅ Checking Python packages
- ✅ Verifying project structure  
- ✅ Downloading required models (with permission)
- ✅ Setting up data directories

## Running the Project

### First-Time Setup
```bash
# Run the interactive setup wizard
python setup_check.py

# Follow the prompts to download models
# This may take 5-10 minutes depending on internet speed
```

### Regular Usage
```bash
# After setup, models are cached - starts immediately
python test_runner.py
```

### Quick Testing
```bash
# Test the installation
python quick_test.py
```

## 🚫 No More Download Spam!

The new system eliminates repetitive downloads:

### ❌ Before (Every Run):
```
Downloading facebook/hubert-large-ls960-ft...
Fetching 12 files: 100%|████████| 12/12 [05:23<00:00, ...]
Downloading facebook/hubert-base-ls960...
Fetching 8 files: 100%|████████| 8/8 [02:15<00:00, ...]
```

### ✅ After (One-Time Setup):
```
📦 MODEL STATUS:
  ✅ hubert-large-ls960-ft [1.2GB] (cached 3 days ago)
  ✅ hubert-base-ls960 [360MB] (cached 3 days ago)

✅ All models are cached and up-to-date
✅ PROJECT READY TO RUN!
```

## 🔧 Configuration Options

### Environment Variables
```bash
# Force offline mode (no downloads)
export TRANSFORMERS_OFFLINE=1
python test_runner.py

# Disable interactive prompts (auto-download)
export AUTO_DOWNLOAD=1
python test_runner.py

# Use specific cache directory
export HF_CACHE_DIR="/path/to/custom/cache"
python test_runner.py
```

### Programmatic Configuration
```python
from src.core.model_loader import ModelConfig

# Interactive mode with user prompts
config = ModelConfig().enable_interactive_mode()

# Auto-download mode (no prompts)
config = ModelConfig().enable_auto_download()

# Offline-only mode
config = ModelConfig().enable_offline_mode()
```

## 📁 Project Structure

The project has a well-organized folder structure for different types of outputs and operations:

```
physics_feature_test_project/
├── 📁 data/                    # Input audio files organized by user
├── 📁 src/                     # Source code modules and utilities
├── 📁 results/                 # Main analysis results and CSV outputs
├── 📁 checkpoints/             # Processing checkpoints for recovery
├── 📁 logs/                    # Comprehensive system logging
├── 📁 output/                  # Analysis reports and summaries  
├── 📁 plots/                   # Statistical plots and visualizations
├── 📁 quarantine/              # Quarantined suspicious files
├── 📁 visualizations/          # Enhanced interactive visualizations
├── 📁 cache/                   # Cached model outputs and intermediate results
├── 📁 venv/                    # Python virtual environment
├── 📄 test_runner.py           # Main execution script
├── 📄 FOLDER_INDEX.md          # Detailed folder documentation
└── 📄 requirements.txt         # Python dependencies
```

### 🗂️ Folder Management

The project includes an automated folder management system that:

- **Automatically creates** all necessary directories and subdirectories
- **Populates folders** with documentation and configuration files
- **Maintains structure** while preserving existing content
- **Provides detailed documentation** for each folder's purpose

Key folders and their purposes:

- **`results/`**: Contains the main CSV analysis outputs and error logs
- **`checkpoints/`**: Stores processing checkpoints for recovery from interruptions
- **`logs/`**: Comprehensive logging system with separate logs for different components
- **`output/`**: Analysis reports, summaries, and session documentation
- **`plots/`**: High-quality statistical plots and visualizations (300 DPI)
- **`quarantine/`**: Security system for suspicious or corrupted files
- **`visualizations/`**: Interactive dashboards and enhanced visualizations
- **`cache/`**: Model outputs and intermediate processing results

### 📋 Folder Documentation

Each folder contains:
- **README.md**: Detailed purpose and usage instructions
- **Configuration files**: JSON configurations for logging, plotting, etc.
- **Status files**: Real-time status and statistics
- **Log files**: Component-specific logging and session tracking

## 🎯 Smart Caching Features

### Cache Validation
- ✅ **File completeness check**: Ensures all model files are present
- ✅ **Integrity verification**: Validates file sizes and checksums
- ✅ **Age tracking**: Monitors cache freshness
- ✅ **Access logging**: Tracks when models were last used

### Intelligent Fallbacks
1. **Primary model** (hubert-large-ls960-ft) - Best quality
2. **Fallback model** (hubert-base-ls960) - Good quality, smaller
3. **Emergency model** (wav2vec2-base-960h) - Basic functionality

### Network Awareness
- 🌐 **Online**: Downloads missing models, checks for updates
- 🔌 **Offline**: Uses cached models, works without internet
- 📡 **Limited bandwidth**: Offers smaller model alternatives
- ⚡ **Fast connection**: Downloads largest models for best quality

## Troubleshooting

### Issue: Setup wizard not responding
**Solution:**
```bash
# Force non-interactive mode
python test_runner.py --no-interactive

# Or use environment variable
export NO_INTERACTIVE=1
python test_runner.py
```

### Issue: Models won't download
**Solution:**
```bash
# Check internet connection
python -c "import socket; socket.create_connection(('huggingface.co', 443), timeout=5)"

# Clear cache and retry
rm -rf ~/.cache/voice_models
python setup_check.py
```

### Issue: Cache corruption
**Solution:**
```bash
# The setup wizard will detect and fix automatically
python setup_check.py

# Or manually clear cache
rm -rf ~/.cache/voice_models
```

### Issue: Permission errors
**Solution:**
```bash
# Use different cache directory
export HF_CACHE_DIR="./local_cache"
python setup_check.py
```

## 📊 Results

After processing, results are saved in timestamped directories:

```
results/
└── run_20240115_103015/
    ├── results_final.csv       # Main results
    ├── results_final.xlsx      # Excel with multiple sheets
    ├── analysis.json           # Statistical analysis
    └── plots/                  # Visualizations (if matplotlib available)
        ├── processing_time.png
        ├── physics_features.png
        └── quality_distribution.png
```

## 🚨 Troubleshooting

### Issue: Still seeing HTTP debug logs
**Solution:** Make sure you're running the updated version. Try:
```bash
python quick_test.py
```

### Issue: Model download fails
**Solution:** The system will automatically:
1. Try cached version first
2. Retry downloads with exponential backoff
3. Fall back to smaller models
4. Use offline mode if no network

### Issue: Out of memory
**Solution:** The system automatically:
1. Detects available memory
2. Enables CPU offloading if needed
3. Uses smaller models as fallbacks

### Issue: Network connectivity problems
**Solution:** 
1. Run setup wizard once with internet to download models
2. Subsequent runs work offline automatically
3. Or set `force_offline=True` in config

## 🔍 Advanced Usage

### Complete Offline Setup
```bash
# Download everything once (with internet)
python setup_check.py

# Then work offline forever
export TRANSFORMERS_OFFLINE=1
python test_runner.py  # Works without internet
```

### Custom Model Configuration
```python
from src.core.model_loader import ModelConfig, DeviceType, OptimizationLevel

config = ModelConfig(
    model_path="your/custom/hubert/model",
    cache_dir="./custom_cache",
    device_preference=DeviceType.CUDA,
    optimization_level=OptimizationLevel.AGGRESSIVE,
    cache_expiry_days=30
).enable_interactive_mode()
```

### Automated CI/CD Setup
```bash
# Non-interactive mode for automated systems
export AUTO_DOWNLOAD=1
export NO_INTERACTIVE=1
python test_runner.py
```

## 📈 Performance Improvements

- **🚫 No HTTP Debug Spam**: Clean terminal output
- **⚡ Smart Caching**: Models download once, run forever
- **🔄 Intelligent Retries**: Robust downloading with fallbacks
- **📡 Network Aware**: Detects and adapts to connectivity
- **💾 Memory Efficient**: Automatic memory management
- **⏱️ Fast Startup**: Cached models load in seconds
- **🎯 User Control**: Interactive choices for downloads

## 🛠️ Project Management

The project includes a comprehensive management utility that handles all aspects of project organization, cleanup, and reset functionality.

### Unified Project Manager

All project management tasks are handled by the `project_manager.py` utility:

```bash
# Activate virtual environment
venv\Scripts\activate

# Run the project manager
python project_manager.py
```

### Features

#### 📁 Folder Management
- **Automatic Structure Creation**: Sets up all necessary directories and subdirectories
- **Documentation Generation**: Creates README files and configuration for each folder
- **Status Monitoring**: Real-time analysis of folder contents and status
- **Validation**: Ensures proper folder structure and expected files

#### 🧹 Cleanup and Maintenance
- **Safe Cleanup**: Removes Python cache, temporary files, and old logs
- **Custom Cleanup**: Choose specific categories of files to remove ✅ **Fixed in v2.0**
- **Analysis Mode**: Preview changes before executing
- **Space Optimization**: Detailed reports on potential space savings

#### 🔄 Complete Project Reset
- **Full Reset Capability**: Returns project to pre-test_runner.py state
- **Cache Clearing**: Removes all cached models and processing results
- **External Cache Management**: Clears voice model caches in parent directory
- **Safety Features**: Requires explicit confirmation for destructive operations

### Menu Options

```
📋 PROJECT MANAGER MENU:
  1. Test and setup folder management system
  2. Show current project structure
  3. Analyze unnecessary files (safe cleanup)
  4. Perform safe cleanup
  5. Analyze complete project reset
  6. Perform COMPLETE PROJECT RESET (⚠️ DANGEROUS)
  7. Custom cleanup options  ✅ Enhanced with better validation
  8. Exit
```

### 🔧 Enhanced Synchronization (v2.0)

The project now includes advanced synchronization features to ensure safe concurrent operation:

#### **Cache Synchronization**
- **Automatic lock mechanism** prevents cache conflicts
- **Process ID tracking** for safe multi-instance operation
- **Cache integrity validation** on startup
- **Corruption detection** and warning system

#### **Mode Coordination**
All processing modes (1-4) now coordinate properly:
- **Mode 1**: Traditional Processing - Basic synchronization
- **Mode 2**: Enhanced Pipeline - Enhanced coordination
- **Mode 3**: Lightweight Pipeline - Lightweight coordination  
- **Mode 4**: Batch Processing - Advanced resource coordination

#### **Output File Protection**
- **Exclusive access validation** before processing
- **Permission conflict detection** and resolution
- **Automatic recovery** from interrupted sessions

### 🎯 Custom Cleanup Improvements

**Fixed Issues (v2.0):**
- ✅ Categories 4 and 5 now work correctly
- ✅ Enhanced input validation with detailed error messages
- ✅ Size calculation and preview before cleanup
- ✅ Better user experience with comprehensive feedback

**Example Usage:**
```bash
python project_manager.py
# Choose option 7 - Custom cleanup options
# Select categories: 1,4,5 (all options now working)

Selected categories:
  1. Python Cache: 4 items (2.1 MB)
  4. Cache Files: 12 items (8.7 MB)  
  5. Empty Directories: 13 items (0.0 MB)

Total: 29 items, 10.8 MB
Proceed with cleanup? (y/n):
```

### Project Reset Functionality

The project manager can completely reset the project by removing:

**Project Directories:**
- `results/`, `checkpoints/`, `logs/`, `output/`, `plots/`, `quarantine/`, `cache/`, `visualizations/`

**External Caches:**
- Voice model caches in parent directory
- HuggingFace and transformer model caches
- User cache directories

**Generated Files:**
- Documentation files (FOLDER_INDEX.md, etc.)
- Visualization HTML files
- Generated plots and images

After reset, the project is ready for fresh `test_runner.py` execution.

### 📁 Cache Organization

The cache system includes organized subdirectories for future development:

```
cache/
├── *.pkl files           # Current active cache (MD5 hashes)
├── README.md            # Cache documentation
├── cache_status.json    # Organization metadata
├── hubert/              # Reserved: HuBERT embeddings only
├── features/            # Reserved: Computed features only
└── processing/          # Reserved: Processing state
```

**Current Status**: All cache files are stored in the root `cache/` directory. Subdirectories are documented and ready for future hierarchical organization improvements.

### 🚨 Troubleshooting

#### Issue: Cache lock persists after abnormal termination
**Solution:** Remove the lock file manually
```bash
rm cache/.cache_lock
python test_runner.py
```

#### Issue: Custom cleanup not working
**Solution:** Use the enhanced v2.0 system with proper validation
```bash
python project_manager.py
# Choose option 7 - All categories (1-5) now work correctly
```

#### Issue: Synchronization warnings
**Solution:** Review the synchronization analysis
```bash
# Check detailed analysis
cat SYNCHRONIZATION_ANALYSIS.md
```

#### Issue: Cache conflicts between processing modes
**Solution:** The system now uses mode-specific caching (Fixed in v3.0)
```bash
# Each processing mode now uses separate cache coordination
echo "4" | python test_runner.py  # Batch processing (cache: batch)
echo "2" | python test_runner.py  # Pipeline processing (cache: pipeline)
# No conflicts - each mode has its own cache space
```

#### Issue: Processing modes creating duplicate cache files
**Solution:** Enhanced cache coordination system prevents conflicts
```bash
# Check cache coordination documentation
cat CACHE_COORDINATION_FIXES.md
```

## 🆕 Latest Updates (v3.0)

### ✅ **Cache Coordination Fixes** 
- **Fixed cache conflicts** between different processing modes
- **Mode-specific caching** ensures proper cache isolation
- **Enhanced cache validation** prevents incompatible cache usage
- **Backward compatibility** with existing cache files

### ✅ **Custom Cleanup Fixes**
- **Fixed custom cleanup options** that weren't working properly
- **Enhanced error handling** with detailed feedback
- **Real-time progress tracking** during cleanup operations
- **Comprehensive logging** and statistics

### 📋 **Processing Mode Coordination**
```bash
# All processing modes now work with perfect coordination:
echo "1" | python test_runner.py  # Traditional (cache: traditional)
echo "2" | python test_runner.py  # Enhanced Pipeline (cache: pipeline)  
echo "3" | python test_runner.py  # Lightweight (cache: lightweight)
echo "4" | python test_runner.py  # Batch Processing (cache: batch)
```

**For complete details, see:** `CACHE_COORDINATION_FIXES.md`

## 📝 Notes

- **First run**: Interactive setup downloads models (5-10 minutes)
- **Subsequent runs**: Uses cache (starts in seconds)
- **Offline capable**: Works without internet after first download
- **Auto-fallback**: Uses smaller models if main model fails
- **Clean output**: No more HTTP debug spam in terminal
- **User control**: You decide when to download or update

---

**Ready to run with intelligent caching and user control! 🎉**