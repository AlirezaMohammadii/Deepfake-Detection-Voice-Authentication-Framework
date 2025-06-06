# Physics-Based Deepfake Detection System

A comprehensive system for detecting AI-generated speech using physics-inspired dynamics analysis in neural embedding spaces. This project implements the VoiceRadar framework for audio authenticity verification through advanced signal processing and machine learning techniques.

## 🎯 Overview

This system analyzes audio files to distinguish between genuine human speech and AI-generated deepfakes using:

- **VoiceRadar Physics**: Novel translational, rotational, and vibrational dynamics analysis
- **Neural Embeddings**: HuBERT-based feature extraction with physics-inspired interpretation
- **Multi-Modal Processing**: Comprehensive feature extraction including traditional audio features
- **Production-Ready**: Enterprise-grade robustness with advanced error handling and recovery

## 🚀 Quick Start

### 1. Setup
```bash
# Clone and navigate
git clone <repository-url>
cd physics_feature_test_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
Place audio files in the `data/` directory:
```
data/
├── user_001/
│   ├── genuine_audio_001.wav
│   ├── deepfake_tts_001.wav
│   └── ...
└── user_002/
    ├── genuine_audio_002.wav
    └── ...
```

### 3. Run Analysis
```bash
# Quick test
python quick_test.py

# Full analysis (recommended for first run)
echo "4" | python test_runner.py
```

### 4. View Results
- **CSV Results**: `results/physics_features_summary.csv`
- **Interactive Dashboard**: `visualizations/interactive/comprehensive_dashboard.html`
- **Statistical Analysis**: `visualizations/reports/summary_report.md`

## 📊 Key Features

### Physics-Based Detection
- **Translational Dynamics (Δf_t)**: Overall drift in embedding space
- **Rotational Dynamics (Δf_r)**: Principal component rotation rate (strongest discriminator)
- **Vibrational Dynamics (Δf_v)**: High-frequency oscillations  
- **Total Dynamics (Δf_total)**: Combined dynamic signature

### Processing Modes
1. **Traditional** - Basic sequential processing (small datasets <50 files)
2. **Enhanced Pipeline** - Advanced validation and recovery (50-200 files) ⭐ **Recommended**
3. **Lightweight** - Fast processing with essential features
4. **Batch Processing** - Optimized for large datasets (200+ files)

### Smart System Features
- **Intelligent Caching**: Models download once, process forever
- **Checkpoint Recovery**: Resume interrupted processing
- **Security Validation**: File integrity and security checks
- **Resource Management**: Memory and CPU usage control
- **Advanced Visualization**: Interactive plots and statistical analysis

## 🎛️ Advanced Usage

### Interactive Processing Mode Selection
```bash
python test_runner.py
# Choose from 4 processing modes based on your dataset size and requirements
```

### Project Management
```bash
python project_manager.py
# Comprehensive project management:
# - Folder setup and organization
# - Safe cleanup and maintenance  
# - Complete project reset capabilities
```

### Custom Configuration
```python
# Example: Batch processing with custom settings
from test_runner import BatchConfig
config = BatchConfig(
    batch_size=16,
    max_concurrent_batches=2,
    memory_efficient_mode=True
)
```

## 📈 Research Results

Based on analysis of 40 audio samples (24 genuine, 16 TTS deepfakes):

### Key Findings
- **Rotational Dynamics**: Most discriminative feature (+2.8% higher in TTS, p<0.05)
- **Processing Success**: 100% success rate with robust error handling
- **Performance**: 2.49s average processing time per file
- **Statistical Significance**: Physics features show statistically significant discrimination

### Detection Capabilities
- **TTS Deepfakes**: Strong discrimination through rotational dynamics
- **Synthesis Artifacts**: Detected through embedding space physics analysis
- **Real-time Processing**: Optimized for production deployment

## 🛠️ System Requirements

### Minimum Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 10GB free space
- **CPU**: Multi-core processor

### Recommended Setup
- **RAM**: 32GB for large datasets
- **GPU**: NVIDIA RTX 3080+ for acceleration
- **Storage**: 50GB SSD for optimal performance

## 📚 Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** - Comprehensive usage instructions, troubleshooting, and examples
- **[TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md)** - Architecture, algorithms, and development guide
- **[CHANGELOG.md](CHANGELOG.md)** - Version history, bug fixes, and enhancements

## 🔧 Project Management

### Setup and Maintenance
```bash
python project_manager.py
# Menu options:
# 1. Setup folder management system
# 2. Show current project structure
# 3. Analyze unnecessary files
# 4. Perform safe cleanup
# 5. Analyze complete project reset
# 6. Perform complete project reset
```

### Fresh Start
To return the project to a clean state:
```bash
python project_manager.py
# Choose option 6 - Complete project reset
# Removes all generated files and caches
# Preserves source code and configuration
```

## 🚨 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Test all imports
python test_imports.py
```

**Model Download Issues**
```bash
# Clear cache and retry
python project_manager.py  # Option 4 - Safe cleanup
python test_runner.py
```

**Memory Issues**
```bash
# Use lightweight mode
echo "3" | python test_runner.py
```

**Processing Failures**
```bash
# Check logs for detailed error information
# Logs are in logs/ directory with timestamps
```

## 🎯 Performance Optimization

### Dataset Size Guidelines
- **Small (<50 files)**: Mode 1 or 2
- **Medium (50-200 files)**: Mode 2 (Enhanced Pipeline)
- **Large (200+ files)**: Mode 4 (Batch Processing)

### Hardware Optimization
- **CPU**: Adjust concurrency based on cores
- **GPU**: CUDA acceleration for HuBERT processing
- **Memory**: Use batch processing for large datasets
- **Storage**: SSD recommended for cache and results

## 🔒 Security Features

- **Input Validation**: Comprehensive file and content validation
- **Quarantine System**: Suspicious files isolated automatically
- **Resource Limits**: Memory and processing time constraints
- **Safe Processing**: Robust error handling prevents system compromise

## 📞 Support

### Getting Help
1. **Check documentation**: USER_GUIDE.md for detailed instructions
2. **Run diagnostics**: `python quick_test.py` and `python test_imports.py`
3. **Review logs**: Check `logs/` directory for error details
4. **System validation**: Use project_manager.py for system health checks

### Reporting Issues
When reporting issues, include:
- Operating system and Python version
- Complete error messages from logs
- Sample data characteristics (size, format, duration)
- Hardware specifications (RAM, CPU, GPU)

## 📜 License

See [LICENSE](LICENSE) file for details.

## 🙏 Citation

If you use this system in research, please cite:

```bibtex
@software{voiceradar_deepfake_detection,
  title={VoiceRadar: Physics-Based Deepfake Detection Using Neural Embedding Dynamics},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

---

**Ready to get started?** Run `python quick_test.py` to verify your setup, then `echo "4" | python test_runner.py` for your first analysis!