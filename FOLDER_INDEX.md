# Project Folder Structure Index

**Generated:** 2025-06-08 16:31:43  
**Session ID:** 20250608_163143

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

### 📁 results/
**Purpose:** Main analysis results and CSV outputs

**Subdirectories:**
- `checkpoints/`

**Auto-populated:** Yes

### 📁 checkpoints/
**Purpose:** Processing checkpoints for recovery and resumption

**Auto-populated:** Yes

### 📁 logs/
**Purpose:** Comprehensive logging for all system components

**Subdirectories:**
- `feature_extraction/`
- `model_performance/`
- `system/`
- `sessions/`

**Auto-populated:** Yes

### 📁 output/
**Purpose:** Analysis reports, summaries, and documentation

**Subdirectories:**
- `feature_summaries/`
- `analysis_reports/`
- `session_reports/`

**Auto-populated:** Yes

### 📁 plots/
**Purpose:** Matplotlib plots and statistical visualizations

**Subdirectories:**
- `physics_analysis/`
- `performance/`
- `validation/`
- `comparisons/`

**Auto-populated:** Yes

### 📁 quarantine/
**Purpose:** Quarantined files that failed security validation

**Subdirectories:**
- `logs/`

**Auto-populated:** Yes

### 📁 cache/
**Purpose:** Cached model outputs and intermediate results

**Subdirectories:**
- `hubert/`
- `features/`
- `processing/`

**Auto-populated:** Yes

### 📁 visualizations/
**Purpose:** Enhanced interactive and static visualizations

**Subdirectories:**
- `static/`
- `interactive/`
- `reports/`

**Auto-populated:** No

## Usage Guidelines

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
- Folders created: 0
- Subdirectories created: 0
- Documentation files created: 3

## Maintenance
- Logs are retained for 30 days
- Checkpoints are cleared on successful completion
- Cache files are managed automatically
- Quarantine files require manual review

## Last Updated
2025-06-08 16:31:43 (Session: 20250608_163143)
