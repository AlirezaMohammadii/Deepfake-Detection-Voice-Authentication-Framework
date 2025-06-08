# 🔬 Physics-Based Deepfake Detection: Enhanced Analysis Report

## 📊 Executive Summary

**Generated:** 2025-06-08T16:14:35.076675  
**Total Samples:** 40  
**Success Rate:** 100.0%

### Key Statistics
- **Genuine Audio:** 24 samples
- **Deepfake Audio:** 16 samples
- **Analysis Version:** 2.0_enhanced

## 🎯 Key Findings

- Marginal significance in physics_delta_fr_revised (p=0.0668) - potential discriminator

## 📈 Feature Discrimination Ranking

| Rank | Feature | Discrimination Score | P-value | Significance |
|------|---------|---------------------|---------|-------------|
| 1 | Rotational Frequency (Δf_r) | 0.628 | 0.0668 | ⚠️ Marginal |
| 2 | Vibrational Frequency (Δf_v) | 0.442 | 0.2026 | ❌ Not Significant |
| 3 | Total Frequency (Δf_total) | 0.012 | 0.9722 | ❌ Not Significant |
| 4 | Translational Frequency (Δf_t) | 0.010 | 0.9756 | ❌ Not Significant |

## 💡 Recommendations

- ⚠️ Rotational Frequency (Δf_r) shows promise but needs larger sample size
- 📊 Increase sample size to ~126 for 80% statistical power
- 🔬 Consider multivariate analysis combining top discriminating features
- 🤖 Evaluate machine learning models for feature combination
