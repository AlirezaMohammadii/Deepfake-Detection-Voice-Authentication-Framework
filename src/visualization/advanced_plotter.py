"""
Advanced Visualization Module for Physics Features Analysis
Generates publication-quality plots and interactive visualizations
Enhanced version with improved layout and professional presentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Optional, Tuple, Any
import warnings
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import time
from datetime import datetime

# Configure plotting parameters
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class AdvancedPhysicsPlotter:
    """
    Enhanced plotting system for physics-based deepfake detection analysis
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.static_dir = self.output_dir / "static"
        self.interactive_dir = self.output_dir / "interactive"
        self.reports_dir = self.output_dir / "reports"
        
        for subdir in [self.static_dir, self.interactive_dir, self.reports_dir]:
            subdir.mkdir(exist_ok=True)
        
        # Enhanced matplotlib configuration for publication quality
        plt.rcParams.update({
            'figure.figsize': [16, 10],
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'dejavuserif'
        })
        
        # Enhanced color schemes for professional appearance
        self.colors = {
            'genuine': '#2E8B57',      # Sea Green
            'deepfake_tts': '#DC143C',  # Crimson
            'deepfake_vc': '#FF8C00',   # Dark Orange
            'deepfake_replay': '#4B0082', # Indigo
            'primary': '#1f77b4',       # Blue
            'secondary': '#ff7f0e',     # Orange
            'accent': '#2ca02c',        # Green
            'background': '#f8f9fa',    # Light gray
            'text': '#2c3e50'           # Dark blue-gray
        }
        
        # Physics feature metadata with enhanced descriptions
        self.physics_features = {
            'physics_delta_ft_revised': {
                'label': 'Translational Frequency (Δf_t)',
                'unit': 'Hz',
                'description': 'Overall drift in embedding space',
                'interpretation': 'Lower values indicate stable voice characteristics'
            },
            'physics_delta_fr_revised': {
                'label': 'Rotational Frequency (Δf_r)', 
                'unit': 'Hz',
                'description': 'Principal component rotation rate',
                'interpretation': 'Higher values indicate algorithmic artifacts (strongest discriminator)'
            },
            'physics_delta_fv_revised': {
                'label': 'Vibrational Frequency (Δf_v)',
                'unit': 'Hz', 
                'description': 'High-frequency oscillations',
                'interpretation': 'Reflects synthesis algorithm stability'
            },
            'physics_delta_f_total_revised': {
                'label': 'Total Frequency (Δf_total)',
                'unit': 'Hz',
                'description': 'Combined dynamic signature',
                'interpretation': 'Composite measure of all dynamics'
            }
        }
    
    def load_and_prepare_data(self, csv_path: str) -> pd.DataFrame:
        """Load and prepare data for visualization with enhanced validation"""
        try:
            df = pd.read_csv(csv_path)
            
            # Ensure proper data types
            physics_cols = [col for col in df.columns if col.startswith('physics_')]
            
            # Function to convert tensor values and other formats to numeric
            def convert_tensor_to_numeric(val):
                """Convert tensor values and other formats to numeric"""
                if val is None or pd.isna(val):
                    return float('nan')
                
                # Handle tensor objects
                if hasattr(val, 'item'):  # PyTorch tensor
                    return float(val.item())
                elif hasattr(val, 'numpy'):  # NumPy array or other array-like
                    return float(val.numpy() if hasattr(val, 'numpy') else val)
                
                # Handle string representations of tensors
                if isinstance(val, str):
                    # Extract numeric value from tensor string
                    if 'tensor(' in val:
                        import re
                        matches = re.findall(r'tensor\(([0-9.e-]+)\)', val)
                        if matches:
                            return float(matches[0])
                    # Try direct conversion
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return float('nan')
                
                # Handle direct numeric values
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return float('nan')
            
            # Apply tensor conversion to physics columns
            for col in physics_cols:
                df[col] = df[col].apply(convert_tensor_to_numeric)
            
            # Also handle other potentially problematic columns
            numeric_cols = ['audio_duration_s', 'processing_time', 'hubert_seq_len_frames', 
                           'hubert_embedding_dim'] + [col for col in df.columns if col.startswith('bayesian_')]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter only successful results
            df_success = df[df['status'] == 'success'].copy()
            
            print(f"✓ Loaded {len(df)} total samples, {len(df_success)} successful")
            print(f"✓ File types: {df_success['file_type'].value_counts().to_dict()}")
            
            # Check if we have valid physics data
            physics_data_summary = {}
            for col in physics_cols:
                valid_count = df_success[col].notna().sum()
                physics_data_summary[col] = valid_count
            
            print(f"✓ Physics features validity: {physics_data_summary}")
            
            return df_success
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def create_enhanced_dashboard(self, df: pd.DataFrame) -> str:
        """Create enhanced professional dashboard with improved layout"""
        print("🎨 Generating enhanced comprehensive dashboard...")
        
        # Create enhanced subplots with better spacing
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=[
                '📊 Physics Features Distribution by Type',
                '🔗 Feature Correlation Matrix', 
                '🎻 Statistical Distribution Analysis',
                '📈 Translational Dynamics Evolution',
                '🔄 Rotational Dynamics Evolution (Key Discriminator)',
                '🌊 Vibrational Dynamics Evolution',
                '📊 Statistical Significance Analysis',
                '🎯 2D Feature Space (PCA Projection)',
                '⚡ System Performance Metrics',
                '📋 Dataset Overview',
                '🔍 Discrimination Analysis',
                '📊 Processing Statistics'
            ],
            specs=[
                [{"type": "box"}, {"type": "heatmap"}, {"type": "violin"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "histogram"}],
                [{"type": "table"}, {"type": "bar"}, {"type": "table"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            column_widths=[0.33, 0.33, 0.34],
            row_heights=[0.22, 0.22, 0.22, 0.34]
        )
        
        physics_cols = [col for col in df.columns if col.startswith('physics_delta_')]
        
        # 1. Enhanced Box plots for feature distributions
        colors_list = [self.colors['genuine'], self.colors['deepfake_tts']]
        for file_type in df['file_type'].unique():
            for i, col in enumerate(physics_cols):
                data = df[df['file_type'] == file_type][col]
                fig.add_trace(
                    go.Box(
                        y=data,
                        name=f"{file_type.replace('_', ' ').title()}",
                        boxpoints='outliers',
                        marker_color=self.colors.get(file_type, '#1f77b4'),
                        showlegend=(i == 0),
                        hovertemplate=f"<b>{file_type}</b><br>" +
                                    f"Value: %{{y:.4f}} Hz<br>" +
                                    f"<extra></extra>"
                    ),
                    row=1, col=1
                )
        
        # 2. Enhanced Correlation heatmap with better annotations
        corr_matrix = df[physics_cols].corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=[self.physics_features[col]['label'].split('(')[1].replace(')', '') for col in corr_matrix.columns],
                y=[self.physics_features[col]['label'].split('(')[1].replace(')', '') for col in corr_matrix.columns],
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="<b>Correlation</b><br>" +
                            "Features: %{x} vs %{y}<br>" +
                            "Correlation: %{z:.3f}<br>" +
                            "<extra></extra>",
                showscale=True,
                colorbar=dict(title="Correlation", x=0.65)
            ),
            row=1, col=2
        )
        
        # 3. Enhanced Violin plots with statistical annotations
        for file_type in df['file_type'].unique():
            for i, col in enumerate(physics_cols):
                data = df[df['file_type'] == file_type][col]
                fig.add_trace(
                    go.Violin(
                        y=data,
                        name=f"{file_type.replace('_', ' ').title()}",
                        box_visible=True,
                        line_color=self.colors.get(file_type, '#1f77b4'),
                        fillcolor=self.colors.get(file_type, '#1f77b4'),
                        opacity=0.7,
                        showlegend=False,
                        hovertemplate=f"<b>{file_type}</b><br>" +
                                    f"Value: %{{y:.4f}} Hz<br>" +
                                    f"Distribution shape visible<br>" +
                                    f"<extra></extra>"
                    ),
                    row=1, col=3
                )
        
        # 4-6. Enhanced Feature evolution scatter plots with trend lines
        evolution_titles = ['Translational', 'Rotational (Key)', 'Vibrational']
        for idx, col in enumerate(physics_cols[:3]):
            row_idx = 2
            col_idx = idx + 1
            
            for file_type in df['file_type'].unique():
                subset = df[df['file_type'] == file_type]
                
                # Add scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=subset.index,
                        y=subset[col],
                        mode='markers',
                        name=f"{file_type.replace('_', ' ').title()}",
                        marker=dict(
                            color=self.colors.get(file_type, '#1f77b4'),
                            size=8,
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        showlegend=False,
                        hovertemplate=f"<b>{file_type}</b><br>" +
                                    f"Sample: %{{x}}<br>" +
                                    f"{self.physics_features[col]['label']}: %{{y:.4f}} Hz<br>" +
                                    f"{self.physics_features[col]['interpretation']}<br>" +
                                    f"<extra></extra>"
                    ),
                    row=row_idx, col=col_idx
                )
                
                # Add trend line
                if len(subset) > 1:
                    z = np.polyfit(subset.index, subset[col], 1)
                    p = np.poly1d(z)
                    fig.add_trace(
                        go.Scatter(
                            x=subset.index,
                            y=p(subset.index),
                            mode='lines',
                            name=f"{file_type} trend",
                            line=dict(color=self.colors.get(file_type, '#1f77b4'), width=2, dash='dash'),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=row_idx, col=col_idx
                    )
        
        # 7. Enhanced Statistical significance with interpretation
        p_values = []
        feature_names = []
        effect_sizes = []
        significance_levels = []
        
        for col in physics_cols:
            genuine_data = df[df['file_type'] == 'genuine'][col].dropna()
            deepfake_data = df[df['file_type'] == 'deepfake_tts'][col].dropna()
            
            if len(genuine_data) > 1 and len(deepfake_data) > 1:  # Need at least 2 samples for t-test
                try:
                    # Check for valid data
                    if (genuine_data.var() > 0 or deepfake_data.var() > 0):  # At least one group has variance
                        _, p_val = stats.ttest_ind(genuine_data, deepfake_data)
                        
                        # Calculate effect size (Cohen's d)
                        pooled_std = np.sqrt(
                            ((len(genuine_data) - 1) * genuine_data.var() + 
                             (len(deepfake_data) - 1) * deepfake_data.var()) / 
                            (len(genuine_data) + len(deepfake_data) - 2)
                        )
                        
                        if pooled_std > 0:
                            effect_size = abs(genuine_data.mean() - deepfake_data.mean()) / pooled_std
                        else:
                            effect_size = 0
                        
                        # Check for valid p-value
                        if not np.isnan(p_val) and p_val > 0:
                            p_values.append(-np.log10(p_val))
                            feature_names.append(col.split('_')[-2])
                            effect_sizes.append(effect_size)
                            
                            # Determine significance level
                            if p_val < 0.001:
                                significance_levels.append('***')
                            elif p_val < 0.01:
                                significance_levels.append('**')
                            elif p_val < 0.05:
                                significance_levels.append('*')
                            elif p_val < 0.1:
                                significance_levels.append('†')
                            else:
                                significance_levels.append('ns')
                        else:
                            print(f"   ⚠️  Invalid p-value for {col}: {p_val}")
                    else:
                        print(f"   ⚠️  No variance in data for {col}")
                except Exception as e:
                    print(f"   ⚠️  Error in statistical test for {col}: {e}")
            else:
                print(f"   ⚠️  Insufficient samples for {col}: genuine={len(genuine_data)}, deepfake={len(deepfake_data)}")
        
        # Add significance bar chart only if we have valid data
        if p_values and feature_names:
            colors_significance = ['#e74c3c' if p > 1.3 else '#f39c12' if p > 1 else '#95a5a6' for p in p_values]
            fig.add_trace(
                go.Bar(
                    x=feature_names,
                    y=p_values,
                    marker_color=colors_significance,
                    name='Statistical Significance',
                    text=[f'p={10**(-p):.3f}<br>{sig}' for p, sig in zip(p_values, significance_levels)],
                    textposition='outside',
                    hovertemplate="<b>%{x}</b><br>" +
                                "-log10(p-value): %{y:.2f}<br>" +
                                "Significance: %{text}<br>" +
                                "<extra></extra>",
                    showlegend=False
                ),
                row=3, col=1
            )
        else:
            # Add placeholder if no statistical analysis available
            fig.add_annotation(
                text="Statistical analysis<br>unavailable due to<br>insufficient valid data",
                xref="x7", yref="y7",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12, color="orange"),
                row=3, col=1
            )
        
        # 8. Enhanced 2D PCA visualization
        if len(physics_cols) >= 2:
            # Prepare data for PCA by handling NaN values
            pca_data = df[physics_cols].copy()
            
            # Check for NaN values and handle them
            if pca_data.isnull().any().any():
                print("⚠️  Found NaN values in physics features, handling them for PCA...")
                
                # Option 1: Drop rows with any NaN values
                pca_data_clean = pca_data.dropna()
                df_pca = df.loc[pca_data_clean.index].copy()
                
                # If too many rows would be dropped, use imputation instead
                if len(pca_data_clean) < len(pca_data) * 0.5:  # Less than 50% of data would remain
                    print("   Using mean imputation for missing values...")
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='mean')
                    pca_data_clean = pd.DataFrame(
                        imputer.fit_transform(pca_data),
                        index=pca_data.index,
                        columns=pca_data.columns
                    )
                    df_pca = df.copy()
                else:
                    print(f"   Dropped {len(pca_data) - len(pca_data_clean)} rows with NaN values...")
            else:
                pca_data_clean = pca_data
                df_pca = df.copy()
            
            # Only proceed with PCA if we have sufficient data
            if len(pca_data_clean) >= 2 and len(pca_data_clean.columns) >= 2:
                try:
                    scaler = StandardScaler()
                    pca = PCA(n_components=2)
                    
                    features_scaled = scaler.fit_transform(pca_data_clean)
                    pca_result = pca.fit_transform(features_scaled)
                    
                    explained_variance = pca.explained_variance_ratio_
                    
                    for file_type in df_pca['file_type'].unique():
                        mask = df_pca['file_type'] == file_type
                        # Only use the mask indices that exist in our cleaned data
                        pca_mask = mask.loc[pca_data_clean.index] if len(pca_data_clean) < len(df) else mask
                        
                        if pca_mask.sum() > 0:  # Only plot if we have data for this file type
                            fig.add_trace(
                                go.Scatter(
                                    x=pca_result[pca_mask, 0],
                                    y=pca_result[pca_mask, 1],
                                    mode='markers',
                                    name=f"{file_type.replace('_', ' ').title()}",
                                    marker=dict(
                                        color=self.colors.get(file_type, '#1f77b4'),
                                        size=12,
                                        opacity=0.7,
                                        line=dict(width=2, color='white')
                                    ),
                                    showlegend=False,
                                    hovertemplate=f"<b>{file_type}</b><br>" +
                                                f"PC1: %{{x:.3f}}<br>" +
                                                f"PC2: %{{y:.3f}}<br>" +
                                                f"Explained variance: {explained_variance[0]:.2%} + {explained_variance[1]:.2%}<br>" +
                                                f"<extra></extra>"
                                ),
                                row=3, col=2
                            )
                except Exception as pca_error:
                    print(f"   ⚠️  PCA visualization failed: {pca_error}")
                    # Add a placeholder message
                    fig.add_annotation(
                        text="PCA visualization<br>unavailable due to<br>data quality issues",
                        xref="x4", yref="y4",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=14, color="red"),
                        row=3, col=2
                    )
            else:
                print(f"   ⚠️  Insufficient data for PCA: {len(pca_data_clean)} samples")
                # Add a placeholder message
                fig.add_annotation(
                    text="PCA visualization<br>requires more data<br>points",
                    xref="x4", yref="y4", 
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="orange"),
                    row=3, col=2
                )
        
        # 9. Enhanced Processing performance
        if 'processing_time' in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df['processing_time'],
                    nbinsx=15,
                    marker_color='rgba(26, 118, 255, 0.7)',
                    name='Processing Time Distribution',
                    hovertemplate="<b>Processing Time</b><br>" +
                                "Time: %{x:.2f}s<br>" +
                                "Count: %{y}<br>" +
                                "<extra></extra>",
                    showlegend=False
                ),
                row=3, col=3
            )
        else:
            # Show alternative metrics if processing time not available
            fig.add_trace(
                go.Bar(
                    x=['Total Samples', 'Success Rate'],
                    y=[len(df), (df['status'] == 'success').mean() * 100 if 'status' in df.columns else 100],
                    marker_color=['rgba(26, 118, 255, 0.7)', 'rgba(26, 200, 26, 0.7)'],
                    name='System Metrics',
                    text=[f'{len(df)}', f'{(df["status"] == "success").mean() * 100 if "status" in df.columns else 100:.1f}%'],
                    textposition='outside',
                    hovertemplate="<b>%{x}</b><br>" +
                                "Value: %{y}<br>" +
                                "<extra></extra>",
                    showlegend=False
                ),
                row=3, col=3
            )
        
        # 10. Dataset overview table
        overview_data = []
        for file_type in df['file_type'].unique():
            subset = df[df['file_type'] == file_type]
            overview_data.append([
                file_type.replace('_', ' ').title(),
                len(subset),
                f"{len(subset)/len(df)*100:.1f}%",
                f"{subset['audio_duration_s'].mean():.2f}s" if 'audio_duration_s' in df.columns else "N/A"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>File Type</b>', '<b>Count</b>', '<b>Percentage</b>', '<b>Avg Duration</b>'],
                    fill_color='lightblue',
                    align='center',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=list(zip(*overview_data)),
                    fill_color='white',
                    align='center',
                    font=dict(size=11)
                )
            ),
            row=4, col=1
        )
        
        # 11. Discrimination analysis
        discrimination_scores = []
        feature_labels = []
        
        for col in physics_cols:
            genuine_data = df[df['file_type'] == 'genuine'][col].dropna()
            deepfake_data = df[df['file_type'] == 'deepfake_tts'][col].dropna()
            
            if len(genuine_data) > 0 and len(deepfake_data) > 0:
                # Calculate discrimination potential only if we have valid data
                try:
                    genuine_mean = genuine_data.mean()
                    deepfake_mean = deepfake_data.mean()
                    genuine_var = genuine_data.var()
                    deepfake_var = deepfake_data.var()
                    
                    # Check for valid statistics
                    if not (np.isnan(genuine_mean) or np.isnan(deepfake_mean) or 
                           np.isnan(genuine_var) or np.isnan(deepfake_var)):
                        
                        mean_diff = abs(genuine_mean - deepfake_mean)
                        pooled_std = np.sqrt((genuine_var + deepfake_var) / 2)
                        
                        # Avoid division by zero
                        if pooled_std > 0:
                            discrimination = mean_diff / pooled_std
                            discrimination_scores.append(discrimination)
                            feature_labels.append(col.split('_')[-2])
                        else:
                            print(f"   ⚠️  Skipping {col}: zero variance")
                    else:
                        print(f"   ⚠️  Skipping {col}: invalid statistics")
                except Exception as e:
                    print(f"   ⚠️  Error calculating discrimination for {col}: {e}")
        
        # Only create the bar chart if we have discrimination scores
        if discrimination_scores and feature_labels:
            fig.add_trace(
                go.Bar(
                    x=feature_labels,
                    y=discrimination_scores,
                    marker_color=['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(discrimination_scores)],
                    name='Discrimination Potential',
                    text=[f'{score:.3f}' for score in discrimination_scores],
                    textposition='outside',
                    hovertemplate="<b>%{x}</b><br>" +
                                "Discrimination Score: %{y:.3f}<br>" +
                                "Higher = Better Discriminator<br>" +
                                "<extra></extra>",
                    showlegend=False
                ),
                row=4, col=2
            )
        else:
            # Add placeholder if no discrimination scores available
            fig.add_annotation(
                text="Discrimination analysis<br>unavailable due to<br>insufficient valid data",
                xref="x10", yref="y10",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12, color="orange"),
                row=4, col=2
            )
        
        # 12. Processing statistics
        if 'processing_time' in df.columns:
            processing_stats = [
                ['Mean Processing Time', f"{df['processing_time'].mean():.2f}s"],
                ['Total Files Processed', str(len(df))],
                ['Success Rate', '100%'],
                ['Total Processing Time', f"{df['processing_time'].sum():.1f}s"]
            ]
        else:
            processing_stats = [
                ['Total Files Processed', str(len(df))],
                ['Success Rate', f"{(df['status'] == 'success').mean() * 100 if 'status' in df.columns else 100:.1f}%"],
                ['Processing Time', 'Not Available'],
                ['System Status', 'Analysis Complete']
            ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color='lightgreen',
                    align='center',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=list(zip(*processing_stats)),
                    fill_color='white',
                    align='center',
                    font=dict(size=11)
                )
            ),
            row=4, col=3
        )
        
        # Enhanced layout configuration
        fig.update_layout(
            title={
                'text': '<b>🔬 Physics-Based Deepfake Detection: Enhanced Comprehensive Analysis Dashboard</b>',
                'x': 0.5,
                'font': {'size': 24, 'color': self.colors['text']}
            },
            height=1600,  # Increased height for better spacing
            showlegend=True,
            template='plotly_white',
            font=dict(family="Times New Roman, serif", size=11, color=self.colors['text']),
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            )
        )
        
        # Update subplot titles with better formatting
        for i in range(12):
            fig.layout.annotations[i].update(font=dict(size=13, color=self.colors['text']))
        
        # Add significance threshold line to statistical analysis (row 3, col 1)
        if feature_names:  # Only add if we have feature names
            fig.add_shape(
                type="line",
                x0=-0.5, x1=len(feature_names)-0.5,
                y0=1.3, y1=1.3,
                line=dict(color="red", width=2, dash="dash"),
                row=3, col=1
            )
            
            # Add annotation for significance threshold
            fig.add_annotation(
                x=0.5, y=1.35,
                text="p = 0.05 threshold",
                showarrow=False,
                font=dict(size=10, color="red"),
                xref="x7", yref="y7"  # Reference to subplot (3,1)
            )
        
        # Add PCA variance explanation only if PCA was successful
        pca_successful = False
        try:
            # Check if PCA was performed and explained_variance is available
            if len(physics_cols) >= 2 and 'explained_variance' in locals():
                fig.add_annotation(
                    x=0.5, y=0.35,
                    text=f"PC1: {explained_variance[0]:.1%} variance<br>PC2: {explained_variance[1]:.1%} variance",
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="gray",
                    borderwidth=1,
                    xref="paper", yref="paper"
                )
                pca_successful = True
        except:
            pass  # PCA explanation not available
        
        if not pca_successful and len(physics_cols) >= 2:
            # Add a note that PCA analysis was not available
            fig.add_annotation(
                x=0.5, y=0.35,
                text="PCA analysis: Data quality issues prevented variance calculation",
                showarrow=False,
                font=dict(size=10, color="orange"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="orange",
                borderwidth=1,
                xref="paper", yref="paper"
            )
        
        # Save enhanced dashboard
        dashboard_path = self.interactive_dir / "comprehensive_dashboard.html"
        
        # Custom HTML template for better presentation
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'physics_analysis_dashboard',
                'height': 1600,
                'width': 1200,
                'scale': 2
            }
        }
        
        fig.write_html(
            str(dashboard_path),
            config=config,
            include_plotlyjs=True,
            div_id="dashboard",
            full_html=True
        )
        
        print(f"✓ Enhanced comprehensive dashboard saved to: {dashboard_path}")
        return str(dashboard_path)
    
    def create_enhanced_physics_analysis_plots(self, df: pd.DataFrame):
        """Create enhanced detailed physics analysis plots with professional formatting"""
        print("📈 Generating enhanced physics analysis plots...")
        
        physics_cols = [col for col in df.columns if col.startswith('physics_delta_')]
        
        # 1. Enhanced Feature Distribution Analysis with Statistical Tests
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('🔬 Physics Features: Enhanced Distribution Analysis', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        for idx, col in enumerate(physics_cols):
            row, col_idx = divmod(idx, 2)
            ax = axes[row, col_idx]
            
            # Enhanced histograms with statistical annotations
            genuine_data = df[df['file_type'] == 'genuine'][col]
            deepfake_data = df[df['file_type'] == 'deepfake_tts'][col]
            
            # Plot distributions
            ax.hist(genuine_data, alpha=0.7, label='Genuine', bins=12, 
                   color=self.colors['genuine'], density=True, edgecolor='white')
            ax.hist(deepfake_data, alpha=0.7, label='Deepfake TTS', bins=12, 
                   color=self.colors['deepfake_tts'], density=True, edgecolor='white')
            
            # Add statistical test results
            if len(genuine_data) > 0 and len(deepfake_data) > 0:
                statistic, p_value = stats.ttest_ind(genuine_data, deepfake_data)
                effect_size = abs(genuine_data.mean() - deepfake_data.mean()) / np.sqrt(
                    ((len(genuine_data) - 1) * genuine_data.var() + 
                     (len(deepfake_data) - 1) * deepfake_data.var()) / 
                    (len(genuine_data) + len(deepfake_data) - 2)
                )
                
                # Add mean lines
                ax.axvline(genuine_data.mean(), color=self.colors['genuine'], 
                          linestyle='--', linewidth=2, label=f'Genuine μ = {genuine_data.mean():.4f}')
                ax.axvline(deepfake_data.mean(), color=self.colors['deepfake_tts'], 
                          linestyle='--', linewidth=2, label=f'Deepfake μ = {deepfake_data.mean():.4f}')
                
                # Statistical annotation
                significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '†' if p_value < 0.1 else 'ns'
                stats_text = f'p = {p_value:.4f} {significance}\nCohen\'s d = {effect_size:.3f}\nΔμ = {abs(genuine_data.mean() - deepfake_data.mean()):.4f}'
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                       verticalalignment='top', fontsize=9, fontweight='bold')
            
            ax.set_title(f'{self.physics_features[col]["label"]}\n{self.physics_features[col]["interpretation"]}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{self.physics_features[col]["unit"]}', fontsize=12)
            ax.set_ylabel('Probability Density', fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.static_dir / "enhanced_physics_distributions.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 2. Enhanced Correlation and Advanced Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('🔗 Physics Features: Advanced Correlation & Interaction Analysis', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Enhanced correlation matrix with significance testing
        corr_matrix = df[physics_cols].corr()
        
        # Calculate p-values for correlations
        n = len(df)
        p_values_corr = np.zeros_like(corr_matrix)
        for i in range(len(physics_cols)):
            for j in range(len(physics_cols)):
                if i != j:
                    r = corr_matrix.iloc[i, j]
                    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                    p_values_corr[i, j] = p_val
        
        # Create enhanced heatmap
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax1, cbar_kws={'label': 'Correlation Coefficient'},
                   fmt='.3f', linewidths=0.5)
        ax1.set_title('Physics Features Correlation Matrix\n(Lower Triangle)', fontsize=14)
        
        # Pairwise scatter plot with regression lines
        col1, col2 = physics_cols[1], physics_cols[0]  # fr vs ft (most discriminative)
        
        for file_type in df['file_type'].unique():
            subset = df[df['file_type'] == file_type]
            ax2.scatter(subset[col1], subset[col2], 
                       label=f"{file_type.replace('_', ' ').title()}", 
                       alpha=0.8, s=80, edgecolors='white', linewidth=1,
                       color=self.colors.get(file_type, '#1f77b4'))
            
            # Add regression line
            if len(subset) > 1:
                z = np.polyfit(subset[col1], subset[col2], 1)
                p = np.poly1d(z)
                x_line = np.linspace(subset[col1].min(), subset[col1].max(), 100)
                ax2.plot(x_line, p(x_line), 
                        color=self.colors.get(file_type, '#1f77b4'), 
                        linestyle='--', alpha=0.8, linewidth=2)
        
        ax2.set_xlabel(self.physics_features[col1]['label'], fontsize=12)
        ax2.set_ylabel(self.physics_features[col2]['label'], fontsize=12)
        ax2.set_title('Most Discriminative Feature Pair\n(with Regression Lines)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Feature importance analysis
        feature_importance = []
        feature_names_short = []
        
        for col in physics_cols:
            genuine_data = df[df['file_type'] == 'genuine'][col]
            deepfake_data = df[df['file_type'] == 'deepfake_tts'][col]
            
            if len(genuine_data) > 0 and len(deepfake_data) > 0:
                # Calculate multiple discrimination metrics
                mean_diff = abs(genuine_data.mean() - deepfake_data.mean())
                pooled_std = np.sqrt((genuine_data.var() + deepfake_data.var()) / 2)
                discrimination = mean_diff / pooled_std
                
                # T-test
                _, p_val = stats.ttest_ind(genuine_data, deepfake_data)
                
                # Combined importance score
                importance = discrimination * (-np.log10(p_val + 1e-10))
                feature_importance.append(importance)
                feature_names_short.append(col.split('_')[-2])
        
        # Plot feature importance
        bars = ax3.bar(feature_names_short, feature_importance, 
                      color=['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'],
                      alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for bar, importance in zip(bars, feature_importance):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{importance:.2f}',
                    ha='center', va='bottom', fontweight='bold')
        
        ax3.set_title('Feature Discrimination Importance\n(Effect Size × -log₁₀(p-value))', fontsize=14)
        ax3.set_ylabel('Importance Score', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Distribution overlap analysis
        overlap_scores = []
        for col in physics_cols:
            genuine_data = df[df['file_type'] == 'genuine'][col]
            deepfake_data = df[df['file_type'] == 'deepfake_tts'][col]
            
            if len(genuine_data) > 0 and len(deepfake_data) > 0:
                # Calculate overlap using histogram intersection
                min_val = min(genuine_data.min(), deepfake_data.min())
                max_val = max(genuine_data.max(), deepfake_data.max())
                bins = np.linspace(min_val, max_val, 20)
                
                hist1, _ = np.histogram(genuine_data, bins=bins, density=True)
                hist2, _ = np.histogram(deepfake_data, bins=bins, density=True)
                
                # Normalize
                hist1 = hist1 / np.sum(hist1)
                hist2 = hist2 / np.sum(hist2)
                
                # Calculate intersection (overlap)
                overlap = np.sum(np.minimum(hist1, hist2))
                overlap_scores.append(1 - overlap)  # Convert to separation score
        
        bars = ax4.bar(feature_names_short, overlap_scores,
                      color=['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'],
                      alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add value labels
        for bar, score in zip(bars, overlap_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        
        ax4.set_title('Distribution Separation Score\n(1 - Histogram Overlap)', fontsize=14)
        ax4.set_ylabel('Separation Score (0-1)', fontsize=12)
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.static_dir / "enhanced_physics_correlation_analysis.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Enhanced physics analysis plots saved to: {self.static_dir}")
    
    def create_comprehensive_statistical_analysis(self, df: pd.DataFrame):
        """Create comprehensive statistical analysis with enhanced visualizations"""
        print("📊 Generating comprehensive statistical analysis...")
        
        physics_cols = [col for col in df.columns if col.startswith('physics_delta_')]
        
        # Enhanced statistical comparison
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('📊 Comprehensive Statistical Analysis: Genuine vs Deepfake TTS', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        stats_results = []
        
        # Individual feature analysis
        for idx, col in enumerate(physics_cols):
            row, col_idx = divmod(idx, 3)
            if row >= 2:  # We only have 2 rows
                break
            ax = axes[row, col_idx]
            
            # Prepare data
            genuine_data = df[df['file_type'] == 'genuine'][col]
            deepfake_data = df[df['file_type'] == 'deepfake_tts'][col]
            
            # Enhanced box plot with individual points
            bp = ax.boxplot([genuine_data, deepfake_data], 
                           labels=['Genuine', 'Deepfake TTS'], 
                           patch_artist=True, notch=True, widths=0.6)
            
            # Color the boxes
            bp['boxes'][0].set_facecolor(self.colors['genuine'])
            bp['boxes'][1].set_facecolor(self.colors['deepfake_tts'])
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_alpha(0.7)
            
            # Add individual data points with jitter
            np.random.seed(42)
            x1 = np.random.normal(1, 0.04, len(genuine_data))
            x2 = np.random.normal(2, 0.04, len(deepfake_data))
            
            ax.scatter(x1, genuine_data, alpha=0.6, color=self.colors['genuine'], 
                      s=30, edgecolors='white', linewidth=0.5)
            ax.scatter(x2, deepfake_data, alpha=0.6, color=self.colors['deepfake_tts'], 
                      s=30, edgecolors='white', linewidth=0.5)
            
            # Statistical tests
            if len(genuine_data) > 0 and len(deepfake_data) > 0:
                # T-test
                statistic, p_value = stats.ttest_ind(genuine_data, deepfake_data)
                
                # Effect size (Cohen's d)
                effect_size = (genuine_data.mean() - deepfake_data.mean()) / np.sqrt(
                    ((len(genuine_data) - 1) * genuine_data.var() + 
                     (len(deepfake_data) - 1) * deepfake_data.var()) / 
                    (len(genuine_data) + len(deepfake_data) - 2)
                )
                
                # Mann-Whitney U test (non-parametric)
                u_statistic, u_p_value = stats.mannwhitneyu(genuine_data, deepfake_data, 
                                                           alternative='two-sided')
                
                # Kolmogorov-Smirnov test
                ks_statistic, ks_p_value = stats.ks_2samp(genuine_data, deepfake_data)
                
                stats_results.append({
                    'feature': col,
                    'genuine_mean': genuine_data.mean(),
                    'genuine_std': genuine_data.std(),
                    'deepfake_mean': deepfake_data.mean(),
                    'deepfake_std': deepfake_data.std(),
                    'mean_difference': deepfake_data.mean() - genuine_data.mean(),
                    'percent_difference': ((deepfake_data.mean() - genuine_data.mean()) / genuine_data.mean()) * 100,
                    't_statistic': statistic,
                    't_p_value': p_value,
                    'effect_size': effect_size,
                    'u_p_value': u_p_value,
                    'ks_p_value': ks_p_value,
                    'significant_t': p_value < 0.05,
                    'significant_u': u_p_value < 0.05,
                    'significant_ks': ks_p_value < 0.05
                })
                
                # Enhanced statistical annotation
                significance_t = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '†' if p_value < 0.1 else 'ns'
                effect_interpretation = 'Large' if abs(effect_size) > 0.8 else 'Medium' if abs(effect_size) > 0.5 else 'Small' if abs(effect_size) > 0.2 else 'Negligible'
                
                stats_text = f't-test: p = {p_value:.4f} {significance_t}\nCohen\'s d = {effect_size:.3f} ({effect_interpretation})\nMann-Whitney: p = {u_p_value:.4f}\nΔ% = {((deepfake_data.mean() - genuine_data.mean()) / genuine_data.mean()) * 100:+.1f}%'
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                       verticalalignment='top', fontsize=9, fontweight='bold')
            
            ax.set_title(f'{self.physics_features[col]["label"]}\n{self.physics_features[col]["interpretation"]}', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{self.physics_features[col]["unit"]}', fontsize=11)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots if any
        for idx in range(len(physics_cols), 6):
            row, col_idx = divmod(idx, 3)
            if row < 2:
                fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout()
        plt.savefig(self.static_dir / "comprehensive_statistical_comparison.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save detailed statistical results
        stats_df = pd.DataFrame(stats_results)
        stats_df.to_csv(self.reports_dir / "detailed_statistical_analysis.csv", index=False)
        
        print(f"✓ Comprehensive statistical analysis saved to: {self.static_dir}")
        return stats_results
    
    def create_enhanced_performance_analysis(self, df: pd.DataFrame):
        """Create enhanced system performance analysis with detailed metrics"""
        print("⚡ Generating enhanced performance analysis...")
        
        # Check for required columns
        has_processing_time = 'processing_time' in df.columns
        has_file_size = 'file_size_mb' in df.columns
        has_file_type = 'file_type' in df.columns
        
        # Enhanced performance visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('⚡ Enhanced System Performance Analysis', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Processing time distribution with detailed statistics
        if has_processing_time:
            processing_times = df['processing_time']
            
            # Enhanced histogram with statistics
            n, bins, patches = ax1.hist(processing_times, bins=20, alpha=0.7, 
                                       color='skyblue', edgecolor='white', linewidth=1)
            
            # Color code bins
            for i, (patch, bin_center) in enumerate(zip(patches, bins[:-1])):
                if bin_center < processing_times.median():
                    patch.set_facecolor('#2ecc71')  # Green for fast
                elif bin_center < processing_times.quantile(0.75):
                    patch.set_facecolor('#f39c12')  # Orange for medium
                else:
                    patch.set_facecolor('#e74c3c')  # Red for slow
            
            # Add statistical lines
            ax1.axvline(processing_times.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {processing_times.mean():.2f}s')
            ax1.axvline(processing_times.median(), color='blue', linestyle='--', 
                       linewidth=2, label=f'Median: {processing_times.median():.2f}s')
            
            # Add percentile lines
            p95 = processing_times.quantile(0.95)
            ax1.axvline(p95, color='orange', linestyle=':', 
                       linewidth=2, label=f'95th %ile: {p95:.2f}s')
            
            ax1.set_title('Processing Time Distribution\n(Color-coded by Performance)', fontsize=14)
            ax1.set_xlabel('Processing Time (seconds)', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add performance statistics text
            stats_text = f'Min: {processing_times.min():.2f}s\nMax: {processing_times.max():.2f}s\nStd: {processing_times.std():.2f}s\nCV: {processing_times.std()/processing_times.mean():.2f}'
            ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                    verticalalignment='top', horizontalalignment='right', fontsize=10)
        else:
            # Fallback: Show sample count distribution
            if has_file_type:
                file_counts = df['file_type'].value_counts()
                bars = ax1.bar(range(len(file_counts)), file_counts.values, 
                              color=[self.colors.get(ft, '#1f77b4') for ft in file_counts.index],
                              alpha=0.8, edgecolor='white', linewidth=2)
                
                # Add value labels
                for bar, count in zip(bars, file_counts.values):
                    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                            f'{count}',
                            ha='center', va='bottom', fontweight='bold')
                
                ax1.set_title('Sample Count by File Type\n(Processing time data not available)', fontsize=14)
                ax1.set_xlabel('File Type', fontsize=12)
                ax1.set_ylabel('Number of Samples', fontsize=12)
                ax1.set_xticks(range(len(file_counts)))
                ax1.set_xticklabels([ft.replace('_', ' ').title() for ft in file_counts.index], rotation=45)
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'Processing time data\nnot available', 
                        transform=ax1.transAxes, ha='center', va='center',
                        fontsize=16, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
                ax1.set_title('Processing Time Distribution', fontsize=14)
        
        # 2. Processing efficiency by file type
        if has_file_type and has_processing_time:
            file_type_performance = df.groupby('file_type')['processing_time'].agg(['mean', 'std', 'count'])
            
            # Bar plot with error bars
            x_pos = range(len(file_type_performance))
            bars = ax2.bar(x_pos, file_type_performance['mean'], 
                          yerr=file_type_performance['std'],
                          capsize=5, alpha=0.8, edgecolor='white', linewidth=2,
                          color=[self.colors.get(ft, '#1f77b4') for ft in file_type_performance.index])
            
            # Add value labels on bars
            for i, (bar, mean_val, count) in enumerate(zip(bars, file_type_performance['mean'], file_type_performance['count'])):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{mean_val:.2f}s\n(n={count})',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax2.set_title('Processing Time by File Type\n(with Standard Deviation)', fontsize=14)
            ax2.set_xlabel('File Type', fontsize=12)
            ax2.set_ylabel('Average Processing Time (seconds)', fontsize=12)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([ft.replace('_', ' ').title() for ft in file_type_performance.index], rotation=45)
            ax2.grid(True, alpha=0.3)
        elif has_file_type:
            # Show file type distribution instead
            file_counts = df['file_type'].value_counts()
            colors = [self.colors.get(ft, '#1f77b4') for ft in file_counts.index]
            
            wedges, texts, autotexts = ax2.pie(file_counts.values, labels=[ft.replace('_', ' ').title() for ft in file_counts.index], 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            
            ax2.set_title('File Type Distribution\n(Processing time not available)', fontsize=14)
        else:
            ax2.text(0.5, 0.5, 'File type data\nnot available', 
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=16, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            ax2.set_title('Processing Efficiency by File Type', fontsize=14)
        
        # 3. File size vs processing time relationship
        if has_file_size and has_processing_time and has_file_type:
            for file_type in df['file_type'].unique():
                subset = df[df['file_type'] == file_type]
                ax3.scatter(subset['file_size_mb'], subset['processing_time'],
                          label=f"{file_type.replace('_', ' ').title()}", 
                          alpha=0.7, s=60, edgecolors='white', linewidth=1,
                          color=self.colors.get(file_type, '#1f77b4'))
            
            # Add trend line for all data
            if len(df) > 1:
                z = np.polyfit(df['file_size_mb'], df['processing_time'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(df['file_size_mb'].min(), df['file_size_mb'].max(), 100)
                ax3.plot(x_line, p(x_line), 'k--', alpha=0.8, linewidth=2, 
                        label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
                
                # Calculate correlation
                correlation = df['file_size_mb'].corr(df['processing_time'])
                ax3.text(0.05, 0.95, f'Correlation: r={correlation:.3f}',
                        transform=ax3.transAxes,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                        verticalalignment='top', fontsize=12, fontweight='bold')
            
            ax3.set_title('File Size vs Processing Time\n(with Correlation Analysis)', fontsize=14)
            ax3.set_xlabel('File Size (MB)', fontsize=12)
            ax3.set_ylabel('Processing Time (seconds)', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        elif has_file_size and has_file_type:
            # Show file size distribution by type
            for file_type in df['file_type'].unique():
                subset = df[df['file_type'] == file_type]
                ax3.hist(subset['file_size_mb'], alpha=0.7, 
                        label=f"{file_type.replace('_', ' ').title()}", 
                        bins=10, edgecolor='white',
                        color=self.colors.get(file_type, '#1f77b4'))
            
            ax3.set_title('File Size Distribution by Type\n(Processing time not available)', fontsize=14)
            ax3.set_xlabel('File Size (MB)', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'File size or processing\ntime data not available', 
                    transform=ax3.transAxes, ha='center', va='center',
                    fontsize=16, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            ax3.set_title('File Size vs Processing Time', fontsize=14)
        
        # 4. Success rate and system metrics
        success_rate = (df['status'] == 'success').mean() * 100 if 'status' in df.columns else 100.0
        total_files = len(df)
        
        # Prepare metrics data
        metrics_data = [
            ['Total Files Processed', f'{total_files:,}'],
            ['Success Rate', f'{success_rate:.1f}%'],
        ]
        
        # Add processing time metrics if available
        if has_processing_time:
            total_time = df['processing_time'].sum()
            throughput = total_files / total_time  # files per second
            
            metrics_data.extend([
                ['Total Processing Time', f'{total_time:.1f}s'],
                ['Average Time per File', f'{df["processing_time"].mean():.2f}s'],
                ['Throughput', f'{throughput:.3f} files/sec'],
                ['Efficiency Score', f'{(success_rate * throughput):.2f}']
            ])
        else:
            metrics_data.extend([
                ['Processing Time', 'Not Available'],
                ['Average Time per File', 'Not Available'],
                ['Throughput', 'Not Available'],
                ['Efficiency Score', f'{success_rate:.1f}']
            ])
        
        # Create table plot
        ax4.axis('off')
        table = ax4.table(cellText=metrics_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(metrics_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f9f9f9' if i % 2 == 0 else 'white')
                    if j == 1:  # Value column
                        cell.set_text_props(weight='bold')
        
        ax4.set_title('System Performance Metrics\n(Comprehensive Overview)', fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.savefig(self.static_dir / "enhanced_performance_analysis.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Enhanced performance analysis saved to: {self.static_dir}")
    
    def generate_enhanced_summary_report(self, df: pd.DataFrame, stats_results: List[Dict]) -> Dict[str, Any]:
        """Generate enhanced comprehensive summary report with detailed analysis"""
        print("📋 Generating enhanced summary report...")
        
        physics_cols = [col for col in df.columns if col.startswith('physics_delta_')]
        
        # Enhanced analysis report
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_samples': len(df),
                'genuine_samples': len(df[df['file_type'] == 'genuine']),
                'deepfake_samples': len(df[df['file_type'] == 'deepfake_tts']),
                'success_rate': (df['status'] == 'success').mean() * 100,
                'analysis_version': '2.0_enhanced'
            },
            'physics_features_summary': {},
            'statistical_analysis': [],
            'key_findings': [],
            'discrimination_ranking': [],
            'recommendations': []
        }
        
        # Enhanced feature analysis
        for col in physics_cols:
            genuine_data = df[df['file_type'] == 'genuine'][col]
            deepfake_data = df[df['file_type'] == 'deepfake_tts'][col]
            
            if len(genuine_data) > 0 and len(deepfake_data) > 0:
                # Calculate discrimination potential
                mean_diff = abs(genuine_data.mean() - deepfake_data.mean())
                pooled_std = np.sqrt((genuine_data.var() + deepfake_data.var()) / 2)
                discrimination = mean_diff / pooled_std
                
                # Statistical tests
                _, p_value = stats.ttest_ind(genuine_data, deepfake_data)
                
                report['physics_features_summary'][col] = {
                    'overall_stats': {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max()
                    },
                    'genuine_stats': {
                        'mean': genuine_data.mean(),
                        'std': genuine_data.std(),
                        'count': len(genuine_data)
                    },
                    'deepfake_stats': {
                        'mean': deepfake_data.mean(),
                        'std': deepfake_data.std(),
                        'count': len(deepfake_data)
                    },
                    'discrimination_potential': discrimination,
                    'statistical_significance': p_value,
                    'practical_significance': 'High' if discrimination > 0.5 else 'Medium' if discrimination > 0.2 else 'Low'
                }
        
        # Enhanced statistical analysis
        for stat in stats_results:
            analysis_entry = {
                'feature': stat['feature'],
                'p_value': stat['t_p_value'],
                'effect_size': stat['effect_size'],
                'significant': stat['significant_t'],
                'mann_whitney_p': stat['u_p_value'],
                'ks_test_p': stat['ks_p_value'],
                'percent_difference': stat['percent_difference'],
                'interpretation': self._interpret_effect_size(stat['effect_size'])
            }
            report['statistical_analysis'].append(analysis_entry)
        
        # Enhanced key findings with detailed analysis
        for stat in stats_results:
            if stat['significant_t']:
                interpretation = self._interpret_effect_size(stat['effect_size'])
                finding = f"Significant difference in {stat['feature']} (p={stat['t_p_value']:.4f}, d={stat['effect_size']:.3f}, {interpretation} effect)"
                report['key_findings'].append(finding)
            elif stat['t_p_value'] < 0.1:
                finding = f"Marginal significance in {stat['feature']} (p={stat['t_p_value']:.4f}) - potential discriminator"
                report['key_findings'].append(finding)
        
        # Create discrimination ranking
        discrimination_scores = []
        for col in physics_cols:
            if col in report['physics_features_summary']:
                score_data = report['physics_features_summary'][col]
                combined_score = score_data['discrimination_potential'] * (-np.log10(score_data['statistical_significance'] + 1e-10))
                
                discrimination_scores.append({
                    'feature': col,
                    'feature_name': self.physics_features[col]['label'],
                    'discrimination_score': score_data['discrimination_potential'],
                    'p_value': score_data['statistical_significance'],
                    'combined_score': combined_score,
                    'interpretation': self.physics_features[col]['interpretation']
                })
        
        # Sort by combined score
        discrimination_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        report['discrimination_ranking'] = discrimination_scores
        
        # Enhanced recommendations
        best_discriminator = discrimination_scores[0] if discrimination_scores else None
        if best_discriminator:
            if best_discriminator['p_value'] < 0.05:
                report['recommendations'].append(f"🎯 Use {best_discriminator['feature_name']} as primary discriminator (statistically significant)")
            elif best_discriminator['p_value'] < 0.1:
                report['recommendations'].append(f"⚠️ {best_discriminator['feature_name']} shows promise but needs larger sample size")
            else:
                report['recommendations'].append(f"🔍 {best_discriminator['feature_name']} requires further investigation")
        
        # Sample size recommendations
        current_power = self._calculate_statistical_power(df)
        if current_power < 0.8:
            recommended_sample_size = self._estimate_required_sample_size(df)
            report['recommendations'].append(f"📊 Increase sample size to ~{recommended_sample_size} for 80% statistical power")
        
        # Feature combination recommendations
        if len(physics_cols) > 1:
            report['recommendations'].append("🔬 Consider multivariate analysis combining top discriminating features")
            report['recommendations'].append("🤖 Evaluate machine learning models for feature combination")
        
        # Save enhanced report
        with open(self.reports_dir / "enhanced_analysis_summary.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create markdown report
        self._create_enhanced_markdown_report(report)
        
        print(f"✓ Enhanced summary report saved to: {self.reports_dir}")
        return report
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_effect = abs(effect_size)
        if abs_effect >= 0.8:
            return "Large"
        elif abs_effect >= 0.5:
            return "Medium"
        elif abs_effect >= 0.2:
            return "Small"
        else:
            return "Negligible"
    
    def _calculate_statistical_power(self, df: pd.DataFrame) -> float:
        """Calculate statistical power for the current sample"""
        # Simplified power calculation
        genuine_n = len(df[df['file_type'] == 'genuine'])
        deepfake_n = len(df[df['file_type'] == 'deepfake_tts'])
        
        # Assume medium effect size (0.5) for power calculation
        from scipy.stats import norm
        alpha = 0.05
        effect_size = 0.5
        
        n_harmonic = 2 / (1/genuine_n + 1/deepfake_n)
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = effect_size * np.sqrt(n_harmonic/4) - z_alpha
        power = norm.cdf(z_beta)
        
        return max(0, min(1, power))
    
    def _estimate_required_sample_size(self, df: pd.DataFrame) -> int:
        """Estimate required sample size for 80% power"""
        # Simplified sample size calculation
        alpha = 0.05
        power = 0.8
        effect_size = 0.5  # Assume medium effect size
        
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        total_n = int(np.ceil(n_per_group * 2))
        
        return total_n
    
    def _create_enhanced_markdown_report(self, report: Dict[str, Any]):
        """Create enhanced markdown report"""
        markdown_content = f"""# 🔬 Physics-Based Deepfake Detection: Enhanced Analysis Report

## 📊 Executive Summary

**Generated:** {report['metadata']['generated_at']}  
**Total Samples:** {report['metadata']['total_samples']}  
**Success Rate:** {report['metadata']['success_rate']:.1f}%

### Key Statistics
- **Genuine Audio:** {report['metadata']['genuine_samples']} samples
- **Deepfake Audio:** {report['metadata']['deepfake_samples']} samples
- **Analysis Version:** {report['metadata']['analysis_version']}

## 🎯 Key Findings

"""
        
        for finding in report['key_findings']:
            markdown_content += f"- {finding}\n"
        
        markdown_content += "\n## 📈 Feature Discrimination Ranking\n\n"
        markdown_content += "| Rank | Feature | Discrimination Score | P-value | Significance |\n"
        markdown_content += "|------|---------|---------------------|---------|-------------|\n"
        
        for i, feature in enumerate(report['discrimination_ranking'], 1):
            significance = "✅ Significant" if feature['p_value'] < 0.05 else "⚠️ Marginal" if feature['p_value'] < 0.1 else "❌ Not Significant"
            markdown_content += f"| {i} | {feature['feature_name']} | {feature['discrimination_score']:.3f} | {feature['p_value']:.4f} | {significance} |\n"
        
        markdown_content += "\n## 💡 Recommendations\n\n"
        for rec in report['recommendations']:
            markdown_content += f"- {rec}\n"
        
        # Save markdown report
        with open(self.reports_dir / "enhanced_analysis_report.md", 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def generate_all_visualizations(self, csv_path: str) -> Dict[str, str]:
        """Generate all enhanced visualizations and analysis"""
        print("🎨 Starting enhanced visualization suite...")
        
        # Load and validate data
        df = self.load_and_prepare_data(csv_path)
        if df.empty:
            print("❌ No valid data found for visualization")
            return {}
        
        # Generate all enhanced visualizations
        dashboard_path = self.create_enhanced_dashboard(df)
        self.create_enhanced_physics_analysis_plots(df)
        stats_results = self.create_comprehensive_statistical_analysis(df)
        self.create_enhanced_performance_analysis(df)
        report = self.generate_enhanced_summary_report(df, stats_results)
        
        # Return summary of generated files
        return {
            'dashboard_path': dashboard_path,
            'static_dir': str(self.static_dir),
            'interactive_dir': str(self.interactive_dir),
            'reports_dir': str(self.reports_dir),
            'summary_report': report
        }

if __name__ == "__main__":
    # Test the visualization system
    plotter = AdvancedPhysicsPlotter("visualizations")
    
    # Example usage
    csv_path = "results/physics_features_summary.csv"
    if Path(csv_path).exists():
        results = plotter.generate_all_visualizations(csv_path)
        print("\\nVisualization system test completed successfully!")
    else:
        print(f"CSV file not found: {csv_path}")
        print("Please run test_runner.py first to generate results.") 