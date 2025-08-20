"""
EEG Correlation Analysis and Visualization
Rice Datathon 2025 - Neurotech Track

This script generates correlation matrices and visualizations for EEG frequency bands.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = Path('../data/Train_and_Validate_EEG.csv')
OUTPUT_PATH = Path('../results/visualizations/')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_and_clean_data(filepath):
    """Load EEG data and perform initial cleaning."""
    print("Loading EEG data...")
    df = pd.read_csv(filepath)
    
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Filter out missing values in key columns
    df_clean = df.dropna(subset=['IQ', 'education'])
    
    print(f"Data shape after cleaning: {df_clean.shape}")
    return df_clean


def visualize_missing_values(df, save_path=None):
    """Create a bar plot of missing values by column."""
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False).head(20)
    
    if len(missing_counts) == 0:
        print("No missing values found!")
        return
    
    plt.figure(figsize=(12, 6))
    ax = missing_counts.plot(kind='bar', color='lightblue', edgecolor='black')
    
    plt.title('Top 20 Columns with Missing Values', fontsize=14, fontweight='bold')
    plt.xlabel('Column Names', fontsize=12)
    plt.ylabel('Number of Missing Values', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(missing_counts):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved missing values plot to {save_path}")
    
    plt.show()


def remove_coherence_features(df):
    """Remove coherence variables to reduce dimensionality."""
    non_coh_columns = [col for col in df.columns if 'COH' not in col]
    df_filtered = df[non_coh_columns]
    print(f"Removed {len(df.columns) - len(non_coh_columns)} coherence features")
    return df_filtered


def create_correlation_heatmap(df, columns_pattern, title, save_path=None, figsize=(12, 10)):
    """Create a correlation heatmap for specific columns."""
    # Filter columns based on pattern
    filtered_cols = [col for col in df.columns if columns_pattern in col]
    
    if len(filtered_cols) == 0:
        print(f"No columns found matching pattern: {columns_pattern}")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[filtered_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=figsize)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    
    # Create heatmap with custom colormap
    sns.heatmap(corr_matrix, 
                mask=mask,
                cmap='coolwarm',
                vmin=-1, vmax=1,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                fmt='.2f')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('')
    plt.ylabel('')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation heatmap to {save_path}")
    
    plt.show()
    
    return corr_matrix


def analyze_frequency_bands(df, save_plots=True):
    """Analyze correlations for different EEG frequency bands."""
    frequency_bands = [
        ('theta', 'Theta Band (4-8 Hz)'),
        ('alpha', 'Alpha Band (8-13 Hz)'),
        ('beta', 'Beta Band (13-30 Hz)'),
        ('highbeta', 'High Beta Band (20-30 Hz)'),
        ('delta', 'Delta Band (0.5-4 Hz)'),
        ('gamma', 'Gamma Band (30-100 Hz)')
    ]
    
    correlation_stats = {}
    
    for band_name, band_title in frequency_bands:
        print(f"\nAnalyzing {band_title}...")
        
        save_path = OUTPUT_PATH / f'correlation_{band_name}.png' if save_plots else None
        
        corr_matrix = create_correlation_heatmap(
            df, 
            band_name,
            f'Correlation Matrix: AB {band_title} Electrodes',
            save_path=save_path
        )
        
        if corr_matrix is not None:
            # Calculate correlation statistics
            corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            correlation_stats[band_name] = {
                'mean_correlation': np.mean(corr_values),
                'std_correlation': np.std(corr_values),
                'max_correlation': np.max(corr_values),
                'min_correlation': np.min(corr_values)
            }
    
    return correlation_stats


def analyze_demographic_correlations(df, save_path=None):
    """Analyze correlations between demographic variables."""
    # Encode categorical variables
    df_copy = df.copy()
    if 'sex' in df_copy.columns:
        df_copy['sex'] = df_copy['sex'].map({'M': 0, 'F': 1})
    
    # Select demographic columns
    demo_cols = ['age', 'sex', 'IQ', 'education']
    available_cols = [col for col in demo_cols if col in df_copy.columns]
    
    if len(available_cols) < 2:
        print("Not enough demographic columns for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = df_copy[available_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='coolwarm',
                vmin=-1, vmax=1,
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                fmt='.3f')
    
    plt.title('Demographic Variables Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved demographic correlation matrix to {save_path}")
    
    plt.show()
    
    return corr_matrix


def create_correlation_summary(correlation_stats):
    """Create a summary visualization of correlation statistics across frequency bands."""
    if not correlation_stats:
        print("No correlation statistics to summarize")
        return
    
    # Prepare data for plotting
    bands = list(correlation_stats.keys())
    means = [correlation_stats[band]['mean_correlation'] for band in bands]
    stds = [correlation_stats[band]['std_correlation'] for band in bands]
    
    # Create bar plot with error bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mean correlations with standard deviation
    x_pos = np.arange(len(bands))
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Frequency Band', fontsize=12)
    ax1.set_ylabel('Mean Correlation', fontsize=12)
    ax1.set_title('Average Correlation by Frequency Band', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([b.capitalize() for b in bands])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}', ha='center', va='bottom')
    
    # Range of correlations (min to max)
    mins = [correlation_stats[band]['min_correlation'] for band in bands]
    maxs = [correlation_stats[band]['max_correlation'] for band in bands]
    
    ax2.bar(x_pos - 0.2, mins, 0.4, label='Min', color='lightcoral', edgecolor='black')
    ax2.bar(x_pos + 0.2, maxs, 0.4, label='Max', color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Frequency Band', fontsize=12)
    ax2.set_ylabel('Correlation Value', fontsize=12)
    ax2.set_title('Correlation Range by Frequency Band', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([b.capitalize() for b in bands])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'correlation_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nCorrelation Statistics Summary:")
    summary_df = pd.DataFrame(correlation_stats).T
    summary_df.index.name = 'Frequency Band'
    print(summary_df.round(3))
    
    return summary_df


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("EEG CORRELATION ANALYSIS")
    print("Rice Datathon 2025 - Neurotech Track")
    print("="*60)
    
    # Load and clean data
    df = load_and_clean_data(DATA_PATH)
    
    # Visualize missing values
    print("\n1. Analyzing missing values...")
    visualize_missing_values(df, save_path=OUTPUT_PATH / 'missing_values.png')
    
    # Remove coherence features
    print("\n2. Removing coherence features...")
    df_filtered = remove_coherence_features(df)
    
    # Analyze frequency bands
    print("\n3. Analyzing frequency band correlations...")
    correlation_stats = analyze_frequency_bands(df_filtered, save_plots=True)
    
    # Analyze demographic correlations
    print("\n4. Analyzing demographic correlations...")
    analyze_demographic_correlations(df_filtered, save_path=OUTPUT_PATH / 'demographic_correlations.png')
    
    # Create summary
    print("\n5. Creating correlation summary...")
    summary_df = create_correlation_summary(correlation_stats)
    
    # Save summary to CSV
    summary_df.to_csv(OUTPUT_PATH / 'correlation_summary.csv')
    
    print("\n" + "="*60)
    print("Analysis complete! Results saved to:", OUTPUT_PATH)
    print("="*60)


if __name__ == "__main__":
    main()