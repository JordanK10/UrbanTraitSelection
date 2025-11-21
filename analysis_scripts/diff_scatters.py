#!/usr/bin/env python3
"""
diff_scatters.py

Generates scatter plots comparing empirical growth rates and initial income
against differences between various growth rate measures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

# Check for 'null' argument to switch directories
import os
import sys

if 'null' in sys.argv:
    INPUT_DIR = 'output_terms_null'
    BASE_OUTPUT_DIR = 'plots_null'
else:
    INPUT_DIR = 'output_terms'
    BASE_OUTPUT_DIR = 'plots'

def load_data():
    """Load and prepare the data from CSV files."""
    print("Loading data...")
    
    # Load community-level data
    try:
        cm_data = pd.read_csv(os.path.join(INPUT_DIR, 'bg_cm_exported_terms.csv'))
        print(f"Loaded community data: {cm_data.shape[0]} rows, {cm_data.shape[1]} columns")
    except FileNotFoundError:
        print("Error: bg_cm_exported_terms.csv not found")
        return None, None
    
    # Load tract-level data
    try:
        tr_data = pd.read_csv(os.path.join(INPUT_DIR, 'bg_tr_exported_terms.csv'))
        print(f"Loaded tract data: {tr_data.shape[0]} rows, {tr_data.shape[1]} columns")
    except FileNotFoundError:
        print("Warning: bg_tr_exported_terms.csv not found, using only community data")
        tr_data = None
    
    return cm_data, tr_data

def prepare_variables(data, level_name):
    """Extract and calculate the required variables from the data."""
    print(f"Preparing variables for {level_name} level...")
    
    # Create a working copy
    df = data.copy()
    
    # Extract empirical growth rate from pre-calculated column
    if 'AvgG_emp_cm' in df.columns:
        df['empirical_growth_rate'] = df['AvgG_emp_cm']
    else:
        # Look for other possible empirical growth rate columns
        emp_cols = [col for col in df.columns if 'AvgG_emp' in col]
        if emp_cols:
            df['empirical_growth_rate'] = df[emp_cols[0]]
        else:
            print(f"Warning: Could not find empirical growth rate column (AvgG_emp_*) for {level_name}")
            df['empirical_growth_rate'] = np.nan
    
    # Population averaged growth rate
    if 'AvgG_pop_cm' in df.columns:
        df['pop_averaged_growth_rate'] = df['AvgG_pop_cm']
    else:
        pop_avg_cols = [col for col in df.columns if 'AvgG_pop' in col]
        if pop_avg_cols:
            df['pop_averaged_growth_rate'] = df[pop_avg_cols[0]]
        else:
            print(f"Warning: Could not find population averaged growth rate for {level_name}")
            df['pop_averaged_growth_rate'] = np.nan
    
    # Income averaged growth rate
    if 'AvgG_inc_cm' in df.columns:
        df['inc_averaged_growth_rate'] = df['AvgG_inc_cm']
    else:
        inc_avg_cols = [col for col in df.columns if 'AvgG_inc' in col]
        if inc_avg_cols:
            df['inc_averaged_growth_rate'] = df[inc_avg_cols[0]]
        else:
            print(f"Warning: Could not find income averaged growth rate for {level_name}")
            df['inc_averaged_growth_rate'] = np.nan
    
    # Initial income (using log average income initial)
    if 'LogAvgIncInitial_cm' in df.columns:
        df['initial_income'] = df['LogAvgIncInitial_cm']
    else:
        initial_inc_cols = [col for col in df.columns if 'LogAvgInc' in col and 'Initial' in col]
        if initial_inc_cols:
            df['initial_income'] = df[initial_inc_cols[0]]
        else:
            print(f"Warning: Could not find initial income for {level_name}")
            df['initial_income'] = np.nan
    
    # Calculate difference variables
    df['pop_avg_minus_empirical'] = df['pop_averaged_growth_rate'] - df['empirical_growth_rate']
    df['inc_avg_minus_empirical'] = df['inc_averaged_growth_rate'] - df['empirical_growth_rate']
    df['pop_avg_minus_inc_avg'] = df['pop_averaged_growth_rate'] - df['inc_averaged_growth_rate']
    
    # Add level identifier
    df['level'] = level_name
    
    # Select only the variables we need
    key_vars = ['empirical_growth_rate', 'pop_averaged_growth_rate', 'inc_averaged_growth_rate', 
                'initial_income', 'level',
                'pop_avg_minus_empirical', 'inc_avg_minus_empirical', 'pop_avg_minus_inc_avg']
    
    # Add UnitName if available for identification
    if 'UnitName' in df.columns:
        key_vars.append('UnitName')
    
    return df[key_vars]

def add_regression_line_and_stats(ax, x, y, color='red'):
    """Add regression line and statistics to a plot."""
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return
    
    # Calculate regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    
    # Add regression line
    line_x = np.array([x_clean.min(), x_clean.max()])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color=color, linestyle='--', alpha=0.8, linewidth=2)
    
    # Add statistics text
    r_squared = r_value ** 2
    stats_text = f'Slope = {slope:.4f}\nIntercept = {intercept:.4f}\nR² = {r_squared:.3f}\np = {p_value:.3f}\nn = {len(x_clean)}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def create_scatter_plot(data, x_var, y_var, title, filename, figsize=(10, 8)):
    """Create a scatter plot with linear fit and save it."""
    print(f"Creating plot: {title}")
    
    # Filter out rows where any of the key variables are NaN or infinite
    mask = (
        data[x_var].notna() & 
        data[y_var].notna() & 
        np.isfinite(data[x_var]) &
        np.isfinite(data[y_var])
    )
    
    plot_data = data[mask].copy()
    
    if len(plot_data) < 2:
        print(f"  Skipping {title} - insufficient data points ({len(plot_data)})")
        return
    
    # Determine aggregation level from data
    if 'level' in plot_data.columns:
        level = plot_data['level'].iloc[0]
        agg_level = 'cm' if 'Community' in level else 'tr'
    else:
        agg_level = 'cm'  # default
    
    # Define mathematical notation mapping
    math_labels = {
        'empirical_growth_rate': rf'$\gamma_{{{agg_level}}}$',
        'pop_averaged_growth_rate': rf'$\gamma_{{{agg_level}}}^p$',
        'inc_averaged_growth_rate': rf'$\gamma_{{{agg_level}}}^i$', 
        'population_growth_rate': r'$\gamma_p$',
        'initial_income': r'$\ln\bar{y}_0$',
        'pop_avg_minus_empirical': rf'$\gamma_{{{agg_level}}}^p - \gamma_{{{agg_level}}}$',
        'inc_avg_minus_empirical': rf'$\gamma_{{{agg_level}}}^i - \gamma_{{{agg_level}}}$',
        'pop_avg_minus_inc_avg': rf'$\gamma_{{{agg_level}}}^p - \gamma_{{{agg_level}}}^i$'
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot - simple black points
    ax.scatter(plot_data[x_var], plot_data[y_var], 
               alpha=0.6, s=50, color='darkblue', edgecolors='black', linewidth=0.5)
    
    # Add regression line and statistics
    add_regression_line_and_stats(ax, plot_data[x_var].values, plot_data[y_var].values)
    
    # Add horizontal line at y=0 for difference plots
    if 'minus' in y_var:
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Formatting with mathematical notation
    x_label = math_labels.get(x_var, x_var.replace('_', ' ').title())
    y_label = math_labels.get(y_var, y_var.replace('_', ' ').title())
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'diff_scatter_plots', filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(BASE_OUTPUT_DIR, 'diff_scatter_plots', filename)}")

def create_all_plots(data):
    """Create all requested difference scatter plots."""
    print("Creating all difference scatter plots...")
    
    # Create output directory
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, 'diff_scatter_plots'), exist_ok=True)
    
    # Split data by level
    levels = data['level'].unique()
    print(f"Creating plots for levels: {levels}")
    
    plot_configs = [
        # Empirical growth rate vs difference plots
        {
            'x_var': 'empirical_growth_rate',
            'y_var': 'pop_avg_minus_empirical',
            'title_template': 'Empirical Growth Rate vs (Pop Avg - Empirical) - {level}',
            'filename_template': 'empirical_vs_pop_minus_emp_{level_short}.png'
        },
        
        {
            'x_var': 'empirical_growth_rate',
            'y_var': 'inc_avg_minus_empirical',
            'title_template': 'Empirical Growth Rate vs (Inc Avg - Empirical) - {level}',
            'filename_template': 'empirical_vs_inc_minus_emp_{level_short}.png'
        },
        
        {
            'x_var': 'inc_averaged_growth_rate',
            'y_var': 'pop_avg_minus_inc_avg',
            'title_template': 'Income Averaged Growth Rate vs (Pop Avg - Inc Avg) - {level}',
            'filename_template': 'inc_avg_vs_pop_minus_inc_{level_short}.png'
        },
        
        # Initial income vs difference plots
        {
            'x_var': 'initial_income',
            'y_var': 'pop_avg_minus_empirical',
            'title_template': 'Initial Income vs (Pop Avg - Empirical) - {level}',
            'filename_template': 'initial_vs_pop_minus_emp_{level_short}.png'
        },
        
        {
            'x_var': 'initial_income',
            'y_var': 'inc_avg_minus_empirical',
            'title_template': 'Initial Income vs (Inc Avg - Empirical) - {level}',
            'filename_template': 'initial_vs_inc_minus_emp_{level_short}.png'
        },
        
        {
            'x_var': 'initial_income',
            'y_var': 'pop_avg_minus_inc_avg',
            'title_template': 'Initial Income vs (Pop Avg - Inc Avg) - {level}',
            'filename_template': 'initial_vs_pop_minus_inc_{level_short}.png'
        }
    ]
    
    # Create plots for each level
    for level in levels:
        level_data = data[data['level'] == level].copy()
        level_short = level.lower()[:2]  # 'co' for community, 'tr' for tract
        
        print(f"\nCreating plots for {level} level ({len(level_data)} observations)...")
        
        for config in plot_configs:
            # Create level-specific title and filename
            title = config['title_template'].format(level=level)
            filename = config['filename_template'].format(level_short=level_short)
            
            create_scatter_plot(
                level_data, 
                config['x_var'], 
                config['y_var'], 
                title, 
                filename
            )

def print_summary_statistics(data):
    """Print summary statistics for all variables."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    variables = ['empirical_growth_rate', 'pop_averaged_growth_rate', 'inc_averaged_growth_rate', 
                 'initial_income', 'pop_avg_minus_empirical', 'inc_avg_minus_empirical', 'pop_avg_minus_inc_avg']
    
    for var in variables:
        if var in data.columns:
            print(f"\n{var.replace('_', ' ').title()}:")
            print(f"  Mean: {data[var].mean():.4f}")
            print(f"  Std:  {data[var].std():.4f}")
            print(f"  Min:  {data[var].min():.4f}")
            print(f"  Max:  {data[var].max():.4f}")
            print(f"  N:    {data[var].count()}")

def main():
    """Main execution function."""
    print("="*60)
    print("DIFFERENCE SCATTER PLOTS ANALYSIS")
    print("="*60)
    
    # Load data
    cm_data, tr_data = load_data()
    if cm_data is None:
        print("Error: Could not load data files")
        return
    
    # Prepare data
    datasets = []
    
    # Process community data
    cm_processed = prepare_variables(cm_data, 'Community')
    if not cm_processed.empty:
        datasets.append(cm_processed)
        print(f"Community data: {len(cm_processed)} observations")
    
    # Process tract data if available
    if tr_data is not None:
        tr_processed = prepare_variables(tr_data, 'Tract')
        if not tr_processed.empty:
            datasets.append(tr_processed)
            print(f"Tract data: {len(tr_processed)} observations")
    
    if not datasets:
        print("Error: No usable data found")
        return
    
    # Combine datasets
    combined_data = pd.concat(datasets, ignore_index=True)
    print(f"Combined data: {len(combined_data)} total observations")
    
    # Print summary statistics
    print_summary_statistics(combined_data)
    
    # Create plots
    create_all_plots(combined_data)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    # Calculate total plots generated
    num_levels = len(combined_data['level'].unique())
    total_plots = 6 * num_levels
    
    print(f"Generated {total_plots} difference scatter plots in {BASE_OUTPUT_DIR}/diff_scatter_plots/ directory")
    print(f"({6} plot types × {num_levels} data levels)")
    print("Each plot includes:")
    print("- Scatter points showing relationship")
    print("- Linear regression line")
    print("- Horizontal line at y=0 for difference plots")
    print("- Slope, intercept, R², p-value, and sample size in top-left corner")

if __name__ == "__main__":
    main() 