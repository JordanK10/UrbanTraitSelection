#!/usr/bin/env python3
"""
variance_by_level.py

Creates a plot showing the variance in population growth rate across different 
levels of geographic aggregation (block group, tract, community, county, state).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Check for 'null' argument to switch directories
if 'null' in sys.argv:
    INPUT_DIR = 'output_terms_null'
    BASE_OUTPUT_DIR = 'plots_null'
else:
    INPUT_DIR = 'output_terms'
    BASE_OUTPUT_DIR = 'plots'

# Custom colors from plotPrice.py
custom_purple = '#633673'
custom_orange = '#E77429'

def load_data_by_level(level):
    """Load and prepare the data for a specific level."""
    print(f"Loading {level}-level data...")
    
    filename = os.path.join(INPUT_DIR, f'bg_{level}_exported_terms.csv')
    try:
        data = pd.read_csv(filename)
        print(f"Loaded {level} data: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return None

def extract_growth_rate_stats(data, level_suffix):
    """Extract statistics for both empirical and population growth rates."""
    print(f"Extracting growth rate statistics for {level_suffix} level...")
    
    # Try different possible column names for empirical growth rate
    possible_emp_growth_cols = [
        f'AvgG_emp_{level_suffix}',
        'AvgG_emp',
        'empirical_growth_rate'
    ]
    
    # Try different possible column names for population growth rate  
    possible_pop_growth_cols = [
        f'PopG_{level_suffix}',
        f'PopFracChange_{level_suffix}',
        'PopG',
        'PopFracChange',
        'population_growth_rate'
    ]
    
    stats = {}
    
    # Extract empirical growth rate statistics
    emp_col = None
    for col_pattern in possible_emp_growth_cols:
        matching_cols = [col for col in data.columns if col_pattern in col]
        if matching_cols:
            emp_col = matching_cols[0]
            break
    
    if emp_col:
        print(f"Using {emp_col} for empirical growth rate")
        emp_data = pd.to_numeric(data[emp_col], errors='coerce').dropna()
        if len(emp_data) > 1: # Need more than 1 data point for CI
            std_val = emp_data.std()
            n_val = len(emp_data)
            stats['empirical'] = {
                'mean': emp_data.mean(),
                'std': std_val,
                'variance': emp_data.var(),
                'n': n_val,
                'ci_95_margin': 1.96 * (std_val / np.sqrt(n_val))
            }
            print(f"Empirical growth stats for {level_suffix}: mean={stats['empirical']['mean']:.6f}, std={stats['empirical']['std']:.6f}, n={stats['empirical']['n']}")
        else:
            stats['empirical'] = None
    else:
        print(f"Warning: Could not find empirical growth rate column for {level_suffix}")
        stats['empirical'] = None
    
    # Extract population growth rate statistics
    pop_col = None
    for col_pattern in possible_pop_growth_cols:
        matching_cols = [col for col in data.columns if col_pattern in col]
        if matching_cols:
            pop_col = matching_cols[0]
            break
    
    if pop_col:
        print(f"Using {pop_col} for population growth rate")
        pop_data = pd.to_numeric(data[pop_col], errors='coerce').dropna()
        if len(pop_data) > 1: # Need more than 1 data point for CI
            std_val = pop_data.std()
            n_val = len(pop_data)
            stats['population'] = {
                'mean': pop_data.mean(),
                'std': std_val,
                'variance': pop_data.var(),
                'n': n_val,
                'ci_95_margin': 1.96 * (std_val / np.sqrt(n_val))
            }
            print(f"Population growth stats for {level_suffix}: mean={stats['population']['mean']:.6f}, std={stats['population']['std']:.6f}, n={stats['population']['n']}")
        else:
            stats['population'] = None
    else:
        print(f"Warning: Could not find population growth rate column for {level_suffix}")
        stats['population'] = None
    
    return stats

def create_line_plot_with_confidence(stats_data, figsize=(4.5, 1.75)):
    """Create a line plot showing mean growth rates with confidence intervals."""
    print("Creating line plot with confidence intervals...")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom colors
    custom_purple = '#633673'
    custom_orange = '#E77429'
    
    # Extract data for plotting
    levels = []
    emp_means = []
    emp_stds = []
    pop_means = []
    pop_stds = []
    emp_ci_margins = []
    pop_ci_margins = []
    
    for level, stats in stats_data.items():
        levels.append(level)
        
        # Empirical growth rate data
        if stats['empirical'] is not None:
            emp_means.append(stats['empirical']['mean'])
            emp_stds.append(stats['empirical']['std'])
            emp_ci_margins.append(stats['empirical']['ci_95_margin'])
        else:
            emp_means.append(np.nan)
            emp_stds.append(np.nan)
            emp_ci_margins.append(np.nan)
        
        # Population growth rate data
        if stats['population'] is not None:
            pop_means.append(np.nan)
            pop_stds.append(np.nan)
            pop_ci_margins.append(np.nan)
    
    if not levels:
        print("Warning: No valid data to plot")
        return fig, ax
    
    # Create y positions (flipped axes)
    y_positions = range(len(levels))
    
    # Plot empirical growth rate
    emp_means_clean = np.array(emp_means)
    emp_ci_margins_clean = np.array(emp_ci_margins)
    valid_emp = ~np.isnan(emp_means_clean)
    
    if np.any(valid_emp):
        ax.plot(emp_means_clean[valid_emp], np.array(y_positions)[valid_emp], 
                color=custom_purple, marker='o', linewidth=2, markersize=6, 
                label='Empirical Growth Rate')
        ax.fill_betweenx(np.array(y_positions)[valid_emp], 
                        (emp_means_clean - emp_ci_margins_clean)[valid_emp],
                        (emp_means_clean + emp_ci_margins_clean)[valid_emp],
                        color=custom_purple, alpha=0.2)
    
    # Plot population growth rate
    pop_means_clean = np.array(pop_means)
    pop_ci_margins_clean = np.array(pop_ci_margins)
    valid_pop = ~np.isnan(pop_means_clean)
    
    if np.any(valid_pop):
        ax.plot(pop_means_clean[valid_pop], np.array(y_positions)[valid_pop], 
                color=custom_orange, marker='s', linewidth=2, markersize=6,
                label='Population Growth Rate')
        ax.fill_betweenx(np.array(y_positions)[valid_pop], 
                        (pop_means_clean - pop_ci_margins_clean)[valid_pop],
                        (pop_means_clean + pop_ci_margins_clean)[valid_pop],
                        color=custom_orange, alpha=0.2)
    
    # Customize the plot (flipped axes)
    ax.set_title('Growth Rates by Aggregation Level\n(Mean with 95% CI)', fontsize=14, fontweight='bold')
    
    # Set y-axis labels (flipped)
    # ax.set_yticks(y_positions)
    # ax.set_yticklabels(levels)
    
    # Add legend
    # ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Add sample sizes as text (flipped axes) at custom x position
    custom_text_x = -0.065  # Fixed x position for text
    for i, (level, stats) in enumerate(stats_data.items()):
        # Create sample size text
        text_parts = []
        text_parts.append(f"n={stats['population']['n']:,}")
        
        if text_parts:
            ax.text(custom_text_x, i+.15, ' / '.join(text_parts), ha='left', va='center', 
                   fontsize=8, color='#666666')
    
    # Style the plot
    ax.set_ylim(top=3.5)
    ax.spines['top'].set_color('#D3D3D3')
    ax.spines['right'].set_color('#D3D3D3')
    ax.spines['left'].set_color('#D3D3D3')
    ax.spines['bottom'].set_color('#D3D3D3')
    ax.grid(True, axis='y', alpha=0.3, color='#D3D3D3')

    ax.tick_params(colors='#333333', labelsize=12)  # 8 * 1.25 = 10
    
    # Add vertical reference line at x=0 (flipped axes)
    ax.axvline(x=0, color='#3D3D3D', alpha=0.75, linewidth=1)
    ax.axhline(y=1, color='#D3D3D3', alpha=0.75, linewidth=1,zorder=-1)
    ax.axhline(y=2, color='#D3D3D3', alpha=0.75, linewidth=1,zorder=-1)
    ax.axhline(y=3, color='#D3D3D3', alpha=0.75, linewidth=1,zorder=-1)
    ax.axhline(y=4, color='#D3D3D3', alpha=0.75, linewidth=1,zorder=-1)

    plt.tight_layout()
    
    return fig, ax

def save_plot(fig, filename):
    """Saves a plot to a file, creating directories if necessary."""
    full_path = os.path.join(BASE_OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    fig.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {full_path}")

def print_summary_statistics(stats_data):
    """Print summary statistics for the growth rate data."""
    print("\n" + "="*60)
    print("GROWTH RATE SUMMARY STATISTICS")
    print("="*60)
    
    for level, stats in stats_data.items():
        print(f"\n{level.upper()} Level:")
        
        if stats['empirical'] is not None:
            emp = stats['empirical']
            print(f"  Empirical Growth Rate:")
            print(f"    Mean: {emp['mean']:.6f}")
            print(f"    Std Dev: {emp['std']:.6f}")
            print(f"    Variance: {emp['variance']:.6f}")
            print(f"    Sample Size: {emp['n']}")
        else:
            print(f"  Empirical Growth Rate: No valid data available")
        
        if stats['population'] is not None:
            pop = stats['population']
            print(f"  Population Growth Rate:")
            print(f"    Mean: {pop['mean']:.6f}")
            print(f"    Std Dev: {pop['std']:.6f}")
            print(f"    Variance: {pop['variance']:.6f}")
            print(f"    Sample Size: {pop['n']}")
        else:
            print(f"  Population Growth Rate: No valid data available")

def main():
    """Main execution function."""
    print("="*60)
    print("GROWTH RATES BY AGGREGATION LEVEL ANALYSIS")
    print("="*60)
    
    # Create output directory for plots
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, 'variance_plots'), exist_ok=True)
    
    # Define levels to process (in order from finest to coarsest)
    levels_to_process = [
        ('bg', 'bg', 'Block Group'),
        ('tr', 'tr', 'Tract'), 
        ('cm', 'cm', 'Community'),
        ('ct', 'ct', 'County')
    ]
    
    stats_data = {}
    
    # Process each level
    for level_code, level_suffix, level_name in levels_to_process:
        print(f"\nProcessing {level_name} level...")
        
        # Load data
        data = load_data_by_level(level_code)
        if data is None:
            stats_data[level_name] = {'empirical': None, 'population': None}
            continue
        
        # Extract growth rate statistics
        stats = extract_growth_rate_stats(data, level_suffix)
        stats_data[level_name] = stats
    
    # Print summary statistics
    print_summary_statistics(stats_data)
    
    # Create plot
    fig, ax = create_line_plot_with_confidence(stats_data)
    
    if fig is not None:
        # Save plot
        save_plot(fig, 'variance_plots/growth_rates_by_level.pdf')
        
        # Show plot
        plt.show()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Generated growth rates by aggregation level plot")
        print("Features:")
        print("- X-axis: Geographic aggregation levels (finest to coarsest)")
        print("- Y-axis: Growth rates (empirical and population)")
        print("- Line colors: Purple (empirical), Orange (population)")
        print("- Shaded region: 95% Confidence Interval of the Mean")
        print("- Sample sizes shown below x-axis")
        print("Plot saved as: plots/variance_plots/growth_rates_by_level.pdf")
    else:
        print("\nError: Could not generate plot due to insufficient data")

if __name__ == "__main__":
    main() 