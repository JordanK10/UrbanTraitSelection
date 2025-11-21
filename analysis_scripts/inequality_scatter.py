#!/usr/bin/env python3
"""
inequality_scatter.py

Creates scatter plots of:
1. Relative Empirical Growth Rate vs Cumulative Income PNC_st
2. Relative Empirical Growth Rate vs Cumulative Population PNC_st

Points are colored by population growth rate with custom colorbar.
Reads community income data and computes cumulative PNC values similar to specialty_histogram2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


# Custom colors
custom_purple = '#633673'
custom_orange = '#E77429'
outline_color = '#3D3D3D'

def load_data(path):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(path)
        print(f"Successfully loaded {path} with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def extract_sum_cm(df, trans_col, sel_col, scale=None):
    """Extract and sum transmitted and selection columns for community data"""
    if df is None:
        return pd.Series([])
    
    # Get the columns, filling missing with 0
    trans_vals = df[trans_col].fillna(0) if trans_col in df.columns else 0
    sel_vals = df[sel_col].fillna(0) if sel_col in df.columns else 0
    
    # Sum the values
    result = trans_vals + sel_vals
    
    # Apply scaling if specified
    if scale is not None:
        result = result * scale
    
    return result

def create_custom_colormap():
    """Create a custom colormap from purple through white to orange"""
    # Define the colors and their positions
    colors = [custom_purple, 'white', custom_orange]
    n_bins = 256
    
    # Create the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return cmap

def create_scatter_plot(x_data, y_data, color_data, x_label, y_label, color_label, title, filename, size_data=None, label=None):
    """Create a scatter plot with specified data and labels, colored by color_data"""
    # Remove NaN, inf, and exactly zero values from x and y data
    valid_mask = (np.isfinite(x_data) & np.isfinite(y_data) & 
                  ~np.isnan(x_data) & ~np.isnan(y_data) & 
                  (x_data != 0) & (y_data != 0))
    
    if size_data is not None:
        valid_mask &= np.isfinite(size_data) & ~np.isnan(size_data)

    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]
    label_clean = label[valid_mask]
    color_clean = color_data[valid_mask]
    
    if size_data is not None:
        size_clean = size_data[valid_mask]
        s_min, s_max = 15, 150
        if size_clean.min() == size_clean.max():
            sizes = pd.Series(s_min, index=size_clean.index)
        else:
            sizes = s_min + ((size_clean - size_clean.min()) / (size_clean.max() - size_clean.min()) * (s_max - s_min))
    else:
        sizes = 25

    print(f"\n{'='*60}")
    print(f"SCATTER PLOT: {title}")
    print(f"{'='*60}")
    print(f"Valid data points: {len(x_clean)}")
    
    if len(x_clean) == 0:
        print("No valid data points to plot!")
        return
    
    # Print data statistics
    print(f"X-axis ({x_label}):")
    print(f"  - Mean: {x_clean.mean():.6f}")
    print(f"  - Std: {x_clean.std():.6f}")
    print(f"  - Min: {x_clean.min():.6f}")
    print(f"  - Max: {x_clean.max():.6f}")
    
    print(f"Y-axis ({y_label}):")
    print(f"  - Mean: {y_clean.mean():.6f}")
    print(f"  - Std: {y_clean.std():.6f}")
    print(f"  - Min: {y_clean.min():.6f}")
    print(f"  - Max: {y_clean.max():.6f}")
    
    print(f"Color variable ({color_label}):")
    print(f"  - Mean: {color_clean.mean():.6f}")
    print(f"  - Std: {color_clean.std():.6f}")
    print(f"  - Min: {color_clean.min():.6f}")
    print(f"  - Max: {color_clean.max():.6f}")
    
    # Calculate correlation
    correlation = np.corrcoef(x_clean, y_clean)[0, 1]
    print(f"Correlation coefficient: {correlation:.6f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Create custom colormap
    custom_cmap = create_custom_colormap()
    
    # Determine color limits to center white at zero
    color_abs_max = max(abs(color_clean.min()), abs(color_clean.max()))
    vmin = -color_abs_max
    vmax = color_abs_max
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(x_clean, y_clean, c=color_clean, cmap=custom_cmap, 
                        alpha=0.8, s=sizes, edgecolors=outline_color, linewidth=0.8,
                        vmin=vmin, vmax=vmax)
    # for x,y,l in zip(x_clean,y_clean,label_clean):

    #     ax.annotate(l,(x,y),fontsize=3,alpha=.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(color_label, fontsize=12, color='#333333')
    cbar.ax.tick_params(labelsize=10, colors='#333333')
    
    # Add correlation text
    # ax.text(0.05, 0.95, f'r = {correlation:.3f}\nn = {len(x_clean)}', 
    #         transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', horizontalalignment='left',
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=outline_color))
    
    # Add reference lines
    ax.axhline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel(x_label, fontsize=14, color='#333333')
    ax.set_ylabel(y_label, fontsize=14, color='#333333')
    # ax.set_title(title, fontsize=16, color='#333333', pad=20)
    
    # Style the plot
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CCCCCC')
    
    ax.tick_params(axis='x', which='major', labelsize=12, colors='#333333')
    ax.tick_params(axis='y', which='major', labelsize=12, colors='#333333')
    

    # input(x_label)
    if x_label == "Cumulative Income PNC_st":
        ax.set_xlim(left=-75)

    # Grid
    # ax.grid(True, alpha=0.3, color='#CCCCCC')
    
    # Save plot
    plt.tight_layout()
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, "inequality_scatter_plots"), exist_ok=True)
    full_filename = os.path.join(os.path.join(BASE_OUTPUT_DIR, "inequality_scatter_plots"), filename)
    plt.savefig(full_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {full_filename}")

def save_plot(fig, filename):
    """Saves a plot to a file, creating directories if necessary."""
    output_dir = os.path.join(BASE_OUTPUT_DIR, "inequality_scatter_plots")
    os.makedirs(output_dir, exist_ok=True)
    full_filename = os.path.join(output_dir, filename)
    fig.savefig(full_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {full_filename}")

# --- Main execution ---
if __name__ == "__main__":
    # Load community data
    cm_data = load_data(os.path.join(INPUT_DIR, "bg_cm_exported_terms.csv"))
    
    if cm_data is None:
        print("Error: Could not load community data file.")
        exit(1)
    
    # Extract relative empirical growth rate
    if 'RelAvgG_emp_cm' in cm_data.columns:
        rel_emp_growth = cm_data['RelAvgG_emp_cm']
        rel_pop_growth = cm_data['RelAvgG_pop_cm']
        rel_inc_growth = cm_data['RelAvgG_inc_cm']
        print(f"Successfully extracted relative empirical growth rate: {len(rel_emp_growth)} values")
    else:
        print("Error: 'RelAvgG_emp_cm' column not found in community data.")
        print("Available columns:", list(cm_data.columns))
        exit(1)
    
    # Extract population growth rate for coloring
    if 'PopG_cm' in cm_data.columns:
        pop_growth = cm_data['LogAvgIncInitial_cm']
        pop_growth = pop_growth-pop_growth.median()
        print(f"Successfully extracted population growth rate: {len(pop_growth)} values")
    else:
        print("Error: 'PopG_cm' column not found in community data.")
        print("Available columns:", list(cm_data.columns))
        exit(1)
    
    # Compute cumulative income PNC_st
    print("\nComputing cumulative income PNC_st...")
    cm_inc_pnc = extract_sum_cm(cm_data, 'Transmitted_Sel_tr_to_cm_inc_PNC_st', 'Sel_cm_from_tr_inc_PNC_st', scale=1)
    
    # Compute cumulative population PNC_st  
    print("Computing cumulative population PNC_st...")
    cm_pop_pnc = extract_sum_cm(cm_data, 'Transmitted_Sel_tr_to_cm_pop_PNC_st', 'Sel_cm_from_tr_pop_PNC_st', scale=1)
    
    pop_initial_cm = cm_data['PopInitial_cm'] if 'PopInitial_cm' in cm_data.columns else None

    cm_label = cm_data['UnitName']

    # Create scatter plots
    print("\nCreating scatter plots...")
    
    # Plot 1: Relative Empirical Growth Rate vs Income PNC_st (axes reversed)
    create_scatter_plot(
        x_data=cm_inc_pnc,
        y_data=rel_inc_growth,
        color_data=pop_growth,
        size_data=pop_initial_cm,
        x_label="Cumulative Income PNC_st",
        y_label="Relative Income-Averaged Growth Rate",
        color_label="Relative Initial Income",
        title="Relative Empirical Growth Rate vs Income PNC_st",
        filename="rel_growth_vs_income_pnc.pdf",
        label = cm_label
    )
    
    # Plot 2: Relative Empirical Growth Rate vs Population PNC_st (axes reversed)
    create_scatter_plot(
        x_data=cm_pop_pnc,
        y_data=rel_pop_growth,
        color_data=pop_growth,
        size_data=pop_initial_cm,
        x_label="Cumulative Population PNC_st", 
        y_label="Relative Population-Averaged Growth Rate",
        color_label="Relative Initial Income",
        title="Relative Empirical Growth Rate vs Population PNC_st",
        filename="rel_growth_vs_population_pnc.pdf",
        label = cm_label
    )
    
    print("\n" + "="*60)
    print("INEQUALITY SCATTER ANALYSIS COMPLETE")
    print("="*60)
    print("Generated plots:")
    print(f"1. {os.path.join(BASE_OUTPUT_DIR, 'inequality_scatter_plots', 'rel_growth_vs_income_pnc.pdf')}")
    print(f"2. {os.path.join(BASE_OUTPUT_DIR, 'inequality_scatter_plots', 'rel_growth_vs_population_pnc.pdf')}") 