#!/usr/bin/env python3
"""
gini_inequality_scatter.py

Creates a scatter plot of cumulative income PNC_st vs change in Gini coefficient.
The Gini coefficient is calculated from block group income data grouped by communities,
comparing initial vs final time periods.

X-axis: Cumulative Income PNC_st (from community data)
Y-axis: Change in Gini coefficient (final_gini - initial_gini) for each community
Color: Population growth rate for visual distinction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import warnings
from scipy import stats
import sys

# Check for 'null' argument to switch directories
if 'null' in sys.argv:
    INPUT_DIR = 'output_terms_null'
    BASE_OUTPUT_DIR = 'plots_null'
else:
    INPUT_DIR = 'output_terms'
    BASE_OUTPUT_DIR = 'plots'

warnings.filterwarnings('ignore')

# Custom colors matching other plots
custom_purple = '#633673'
custom_orange = '#E77429'
outline_color = '#3D3D3D'

def load_data():
    """Load community, block group, and tract data"""
    try:
        # Load community data for cumulative PNC_st and population growth
        cm_data = pd.read_csv(os.path.join(INPUT_DIR, "bg_cm_exported_terms.csv"))
        print(f"Successfully loaded community data: {len(cm_data)} rows")
        
        # Load block group data for Gini calculation
        bg_data = pd.read_csv(os.path.join(INPUT_DIR, "bg_bg_exported_terms.csv"))
        print(f"Successfully loaded block group data: {len(bg_data)} rows")
        
        # Load tract data for Gini calculation
        tr_data = pd.read_csv(os.path.join(INPUT_DIR, "bg_tr_exported_terms.csv"))
        print(f"Successfully loaded tract data: {len(tr_data)} rows")
        
        return cm_data, bg_data, tr_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def calculate_gini_coefficient(incomes, weights=None):
    """
    Calculate Gini coefficient for a distribution of incomes.
    
    Args:
        incomes: Array of income values
        weights: Array of weights (population/households) for each income value
    
    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    if weights is None:
        weights = np.ones_like(incomes)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(incomes) | np.isnan(weights))
    incomes = incomes[valid_mask]
    weights = weights[valid_mask]
    
    if len(incomes) < 2:
        return np.nan
    
    # Sort by income
    sorted_indices = np.argsort(incomes)
    sorted_incomes = incomes[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Calculate weighted cumulative shares
    cum_weights = np.cumsum(sorted_weights)
    cum_income_weights = np.cumsum(sorted_incomes * sorted_weights)
    
    # Normalize to get shares
    total_weight = cum_weights[-1]
    total_income_weight = cum_income_weights[-1]
    
    if total_weight == 0 or total_income_weight == 0:
        return np.nan
    
    cum_pop_share = cum_weights / total_weight
    cum_income_share = cum_income_weights / total_income_weight
    
    # Calculate Gini using the trapezoidal rule
    # Gini = 1 - 2 * Area under Lorenz curve
    # Area under Lorenz curve approximated by trapezoidal rule
    
    # Add point (0,0) at the beginning
    x = np.concatenate([[0], cum_pop_share])
    y = np.concatenate([[0], cum_income_share])
    
    # Calculate area under Lorenz curve using trapezoidal rule
    area_under_lorenz = np.trapz(y, x)
    
    # Gini coefficient
    gini = 1 - 2 * area_under_lorenz
    
    return gini

def prepare_bg_income_data(bg_data):
    """
    Prepare block group data for Gini calculation by extracting initial and final incomes.
    Uses log income columns and converts them to actual incomes.
    """
    print("Preparing block group income data...")
    
    # Look for log income columns and convert to actual incomes
    log_initial_income_col = None
    log_final_income_col = None
    
    # Check for log income column naming patterns
    for col in bg_data.columns:
        if 'LogAvgIncInitial' in col:
            log_initial_income_col = col
        elif 'LogAvgIncFinal' in col:
            log_final_income_col = col
    
    if log_initial_income_col is None or log_final_income_col is None:
        print(f"Warning: Could not find log income columns. Available columns: {bg_data.columns.tolist()}")
        return None
    
    print(f"Using log initial income column: {log_initial_income_col}")
    print(f"Using log final income column: {log_final_income_col}")
    
    # Prepare the data
    bg_prepared = bg_data.copy()
    
    # Ensure we have ParentCommunity for grouping
    if 'ParentCommunity' not in bg_prepared.columns:
        print("Error: ParentCommunity column not found in block group data")
        return None
    
    # Convert log incomes to actual incomes
    bg_prepared['initial_income'] = np.exp(bg_prepared[log_initial_income_col])
    bg_prepared['final_income'] = np.exp(bg_prepared[log_final_income_col])
    
    # Get population/household weights
    pop_col = None
    for col in ['PopInitial_bg', 'Population', 'PopFinal_bg']:
        if col in bg_prepared.columns:
            pop_col = col
            break
    
    if pop_col is None:
        print("Warning: No population column found, using equal weights")
        bg_prepared['weight'] = 1
    else:
        bg_prepared['weight'] = bg_prepared[pop_col]
        print(f"Using weight column: {pop_col}")
    
    # Select relevant columns
    result = bg_prepared[['ParentCommunity', 'initial_income', 'final_income', 'weight']].copy()
    
    # Remove rows with missing community data or invalid incomes
    result = result.dropna(subset=['ParentCommunity'])
    result = result[(result['initial_income'] > 0) & (result['final_income'] > 0)]
    
    print(f"Prepared {len(result)} block group records for Gini calculation")
    return result

def prepare_tr_income_data(tr_data):
    """
    Prepare tract data for Gini calculation by extracting initial and final incomes.
    Uses log income columns and converts them to actual incomes.
    """
    print("Preparing tract income data...")
    
    # Look for log income columns and convert to actual incomes
    log_initial_income_col = None
    log_final_income_col = None
    
    # Check for log income column naming patterns
    for col in tr_data.columns:
        if 'LogAvgIncInitial' in col:
            log_initial_income_col = col
        elif 'LogAvgIncFinal' in col:
            log_final_income_col = col
    
    if log_initial_income_col is None or log_final_income_col is None:
        print(f"Warning: Could not find log income columns in tract data. Available columns: {tr_data.columns.tolist()}")
        return None
    
    print(f"Using log initial income column: {log_initial_income_col}")
    print(f"Using log final income column: {log_final_income_col}")
    
    # Prepare the data
    tr_prepared = tr_data.copy()
    
    # Ensure we have ParentCommunity for grouping
    if 'ParentCommunity' not in tr_prepared.columns:
        print("Error: ParentCommunity column not found in tract data")
        return None
    
    # Convert log incomes to actual incomes
    tr_prepared['initial_income'] = np.exp(tr_prepared[log_initial_income_col])
    tr_prepared['final_income'] = np.exp(tr_prepared[log_final_income_col])
    
    # Get population/household weights
    pop_col = None
    for col in ['PopInitial_tr', 'Population', 'PopFinal_tr']:
        if col in tr_prepared.columns:
            pop_col = col
            break
    
    if pop_col is None:
        print("Warning: No population column found in tract data, using equal weights")
        tr_prepared['weight'] = 1
    else:
        tr_prepared['weight'] = tr_prepared[pop_col]
        print(f"Using weight column: {pop_col}")
    
    # Select relevant columns
    result = tr_prepared[['ParentCommunity', 'initial_income', 'final_income', 'weight']].copy()
    
    # Remove rows with missing community data or invalid incomes
    result = result.dropna(subset=['ParentCommunity'])
    result = result[(result['initial_income'] > 0) & (result['final_income'] > 0)]
    
    print(f"Prepared {len(result)} tract records for Gini calculation")
    return result

def calculate_community_gini_changes(bg_income_data):
    """
    Calculate Gini coefficient change for each community.
    
    Returns DataFrame with columns: ParentCommunity, initial_gini, final_gini, gini_change
    """
    print("Calculating Gini coefficients by community...")
    
    results = []
    
    for community, group in bg_income_data.groupby('ParentCommunity'):
        if len(group) < 2:  # Need at least 2 observations for meaningful Gini
            continue
            
        # Calculate initial Gini
        initial_gini = calculate_gini_coefficient(
            group['initial_income'].values, 
            group['weight'].values
        )
        
        # Calculate final Gini
        final_gini = calculate_gini_coefficient(
            group['final_income'].values, 
            group['weight'].values
        )
        
        # Calculate change
        gini_change = final_gini - initial_gini if pd.notna(final_gini) and pd.notna(initial_gini) else np.nan
        
        results.append({
            'ParentCommunity': community,
            'initial_gini': initial_gini,
            'final_gini': final_gini,
            'gini_change': gini_change,
            'num_block_groups': len(group)
        })
    
    result_df = pd.DataFrame(results)
    print(f"Calculated Gini changes for {len(result_df)} communities")
    
    # Print some statistics
    valid_changes = result_df['gini_change'].dropna()
    if len(valid_changes) > 0:
        print(f"Gini change statistics:")
        print(f"  Mean: {valid_changes.mean():.4f}")
        print(f"  Std: {valid_changes.std():.4f}")
        print(f"  Min: {valid_changes.min():.4f}")
        print(f"  Max: {valid_changes.max():.4f}")
    
    return result_df

def extract_cumulative_income_pnc(cm_data):
    """Extract cumulative income PNC_st from community data"""
    print("Extracting cumulative income PNC_st...")
    
    # Look for the PNC income columns
    transmitted_col = 'Transmitted_Sel_tr_to_cm_inc_PNC_st'
    selection_col = 'Sel_cm_from_tr_inc_PNC_st'
    
    if transmitted_col not in cm_data.columns or selection_col not in cm_data.columns:
        print(f"Warning: Expected PNC columns not found. Available columns: {cm_data.columns.tolist()}")
        return None
    
    # Calculate cumulative PNC_st (transmitted + selection)
    transmitted_vals = cm_data[transmitted_col].fillna(0)
    selection_vals = cm_data[selection_col].fillna(0)
    
    cumulative_pnc = transmitted_vals + selection_vals
    
    print(f"Cumulative income PNC_st statistics:")
    print(f"  Mean: {cumulative_pnc.mean():.4f}")
    print(f"  Std: {cumulative_pnc.std():.4f}")
    print(f"  Min: {cumulative_pnc.min():.4f}")
    print(f"  Max: {cumulative_pnc.max():.4f}")
    
    return cumulative_pnc

def extract_transmitted_income_pnc(cm_data):
    """Extract transmitted income PNC_st from community data"""
    print("Extracting transmitted income PNC_st...")
    
    transmitted_col = 'Transmitted_Sel_tr_to_cm_inc_PNC_st'
    
    if transmitted_col not in cm_data.columns:
        print(f"Warning: Expected PNC column not found: {transmitted_col}. Available columns: {cm_data.columns.tolist()}")
        return None
    
    transmitted_vals = cm_data[transmitted_col].fillna(0)
    
    print(f"Transmitted income PNC_st statistics:")
    print(f"  Mean: {transmitted_vals.mean():.4f}")
    print(f"  Std: {transmitted_vals.std():.4f}")
    print(f"  Min: {transmitted_vals.min():.4f}")
    print(f"  Max: {transmitted_vals.max():.4f}")
    
    return transmitted_vals

def extract_selection_income_pnc(cm_data):
    """Extract selection income PNC_st from community data"""
    print("Extracting selection income PNC_st...")
    
    selection_col = 'Sel_cm_from_tr_inc_PNC_st'
    
    if selection_col not in cm_data.columns:
        print(f"Warning: Expected PNC column not found: {selection_col}. Available columns: {cm_data.columns.tolist()}")
        return None

    selection_vals = cm_data[selection_col].fillna(0)
    
    print(f"Selection income PNC_st statistics:")
    print(f"  Mean: {selection_vals.mean():.4f}")
    print(f"  Std: {selection_vals.std():.4f}")
    print(f"  Min: {selection_vals.min():.4f}")
    print(f"  Max: {selection_vals.max():.4f}")
    
    return selection_vals

def create_custom_colormap():
    """Create a custom colormap from purple through white to orange"""
    colors = [custom_purple, 'white', custom_orange]
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return cmap

def create_gini_scatter_plot(merged_data):
    """Create the main scatter plot"""
    
    # Filter out cumulative PNC_st values less than -100
    filtered_data = merged_data[merged_data['cumulative_income_pnc'] >= -100].copy()
    
    # Remove NaN and infinite values
    valid_mask = (np.isfinite(filtered_data['cumulative_income_pnc']) & 
                  np.isfinite(filtered_data['gini_change']) & 
                  np.isfinite(filtered_data['color_variable']) &
                  ~np.isnan(filtered_data['cumulative_income_pnc']) & 
                  ~np.isnan(filtered_data['gini_change']) &
                  ~np.isnan(filtered_data['color_variable']))
    
    clean_data = filtered_data[valid_mask].copy()
    
    print(f"\n{'='*60}")
    print(f"GINI INEQUALITY SCATTER PLOT")
    print(f"{'='*60}")
    print(f"Valid data points (after filtering cumulative PNC_st >= -100): {len(clean_data)}")
    
    if len(clean_data) == 0:
        print("No valid data points to plot!")
        return
    
    # Print data statistics
    print(f"X-axis (Cumulative Income PNC_st):")
    print(f"  - Mean: {clean_data['cumulative_income_pnc'].mean():.6f}")
    print(f"  - Std: {clean_data['cumulative_income_pnc'].std():.6f}")
    print(f"  - Min: {clean_data['cumulative_income_pnc'].min():.6f}")
    print(f"  - Max: {clean_data['cumulative_income_pnc'].max():.6f}")
    
    print(f"Y-axis (Gini Change):")
    print(f"  - Mean: {clean_data['gini_change'].mean():.6f}")
    print(f"  - Std: {clean_data['gini_change'].std():.6f}")
    print(f"  - Min: {clean_data['gini_change'].min():.6f}")
    print(f"  - Max: {clean_data['gini_change'].max():.6f}")
    
    print(f"Color variable (Population Growth):")
    print(f"  - Mean: {clean_data['color_variable'].mean():.6f}")
    print(f"  - Std: {clean_data['color_variable'].std():.6f}")
    print(f"  - Min: {clean_data['color_variable'].min():.6f}")
    print(f"  - Max: {clean_data['color_variable'].max():.6f}")
    
    # Calculate correlation
    correlation = np.corrcoef(clean_data['cumulative_income_pnc'], clean_data['gini_change'])[0, 1]
    print(f"Correlation coefficient: {correlation:.6f}")
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        clean_data['cumulative_income_pnc'], clean_data['gini_change']
    )
    r_squared = r_value**2
    
    print(f"Linear regression statistics:")
    print(f"  - Slope: {slope:.6f}")
    print(f"  - R²: {r_squared:.6f}")
    print(f"  - P-value: {p_value:.6f}")
    
    # Calculate 95% confidence interval for regression line
    def calculate_prediction_interval(x, y, x_pred, confidence=0.95):
        """Calculate prediction interval for linear regression"""
        n = len(x)
        x_mean = np.mean(x)
        
        # Calculate standard error of prediction
        residuals = y - (slope * x + intercept)
        mse = np.sum(residuals**2) / (n - 2)
        
        # Standard error for prediction
        se_pred = np.sqrt(mse * (1 + 1/n + (x_pred - x_mean)**2 / np.sum((x - x_mean)**2)))
        
        # t-value for confidence interval
        t_val = stats.t.ppf((1 + confidence) / 2, n - 2)
        
        # Prediction interval
        y_pred = slope * x_pred + intercept
        margin = t_val * se_pred
        
        return y_pred, y_pred - margin, y_pred + margin
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Create custom colormap
    custom_cmap = create_custom_colormap()
    
    # Determine color limits to center white at zero
    color_abs_max = max(abs(clean_data['color_variable'].min()), abs(clean_data['color_variable'].max()))
    vmin = -color_abs_max
    vmax = color_abs_max
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(clean_data['cumulative_income_pnc'], clean_data['gini_change'], 
                        c=clean_data['color_variable'], cmap=custom_cmap, 
                        alpha=0.8, s=50, edgecolors=outline_color, linewidth=0.8,
                        vmin=vmin, vmax=vmax)
    
    # Add regression line with 95% confidence interval
    x_range = np.linspace(clean_data['cumulative_income_pnc'].min(), 
                         clean_data['cumulative_income_pnc'].max(), 100)
    y_pred, y_lower, y_upper = calculate_prediction_interval(
        clean_data['cumulative_income_pnc'].values, 
        clean_data['gini_change'].values, 
        x_range
    )
    
    # Plot confidence interval first (so it's behind the line)
    ax.fill_between(x_range, y_lower, y_upper, alpha=0.2, color='#3D3D3D', label='95% CI')
    
    # Plot regression line
    ax.plot(x_range, y_pred, color='#3D3D3D', linewidth=2, alpha=0.8, linestyle='-')
    
    # Add statistics text box in top left
    stats_text = f'Slope: {slope:.4f}\nR² = {r_squared:.4f}\np = {p_value:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Relative Population Growth', fontsize=12, color='#333333')
    cbar.ax.tick_params(labelsize=10, colors='#333333')
    
    # Add reference lines
    ax.axhline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)
    
    # Set labels
    ax.set_xlabel('Cumulative Income PNC_st', fontsize=14, color='#333333')
    ax.set_ylabel('Change in Gini Coefficient', fontsize=14, color='#333333')
    
    # Style the plot
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CCCCCC')
    
    ax.tick_params(axis='x', which='major', labelsize=12, colors='#333333')
    ax.tick_params(axis='y', which='major', labelsize=12, colors='#333333')
    
    # Set x-axis limit similar to other plots
    ax.set_xlim(left=-75)
    
    # Save plot
    plt.tight_layout()
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, "inequality_scatter_plots"), exist_ok=True)
    filename = os.path.join(BASE_OUTPUT_DIR, "inequality_scatter_plots/gini_change_vs_income_pnc.pdf")
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")

def create_gini_scatter_plot_tract(merged_data):
    """Create the tract-level Gini scatter plot"""
    
    # Filter out cumulative PNC_st values less than -100
    filtered_data = merged_data[merged_data['cumulative_income_pnc'] >= -100].copy()
    
    # Remove NaN and infinite values
    valid_mask = (np.isfinite(filtered_data['cumulative_income_pnc']) & 
                  np.isfinite(filtered_data['gini_change']) & 
                  np.isfinite(filtered_data['color_variable']) &
                  ~np.isnan(filtered_data['cumulative_income_pnc']) & 
                  ~np.isnan(filtered_data['gini_change']) &
                  ~np.isnan(filtered_data['color_variable']))
    
    clean_data = filtered_data[valid_mask].copy()
    
    print(f"\n{'='*60}")
    print(f"TRACT-LEVEL GINI INEQUALITY SCATTER PLOT")
    print(f"{'='*60}")
    print(f"Valid data points (after filtering cumulative PNC_st >= -100): {len(clean_data)}")
    
    if len(clean_data) == 0:
        print("No valid data points to plot!")
        return
    
    # Print data statistics
    print(f"X-axis (Cumulative Income PNC_st):")
    print(f"  - Mean: {clean_data['cumulative_income_pnc'].mean():.6f}")
    print(f"  - Std: {clean_data['cumulative_income_pnc'].std():.6f}")
    print(f"  - Min: {clean_data['cumulative_income_pnc'].min():.6f}")
    print(f"  - Max: {clean_data['cumulative_income_pnc'].max():.6f}")
    
    print(f"Y-axis (Gini Change):")
    print(f"  - Mean: {clean_data['gini_change'].mean():.6f}")
    print(f"  - Std: {clean_data['gini_change'].std():.6f}")
    print(f"  - Min: {clean_data['gini_change'].min():.6f}")
    print(f"  - Max: {clean_data['gini_change'].max():.6f}")
    
    print(f"Color variable (Population Growth):")
    print(f"  - Mean: {clean_data['color_variable'].mean():.6f}")
    print(f"  - Std: {clean_data['color_variable'].std():.6f}")
    print(f"  - Min: {clean_data['color_variable'].min():.6f}")
    print(f"  - Max: {clean_data['color_variable'].max():.6f}")
    
    # Calculate correlation
    correlation = np.corrcoef(clean_data['cumulative_income_pnc'], clean_data['gini_change'])[0, 1]
    print(f"Correlation coefficient: {correlation:.6f}")
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        clean_data['cumulative_income_pnc'], clean_data['gini_change']
    )
    r_squared = r_value**2
    
    print(f"Linear regression statistics:")
    print(f"  - Slope: {slope:.6f}")
    print(f"  - R²: {r_squared:.6f}")
    print(f"  - P-value: {p_value:.6f}")
    
    # Calculate 95% confidence interval for regression line
    def calculate_prediction_interval(x, y, x_pred, confidence=0.95):
        """Calculate prediction interval for linear regression"""
        n = len(x)
        x_mean = np.mean(x)
        
        # Calculate standard error of prediction
        residuals = y - (slope * x + intercept)
        mse = np.sum(residuals**2) / (n - 2)
        
        # Standard error for prediction
        se_pred = np.sqrt(mse * (1 + 1/n + (x_pred - x_mean)**2 / np.sum((x - x_mean)**2)))
        
        # t-value for confidence interval
        t_val = stats.t.ppf((1 + confidence) / 2, n - 2)
        
        # Prediction interval
        y_pred = slope * x_pred + intercept
        margin = t_val * se_pred
        
        return y_pred, y_pred - margin, y_pred + margin
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Create custom colormap
    custom_cmap = create_custom_colormap()
    
    # Determine color limits to center white at zero
    color_abs_max = max(abs(clean_data['color_variable'].min()), abs(clean_data['color_variable'].max()))
    vmin = -color_abs_max
    vmax = color_abs_max
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(clean_data['cumulative_income_pnc'], clean_data['gini_change'], 
                        c=clean_data['color_variable'], cmap=custom_cmap, 
                        alpha=0.8, s=50, edgecolors=outline_color, linewidth=0.8,
                        vmin=vmin, vmax=vmax)
    
    # Add regression line with 95% confidence interval
    x_range = np.linspace(clean_data['cumulative_income_pnc'].min(), 
                         clean_data['cumulative_income_pnc'].max(), 100)
    y_pred, y_lower, y_upper = calculate_prediction_interval(
        clean_data['cumulative_income_pnc'].values, 
        clean_data['gini_change'].values, 
        x_range
    )
    
    # Plot confidence interval first (so it's behind the line)
    ax.fill_between(x_range, y_lower, y_upper, alpha=0.2, color='#3D3D3D', label='95% CI')
    
    # Plot regression line
    ax.plot(x_range, y_pred, color='#3D3D3D', linewidth=2, alpha=0.8, linestyle='-')
    
    # Add statistics text box in top left
    stats_text = f'Slope: {slope:.4f}\nR² = {r_squared:.4f}\np = {p_value:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Relative Population Growth', fontsize=12, color='#333333')
    cbar.ax.tick_params(labelsize=10, colors='#333333')
    
    # Add reference lines
    ax.axhline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)
    
    # Set labels
    ax.set_xlabel('Cumulative Income PNC_st', fontsize=14, color='#333333')
    ax.set_ylabel('Change in Gini Coefficient (Tract Level)', fontsize=14, color='#333333')
    
    # Style the plot
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CCCCCC')
    
    ax.tick_params(axis='x', which='major', labelsize=12, colors='#333333')
    ax.tick_params(axis='y', which='major', labelsize=12, colors='#333333')
    
    # Set x-axis limit similar to other plots
    # ax.set_xlim(left=-75)
    
    # Save plot
    plt.tight_layout()
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, "inequality_scatter_plots"), exist_ok=True)
    filename = os.path.join(BASE_OUTPUT_DIR, "inequality_scatter_plots/gini_change_vs_income_pnc_tract.pdf")
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")

def create_transmitted_vs_bg_gini_scatter_plot(merged_data):
    """Create the Transmitted Selection vs Block Group Gini scatter plot"""
    
    # Remove NaN and infinite values
    valid_mask = (np.isfinite(merged_data['transmitted_pnc']) & 
                  np.isfinite(merged_data['gini_change']) & 
                  np.isfinite(merged_data['color_variable']) &
                  ~np.isnan(merged_data['transmitted_pnc']) & 
                  ~np.isnan(merged_data['gini_change']) &
                  ~np.isnan(merged_data['color_variable']))
    
    clean_data = merged_data[valid_mask].copy()
    
    print(f"\n{'='*60}")
    print(f"TRANSMITTED VS BG GINI SCATTER PLOT")
    print(f"{'='*60}")
    print(f"Valid data points: {len(clean_data)}")
    
    if len(clean_data) == 0:
        print("No valid data points to plot!")
        return
        
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        clean_data['transmitted_pnc'], clean_data['gini_change']
    )
    r_squared = r_value**2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Create scatter plot
    scatter = ax.scatter(clean_data['transmitted_pnc'], clean_data['gini_change'], 
                        c=clean_data['color_variable'], cmap=create_custom_colormap(), 
                        alpha=0.8, s=50, edgecolors=outline_color, linewidth=0.8,
                        vmin=-max(abs(clean_data['color_variable'].min()), abs(clean_data['color_variable'].max())), 
                        vmax=max(abs(clean_data['color_variable'].min()), abs(clean_data['color_variable'].max())))

    # Add regression line with 95% confidence interval
    x_range = np.linspace(clean_data['transmitted_pnc'].min(), clean_data['transmitted_pnc'].max(), 100)
    y_pred = slope * x_range + intercept
    ax.plot(x_range, y_pred, color='#3D3D3D', linewidth=2, alpha=0.8, linestyle='-')
    
    # Add stats text
    stats_text = f'Slope: {slope:.4f}\nR² = {r_squared:.4f}\np = {p_value:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Transmitted Income PNC_st', fontsize=14, color='#333333')
    ax.set_ylabel('Change in Gini Coefficient (Block Group)', fontsize=14, color='#333333')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "inequality_scatter_plots/transmitted_vs_bg_gini.pdf"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: plots/inequality_scatter_plots/transmitted_vs_bg_gini.pdf")

def create_selection_vs_tr_gini_scatter_plot(merged_data):
    """Create the Selection vs Tract Gini scatter plot"""
    
    # Remove NaN and infinite values
    valid_mask = (np.isfinite(merged_data['selection_pnc']) & 
                  np.isfinite(merged_data['gini_change']) & 
                  np.isfinite(merged_data['color_variable']) &
                  ~np.isnan(merged_data['selection_pnc']) & 
                  ~np.isnan(merged_data['gini_change']) &
                  ~np.isnan(merged_data['color_variable']))
    
    clean_data = merged_data[valid_mask].copy()
    
    print(f"\n{'='*60}")
    print(f"SELECTION VS TRACT GINI SCATTER PLOT")
    print(f"{'='*60}")
    print(f"Valid data points: {len(clean_data)}")
    
    if len(clean_data) == 0:
        print("No valid data points to plot!")
        return
        
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        clean_data['selection_pnc'], clean_data['gini_change']
    )
    r_squared = r_value**2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Create scatter plot
    scatter = ax.scatter(clean_data['selection_pnc'], clean_data['gini_change'], 
                        c=clean_data['color_variable'], cmap=create_custom_colormap(), 
                        alpha=0.8, s=50, edgecolors=outline_color, linewidth=0.8,
                        vmin=-max(abs(clean_data['color_variable'].min()), abs(clean_data['color_variable'].max())), 
                        vmax=max(abs(clean_data['color_variable'].min()), abs(clean_data['color_variable'].max())))

    # Add regression line with 95% confidence interval
    x_range = np.linspace(clean_data['selection_pnc'].min(), clean_data['selection_pnc'].max(), 100)
    y_pred = slope * x_range + intercept
    ax.plot(x_range, y_pred, color='#3D3D3D', linewidth=2, alpha=0.8, linestyle='-')
    
    # Add stats text
    stats_text = f'Slope: {slope:.4f}\nR² = {r_squared:.4f}\np = {p_value:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    ax.set_xlabel('Selection Income PNC_st', fontsize=14, color='#333333')
    ax.set_ylabel('Change in Gini Coefficient (Tract Level)', fontsize=14, color='#333333')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "inequality_scatter_plots/selection_vs_tr_gini.pdf"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: plots/inequality_scatter_plots/selection_vs_tr_gini.pdf")

# --- Main execution ---
if __name__ == "__main__":
    print("Starting Gini inequality scatter analysis...")
    
    # Load data
    cm_data, bg_data, tr_data = load_data()
    if cm_data is None or bg_data is None or tr_data is None:
        print("Error: Could not load required data files.")
        exit(1)
    
    # Extract cumulative income PNC_st from community data (shared for both plots)
    cumulative_income_pnc = extract_cumulative_income_pnc(cm_data)
    if cumulative_income_pnc is None:
        print("Error: Could not extract cumulative income PNC_st.")
        exit(1)
    
    # Get population growth for coloring (shared for both plots)
    pop_growth_col = None
    for col in ['RelPopG_cm', 'PopG_cm', 'LogAvgIncInitial_cm']:
        if col in cm_data.columns:
            pop_growth_col = col
            break
    
    if pop_growth_col is None:
        print("Warning: Could not find population growth column, using zeros")
        color_variable = pd.Series(0, index=cm_data.index)
    else:
        color_variable = cm_data[pop_growth_col] - cm_data[pop_growth_col].mean()
        print(f"Using color variable: {pop_growth_col} (de-meaned)")
    
    # Prepare community plot data (shared for both plots)
    if 'UnitName' in cm_data.columns:
        cm_plot_data = pd.DataFrame({
            'community': cm_data['UnitName'],
            'cumulative_income_pnc': cumulative_income_pnc,
            'color_variable': color_variable
        })
    else:
        print("Error: UnitName column not found in community data")
        exit(1)
    
    # === BLOCK GROUP LEVEL ANALYSIS ===
    print("\n" + "="*60)
    print("PROCESSING BLOCK GROUP LEVEL DATA")
    print("="*60)
    
    # Prepare block group income data for Gini calculation
    bg_income_data = prepare_bg_income_data(bg_data)
    if bg_income_data is not None:
        # Calculate Gini changes by community for block groups
        bg_gini_results = calculate_community_gini_changes(bg_income_data)
        if len(bg_gini_results) > 0:
            # Merge with community data for block group plot
            bg_merged_data = pd.merge(
                bg_gini_results, 
                cm_plot_data, 
                left_on='ParentCommunity', 
                right_on='community', 
                how='inner'
            )
            
            print(f"Successfully merged block group data for {len(bg_merged_data)} communities")
            
            # Create the block group scatter plot
            create_gini_scatter_plot(bg_merged_data)
        else:
            print("Error: No valid block group Gini calculations.")
    else:
        print("Error: Could not prepare block group income data.")
    
    # === TRACT LEVEL ANALYSIS ===
    print("\n" + "="*60)
    print("PROCESSING TRACT LEVEL DATA")
    print("="*60)
    
    # Prepare tract income data for Gini calculation
    tr_income_data = prepare_tr_income_data(tr_data)
    if tr_income_data is not None:
        # Calculate Gini changes by community for tracts
        tr_gini_results = calculate_community_gini_changes(tr_income_data)
        if len(tr_gini_results) > 0:
            # Merge with community data for tract plot
            tr_merged_data = pd.merge(
                tr_gini_results, 
                cm_plot_data, 
                left_on='ParentCommunity', 
                right_on='community', 
                how='inner'
            )
            
            print(f"Successfully merged tract data for {len(tr_merged_data)} communities")
            
            # Create the tract scatter plot
            create_gini_scatter_plot_tract(tr_merged_data)
        else:
            print("Error: No valid tract Gini calculations.")
    else:
        print("Error: Could not prepare tract income data.")

    # === NEW PLOTS: TRANSMITTED vs BG and SELECTION VS TR ===
    print("\n" + "="*60)
    print("PROCESSING NEW PLOTS: TRANSMITTED vs BG GINI, SELECTION vs TR GINI")
    print("="*60)

    # Filter cm_data based on original cumulative PNC condition
    cm_data_filtered = cm_data[cumulative_income_pnc >= -100].copy()
    print(f"Filtered community data to {len(cm_data_filtered)} rows based on cumulative_income_pnc >= -100")
    
    # Re-calculate color variable for filtered data
    if pop_growth_col:
        color_variable_filtered = cm_data_filtered[pop_growth_col] - cm_data_filtered[pop_growth_col].mean()
    else:
        color_variable_filtered = pd.Series(0, index=cm_data_filtered.index)

    # Extract transmitted and selection terms from filtered data
    transmitted_pnc = extract_transmitted_income_pnc(cm_data_filtered)
    selection_pnc = extract_selection_income_pnc(cm_data_filtered)

    # Note: Gini results for BG and TR are already calculated and can be reused.

    # PLOT 1: Transmitted vs BG Gini
    if transmitted_pnc is not None and bg_gini_results is not None:
        cm_plot_data_transmitted = pd.DataFrame({
            'community': cm_data_filtered['UnitName'],
            'transmitted_pnc': transmitted_pnc,
            'color_variable': color_variable_filtered
        })
        transmitted_merged_data = pd.merge(
            bg_gini_results, 
            cm_plot_data_transmitted, 
            left_on='ParentCommunity', 
            right_on='community', 
            how='inner'
        )
        create_transmitted_vs_bg_gini_scatter_plot(transmitted_merged_data)
    
    # PLOT 2: Selection vs TR Gini
    if selection_pnc is not None and tr_gini_results is not None:
        cm_plot_data_selection = pd.DataFrame({
            'community': cm_data_filtered['UnitName'],
            'selection_pnc': selection_pnc,
            'color_variable': color_variable_filtered
        })
        selection_merged_data = pd.merge(
            tr_gini_results, 
            cm_plot_data_selection, 
            left_on='ParentCommunity', 
            right_on='community', 
            how='inner'
        )
        create_selection_vs_tr_gini_scatter_plot(selection_merged_data)
    
    print("\n" + "="*60)
    print("GINI INEQUALITY SCATTER ANALYSIS COMPLETE")
    print("="*60)
    print("Generated plots:")
    print("  - Block Group: plots/inequality_scatter_plots/gini_change_vs_income_pnc.pdf")
    print("  - Tract Level: plots/inequality_scatter_plots/gini_change_vs_income_pnc_tract.pdf") 
    print("  - Transmitted vs BG Gini: plots/inequality_scatter_plots/transmitted_vs_bg_gini.pdf")
    print("  - Selection vs TR Gini: plots/inequality_scatter_plots/selection_vs_tr_gini.pdf") 