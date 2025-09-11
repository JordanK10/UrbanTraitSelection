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
from scipy.stats import t
import statsmodels.api as sm
warnings.filterwarnings('ignore')

# Custom colors matching other plots
custom_purple = '#633673'
custom_orange = '#E77429'
outline_color = '#3D3D3D'
purple_dot = '#caa8d6'
orange_dot = '#f2b58c'

def load_data():
    """Load community, block group, and tract data"""
    try:
        # Load community data for cumulative PNC_st and population growth
        cm_data = pd.read_csv("output_terms/bg_cm_exported_terms.csv")
        print(f"Successfully loaded community data: {len(cm_data)} rows")
        
        # Load block group data for Gini calculation
        bg_data = pd.read_csv("output_terms/bg_bg_exported_terms.csv")
        print(f"Successfully loaded block group data: {len(bg_data)} rows")
        
        # Load tract data for Gini calculation
        tr_data = pd.read_csv("output_terms/bg_tr_exported_terms.csv")
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

def calculate_gini_changes(income_data, group_by_col):
    """
    Calculate Gini coefficient change for each group defined by group_by_col.
    
    Returns DataFrame with columns: [group_by_col], initial_gini, final_gini, gini_change
    """
    print(f"Calculating Gini coefficients by {group_by_col}...")
    
    results = []
    
    if income_data is None or group_by_col not in income_data.columns:
        print(f"Error: Cannot calculate Gini changes. Input data is None or missing group_by column '{group_by_col}'.")
        return pd.DataFrame()

    for group_name, group in income_data.groupby(group_by_col):
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
        gini_change = final_gini-initial_gini if pd.notna(final_gini) and pd.notna(initial_gini) else np.nan
        
        results.append({
            group_by_col: group_name,
            'initial_gini': initial_gini,
            'final_gini': final_gini,
            'gini_change': gini_change,
            'num_members': len(group)
        })
    
    result_df = pd.DataFrame(results)
    print(f"Calculated Gini changes for {len(result_df)} groups.")
    
    # Print some statistics
    valid_changes = result_df['gini_change'].dropna()
    if len(valid_changes) > 0:
        print(f"Gini change statistics:")
        print(f"  Mean: {valid_changes.mean():.4f}")
        print(f"  Std: {valid_changes.std():.4f}")
        print(f"  Min: {valid_changes.min():.4f}")
        print(f"  Max: {valid_changes.max():.4f}")
    
    return result_df

def prepare_bg_income_data_by_tract(bg_data):
    """
    Prepare block group data for Gini calculation, for grouping by tracts.
    """
    print("Preparing block group income data (for tract grouping)...")
    
    log_initial_income_col, log_final_income_col = None, None
    for col in bg_data.columns:
        if 'LogAvgIncInitial' in col:
            log_initial_income_col = col
        elif 'LogAvgIncFinal' in col:
            log_final_income_col = col
    
    if log_initial_income_col is None or log_final_income_col is None:
        print(f"Warning: Could not find log income columns in block group data. Available columns: {bg_data.columns.tolist()}")
        return None
    
    bg_prepared = bg_data.copy()
    
    if 'ParentTract' not in bg_prepared.columns:
        print("Error: ParentTract column not found in block group data")
        return None
    
    bg_prepared['initial_income'] = np.exp(bg_prepared[log_initial_income_col])
    bg_prepared['final_income'] = np.exp(bg_prepared[log_final_income_col])
    
    pop_col = 'PopInitial_bg' if 'PopInitial_bg' in bg_prepared.columns else None
    if pop_col:
        bg_prepared['weight'] = bg_prepared[pop_col]
    else:
        bg_prepared['weight'] = 1

    result = bg_prepared[['ParentTract', 'initial_income', 'final_income', 'weight']].copy()
    result = result.dropna(subset=['ParentTract'])
    result = result[(result['initial_income'] > 0) & (result['final_income'] > 0)]
    
    print(f"Prepared {len(result)} block group records for Gini calculation (grouped by tract)")
    return result

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

def create_scatter_plot(data, x_col, y_col, color_col, title, xlabel, ylabel, output_filename, 
                      size_col=None, color_label='Color Variable', filter_col=None, filter_threshold=None, bifurcate_on_color=False, ylim=None, xlim=None):
    """
    Generates a scatter plot with a linear regression fit and confidence interval.

    Args:
        data (pd.DataFrame): The dataframe containing the plot data.
        x_col (str): The name of the column for the x-axis.
        y_col (str): The name of the column for the y-axis.
        color_col (str): The name of the column for coloring points.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        output_filename (str): The path to save the output PDF file.
        size_col (str, optional): The column to scale point sizes by and use for weighting regression.
        color_label (str, optional): The label for the color bar.
        filter_col (str, optional): The column to filter data on. Defaults to None.
        bifurcate_on_color (bool, optional): If True, splits data by color value for separate fits.
    """
    # 1. Filter data
    if filter_col and filter_threshold is not None:
        plot_data = data[data[filter_col] >= filter_threshold].copy()
    else:
        plot_data = data.copy()

    if plot_data.empty:
        print(f"Warning: No data to plot for {title} after filtering. Skipping plot.")
        return

    # Drop rows with NaN in essential columns for plotting and regression
    essential_cols = [x_col, y_col, color_col]
    if size_col:
        essential_cols.append(size_col)
    
    plot_data.dropna(subset=essential_cols, inplace=True)

    if plot_data.empty:
        print(f"Warning: No data to plot for {title} after dropping NaNs. Skipping plot.")
        return

    # 2. Prepare point sizes
    if size_col and size_col in plot_data.columns:
        pop = plot_data[size_col]
        s_min, s_max = 15, 200
        # Handle case where all population values are the same
        if pop.min() == pop.max():
            sizes = pd.Series(s_min, index=plot_data.index)
        else:
            sizes = s_min + ((pop - pop.min()) / (pop.max() - pop.min()) * (s_max - s_min))
        plot_data['point_size'] = sizes
    else:
        plot_data['point_size'] = 50 # Default size

    # 3. Setup plot
    fig, ax = plt.subplots(figsize=(3.25, 2.9))
    fig.patch.set_facecolor('white')

    
    for spine in ax.spines.values():
        spine.set_edgecolor(outline_color)
        spine.set_linewidth(1.5)

    # 4. Bifurcate data for coloring and fitting
    low_income_data = plot_data[plot_data[color_col] <= 0]
    high_income_data = plot_data[plot_data[color_col] > 0]

    # 5. Scatter plot (with bifurcated colors and scaled sizes)
    ax.scatter(low_income_data[x_col], low_income_data[y_col], color=purple_dot, alpha=0.25, s=low_income_data['point_size'], edgecolors=outline_color, linewidth=0.8, label='Below/Eq. Median Income')
    ax.scatter(high_income_data[x_col], high_income_data[y_col], color=orange_dot, alpha=0.25, s=high_income_data['point_size'], edgecolors=outline_color, linewidth=0.8, label='Above Median Income')
    
    # --- Fit Lines and Stats Text Box ---
    # Full data fit
    y = plot_data[y_col]
    X = sm.add_constant(plot_data[[x_col]])
    weights = plot_data[size_col] if size_col else None
    
    wls_model = sm.WLS(y, X, weights=weights)
    wls_fit = wls_model.fit()
    
    slope = wls_fit.params[x_col]
    intercept = wls_fit.params['const']
    r_squared = wls_fit.rsquared
    p_value = wls_fit.pvalues[x_col]

    stats_text = f'Full Fit:\n  m={slope:.3f}, R²={r_squared:.3f}'

    # Plot the main fit line for the full dataset
    x_pred_full = pd.DataFrame({x_col: np.linspace(plot_data[x_col].min(), plot_data[x_col].max(), 100)})
    x_pred_full_sm = sm.add_constant(x_pred_full, has_constant='add')
    y_pred_full = wls_fit.predict(x_pred_full_sm)
    ax.plot(x_pred_full[x_col], y_pred_full, color='#3D3D3D', linestyle='-', linewidth=2, label='Full Fit')
    
    # Confidence Interval for full fit
    pred_full = wls_fit.get_prediction(x_pred_full_sm)
    ci_full_df = pred_full.summary_frame(alpha=0.05)
    ax.fill_between(x_pred_full[x_col], ci_full_df['mean_ci_lower'], ci_full_df['mean_ci_upper'], color='#B0B0B0', alpha=0.3)

    if bifurcate_on_color:
        # Fit for relatively growing subset
        if len(high_income_data) > 2:
            y_high = high_income_data[y_col]
            X_high = sm.add_constant(high_income_data[[x_col]])
            weights_high = high_income_data[size_col] if size_col else None

            wls_model_high = sm.WLS(y_high, X_high, weights=weights_high)
            wls_fit_high = wls_model_high.fit()
            
            x_pred_high = pd.DataFrame({x_col: np.linspace(high_income_data[x_col].min(), high_income_data[x_col].max(), 100)})
            x_pred_high_sm = sm.add_constant(x_pred_high, has_constant='add')
            y_pred_high = wls_fit_high.predict(x_pred_high_sm)
            ax.plot(x_pred_high[x_col], y_pred_high, color=custom_orange, linestyle='--', linewidth=2)
            
            pred_high = wls_fit_high.get_prediction(x_pred_high_sm)
            ci_high_df = pred_high.summary_frame(alpha=0.05)
            ax.fill_between(x_pred_high[x_col], ci_high_df['mean_ci_lower'], ci_high_df['mean_ci_upper'], color=custom_orange, alpha=0.2)
            
            stats_text += f'\nAbove Median Fit:\n  m={wls_fit_high.params[x_col]:.3f}, R²={wls_fit_high.rsquared:.3f}'
        
        # Fit for relatively shrinking subset
        if len(low_income_data) > 2:
            y_low = low_income_data[y_col]
            X_low = sm.add_constant(low_income_data[[x_col]])
            weights_low = low_income_data[size_col] if size_col else None

            wls_model_low = sm.WLS(y_low, X_low, weights=weights_low)
            wls_fit_low = wls_model_low.fit()

            x_pred_low = pd.DataFrame({x_col: np.linspace(low_income_data[x_col].min(), low_income_data[x_col].max(), 100)})
            x_pred_low_sm = sm.add_constant(x_pred_low, has_constant='add')
            y_pred_low = wls_fit_low.predict(x_pred_low_sm)
            ax.plot(x_pred_low[x_col], y_pred_low, color=custom_purple, linestyle='--', linewidth=2)

            pred_low = wls_fit_low.get_prediction(x_pred_low_sm)
            ci_low_df = pred_low.summary_frame(alpha=0.05)
            ax.fill_between(x_pred_low[x_col], ci_low_df['mean_ci_lower'], ci_low_df['mean_ci_upper'], color=custom_purple, alpha=0.2)
            
            stats_text += f'\nBelow/Eq. Median Fit:\n  m={wls_fit_low.params[x_col]:.3f}, R²={wls_fit_low.rsquared:.3f}'
        
    # ax.text(0.7, 0.3, stats_text, transform=ax.transAxes, fontsize=7,
    #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 8. Labels, title, grid, and style
    # ax.set_xlabel(xlabel, fontsize=14, color='#333333')
    # ax.set_ylabel(ylabel, fontsize=14, color='#333333')
    ax.axhline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CCCCCC')
    ax.tick_params(axis='x', which='major', labelsize=12, colors='#333333')
    ax.tick_params(axis='y', which='major', labelsize=12, colors='#333333')
    
    # 9. Legend
    # ax.legend(loc='lower right', fontsize=10)
    
    # if ylim:
    #     ax.set_ylim(bottom = ylim[0], top = ylim[1])
    # if xlim:
    #     ax.set_xlim(left = xlim[0], right = xlim[1])

    # 10. Save figure
    output_dir = "gini_scatter_plots"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_filename)
    
    plt.tight_layout()
    plt.savefig(full_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Plot saved to {full_path}")


def main():
    # Load data
    cm_data, bg_data, tr_data = load_data()
    if cm_data is None:
        return

    # --- Gini Calculations (Community-level) ---
    # Prepare BG income data and calculate Gini change (across BGs, grouped by community)
    bg_income_community_grouped = prepare_bg_income_data(bg_data)
    gini_change_bg_by_community = calculate_gini_changes(bg_income_community_grouped, 'ParentCommunity')
    if not gini_change_bg_by_community.empty:
        gini_change_bg_by_community.rename(columns={'gini_change': 'gini_change_bg', 'ParentCommunity': 'UnitName'}, inplace=True)

    # Prepare TR income data and calculate Gini change (across TRs, grouped by community)
    tr_income_community_grouped = prepare_tr_income_data(tr_data)
    gini_change_tr_by_community = calculate_gini_changes(tr_income_community_grouped, 'ParentCommunity')
    if not gini_change_tr_by_community.empty:
        gini_change_tr_by_community.rename(columns={'gini_change': 'gini_change_tr', 'ParentCommunity': 'UnitName'}, inplace=True)

    # --- Gini Calculations (Tract-level) ---
    # Prepare BG income data and calculate Gini change (across BGs, grouped by tract)
    bg_income_tract_grouped = prepare_bg_income_data_by_tract(bg_data)
    gini_change_bg_by_tract = calculate_gini_changes(bg_income_tract_grouped, 'ParentTract')
    if not gini_change_bg_by_tract.empty:
        gini_change_bg_by_tract.rename(columns={'gini_change': 'gini_change_bg_by_tract', 'ParentTract': 'UnitName'}, inplace=True)


    # --- Data Merging and Preparation ---
    # To ensure data alignment, we'll use a series of inner merges.
    # This guarantees that we only analyze communities where all data sources are present.
    if gini_change_bg_by_community.empty or gini_change_tr_by_community.empty:
        print("Error: Could not calculate community-level Gini changes. Aborting.")
        return
        
    # Merge community data with BG Gini, then with TR Gini
    all_cm_data = pd.merge(cm_data, gini_change_bg_by_community, on='UnitName', how='inner')
    all_cm_data = pd.merge(all_cm_data, gini_change_tr_by_community, on='UnitName', how='inner')

    # Add bifurcation column based on median initial income for community plots
    if 'LogAvgIncInitial_cm' in all_cm_data.columns:
        median_income_cm = all_cm_data['LogAvgIncInitial_cm'].median()
        all_cm_data['bifurcate_col_cm'] = all_cm_data['LogAvgIncInitial_cm'] > median_income_cm
        print(f"Community median initial log income: {median_income_cm:.4f}")
    else:
        print("Warning: 'LogAvgIncInitial_cm' not found for community bifurcation.")
        all_cm_data['bifurcate_col_cm'] = False

    if all_cm_data.empty:
        print("Error: No common communities found after merging all data sources. Cannot create plots.")
        return

    # Merge tract data with the new tract-level Gini results
    if not gini_change_bg_by_tract.empty and tr_data is not None:
        all_tr_data = pd.merge(tr_data, gini_change_bg_by_tract, on='UnitName', how='inner')
        # Add bifurcation column based on median initial income for tract plots
        if 'LogAvgIncInitial_tr' in all_tr_data.columns:
            median_income_tr = all_tr_data['LogAvgIncInitial_tr'].median()
            all_tr_data['bifurcate_col_tr'] = all_tr_data['LogAvgIncInitial_tr'] > median_income_tr
            print(f"Tract median initial log income: {median_income_tr:.4f}")
        else:
            print("Warning: 'LogAvgIncInitial_tr' not found for tract bifurcation.")
            all_tr_data['bifurcate_col_tr'] = False
    else:
        all_tr_data = pd.DataFrame()
        print("Warning: Could not create merged tract-level data for new plot.")


    # Extract PNC components from the fully merged and aligned community data
    all_cm_data['cumulative_income_pnc'] = extract_cumulative_income_pnc(all_cm_data)/100
    all_cm_data['transmitted_income_pnc'] = extract_transmitted_income_pnc(all_cm_data)/100
    all_cm_data['selection_income_pnc'] = extract_selection_income_pnc(all_cm_data)/100

    # Define plot configurations
    output_dir = "gini_scatter_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for PopInitial_cm column for weighting/sizing
    if 'PopInitial_cm' not in all_cm_data.columns:
        print("Warning: 'PopInitial_cm' not found in data. Regressions will not be weighted.")
        size_col_cm = None
    else:
        size_col_cm = 'PopInitial_cm'

    plot_configs = [
        {
            'data': all_cm_data,
            'x': 'cumulative_income_pnc',
            'y': 'gini_change_bg',
            'title': 'Cumulative PNC vs. Block Group Gini Change',
            'xlabel': 'Cumulative Income PNC (Community)',
            'ylabel': 'Change in Block Group Gini Coefficient',
            'color': 'bifurcate_col_cm',
            'color_label': 'Initial Income vs. Median',
            'output': os.path.join(output_dir, 'cumulative_pnc_vs_bg_gini.pdf'),
            'filter_col': 'cumulative_income_pnc',
            'filter_threshold': -500,
            'bifurcate_on_color': True,
            'size_col': size_col_cm,
            'ylim': None,
            'xlim': None
        },
        {
            'data': all_cm_data,
            'x': 'cumulative_income_pnc',
            'y': 'gini_change_tr',
            'title': 'Cumulative PNC vs. Tract Gini Change',
            'xlabel': 'Cumulative Income PNC (Community)',
            'ylabel': 'Change in Tract Gini Coefficient',
            'color': 'bifurcate_col_cm',
            'color_label': 'Initial Income vs. Median',
            'output': os.path.join(output_dir, 'cumulative_pnc_vs_tr_gini.pdf'),
            'filter_col': 'cumulative_income_pnc',
            'filter_threshold': -500,
            'bifurcate_on_color': True,
            'size_col': size_col_cm,
            'ylim': None,
            'xlim': None
        },
        {
            'data': all_cm_data,
            'x': 'transmitted_income_pnc',
            'y': 'gini_change_bg',
            'title': 'Transmitted PNC vs. Block Group Gini Change',
            'xlabel': 'Transmitted Income PNC (Community)',
            'ylabel': 'Change in Block Group Gini Coefficient',
            'color': 'bifurcate_col_cm',
            'color_label': 'Initial Income vs. Median',
            'output': os.path.join(output_dir, 'transmitted_pnc_vs_bg_gini.pdf'),
            'filter_col': 'cumulative_income_pnc',
            'filter_threshold': -500,
            'bifurcate_on_color': True,
            'size_col': size_col_cm,
            'ylim': None,
            'xlim': None
        },
        {
            'data': all_cm_data,
            'x': 'selection_income_pnc',
            'y': 'gini_change_tr',
            'title': 'Selection PNC vs. Tract Gini Change',
            'xlabel': 'Selection Income PNC (Community)',
            'ylabel': 'Change in Tract Gini Coefficient',
            'color': 'bifurcate_col_cm',
            'color_label': 'Initial Income vs. Median',
            'output': os.path.join(output_dir, 'selection_pnc_vs_tr_gini.pdf'),
            'filter_col': 'cumulative_income_pnc',
            'filter_threshold': -500,
            'bifurcate_on_color': True,
            'size_col': size_col_cm,
            'ylim': [-.085,.08],
            'xlim': [-.4,.2]
        }
    ]

    # Add the new tract-level plots if data is available
    if not all_tr_data.empty:
        if 'PopInitial_tr' not in all_tr_data.columns:
            print("Warning: 'PopInitial_tr' not found in tract data. New plots will not be weighted.")
            size_col_tr = None
        else:
            size_col_tr = 'PopInitial_tr'

        # Plot 6: Tract Selection vs. Change in BG Gini
        selection_col_tr = 'Sel_tr_from_bg_inc'
        if selection_col_tr in all_tr_data.columns and 'gini_change_bg_by_tract' in all_tr_data.columns:
            plot_configs.append({
                'data': all_tr_data,
                'x': selection_col_tr,
                'y': 'gini_change_bg_by_tract',
                'title': 'Tract Selection vs. Change in BG Gini',
                'xlabel': 'Selection Term (Tract, from BGs)',
                'ylabel': 'Change in Gini Coefficient (from BGs)',
                'color': 'bifurcate_col_tr',
                'color_label': 'Initial Income vs. Median',
                'output': os.path.join(output_dir, 'tract_selection_vs_bg_gini.pdf'),
                'filter_col': None,
                'filter_threshold': None,
                'bifurcate_on_color': True,
                'size_col': size_col_tr,
                'ylim': [-.085,.08],
                'xlim': None
            })
        else:
            print(f"Warning: Skipping 'Tract Selection vs. Change in BG Gini' plot. Missing columns: '{selection_col_tr}' or 'gini_change_bg_by_tract'.")

    # Plot 6: Tract Selection PNC vs. Change in BG Gini
    selection_pnc_col_tr = 'Sel_tr_from_bg_inc_PNC_st'
    if selection_pnc_col_tr in all_tr_data.columns and 'gini_change_bg_by_tract' in all_tr_data.columns:
        # Prepare and scale the PNC column for consistency
        all_tr_data['selection_pnc_st_tr'] = all_tr_data[selection_pnc_col_tr] / 100
        
        plot_configs.append({
            'data': all_tr_data,
            'x': 'selection_pnc_st_tr',
            'y': 'gini_change_bg_by_tract',
            'title': 'Tract Selection PNC vs. Change in BG Gini',
            'xlabel': 'Selection PNC (Tract, from BGs, State-Normalized)',
            'ylabel': 'Change in Gini Coefficient (from BGs)',
            'color': 'bifurcate_col_tr',
            'color_label': 'Initial Income vs. Median',
            'output': os.path.join(output_dir, 'tract_selection_pnc_vs_bg_gini.pdf'),
            'filter_col': None,
            'filter_threshold': None,
            'bifurcate_on_color': True,
            'size_col': size_col_tr,
            'ylim': [-.085,.08],
            'xlim': None
        })
    else:
        print(f"Warning: Skipping 'Tract Selection PNC vs. Change in BG Gini' plot. Missing columns: '{selection_pnc_col_tr}' or 'gini_change_bg_by_tract'.")

    # Generate plots
    for config in plot_configs:
        # The filter is the same for all plots in this case
        create_scatter_plot(
            data=config['data'],
            x_col=config['x'],
            y_col=config['y'],
            color_col=config['color'],
            title=config['title'],
            xlabel=config['xlabel'],
            ylabel=config['ylabel'],
            output_filename=config['output'].split('/')[-1], # Use the filename from config
            color_label=config['color_label'],
            filter_col=config['filter_col'],
            filter_threshold=config['filter_threshold'],
            bifurcate_on_color=config.get("bifurcate_on_color", False),
            size_col=config.get("size_col"),
            ylim=config["ylim"],
            xlim=config["xlim"]
        )

if __name__ == "__main__":
    main() 