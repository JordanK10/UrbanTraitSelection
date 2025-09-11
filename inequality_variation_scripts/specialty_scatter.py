#!/usr/bin/env python3
"""
specialty_scatter.py

Creates specialty scatter plots comparing empirical growth rates with:
1. (Income-averaged growth rate) - (Empirical growth rate) 
2. (Population-averaged growth rate) - (Empirical growth rate)

Points for the same communities are connected with lines.
Generates separate plots for full MSA and each county.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Custom colors from plotPrice.py
custom_purple = '#633673'
custom_orange = '#E77429'
purple_dot = '#caa8d6'
orange_dot = '#f2b58c'

def load_data_by_level(level):
    """Load and prepare the data for a specific level."""
    print(f"Loading {level}-level data...")
    
    filename = f'output_terms/bg_{level}_exported_terms.csv'
    try:
        data = pd.read_csv(filename)
        print(f"Loaded {level} data: {data.shape[0]} rows, {data.shape[1]} columns")
        return data, level
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return None, level

def prepare_variables(data, level_suffix):
    """Extract and calculate the required variables from the data."""
    print(f"Preparing variables for {level_suffix} level...")
    
    # Create a working copy
    df = data.copy()
    
    # Empirical growth rate
    if f'AvgG_emp_{level_suffix}' in df.columns:
        df['empirical_growth_rate'] = df[f'AvgG_emp_{level_suffix}']
    else:
        emp_cols = [col for col in df.columns if 'AvgG_emp' in col]
        if emp_cols:
            df['empirical_growth_rate'] = df[emp_cols[0]]
            print(f"Using {emp_cols[0]} for empirical growth rate")
        else:
            print("Warning: Could not find empirical growth rate column")
            df['empirical_growth_rate'] = np.nan
    
    # Income-averaged growth rate
    if f'AvgG_inc_{level_suffix}' in df.columns:
        df['inc_averaged_growth_rate'] = df[f'AvgG_inc_{level_suffix}']
    else:
        inc_avg_cols = [col for col in df.columns if 'AvgG_inc' in col]
        if inc_avg_cols:
            df['inc_averaged_growth_rate'] = df[inc_avg_cols[0]]
            print(f"Using {inc_avg_cols[0]} for income averaged growth rate")
        else:
            print("Warning: Could not find income averaged growth rate")
            df['inc_averaged_growth_rate'] = np.nan
    
    # Population-averaged growth rate
    if f'AvgG_pop_{level_suffix}' in df.columns:
        df['pop_averaged_growth_rate'] = df[f'AvgG_pop_{level_suffix}']
    else:
        pop_avg_cols = [col for col in df.columns if 'AvgG_pop' in col]
        if pop_avg_cols:
            df['pop_averaged_growth_rate'] = df[pop_avg_cols[0]]
            print(f"Using {pop_avg_cols[0]} for population averaged growth rate")
        else:
            print("Warning: Could not find population averaged growth rate")
            df['pop_averaged_growth_rate'] = np.nan
    
    # Population growth rate (for coloring)
    if f'PopG_{level_suffix}' in df.columns:
        df['population_growth_rate'] = df[f'PopG_{level_suffix}']
    else:
        pop_growth_cols = [col for col in df.columns if 'PopG' in col or 'PopFracChange' in col]
        if pop_growth_cols:
            df['population_growth_rate'] = df[pop_growth_cols[0]]
            print(f"Using {pop_growth_cols[0]} for population growth rate")
        else:
            print("Warning: Could not find population growth rate")
            df['population_growth_rate'] = np.nan
    
    # Add county identifier - try different possible column names
    county_col = None
    possible_county_cols = ['ParentCounty', 'County', 'county', 'COUNTY']
    for col in possible_county_cols:
        if col in df.columns:
            county_col = col
            break
    
    if county_col:
        df['county'] = df[county_col].astype(str)
        print(f"Using {county_col} as county identifier")
    else:
        print("Warning: Could not find county identifier column")
        df['county'] = 'Unknown'
    
    # Calculate difference variables
    df['inc_diff'] = df['inc_averaged_growth_rate']# - df['empirical_growth_rate']
    df['pop_diff'] = df['pop_averaged_growth_rate']# - df['empirical_growth_rate']
    
    # Add community identifier if available
    if 'UnitName' in df.columns:
        df['community_id'] = df['UnitName']
    else:
        # Create a simple index-based identifier
        df['community_id'] = df.index
    
    # Add initial population for filtering
    initial_pop_col = f'PopInitial_{level_suffix}'
    if initial_pop_col in df.columns:
        df['initial_population'] = df[initial_pop_col]
        print(f"Using {initial_pop_col} for initial population")
    else:
        print(f"Warning: Could not find initial population column {initial_pop_col}")
        df['initial_population'] = np.nan
    
    # Filter out rows with missing essential data
    essential_vars = ['empirical_growth_rate', 'inc_diff', 'pop_diff', 'population_growth_rate']
    df_clean = df.dropna(subset=essential_vars)
    
    # Filter by initial population > 5,000
    if 'initial_population' in df_clean.columns:
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['initial_population'] > 5000]
        filtered_count = len(df_clean)
        print(f"Filtered by initial population > 5,000: {filtered_count} communities remaining (removed {initial_count - filtered_count})")
    
    print(f"Data prepared: {len(df_clean)} complete observations out of {len(df)} total")
    
    return df_clean

def get_county_name_mapping():
    """Create a mapping from county codes to readable names."""
    # Common Chicago metro area counties
    county_mapping = {
        '17031': 'Cook County',
        '17043': 'DuPage County', 
        '17089': 'Kane County',
        '17093': 'Kendall County',
        '17097': 'Lake County',
        '17111': 'McHenry County',
        '17197': 'Will County',
        '031': 'Cook County',
        '043': 'DuPage County',
        '089': 'Kane County', 
        '093': 'Kendall County',
        '097': 'Lake County',
        '111': 'McHenry County',
        '197': 'Will County'
    }
    return county_mapping

def create_specialty_scatter(data, title_suffix="", figsize=(5, 4.5)):
    """Create the specialty scatter plot."""
    print(f"Creating specialty scatter plot{title_suffix}...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set clean background
    ax.set_facecolor('white')
    
    # Extract variables
    x_emp = data['empirical_growth_rate']
    y_inc_diff = data['inc_diff']
    y_pop_diff = data['pop_diff']
    pop_growth_color = data['population_growth_rate']
    community_ids = data['community_id']
    
    # First scatter: Empirical vs Income difference (custom purple)
    scatter1 = ax.scatter(x_emp, y_inc_diff, 
                         color=purple_dot, 
                         alpha=.25,  # Half of 0.7
                         s=50, 
                         edgecolors='#3C3C3C',
                         label=r'$\gamma_{cm}^i$',
                         zorder=-100)
    
    # Second scatter: Empirical vs Population difference (custom orange)
    scatter2 = ax.scatter(x_emp, y_pop_diff, 
                         color=orange_dot,
                         alpha=.25,  # Half of 0.7
                         s=50, 
                         edgecolors='#3C3C3C',
                         label=r'$\gamma_{cm}^p$',
                         zorder=-100)
    
    # Draw connecting lines between the same communities
    print("Drawing connecting lines between community pairs...")
    for i, community_id in enumerate(community_ids):
        x_val = x_emp.iloc[i]
        y_inc_val = y_inc_diff.iloc[i]
        y_pop_val = y_pop_diff.iloc[i]
        
        # Draw line from income point to population point
        ax.plot([x_val, x_val], [y_inc_val, y_pop_val], 
                color='black', 
                linewidth=0.5, 
                alpha=0.25, 
                zorder=-101)
    
    # Fit regression lines and confidence intervals
    print("Calculating best fit lines and confidence intervals...")
    
    # Remove NaN values for regression
    valid_inc_mask = ~(np.isnan(x_emp) | np.isnan(y_inc_diff))
    valid_pop_mask = ~(np.isnan(x_emp) | np.isnan(y_pop_diff))
    
    x_emp_inc = x_emp[valid_inc_mask]
    y_inc_diff_clean = y_inc_diff[valid_inc_mask]
    x_emp_pop = x_emp[valid_pop_mask]
    y_pop_diff_clean = y_pop_diff[valid_pop_mask]
    
    # Create x range for plotting fitted lines
    x_range = np.linspace(min(x_emp.min(), x_emp.min()), max(x_emp.max(), x_emp.max()), 100)
    
    # Income regression (purple)
    if len(x_emp_inc) > 3:  # Need at least 4 points for meaningful regression
        # Fit regression
        slope_inc, intercept_inc, r_value_inc, p_value_inc, std_err_inc = stats.linregress(x_emp_inc, y_inc_diff_clean)
        
        # Calculate fitted line
        y_inc_fit = slope_inc * x_range + intercept_inc
        
        # Calculate confidence intervals
        n_inc = len(x_emp_inc)
        x_mean_inc = np.mean(x_emp_inc)
        sxx_inc = np.sum((x_emp_inc - x_mean_inc) ** 2)
        
        # Standard error of prediction
        residuals_inc = y_inc_diff_clean - (slope_inc * x_emp_inc + intercept_inc)
        mse_inc = np.sum(residuals_inc ** 2) / (n_inc - 2)
        
        # 95% confidence interval
        t_val_inc = stats.t.ppf(0.975, n_inc - 2)  # 95% CI
        se_inc = np.sqrt(mse_inc * (1/n_inc + (x_range - x_mean_inc)**2 / sxx_inc))
        ci_inc = t_val_inc * se_inc
        
        # Plot fitted line and confidence interval
        ax.plot(x_range, y_inc_fit, color=custom_purple, linewidth=2, alpha=1, 
                label=f'$m$={slope_inc:.2f},$r²$={r_value_inc**2:.2f}', zorder=4)
        ax.fill_between(x_range, y_inc_fit - ci_inc, y_inc_fit + ci_inc, 
                       color=custom_purple, alpha=0.5, zorder=3)
    
    # Population regression (orange)
    if len(x_emp_pop) > 3:  # Need at least 4 points for meaningful regression
        # Fit regression
        slope_pop, intercept_pop, r_value_pop, p_value_pop, std_err_pop = stats.linregress(x_emp_pop, y_pop_diff_clean)
        
        # Calculate fitted line
        y_pop_fit = slope_pop * x_range + intercept_pop
        
        # Calculate confidence intervals
        n_pop = len(x_emp_pop)
        x_mean_pop = np.mean(x_emp_pop)
        sxx_pop = np.sum((x_emp_pop - x_mean_pop) ** 2)
        
        # Standard error of prediction
        residuals_pop = y_pop_diff_clean - (slope_pop * x_emp_pop + intercept_pop)
        mse_pop = np.sum(residuals_pop ** 2) / (n_pop - 2)
        
        # 95% confidence interval
        t_val_pop = stats.t.ppf(0.975, n_pop - 2)  # 95% CI
        se_pop = np.sqrt(mse_pop * (1/n_pop + (x_range - x_mean_pop)**2 / sxx_pop))
        ci_pop = t_val_pop * se_pop
        
        # Plot fitted line and confidence interval
        ax.plot(x_range, y_pop_fit, color=custom_orange, linewidth=2, alpha=1,
                label=f'$m$={slope_pop:.2f},$r²$={r_value_pop**2:.2f}', zorder=2)
        ax.fill_between(x_range, y_pop_fit - ci_pop, y_pop_fit + ci_pop, 
                       color=custom_orange, alpha=0.5, zorder=1)
    
    ax.set_ylim(-0.032, 0.09)
    ax.set_xlim(-0.022, 0.092)

    # Add reference lines
    ax.axhline(0, color='#3c3c3c', linestyle='-', linewidth=1, alpha=1, zorder=1)
    ax.axvline(0, color='#3c3c3c', linestyle='-', linewidth=1, alpha=1, zorder=1)
    plt.plot([-.02,.09],[-.02,.09],linestyle='--',color='black',linewidth=1,alpha=.5,zorder=1)
    # Styling
    ax.set_xlabel(r'$\gamma_{cm}$', fontsize=14, fontweight='normal')
    ax.set_ylabel(r'$\gamma_{cm}^x$', fontsize=14, fontweight='normal')
    
    # Title with optional suffix
    base_title = 'Empirical vs Income/Population-Averaged Growth Rate Differences'
    full_title = f'{base_title}{title_suffix}'
    # ax.set_title(full_title, fontsize=16, fontweight='bold', pad=20)
    
    # Set xlim for full MSA tracts only
    if 'Full MSA (TR)' in title_suffix:
        ax.set_xlim(right=0.125)
        ax.set_ylim(top=0.15,bottom=-.05)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')
    
    # Style ticks
    ax.tick_params(axis='both', which='major', labelsize=12, colors='#333333')
    
    # Add legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False, 
              fontsize=7.5, facecolor='white', edgecolor='gray', framealpha=0.9)
    
    # Add grid
    # ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Tight layout
    plt.tight_layout()
    
    return fig, ax

def save_plot(fig, filename):
    """Save the plot."""
    print(f"Saving plot to {filename}...")
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # Close to free memory
    print(f"Plot saved successfully")

def print_summary_statistics(data, title="SUMMARY STATISTICS"):
    """Print summary statistics for the variables."""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    variables = ['empirical_growth_rate', 'inc_averaged_growth_rate', 'pop_averaged_growth_rate', 
                 'population_growth_rate', 'inc_diff', 'pop_diff']
    
    for var in variables:
        if var in data.columns:
            var_data = data[var].dropna()
            if len(var_data) > 0:
                print(f"\n{var.replace('_', ' ').title()}:")
                print(f"  Mean: {var_data.mean():.4f}")
                print(f"  Std:  {var_data.std():.4f}")
                print(f"  Min:  {var_data.min():.4f}")
                print(f"  Max:  {var_data.max():.4f}")
                print(f"  N:    {len(var_data)}")

def process_level(level, level_suffix):
    """Process a single geographic level."""
    print(f"\n{'='*60}")
    print(f"PROCESSING {level.upper()} LEVEL")
    print(f"{'='*60}")
    
    # Load data for this level
    data, _ = load_data_by_level(level)
    if data is None:
        print(f"Error: Could not load {level} data")
        return
    
    # Prepare variables
    prepared_data = prepare_variables(data, level_suffix)
    if prepared_data.empty:
        print(f"Error: No usable data after preparation for {level}")
        return
    
    # Get county mapping for readable names
    county_mapping = get_county_name_mapping()
    
    # Print overall summary statistics
    print_summary_statistics(prepared_data, f"FULL MSA {level.upper()} SUMMARY STATISTICS")
    
    # Create plot for full MSA
    unit_name = "communities" if level == "cm" else "tracts"
    print(f"\nCreating plot for Full MSA ({len(prepared_data)} {unit_name})...")
    fig_msa, ax_msa = create_specialty_scatter(prepared_data, title_suffix=f" - Full MSA ({level.upper()})")
    save_plot(fig_msa, f'specialty_scatter_plots/specialty_scatter_full_msa_{level}.pdf')
    
    # Get unique counties and create individual plots
    unique_counties = prepared_data['county'].unique()
    print(f"\nFound {len(unique_counties)} unique counties: {list(unique_counties)}")
    
    county_plots_created = 0
    for county_code in unique_counties:
        if county_code == 'Unknown':
            continue
            
        # Filter data for this county
        county_data = prepared_data[prepared_data['county'] == county_code].copy()
        
        if len(county_data) < 2:  # Need at least 2 points for a meaningful plot
            print(f"Skipping {county_code}: only {len(county_data)} communities")
            continue
        
        # Get readable county name
        county_name = county_mapping.get(county_code, f"County {county_code}")
        
        print(f"\nCreating plot for {county_name} ({len(county_data)} {unit_name})...")
        
        # Print county-specific summary statistics
        print_summary_statistics(county_data, f"{county_name.upper()} {level.upper()} SUMMARY STATISTICS")
        
        # Create county-specific plot
        fig_county, ax_county = create_specialty_scatter(county_data, title_suffix=f" - {county_name} ({level.upper()})")
        
        # Save with safe filename
        safe_county_name = county_name.replace(' ', '_').replace('County', 'County').lower()
        filename = f'specialty_scatter_plots/specialty_scatter_{safe_county_name}_{level}.pdf'
        save_plot(fig_county, filename)
        
        county_plots_created += 1
    
    print(f"\n{'='*60}")
    print(f"{level.upper()} ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(prepared_data)} total {unit_name}")
    print(f"Generated {county_plots_created + 1} plots for {level.upper()} level:")
    print(f"- 1 Full MSA plot")
    print(f"- {county_plots_created} County-specific plots")
    return county_plots_created + 1

def main():
    """Main execution function."""
    print("="*60)
    print("SPECIALTY SCATTER PLOT ANALYSIS")
    print("="*60)
    
    # Create output directory for plots
    os.makedirs('specialty_scatter_plots', exist_ok=True)
    
    # Define levels to process
    levels_to_process = [
        ('cm', 'cm'),  # (level_code, level_suffix)
        ('tr', 'tr')   # (level_code, level_suffix)
    ]
    
    total_plots = 0
    
    # Process each level
    for level, level_suffix in levels_to_process:
        plots_generated = process_level(level, level_suffix)
        if plots_generated:
            total_plots += plots_generated
    
    print("\n" + "="*60)
    print("OVERALL ANALYSIS COMPLETE")
    print("="*60)
    print(f"Generated {total_plots} total plots across all levels")
    print("\nPlot features:")
    print("- Purple points: Empirical vs Income-averaged growth rates")
    print("- Orange points: Empirical vs Population-averaged growth rates")
    print("- Black lines connect points for the same geographic units")
    print("- Trend lines show best fit with 95% confidence intervals")
    print("- R² and slope (m) values shown in legend")
    print("\nAll plots saved in: specialty_scatter_plots/ directory")

if __name__ == "__main__":
    main() 