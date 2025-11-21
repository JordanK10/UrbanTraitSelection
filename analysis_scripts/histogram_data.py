#!/usr/bin/env python3
"""
histogram_data.py

Generates histograms for key growth rate and income variables with distribution fits
for community, tract, and block group level data.

Uses Student t-distribution for all growth rate variables and normal distribution for initial log income.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t
import warnings
warnings.filterwarnings('ignore')
import os
import sys

# Check for 'null' argument to switch directories
if 'null' in sys.argv:
    INPUT_DIR = 'output_terms_null'
    BASE_OUTPUT_DIR = 'plots_null'
else:
    INPUT_DIR = 'output_terms'
    BASE_OUTPUT_DIR = 'plots'

# Set style for better-looking plots
plt.style.use('default')  # Use default instead of seaborn
# Remove seaborn palette setting since we'll use custom colors

def load_data():
    """Load and prepare the data from CSV files."""
    print("Loading data...")
    
    # Load community-level data
    try:
        cm_data = pd.read_csv(os.path.join(INPUT_DIR, 'bg_cm_exported_terms.csv'))
        print(f"Loaded community data: {cm_data.shape[0]} rows, {cm_data.shape[1]} columns")
    except FileNotFoundError:
        print("Error: bg_cm_exported_terms.csv not found")
        return None, None, None
    
    # Load tract-level data
    try:
        tr_data = pd.read_csv(os.path.join(INPUT_DIR, 'bg_tr_exported_terms.csv'))
        print(f"Loaded tract data: {tr_data.shape[0]} rows, {tr_data.shape[1]} columns")
    except FileNotFoundError:
        print("Warning: bg_tr_exported_terms.csv not found")
        tr_data = None
    
    # Load block group-level data
    try:
        bg_data = pd.read_csv(os.path.join(INPUT_DIR, 'bg_bg_exported_terms.csv'))
        print(f"Loaded block group data: {bg_data.shape[0]} rows, {bg_data.shape[1]} columns")
    except FileNotFoundError:
        print("Warning: bg_bg_exported_terms.csv not found")
        bg_data = None
    
    return cm_data, tr_data, bg_data

def prepare_variables(data, level_name):
    """Extract and calculate the required variables from the data."""
    print(f"Preparing variables for {level_name} level...")
    
    # Create a working copy
    df = data.copy()
    
    # Extract empirical growth rate from pre-calculated column
    # Try different naming patterns based on level
    level_suffix = 'cm' if 'Community' in level_name else ('tr' if 'Tract' in level_name else 'bg')
    
    if f'AvgG_emp_{level_suffix}' in df.columns:
        df['empirical_growth_rate'] = df[f'AvgG_emp_{level_suffix}']
    else:
        # Look for other possible empirical growth rate columns
        emp_cols = [col for col in df.columns if 'AvgG_emp' in col]
        if emp_cols:
            df['empirical_growth_rate'] = df[emp_cols[0]]
            print(f"Using {emp_cols[0]} for empirical growth rate in {level_name}")
        else:
            print(f"Warning: Could not find empirical growth rate column (AvgG_emp_*) for {level_name}")
            df['empirical_growth_rate'] = np.nan
    
    # Population averaged growth rate
    if f'AvgG_pop_{level_suffix}' in df.columns:
        df['pop_averaged_growth_rate'] = df[f'AvgG_pop_{level_suffix}']
    else:
        pop_avg_cols = [col for col in df.columns if 'AvgG_pop' in col]
        if pop_avg_cols:
            df['pop_averaged_growth_rate'] = df[pop_avg_cols[0]]
            print(f"Using {pop_avg_cols[0]} for population averaged growth rate in {level_name}")
        else:
            print(f"Warning: Could not find population averaged growth rate for {level_name}")
            df['pop_averaged_growth_rate'] = np.nan
    
    # Income averaged growth rate
    if f'AvgG_inc_{level_suffix}' in df.columns:
        df['inc_averaged_growth_rate'] = df[f'AvgG_inc_{level_suffix}']
    else:
        inc_avg_cols = [col for col in df.columns if 'AvgG_inc' in col]
        if inc_avg_cols:
            df['inc_averaged_growth_rate'] = df[inc_avg_cols[0]]
            print(f"Using {inc_avg_cols[0]} for income averaged growth rate in {level_name}")
        else:
            print(f"Warning: Could not find income averaged growth rate for {level_name}")
            df['inc_averaged_growth_rate'] = np.nan
    
    # Population growth rate
    if f'PopG_{level_suffix}' in df.columns:
        df['population_growth_rate'] = df[f'PopG_{level_suffix}']
    else:
        pop_growth_cols = [col for col in df.columns if 'PopG' in col or 'PopFracChange' in col]
        if pop_growth_cols:
            df['population_growth_rate'] = df[pop_growth_cols[0]]
            print(f"Using {pop_growth_cols[0]} for population growth rate in {level_name}")
        else:
            print(f"Warning: Could not find population growth rate for {level_name}")
            df['population_growth_rate'] = np.nan
    
    # Initial income (using log average income initial)
    if f'LogAvgIncInitial_{level_suffix}' in df.columns:
        df['initial_log_income'] = df[f'LogAvgIncInitial_{level_suffix}']
    else:
        initial_inc_cols = [col for col in df.columns if 'LogAvgInc' in col and 'Initial' in col]
        if initial_inc_cols:
            df['initial_log_income'] = df[initial_inc_cols[0]]
            print(f"Using {initial_inc_cols[0]} for initial log income in {level_name}")
        else:
            print(f"Warning: Could not find initial log income for {level_name}")
            df['initial_log_income'] = np.nan
    
    # Add level identifier
    df['level'] = level_name
    
    # Select only the variables we need
    key_vars = ['empirical_growth_rate', 'pop_averaged_growth_rate', 'inc_averaged_growth_rate', 
                'population_growth_rate', 'initial_log_income', 'level']
    
    # Add UnitName if available for identification
    if 'UnitName' in df.columns:
        key_vars.append('UnitName')
    
    return df[key_vars]

def fit_distribution(data, use_t_dist=False):
    """Fit a normal or t-distribution to the data and return parameters."""
    # Remove NaN and infinite values
    clean_data = data[np.isfinite(data) & ~np.isnan(data)]
    
    if len(clean_data) < 2:
        return None, None, None, None, None
    
    if use_t_dist:
        # Fit t-distribution
        df, loc, scale = t.fit(clean_data)
        # Perform Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.kstest(clean_data, lambda x: t.cdf(x, df, loc, scale))
        return df, loc, scale, ks_stat, ks_p_value
    else:
        # Fit normal distribution
        mu, sigma = norm.fit(clean_data)
        # Perform Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.kstest(clean_data, lambda x: norm.cdf(x, mu, sigma))
        return mu, sigma, None, ks_stat, ks_p_value

def create_histogram_with_fit(data, var_name, level, figsize=(4.3254, 4), n_bins=None):
    """Create a histogram with normal distribution fit."""
    print(f"Creating histogram for {var_name} - {level}")
    
    # Filter out NaN and infinite values
    clean_data = data[np.isfinite(data) & ~np.isnan(data)]
    
    if len(clean_data) < 2:
        print(f"  Skipping {var_name} - {level}: insufficient data points ({len(clean_data)})")
        return None
    
    # Calculate statistics for x-axis limits
    mean_value = clean_data.mean()
    std_value = clean_data.std()
    x_min_limit = mean_value - 3 * std_value
    x_max_limit = mean_value + 3 * std_value
    
    # Determine aggregation level notation
    agg_level = 'cm' if 'Community' in level else 'tr'
    
    # Create figure with clean styling
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set clean background and remove grid initially
    ax.set_facecolor('white')
    
    # Purple color scheme
    purple_color = '#633673'
    
    # Create histogram with purple color and density
    if n_bins is None:
        n_bins = min(50, max(10, int(np.sqrt(len(clean_data)))))  # Adaptive bin count
    counts, bins, patches = ax.hist(clean_data, bins=n_bins, density=True, alpha=0.7, 
                                   color=purple_color, edgecolor='white', linewidth=0.8)
    
    # Determine if we should use t-distribution (all variables except initial log income)
    use_t_dist = var_name != 'initial_log_income'
    
    # Fit distribution
    if use_t_dist:
        df_param, loc, scale, ks_stat, ks_p_value = fit_distribution(clean_data, use_t_dist=True)
    else:
        mu, sigma, _, ks_stat, ks_p_value = fit_distribution(clean_data, use_t_dist=False)
    
    if (use_t_dist and df_param is not None) or (not use_t_dist and mu is not None):
        # Plot distribution fit with purple curve
        x_range = np.linspace(clean_data.min(), clean_data.max(), 200)
        
        if use_t_dist:
            y_fit = t.pdf(x_range, df_param, loc, scale)
            fit_label = 'Fitted Distribution'
            stats_text = f'Student t\nDOF = {df_param:.3f}\nmean = {mean_value:.3f}\nscale = {scale:.3f}'
        else:
            y_fit = norm.pdf(x_range, mu, sigma)
            fit_label = 'Fitted Distribution'
            stats_text = f'Gaussian \nmean = {mean_value:.3f}\nscale = {sigma:.3f}'
        
        # Plot KDE-style curve in darker purple
        ax.plot(x_range, y_fit, color='#4a2654', linewidth=2.5, label=fit_label)
        
        # Add mean line
        ax.axvline(mean_value, color='#3D3D3D', linestyle='-', linewidth=1.5, alpha=0.75, label=f'Mean = {mean_value:.3f}')
        
        # Add reference line at x=0
        ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.75)
        
        # Add statistics text box in upper left with adjusted font size
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Define mathematical notation mapping
    math_labels = {
        'empirical_growth_rate': rf'$\gamma_{{{agg_level}}}$',
        'pop_averaged_growth_rate': rf'$\gamma_{{{agg_level}}}^p$',
        'inc_averaged_growth_rate': rf'$\gamma_{{{agg_level}}}^i$', 
        'population_growth_rate': r'$\gamma_p$',
        'initial_log_income': r'$\ln\bar{y}_0$'
    }
    
    # Formatting with mathematical notation
    x_label = math_labels.get(var_name, var_name.replace('_', ' ').title())
    
    # Clean styling
    # ax.set_xlabel(x_label, fontsize=12, fontweight='normal')
    # ax.set_ylabel('Density', fontsize=12, fontweight='normal')
    ax.set_title(f'n = {len(clean_data):,}', 
                 fontsize=13, fontweight='bold', pad=20)
    
    # Use scientific notation for y-axis of initial log income distribution
    if var_name == 'initial_log_income':
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Set x-axis limits to mean ± 2 standard deviations
    ax.set_xlim(x_min_limit, x_max_limit)
    
    # Style all spines with light gray color
    ax.spines['top'].set_color('#CCCCCC')
    ax.spines['right'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Style ticks with larger font size (1.5x * 1.25 = 1.875x: 10 * 1.875 = 18.75)
    ax.tick_params(axis='x', which='major', labelsize=18.75, colors='#333333', bottom=True, top=False)
    ax.tick_params(axis='y', which='major', labelsize=18.75, colors='#333333',left=True, right=False, labelleft=True)
    
    # No grid for clean look
    ax.grid(False)
    
    # Legend removed as requested
    
    # Save plot as PDF
    level_short = level.lower()[:2]
    filename = f'histogram_{var_name}_{level_short}.pdf'
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'histogram_plots', filename), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {os.path.join(BASE_OUTPUT_DIR, 'histogram_plots', filename)}")
    
    # Return different parameters based on distribution type
    if use_t_dist and df_param is not None:
        return df_param, loc, scale, ks_stat, ks_p_value, 'student_t'
    elif not use_t_dist and mu is not None:
        return mu, sigma, None, ks_stat, ks_p_value, 'normal'
    else:
        return None, None, None, None, None, None

def create_all_histograms(data):
    """Create histograms for all variables at all levels."""
    print("Creating all histograms...")
    
    # Create output directory
    import os
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, 'histogram_plots'), exist_ok=True)
    
    # Variables to plot
    variables = ['empirical_growth_rate', 'pop_averaged_growth_rate', 'inc_averaged_growth_rate', 
                 'population_growth_rate', 'initial_log_income']
    
    # Split data by level
    levels = data['level'].unique()
    print(f"Creating histograms for levels: {levels}")
    
    # Store fit results
    fit_results = []
    
    # Create histograms for each variable and level
    for level in levels:
        level_data = data[data['level'] == level].copy()
        
        print(f"\nCreating histograms for {level} level ({len(level_data)} observations)...")
        
        for var in variables:
            if var in level_data.columns:
                # Use 50 bins for all Block Group level histograms
                bins_to_use = None
                if level == 'Block Group':
                    bins_to_use = 50
                    print(f"  Using 50 bins for Block Group {var}")
                
                fit_params = create_histogram_with_fit(level_data[var], var, level, n_bins=bins_to_use)
                
                if fit_params is not None and fit_params[0] is not None:  # If fit was successful
                    param1, param2, param3, ks_stat, ks_p_value, dist_type = fit_params
                    
                    if dist_type == 'student_t':
                        fit_results.append({
                            'Level': level,
                            'Variable': var,
                            'Distribution': 'Student t',
                            'Degrees of Freedom (df)': param1,
                            'Location (loc)': param2,
                            'Scale': param3,
                            'KS Statistic': ks_stat,
                            'KS P-value': ks_p_value,
                            'Sample Size': len(level_data[var].dropna())
                        })
                    else:  # normal distribution
                        fit_results.append({
                            'Level': level,
                            'Variable': var,
                            'Distribution': 'Normal',
                            'Mean (μ)': param1,
                            'Std Dev (σ)': param2,
                            'Parameter 3': None,  # Keep columns aligned
                            'KS Statistic': ks_stat,
                            'KS P-value': ks_p_value,
                            'Sample Size': len(level_data[var].dropna())
                        })
    
    return fit_results

def save_fit_results_table(fit_results):
    """Save the fit results to a CSV file and print summary."""
    if not fit_results:
        print("No fit results to save.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(fit_results)
    
    # Save to CSV
    results_df.to_csv(os.path.join(BASE_OUTPUT_DIR, 'histogram_plots', 'distribution_fit_results.csv'), index=False)
    print(f"\nSaved fit results to: {os.path.join(BASE_OUTPUT_DIR, 'histogram_plots', 'distribution_fit_results.csv')}")
    
    # Print summary table
    print("\n" + "="*80)
    print("DISTRIBUTION FIT RESULTS")
    print("="*80)
    print(results_df.to_string(index=False, float_format='%.4f'))
    print("\nNote: KS Test - Low p-values (<0.05) suggest data significantly deviates from fitted distribution")
    print("All variables use Student t-distribution except initial log income (normal distribution)")

def print_summary_statistics(data):
    """Print summary statistics for all variables."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    variables = ['empirical_growth_rate', 'pop_averaged_growth_rate', 'inc_averaged_growth_rate', 
                 'population_growth_rate', 'initial_log_income']
    
    levels = data['level'].unique()
    
    for level in levels:
        level_data = data[data['level'] == level]
        print(f"\n{level.upper()} LEVEL:")
        print("-" * 40)
        
        for var in variables:
            if var in level_data.columns:
                var_data = level_data[var].dropna()
                if len(var_data) > 0:
                    print(f"\n{var.replace('_', ' ').title()}:")
                    print(f"  Mean: {var_data.mean():.4f}")
                    print(f"  Std:  {var_data.std():.4f}")
                    print(f"  Min:  {var_data.min():.4f}")
                    print(f"  Max:  {var_data.max():.4f}")
                    print(f"  N:    {len(var_data)}")

def main():
    """Main execution function."""
    print("="*60)
    print("HISTOGRAM ANALYSIS WITH DISTRIBUTION FITS")
    print("="*60)
    
    # Load data
    cm_data, tr_data, bg_data = load_data()
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
    
    # Process block group data if available
    if bg_data is not None:
        bg_processed = prepare_variables(bg_data, 'Block Group')
        if not bg_processed.empty:
            datasets.append(bg_processed)
            print(f"Block group data: {len(bg_processed)} observations")
    
    if not datasets:
        print("Error: No usable data found")
        return
    
    # Combine datasets
    combined_data = pd.concat(datasets, ignore_index=True)
    print(f"Combined data: {len(combined_data)} total observations")
    
    # Print summary statistics
    print_summary_statistics(combined_data)
    
    # Create histograms and fit distributions
    fit_results = create_all_histograms(combined_data)
    
    # Save and display fit results
    save_fit_results_table(fit_results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    # Calculate total plots generated
    num_levels = len(combined_data['level'].unique())
    num_variables = 5  # empirical_growth_rate, pop_averaged_growth_rate, inc_averaged_growth_rate, population_growth_rate, initial_log_income
    total_plots = num_variables * num_levels
    
    print(f"Data levels processed: {list(combined_data['level'].unique())}")
    
    print(f"Generated {total_plots} histogram plots in {os.path.join(BASE_OUTPUT_DIR, 'histogram_plots')}/ directory")
    print(f"({num_variables} variables × {num_levels} data levels)")
    print("Each histogram includes:")
    print("- Data distribution bins")
    print("- Fitted distribution curve (Normal or Student t)")
    print("- Distribution parameters (μ,σ for Normal; df,loc,scale for Student t)")
    print("- Kolmogorov-Smirnov test statistic and p-value")
    print("- Sample size")
    print("\nDistribution types:")
    print("- Growth rate variables: Student t-distribution")
    print("- Initial log income: Normal distribution")
    print("\nFit results saved to: {os.path.join(BASE_OUTPUT_DIR, 'histogram_plots', 'distribution_fit_results.csv')}")

if __name__ == "__main__":
    main() 