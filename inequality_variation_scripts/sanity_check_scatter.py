import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def load_data():
    """Load the decomposition results from CSV files"""
    try:
        # Load community-level data
        cm_data = pd.read_csv('output_terms/bg_cm_exported_terms.csv')
        cm_data['Agg'] = 'cm'
        print(f"Successfully loaded community data with shape: {cm_data.shape}")
        
        # Load tract-level data  
        tr_data = pd.read_csv('output_terms/bg_tr_exported_terms.csv')
        tr_data['Agg'] = 'tr'
        print(f"Successfully loaded tract data with shape: {tr_data.shape}")
        
        # Combine data
        df = pd.concat([cm_data, tr_data], ignore_index=True)
        print(f"Combined data shape: {df.shape}")
        print(f"Available aggregation levels: {df['Agg'].unique()}")
        
        return df
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV files - {e}")
        print("Expected files: 'output_terms/bg_cm_exported_terms.csv' and 'output_terms/bg_tr_exported_terms.csv'")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_sanity_check_data(df, level_name):
    """Prepare LHS and RHS variables for sanity check plots"""
    print(f"\n=== Preparing sanity check data for {level_name.upper()} level ===")
    
    # Clean level name for column matching
    level_clean = level_name.replace('_co', '').replace('_tr', '')
    
    # Find LHS variables (exactly what the sanity checks validate)
    lhs_pop_col = f'AvgG_pop_{level_clean}'
    lhs_inc_col = f'AvgG_inc_{level_clean}'
    
    if lhs_pop_col in df.columns:
        df['lhs_population'] = df[lhs_pop_col]
        print(f"Using {lhs_pop_col} for LHS population decomposition")
    else:
        print(f"Warning: Could not find {lhs_pop_col}")
        df['lhs_population'] = np.nan
        
    if lhs_inc_col in df.columns:
        df['lhs_income'] = df[lhs_inc_col]
        print(f"Using {lhs_inc_col} for LHS income decomposition")
    else:
        print(f"Warning: Could not find {lhs_inc_col}")
        df['lhs_income'] = np.nan

    # Find RHS components
    # 1. Transmitted Growth from base level
    transmitted_pop_col = f'Transmitted_AvgG_bg_to_{level_clean}_pop'
    transmitted_inc_col = f'Transmitted_AvgG_bg_to_{level_clean}_inc'
    
    if transmitted_pop_col in df.columns:
        df['transmitted_pop_growth'] = df[transmitted_pop_col]
        print(f"Found transmitted growth (pop): {transmitted_pop_col}")
    else:
        print(f"Warning: Could not find {transmitted_pop_col}")
        df['transmitted_pop_growth'] = np.nan
        
    if transmitted_inc_col in df.columns:
        df['transmitted_inc_growth'] = df[transmitted_inc_col]
        print(f"Found transmitted growth (inc): {transmitted_inc_col}")
    else:
        print(f"Warning: Could not find {transmitted_inc_col}")
        df['transmitted_inc_growth'] = np.nan

    # 2. Cumulative Selection (direct + transmitted selection)
    cumulative_sel_pop_col = f'Cumulative_Sel_pop_{level_clean}'
    cumulative_sel_inc_col = f'Cumulative_Sel_inc_{level_clean}'
    
    if cumulative_sel_pop_col in df.columns:
        df['cumulative_sel_pop'] = df[cumulative_sel_pop_col]
        print(f"Found cumulative selection (pop): {cumulative_sel_pop_col}")
    else:
        print(f"Warning: Could not find {cumulative_sel_pop_col}")
        df['cumulative_sel_pop'] = np.nan
        
    if cumulative_sel_inc_col in df.columns:
        df['cumulative_sel_inc'] = df[cumulative_sel_inc_col]
        print(f"Found cumulative selection (inc): {cumulative_sel_inc_col}")
    else:
        print(f"Warning: Could not find {cumulative_sel_inc_col}")
        df['cumulative_sel_inc'] = np.nan

    # Calculate complete RHS totals (matching the multi-level sanity check structure)
    df['rhs_pop_total'] = df['transmitted_pop_growth'] + df['cumulative_sel_pop']
    df['rhs_inc_total'] = df['transmitted_inc_growth'] + df['cumulative_sel_inc']

    # Calculate differences for diagnostics
    df['sanity_check_pop'] = df['lhs_population'] - df['rhs_pop_total']
    df['sanity_check_inc'] = df['lhs_income'] - df['rhs_inc_total']

    print(f"=== Summary for {level_name.upper()} ===")
    print(f"Population RHS = Transmitted Growth + Cumulative Selection")
    print(f"Income RHS = Transmitted Growth + Cumulative Selection")
    
    # Report sample values and sanity check quality
    valid_pop = ~(pd.isna(df['lhs_population']) | pd.isna(df['rhs_pop_total']))
    valid_inc = ~(pd.isna(df['lhs_income']) | pd.isna(df['rhs_inc_total']))
    
    if valid_pop.any():
        sanity_pop_rmse = np.sqrt(np.mean(df.loc[valid_pop, 'sanity_check_pop']**2))
        sanity_pop_max_abs = np.abs(df.loc[valid_pop, 'sanity_check_pop']).max()
        print(f"Population sanity check - RMSE: {sanity_pop_rmse:.6f}, Max |diff|: {sanity_pop_max_abs:.6f}")
        
    if valid_inc.any():
        sanity_inc_rmse = np.sqrt(np.mean(df.loc[valid_inc, 'sanity_check_inc']**2))
        sanity_inc_max_abs = np.abs(df.loc[valid_inc, 'sanity_check_inc']).max()
        print(f"Income sanity check - RMSE: {sanity_inc_rmse:.6f}, Max |diff|: {sanity_inc_max_abs:.6f}")
    
    return df

def add_regression_line_and_stats(ax, x, y, color='red'):
    """Add regression line and statistics to plot"""
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if valid_mask.sum() < 2:
        ax.text(0.05, 0.95, 'Insufficient data for regression', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        return
    
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    # Calculate regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    
    # Add regression line
    x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_pred = slope * x_range + intercept
    ax.plot(x_range, y_pred, color=color, linewidth=2, alpha=0.8, label='Regression')
    
    # Add statistics text
    stats_text = f'y = {slope:.4f}x + {intercept:.4f}\n'
    stats_text += f'R² = {r_value**2:.4f}\n'
    stats_text += f'p = {p_value:.4e}\n'
    stats_text += f'n = {len(x_clean)}'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def create_sanity_check_plot(data, x_var, y_var, title, filename, figsize=(10, 8)):
    """Create a sanity check scatter plot with y=x diagonal reference"""
    
    # Filter out NaN values
    mask = ~(pd.isna(data[x_var]) | pd.isna(data[y_var]))
    plot_data = data[mask]
    
    if len(plot_data) == 0:
        print(f"Warning: No valid data for plot {title}")
        return
    
    plt.figure(figsize=figsize)
    
    # Create scatter plot
    plt.scatter(plot_data[x_var], plot_data[y_var], alpha=0.6, s=30, color='blue')
    
    # Add y=x diagonal reference line (perfect sanity check)
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='Perfect Match (y = x)')
    
    # Add y=0 and x=0 reference lines
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add regression line and stats
    add_regression_line_and_stats(plt.gca(), plot_data[x_var].values, plot_data[y_var].values)
    
    plt.xlabel('LHS (Left-Hand Side of Decomposition)')
    plt.ylabel('RHS (Transmitted Growth + Cumulative Selection)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

def create_all_sanity_check_plots(data):
    """Create sanity check plots for both community and tract levels"""
    
    # Define levels to analyze
    levels = ['cm', 'tr']  # community and tract
    
    for level in levels:
        level_suffix = '_co' if level == 'cm' else '_tr'
        level_data = data[data['Agg'] == level].copy()
        
        if len(level_data) == 0:
            print(f"No data found for level {level}")
            continue
        
        print(f"\n=== Processing {level.upper()} level data ===")
        level_data = prepare_sanity_check_data(level_data, level)
        
        # Create output directory
        output_dir = f'plots/sanity_check_{level}'
        
        # 1. Population Decomposition Sanity Check
        create_sanity_check_plot(
            level_data,
            'lhs_population',
            'rhs_pop_total',
            f'{level.upper()} Level: Population Decomposition Sanity Check\nLHS vs RHS (Transmitted Growth + Cumulative Selection)',
            f'{output_dir}/population_sanity_check{level_suffix}.png'
        )
        
        # 2. Income Decomposition Sanity Check  
        create_sanity_check_plot(
            level_data,
            'lhs_income',
            'rhs_inc_total',
            f'{level.upper()} Level: Income Decomposition Sanity Check\nLHS vs RHS (Transmitted Growth + Cumulative Selection)',
            f'{output_dir}/income_sanity_check{level_suffix}.png'
        )

def print_summary_statistics(data):
    """Print summary statistics for sanity check quality"""
    print("\n=== Sanity Check Quality Summary ===")
    
    for level in ['cm', 'tr']:
        level_data = data[data['Agg'] == level]
        if len(level_data) == 0:
            continue
            
        print(f"\n{level.upper()} Level:")
        
        # Population sanity check
        if 'sanity_check_pop' in level_data.columns:
            pop_check = level_data['sanity_check_pop'].dropna()
            if len(pop_check) > 0:
                print(f"  Population Sanity Check (LHS - RHS):")
                print(f"    Mean: {pop_check.mean():.6f}")
                print(f"    Std:  {pop_check.std():.6f}")
                print(f"    RMSE: {np.sqrt(np.mean(pop_check**2)):.6f}")
                print(f"    Max |error|: {np.abs(pop_check).max():.6f}")
                print(f"    N: {len(pop_check)}")
        
        # Income sanity check  
        if 'sanity_check_inc' in level_data.columns:
            inc_check = level_data['sanity_check_inc'].dropna()
            if len(inc_check) > 0:
                print(f"  Income Sanity Check (LHS - RHS):")
                print(f"    Mean: {inc_check.mean():.6f}")
                print(f"    Std:  {inc_check.std():.6f}")
                print(f"    RMSE: {np.sqrt(np.mean(inc_check**2)):.6f}")
                print(f"    Max |error|: {np.abs(inc_check).max():.6f}")
                print(f"    N: {len(inc_check)}")

def main():
    """Main function"""
    print("=== Decomposition Sanity Check Analysis ===")
    print("Plotting LHS vs RHS (Transmitted Growth + Cumulative Selection)")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Print basic info
    print(f"\nDataset shape: {data.shape}")
    print(f"Available aggregation levels: {data['Agg'].unique()}")
    
    # Create sanity check plots
    create_all_sanity_check_plots(data)
    
    # Print summary statistics
    print_summary_statistics(data)
    
    print("\n=== Analysis Complete ===")
    print("Check the 'plots/sanity_check_cm' and 'plots/sanity_check_tr' directories for output plots.")
    print("\nThese plots should show points very close to the y=x diagonal if the decomposition is working correctly.")
    print("The regression line slope should be very close to 1.0 and R² should be very close to 1.0.")

if __name__ == "__main__":
    main() 