import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Check for 'null' argument to switch directories
if 'null' in sys.argv:
    INPUT_DIR = 'output_terms_null'
    BASE_OUTPUT_DIR = 'plots_null'
else:
    INPUT_DIR = 'output_terms'
    BASE_OUTPUT_DIR = 'plots'

# --- Configuration ---
TRACT_INPUT_FILE = os.path.join(INPUT_DIR, 'bg_tr_exported_terms.csv')
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'specialty_scatter_plots')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'scaling_analysis_summary_tract.pdf')
CUSTOM_ORANGE = '#E77429'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_tract_data(filepath):
    """
    Loads tract data and aggregates it to the county level using all three calculation methods.
    """
    print(f"Reading and processing {filepath}...")
    try:
        df = pd.read_csv(filepath, dtype={'ParentCounty': str})
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return None

    pop_col = 'PopFinal_tr'
    selection_col = 'Sel_tr_from_bg_pop'
    county_col = 'ParentCounty'

    if not all(c in df.columns for c in [pop_col, selection_col, county_col]):
        print("Error: Required columns not found.")
        return None

    # --- Perform all three aggregations ---
    county_pop = df.groupby(county_col)[pop_col].sum()
    
    # Case 1: Unweighted Sum
    unweighted_sum = df.groupby(county_col)[selection_col].apply(lambda x: x.abs().sum())
    
    # Case 2: Population-Weighted Sum
    df['weighted_abs_selection'] = df[selection_col].abs() * df[pop_col]
    weighted_sum = df.groupby(county_col)['weighted_abs_selection'].sum()
    
    # Case 3: Population-Weighted Average (Normalized)
    weighted_avg = weighted_sum / county_pop

    # --- Combine into a single DataFrame ---
    df_county = pd.DataFrame({
        'total_pop': county_pop,
        'unweighted_sum': unweighted_sum,
        'weighted_sum': weighted_sum,
        'weighted_avg': weighted_avg
    }).reset_index()

    # --- Log Transformations ---
    # Filter out any zero or negative values before taking the log
    df_county = df_county[
        (df_county['total_pop'] > 0) &
        (df_county['unweighted_sum'] > 0) &
        (df_county['weighted_sum'] > 0) &
        (df_county['weighted_avg'] > 0)
    ].copy()
    
    df_county['log_pop'] = np.log(df_county['total_pop'])
    df_county['log_unweighted_sum'] = np.log(df_county['unweighted_sum'])
    df_county['log_weighted_sum'] = np.log(df_county['weighted_sum'])
    df_county['log_weighted_avg'] = np.log(df_county['weighted_avg'])
    
    return df_county

def create_summary_plot(df):
    """
    Generates the 3-panel summary plot of the scaling analyses.
    """
    if df is None or df.empty:
        print("No data to plot.")
        return

    # --- Setup Figure ---
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.5))
    plot_configs = [
        {'y_col': 'log_unweighted_sum', 'title': 'Case 1: Unweighted Sum', 'ax': axes[0]},
        {'y_col': 'log_weighted_sum', 'title': 'Case 2: Population-Weighted Sum', 'ax': axes[1]},
        {'y_col': 'log_weighted_avg', 'title': 'Case 3: Population-Weighted Average', 'ax': axes[2]}
    ]

    # --- Create Each Plot ---
    for config in plot_configs:
        y_col = config['y_col']
        ax = config['ax']
        
        # Calculate slope
        slope, _ = np.polyfit(df['log_pop'], df[y_col], 1)
        label = f'Tracts (β={slope:.2f})'
        
        # Plot data
        sns.regplot(
            x='log_pop', y=y_col, data=df, ax=ax,
            scatter_kws={'alpha': 0.7, 's': 80, 'color': CUSTOM_ORANGE},
            line_kws={'color': CUSTOM_ORANGE, 'linestyle': '-', 'linewidth': 1}
        )
        
        # Aesthetics
        # ax.set_title(config['title'], fontsize=12, fontweight='bold')
        ax.legend([label], fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # --- Final Figure Aesthetics ---
    # axes[1].set_ylabel('Log(Selection Metric)', fontsize=11)
    # axes[2].set_xlabel('Log(Total County Population)', fontsize=11)
    axes[2].set_ylim(bottom=-7, top=-3)
    # fig.suptitle('Urban Scaling of Selection Dynamics (Tract Level)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 1])
    
    # --- Save ---
    print(f"Saving plot to {OUTPUT_FILE}...")
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    print("Plotting complete.")

if __name__ == '__main__':
    county_data = process_tract_data(TRACT_INPUT_FILE)
    create_summary_plot(county_data)
