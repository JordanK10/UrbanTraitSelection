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
COMMUNITY_INPUT_FILE = os.path.join(INPUT_DIR, 'bg_cm_exported_terms.csv')
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'specialty_scatter_plots')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'log_abs_selection_vs_log_population_comparison.pdf')

# Custom Colors
CUSTOM_PURPLE = '#633673'
CUSTOM_ORANGE = '#E77429'

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# A mapping from FIPS code to County Name for prettier labels
COUNTY_FIPS_MAP = {
    '17031': 'Cook',
    '17043': 'DuPage',
    '17089': 'Kane',
    '17093': 'Kankakee',
    '17097': 'Lake',
    '17111': 'McHenry',
    '17197': 'Will'
}

def process_level_data(filepath, level_suffix, county_col='ParentCounty'):
    """
    Loads and aggregates data for a specific geographic level.
    """
    print(f"Reading and processing {filepath}...")
    try:
        df = pd.read_csv(filepath, dtype={county_col: str})
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return None

    pop_col = f'PopFinal_{level_suffix}'
    if level_suffix == 'tr':
        selection_col = 'Sel_tr_from_bg_pop'
    elif level_suffix == 'cm':
        selection_col = 'Sel_cm_from_tr_pop'
    else:
        print(f"Unknown level suffix: {level_suffix}")
        return None

    if not all(c in df.columns for c in [pop_col, selection_col, county_col]):
        print(f"Error: Required columns not found in {filepath}.")
        return None

    # Aggregate to county level
    county_pop = df.groupby(county_col)[pop_col].sum()
    county_abs_selection = df.groupby(county_col)[selection_col].apply(lambda x: x.abs().sum())

    df_county = pd.DataFrame({
        'total_pop': county_pop,
        'total_abs_selection': county_abs_selection
    }).reset_index()
    
    # Filter for log transformation
    df_filtered = df_county[(df_county['total_pop'] > 0) & (df_county['total_abs_selection'] > 0)].copy()
    if df_filtered.empty:
        return None
        
    df_filtered['log_pop'] = np.log(df_filtered['total_pop'])
    df_filtered['log_selection'] = np.log(df_filtered['total_abs_selection'])
    df_filtered['CountyName'] = df_filtered[county_col].map(COUNTY_FIPS_MAP)
    
    return df_filtered

def create_plot():
    """
    Generates a scatter plot comparing tract and community level selection vs. population.
    """
    df_tract = process_level_data(TRACT_INPUT_FILE, 'tr')
    df_community = process_level_data(COMMUNITY_INPUT_FILE, 'cm')

    if df_tract is None or df_community is None:
        print("Could not process one or both data levels. Aborting plot generation.")
        return

    # --- Calculate Slopes for Legend ---
    # Use numpy's polyfit to get the slope (m) and intercept (b) of the line y = mx + b
    slope_tract, _ = np.polyfit(df_tract['log_pop'], df_tract['log_selection'], 1)
    slope_community, _ = np.polyfit(df_community['log_pop'], df_community['log_selection'], 1)

    tract_label = f'tr (slope={slope_tract:.2f})'
    community_label = f'cm (slope={slope_community:.2f})'

    # --- Plotting ---
    print("Generating plot...")
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Plot Tract Data
    sns.regplot(
        x='log_pop', y='log_selection', data=df_tract, ax=ax,
        scatter_kws={'alpha': 0.5, 's': 100, 'color': CUSTOM_ORANGE},
        line_kws={'color': CUSTOM_ORANGE, 'linestyle': '-'},
        label=tract_label
    )

    # Plot Community Data
    sns.regplot(
        x='log_pop', y='log_selection', data=df_community, ax=ax,
        scatter_kws={'alpha': 0.5, 's': 100, 'color': CUSTOM_PURPLE},
        line_kws={'color': CUSTOM_PURPLE, 'linestyle': '-'},
        label=community_label
    )

    # Add labels to points (optional, can be crowded)
    # for i, row in df_tract.iterrows():
    #     ax.text(row['log_pop'], row['log_selection'] + 0.05, row['CountyName'], fontsize=9, color='darkorange')
    # for i, row in df_community.iterrows():
    #     ax.text(row['log_pop'], row['log_selection'] - 0.1, row['CountyName'], fontsize=9, color='purple')
    
    # --- Aesthetics ---
    # Remove axis labels by setting them to empty strings
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    # ax.grid(True, zorder=0)
    ax.legend(fontsize=12)
    
    # --- Save and Show ---
    print(f"Saving plot to {OUTPUT_FILE}...")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    plt.close()
    print("Plotting complete.")

if __name__ == '__main__':
    create_plot()
