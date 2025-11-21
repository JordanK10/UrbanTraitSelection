import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import re
import sys

# Check for 'null' argument to switch directories
if 'null' in sys.argv:
    INPUT_DIR = 'output_terms_null'
    BASE_OUTPUT_DIR = 'plots_null'
else:
    INPUT_DIR = 'output_terms'
    BASE_OUTPUT_DIR = 'plots'

# --- Configuration ---
HIERARCHY_LEVELS = ['bg', 'tr', 'cm', 'ct', 'st']
BASE_ANALYSIS_LEVEL = 'bg'  # We'll focus on the 'bg' based decomposition
PICKLE_FILE_PATH = os.path.join(INPUT_DIR, "all_decomposition_results.pkl")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "heatmaps_local_dominance/st_summary")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'msa_normalized_selection_summary.pdf')
CUSTOM_ORANGE = '#E77429'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_pickle_data(filepath):
    """Loads data from a pickle file."""
    if not os.path.exists(filepath):
        print(f"Error: Pickle file not found at {filepath}")
        return None
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded decomposition data from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading pickle data from {filepath}: {e}")
        return None

def _plot_summary_heatmap(df_to_plot, title, output_path, cmap, vmin, vmax):
    """A private helper function to generate and save a summary heatmap with consistent styling."""
    n_rows, n_cols = df_to_plot.shape
    figsize = (1.5 * n_cols, 4/5 * n_rows)

    fig, ax = plt.subplots(figsize=figsize)
    
    # Format annotations to 2 decimal places with a percent sign
    annotations = np.array([["{:.1f}%".format(val) for val in row] for row in df_to_plot.values])

    sns.heatmap(
        df_to_plot,
        annot=annotations,
        fmt="",
        cmap=cmap,
        linewidths=.5,
        ax=ax,
        vmin=vmin, 
        vmax=vmax,
        cbar=False,
        annot_kws={"size": 25}
    )
        
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close(fig)

def create_msa_normalized_summary_plot(decomposition_results):
    """
    Creates a heatmap summarizing the mean and std dev of the MSA-Normalized Selection Intensity.
    """
    print("--- Generating MSA-Normalized Selection Summary ---")
    
    # 1. Get the MSA's total growth rate from the state-level data
    try:
        df_st = decomposition_results[BASE_ANALYSIS_LEVEL]['st']
        msa_growth_rate = df_st['AvgG_pop_st'].iloc[0]
        if pd.isna(msa_growth_rate) or msa_growth_rate == 0:
            print("Error: MSA growth rate is zero or NaN. Cannot normalize.")
            return
        msa_growth_rate = abs(msa_growth_rate)
        print(f"Found MSA growth rate: {msa_growth_rate:.4f}")
    except (KeyError, IndexError):
        print("Error: Could not retrieve MSA growth rate from state-level data.")
        return

    # 2. Define levels and the corresponding selection columns
    levels_to_plot = ['tr', 'cm', 'ct', 'st']
    sel_col_map = {
        'tr': 'Sel_tr_from_bg_pop',
        'cm': 'Sel_cm_from_tr_pop',
        'ct': 'Sel_ct_from_cm_pop',
        'st': 'Sel_st_from_ct_pop'
    }

    summary_rows = []

    # 3. Loop through levels to calculate the metric and its stats
    for level in levels_to_plot:
        df_level = decomposition_results[BASE_ANALYSIS_LEVEL].get(level)
        if df_level is None or df_level.empty:
            continue

        sel_col = sel_col_map.get(level)
        pop_col = f'PopInitial_{level}'
        
        if not all(c in df_level.columns for c in [sel_col, pop_col]):
            print(f"Warning: Missing required columns for level '{level}'. Skipping.")
            continue

        # Calculate the MSA-Normalized Selection Intensity for each unit and scale to percent
        df_level['msa_norm_sel'] = (df_level[sel_col].abs() / msa_growth_rate) * 100
        
        # --- Calculate both weighted and unweighted statistics ---
        
        # Always calculate unweighted stats
        unweighted_mean = df_level['msa_norm_sel'].mean()
        unweighted_std = df_level['msa_norm_sel'].std()

        # Calculate population-weighted stats
        weights = df_level[pop_col]
        if weights.isnull().all() or weights.sum() == 0:
            # If weights are invalid, weighted stats are the same as unweighted
            weighted_mean = unweighted_mean
            weighted_std = unweighted_std
        else:
            weighted_mean = np.average(df_level['msa_norm_sel'], weights=weights)
            variance = np.average((df_level['msa_norm_sel'] - weighted_mean)**2, weights=weights)
            weighted_std = np.sqrt(variance)
            
        summary_rows.append({
            'Level': level.capitalize(),
            'Mean (Weighted)': weighted_mean,
            'Std (Weighted)': weighted_std,
            'Mean (Unweighted)': unweighted_mean,
            'Std (Unweighted)': unweighted_std
        })

    if not summary_rows:
        print("Could not generate summary data. Aborting.")
        return

    # 4. Create the final DataFrame for the heatmap
    df_plot = pd.DataFrame(summary_rows).set_index('Level').T
    
    # 5. Plotting, using the same style as the LDR variation summary
    custom_cmap = LinearSegmentedColormap.from_list('custom_white_orange', ['white', CUSTOM_ORANGE])
    
    _plot_summary_heatmap(
        df_to_plot=df_plot,
        title='MSA-Normalized Selection Intensity (%)',
        output_path=OUTPUT_FILE,
        cmap=custom_cmap,
        vmin=0,
        vmax=None
    )
    
    print(f"Successfully saved MSA-Normalized Selection summary to {OUTPUT_FILE}")


if __name__ == "__main__":
    decomposition_data = load_pickle_data(PICKLE_FILE_PATH)
    if decomposition_data:
        create_msa_normalized_summary_plot(decomposition_data)
