import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import re

# --- Configuration ---
HIERARCHY_LEVELS = ['bg', 'tr', 'cm', 'ct', 'st']
BASE_ANALYSIS_LEVEL = 'bg'  # We'll focus on the 'bg' based decomposition
PICKLE_FILE_PATH = "output_terms/all_decomposition_results.pkl"
OUTPUT_DIR = "heatmaps_local_dominance"
TARGET_LEVELS_FOR_PLOTTING = ['cm', 'ct'] # Let's create heatmaps for Community and County levels
CUSTOM_PURPLE = '#633673'
CUSTOM_ORANGE = '#E77429'
CUSTOM_GREY = '#3C3C3C'

# Defines which column in the *child* level DataFrame identifies its parent.
CHILD_TO_PARENT_ID_COL_MAP = {
    'tr': 'ParentCommunity',
    'cm': 'ParentCounty',
    'ct': 'ParentState',
}

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

def clean_column_names(col_name):
    """Creates human-readable labels for heatmap columns from the raw column names."""
    # First, strip any metric suffix like _LDR or _GIR
    col_name_base = re.sub(r'(_LDR|_GIR|_PNC)$', '', col_name)

    # Regular expression to parse the column names
    sel_match = re.match(r"Sel_(\w+)_from_(\w+)", col_name_base)
    trans_sel_match = re.match(r"Transmitted_Sel_(\w+)_to_(\w+)", col_name_base)
    trans_avgg_match = re.match(r"Transmitted_AvgG_(\w+)_to_(\w+)", col_name_base)
    # New regex for transmitted aggregated growth (handles pop/inc variants)
    trans_agg_match = re.match(r"Transmitted_AggG_(?:pop_|inc_)?(\w+)_to_(\w+)", col_name_base)

    if sel_match:
        level = sel_match.group(1).upper()
        return f"Selection ({level})"
    elif trans_sel_match:
        origin = trans_sel_match.group(1).upper()
        return f"Transmitted Sel. ({origin})"
    elif trans_avgg_match:
        base = trans_avgg_match.group(1).upper()
        return f"Transmitted Growth ({base})"
    elif trans_agg_match:
        origin = trans_agg_match.group(1).upper()
        return f"Transmitted Growth ({origin})"
    
    # Fallback for any names that don't match
    return col_name_base.replace('_', ' ').title()

def prepare_ldr_heatmap_data(df_for_level, path_suffix):
    """
    Extracts pre-calculated LDR columns (which are absolute percentages)
    and prepares the data for heatmap plotting.
    """
    if df_for_level.empty:
        return None

    # 1. Find all LDR columns for the given path (_pop or _inc).
    # These are unsigned percentages (0-100).
    ldr_cols = sorted([col for col in df_for_level.columns if col.endswith(f'{path_suffix}_LDR')])
    
    if not ldr_cols:
        return None

    # 2. Extract the LDR data.
    # Set index to UnitName for alignment and for the final plot.
    df_indexed = df_for_level.set_index('UnitName')
    
    df_ldr = df_indexed[ldr_cols].copy()
    
    # 3. Clean up column names for the plot.
    clean_names = [clean_column_names(col) for col in df_ldr.columns]
    df_ldr.columns = clean_names
    
    # 4. Explicitly rename the generic 'Selection (LVL)' to 'Selection'
    sel_col_to_rename = [col for col in df_ldr.columns if col.startswith('Selection (')]
    if sel_col_to_rename:
        df_ldr.rename(columns={sel_col_to_rename[0]: 'Selection'}, inplace=True)

    # These are the desired columns for the final plot, in order.
    final_cols_in_order = ['Selection', 'Transmitted Sel. (CM)', 'Transmitted Sel. (CT)', 'Transmitted Sel. (TR)', 'Transmitted Growth (BG)']

    # --- Reorder columns to place Transmitted Growth last ---
    cols = df_ldr.columns.tolist()
    trans_popwth_col = None
    for col in cols:
        if "Transmitted Growth" in col:
            trans_popwth_col = col
            break
    if trans_popwth_col:
        cols.remove(trans_popwth_col)
        cols.append(trans_popwth_col)
        df_ldr = df_ldr[cols]

    return df_ldr

def prepare_pnc_heatmap_data(df_for_level, target_level, path_suffix, hierarchy_levels_list):
    """
    Prepares a DataFrame with PNC values relative to the immediate parent for plotting.
    """
    if df_for_level.empty:
        return None

    # 1. Determine the parent level to normalize against.
    try:
        current_level_idx = hierarchy_levels_list.index(target_level)
        if current_level_idx == len(hierarchy_levels_list) - 1:
            # This is the highest level (e.g., 'st'), it has no parent.
            # We'll use PNC relative to itself.
            parent_level_for_pnc = target_level
        else:
            parent_level_for_pnc = hierarchy_levels_list[current_level_idx + 1] # Immediate parent
    except ValueError:
        print(f"  Error (prepare_pnc_heatmap_data): Target level '{target_level}' not in hierarchy.")
        return None
    
    # Suffix for the PNC columns to select. E.g., _PNC_ct for a 'cm' level plot.
    # If the target is its own parent (e.g., highest level), the suffix is just _PNC.
    pnc_suffix_for_filter = f"_PNC_{parent_level_for_pnc}" if parent_level_for_pnc != target_level else "_PNC"
    
    # Regex to find all relevant PNC columns for the specified path and parent normalization
    # e.g., ^(Sel|Transmitted).+_pop_PNC_ct$
    pnc_cols_regex = f"^(Sel|Transmitted).+({path_suffix}){pnc_suffix_for_filter}$"
    
    pnc_cols_to_use = [col for col in df_for_level.columns if re.match(pnc_cols_regex, col)]

    if not pnc_cols_to_use:
        print(f"  Warning (prepare_pnc_heatmap_data): No PNC columns matching regex '{pnc_cols_regex}' found for path '{path_suffix}' in level '{target_level}'.")
        return None

    # Construct PNC column names and find the ones that actually exist
    pnc_cols_to_use = sorted(list(set(pnc_cols_to_use)))
    
    if not pnc_cols_to_use or 'UnitName' not in df_for_level.columns:
        return None

    # --- Create the PNC value dataframe ---
    df_pnc = df_for_level[['UnitName'] + pnc_cols_to_use].set_index('UnitName')
    
    # 3. Clean up column names for the plot.
    clean_names = [clean_column_names(col) for col in df_pnc.columns]
    df_pnc.columns = clean_names
    
    # 4. Explicitly rename the generic 'Selection (LVL)' to 'Selection'
    sel_col_to_rename = [col for col in df_pnc.columns if col.startswith('Selection (')]
    if sel_col_to_rename:
        df_pnc.rename(columns={sel_col_to_rename[0]: 'Selection'}, inplace=True)

    # These are the desired columns for the final plot, in order.
    final_cols_in_order = ['Selection', 'Transmitted Sel. (CM)', 'Transmitted Sel. (CT)', 'Transmitted Sel. (TR)', 'Transmitted Growth (BG)']
    
    # --- Reorder columns to place Transmitted Growth last ---
    cols = df_pnc.columns.tolist()
    trans_popwth_col = None
    for col in cols:
        if "Transmitted Growth" in col:
            trans_popwth_col = col
            break
    if trans_popwth_col:
        cols.remove(trans_popwth_col)
        cols.append(trans_popwth_col)
        df_pnc = df_pnc[cols]

    return df_pnc

def get_custom_colormap_and_bounds(df):
    """
    Creates a custom colormap and determines bounds based on selection terms.
    The colormap uses purple-white-orange for values within the bounds of selection
    terms and grey for values outside those bounds (e.g., transmitted growth).
    """
    # Identify the transmitted growth column to exclude it from bound calculation
    trans_popwth_cols = [col for col in df.columns if 'Transmitted Growth' in col]
    selection_cols = [col for col in df.columns if col not in trans_popwth_cols]

    # Determine bounds from selection columns only
    if selection_cols:
        v_bound = df[selection_cols].abs().max().max()
    else:
        v_bound = df.abs().max().max() # Fallback to all columns if no selection cols found

    if pd.isna(v_bound) or v_bound == 0:
        v_bound = 1.0 # A small default to prevent division by zero in Normalize

    vmin, vmax = -v_bound, v_bound

    # Create the colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_purple_orange_grey",
        [CUSTOM_PURPLE, "white", CUSTOM_ORANGE]
    )
    # Set colors for values outside the vmin/vmax range
    custom_cmap.set_over(CUSTOM_GREY)
    custom_cmap.set_under(CUSTOM_GREY)

    return custom_cmap, vmin, vmax

def plot_heatmap(df, title, output_filename=None, ax=None):
    """
    Generates and saves a Local Dominance heatmap with a diverging colormap.
    Can plot to a file or a provided Matplotlib Axes object.
    """
    if df is None or df.empty:
        # print(f"    Cannot plot heatmap for '{title}'; data is empty.")
        return
    
    # Create the custom diverging colormap and bounds
    custom_cmap, vmin, vmax = get_custom_colormap_and_bounds(df)
    
    # If no Axes is provided, create a new figure for a single plot
    if ax is None:
        fig, ax_local = plt.subplots(figsize=(12, max(8, len(df) * 0.6)))
    else:
        ax_local = ax
        fig = ax.get_figure()

    heatmap_kwargs = {
        "annot": True,
        "fmt": ".1f",
        "cmap": custom_cmap,
        "linewidths": .5,
        'cbar_kws': {'label': 'Local Component Dominance (%)', 'extend': 'both'},
        'vmin': vmin,
        'vmax': vmax,
    }
    
    sns.heatmap(df, ax=ax_local, **heatmap_kwargs)

    ax_local.set_title(title, fontsize=16)
    ax_local.set_xlabel("Decomposition Term", fontsize=12)
    ax_local.set_ylabel("Spatial Unit", fontsize=12)
    ax_local.tick_params(axis='x', rotation=45)

    # If we created the figure, we are responsible for layout and saving/closing
    if ax is None:
        plt.tight_layout()
        try:
            plt.savefig(output_filename, format='pdf')
            print(f"    Successfully saved heatmap to {output_filename}")
        except Exception as e:
            print(f"    Error saving heatmap to {output_filename}: {e}")
        plt.close(fig)

def plot_pnc_heatmap(df, title, output_filename=None, ax=None):
    """
    Generates and saves a heatmap of Parent-Normalized Contribution (PNC) values.
    Can plot to a file or a provided Matplotlib Axes object.
    """
    if df is None or df.empty:
        # print(f"    Cannot plot PNC heatmap for '{title}'; data is empty.")
        return

    # Create the custom diverging colormap and bounds
    custom_cmap, vmin, vmax = get_custom_colormap_and_bounds(df)
    
    # If no Axes is provided, create a new figure for a single plot
    if ax is None:
        fig, ax_local = plt.subplots(figsize=(12, max(8, len(df) * 0.6)))
    else:
        ax_local = ax
        fig = ax.get_figure()

    heatmap_kwargs = {
        "annot": True,
        "fmt": ".1f", # One decimal place for percentages
        "cmap": custom_cmap,
        "linewidths": .5,
        'cbar_kws': {'label': 'Parent-Normalized Contribution (%)', 'extend': 'both'},
        'vmin': vmin,
        'vmax': vmax,
    }
    
    sns.heatmap(df, ax=ax_local, **heatmap_kwargs)

    ax_local.set_title(title, fontsize=16)
    ax_local.set_xlabel("Decomposition Term", fontsize=12)
    ax_local.set_ylabel("Spatial Unit", fontsize=12)
    ax_local.tick_params(axis='x', rotation=45)

    # If we created the figure, we are responsible for layout and saving/closing
    if ax is None:
        plt.tight_layout()
        try:
            plt.savefig(output_filename, format='pdf')
            print(f"    Successfully saved heatmap to {output_filename}")
        except Exception as e:
            print(f"    Error saving heatmap to {output_filename}: {e}")
        plt.close(fig)

def plot_histogram(data_series, title, xlabel, output_filename):
    """
    Generates and saves a histogram with a KDE overlay for a given data series.
    """
    if data_series is None or data_series.empty:
        return
        
    data_series = data_series.dropna()
    if data_series.empty:
        # print(f"    Cannot plot histogram for '{title}'; data series is empty after dropping NaNs.")
        return

    # Use clean style without grid
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define the purple color from your reference
    purple_color = '#633673'
    
    # Create histogram with matching style
    n, bins, patches = ax.hist(data_series, bins=30, alpha=0.8, color=purple_color, 
                              edgecolor='white', linewidth=0.8, density=True)
    
    # Add smooth density curve overlay (matching your reference)
    from scipy import stats
    density = stats.gaussian_kde(data_series)
    xs = np.linspace(data_series.min(), data_series.max(), 200)
    density_curve = density(xs)
    ax.plot(xs, density_curve, color='#2d1a3d', linewidth=3)
    
    # Calculate statistics
    mean_val = data_series.mean()
    std_val = data_series.std()
    
    # Add vertical line at mean (matching your reference)
    ax.axvline(mean_val, color='black', linestyle='-', linewidth=2.5, alpha=0.8)
    
    # Style the plot to match your reference
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    
    # Remove top and right spines for clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Style the ticks
    ax.tick_params(axis='both', which='major', labelsize=12, length=5, width=1.5)
    
    # Add statistics text box in upper left (matching your reference style)
    stats_text = f'Mean: {mean_val:.3f}\nStDev: {std_val:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, 
                     edgecolor='lightgray', linewidth=1))
    
    plt.tight_layout()
    
    try:
        plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    except Exception as e:
        print(f"    Error saving histogram to {output_filename}: {e}")
    
    plt.close(fig)

def _plot_summary_heatmap(df_to_plot, title, output_path, cmap, vmin, vmax):
    """A private helper function to generate and save a summary heatmap with consistent styling."""
    n_rows, n_cols = df_to_plot.shape
    figsize = (1.5 * n_cols, n_rows) # User's preferred ratio, adjusted for more rectangular cells

    fig, ax = plt.subplots(figsize=figsize)
    
    # Manually format annotations with percent sign
    annotations = np.array([["{:.1f}%".format(val) for val in row] for row in df_to_plot.values])

    sns.heatmap(
        df_to_plot,
        annot=annotations,
        fmt="",  # Disable default formatting
        cmap=cmap,
        linewidths=.5,
        ax=ax,
        vmin=vmin, 
        vmax=vmax,
        cbar=False,  # Remove colorbar
        annot_kws={"size": 25}  # User's preferred size
    )
        
    ax.set_title(title, fontsize=18)
    
    # Remove all axis labels and ticks
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close(fig)

def create_combined_summary_plot(df, level, output_dir, file_name, plot_title):
    """
    Creates a single, consolidated heatmap for a given region, showing LDR and PNC
    for both population and income paths, using row-normalization for coloring.
    """
    print(f"  Generating combined summary plot for {plot_title} (saving to {output_dir})...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare the four data series
    ldr_pop = prepare_ldr_heatmap_data(df, '_pop')
    ldr_inc = prepare_ldr_heatmap_data(df, '_inc')
    pnc_pop = prepare_pnc_heatmap_data(df, level, '_pop', HIERARCHY_LEVELS)
    pnc_inc = prepare_pnc_heatmap_data(df, level, '_inc', HIERARCHY_LEVELS)

    # Extract the first (and only) row of data for the single entity
    # and handle cases where data might be missing.
    data_series = {
        'LDR (Population)': ldr_pop.iloc[0] if ldr_pop is not None and not ldr_pop.empty else pd.Series(dtype=float),
        'PNC (Population)': pnc_pop.iloc[0] if pnc_pop is not None and not pnc_pop.empty else pd.Series(dtype=float),
        'LDR (Income)': ldr_inc.iloc[0] if ldr_inc is not None and not ldr_inc.empty else pd.Series(dtype=float),
        'PNC (Income)': pnc_inc.iloc[0] if pnc_inc is not None and not pnc_inc.empty else pd.Series(dtype=float),
    }
    
    # 2. Combine into a single DataFrame and clean up
    combined_df = pd.DataFrame(data_series).T.fillna(0)
    
    if combined_df.empty or combined_df.shape[1] == 0:
        print(f"    Cannot create combined plot for {plot_title}; not enough data.")
        return

    # --- NEW: Reorder columns sequentially by spatial scale ---
    canonical_order = [
        'Transmitted Sel. (TR)',
        'Transmitted Sel. (CM)',
        'Transmitted Sel. (CT)',
        'Selection',
        'Transmitted Growth (BG)'
    ]
    final_order = [col for col in canonical_order if col in combined_df.columns]
    combined_df = combined_df[final_order]
    # --- END NEW ---

    # 3. Get colormap and bounds based on selection terms from the raw data
    cmap, vmin, vmax = get_custom_colormap_and_bounds(combined_df)
    output_filename = os.path.join(output_dir, file_name)

    # 4. Plotting (delegated to helper)
    _plot_summary_heatmap(
        df_to_plot=combined_df,
        title=plot_title,
        output_path=output_filename,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

def create_ldr_variation_summary_plot(decomposition_results, output_dir, file_name, plot_title, levels_to_plot, cook_county_only=False):
    """
    Creates a heatmap summarizing the mean and standard deviation of the "Total Selection Dominance"
    across different spatial levels. Total Selection Dominance for a unit is the sum of the LDRs
    of its selection-based growth components.
    """
    summary_rows = []

    for level in levels_to_plot:
        df_level = decomposition_results.get(level)
        if df_level is None or df_level.empty:
            continue
            
        # Filter for Cook County if requested
        if cook_county_only:
            if level == 'tr' or level == 'cm':
                if 'ParentCounty' in df_level.columns:
                    # Handle both 3-digit and 5-digit FIPS codes for Cook County
                    # Tract level uses '031', Community level uses '17031'
                    cook_county_mask = (df_level['ParentCounty'] == '17031') | (df_level['ParentCounty'] == '031')
                    df_level = df_level[cook_county_mask].copy()
            elif level == 'ct':
                if 'UnitName' in df_level.columns:
                    # UnitName for counties is the 5-digit FIPS code, e.g., '17031'
                    df_level = df_level[df_level['UnitName'] == '17031'].copy()
            else: # 'st' level should be skipped for Cook County specific plot
                continue
        
        if df_level.empty:
            continue

        level_summary = {'Level': level.capitalize()}

        for path_suffix, path_name in [('_pop', 'Population'), ('_inc', 'Income')]:
            # Find all LDR columns that represent selection terms
            ldr_sel_cols = [
                c for c in df_level.columns 
                if (c.startswith('Sel_') or c.startswith('Transmitted_Sel_')) 
                and c.endswith(f'{path_suffix}_LDR')
            ]
            

            
            if not ldr_sel_cols:
                level_summary[f'{path_name}_Mean'] = np.nan
                level_summary[f'{path_name}_Std'] = np.nan
                continue

            # For each unit, sum the LDRs of its selection terms
            df_level['TotalSelLDR'] = df_level[ldr_sel_cols].sum(axis=1)
            
            # Get weights (Initial Population) for weighted stats
            pop_col = f'PopInitial_{level}'
            weights = df_level[pop_col] if pop_col in df_level.columns else None

            # Handle edge case with only one unit (e.g., state level)
            if len(df_level) == 1:
                mean = df_level['TotalSelLDR'].iloc[0]
                std = 0.0
            elif weights is not None and not weights.isnull().all() and weights.sum() > 0:
                # Calculate weighted mean and std dev
                mean = np.average(df_level['TotalSelLDR'], weights=weights)
                variance = np.average((df_level['TotalSelLDR'] - mean)**2, weights=weights)
                std = np.sqrt(variance)
            else:
                # Fallback to unweighted if weights are missing or invalid
                mean = df_level['TotalSelLDR'].mean()
                std = df_level['TotalSelLDR'].std()

            level_summary[f'{path_name}_Mean'] = mean
            level_summary[f'{path_name}_Std'] = std
            
        summary_rows.append(level_summary)

    if not summary_rows:
        print(f"Cannot generate LDR variation plot for {plot_title}, no data found.")
        return

    # Create the final DataFrame for the heatmap
    df_plot = pd.DataFrame(summary_rows).set_index('Level').T
    
    # Restructure the index to be a more readable MultiIndex
    df_plot.index = pd.MultiIndex.from_tuples(
        [(name.split('_')[0], name.split('_')[1]) for name in df_plot.index],
        names=['Path', 'Metric']
    )
    
    # --- Plotting (delegated to helper) ---
    # Create the custom colormap from white to orange
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_white_orange', 
        ['white', CUSTOM_ORANGE]
    )
    output_path = os.path.join(output_dir, file_name)

    _plot_summary_heatmap(
        df_to_plot=df_plot,
        title=plot_title,
        output_path=output_path,
        cmap=custom_cmap,
        vmin=0,
        vmax=None
    )
    
    print(f"  Successfully saved LDR variation summary to {output_path}")

def main():
    """
    Main function to orchestrate loading data and generating heatmaps based on different rankings.
    """
    print("--- Starting Plot Generation ---")
    
    decomposition_results = load_pickle_data(PICKLE_FILE_PATH)
    if decomposition_results is None:
        return

    print("\n--- Generating Local Dominance Heatmaps ---")
    dominance_target_levels = [lvl for lvl in HIERARCHY_LEVELS if lvl != BASE_ANALYSIS_LEVEL]

    for level in dominance_target_levels:
        if level not in TARGET_LEVELS_FOR_PLOTTING:
            continue
        try:
            df_level_full = decomposition_results[BASE_ANALYSIS_LEVEL][level]
        except KeyError:
            print(f"Could not find data for base '{BASE_ANALYSIS_LEVEL}' and level '{level}'. Skipping.")
            continue
        
        parent_col = CHILD_TO_PARENT_ID_COL_MAP.get(level)
        if not (parent_col and parent_col in df_level_full.columns):
            print(f"Parent column '{parent_col}' not found for level '{level}'. Cannot create parent-specific plots.")
            continue

        parent_units = df_level_full[parent_col].dropna().unique()
        parent_level_name = parent_col.replace("Parent", "").lower()
        print(f"\n--- Processing level '{level}', generating reports for {len(parent_units)} {parent_level_name}s ---")

        for parent_id in parent_units:
            safe_parent_id_str = str(parent_id).split('.')[0]
            if level == 'ct':
                try:
                    state_fips = df_level_full[df_level_full[parent_col] == parent_id]['ParentState'].iloc[0]
                    safe_parent_id_str = f"{state_fips}{safe_parent_id_str}"
                except (IndexError, KeyError):
                     pass
            
            county_report_dir = os.path.join(OUTPUT_DIR, level, f"{parent_level_name}_{safe_parent_id_str}")
            os.makedirs(county_report_dir, exist_ok=True)
            print(f"  Creating report directory: {county_report_dir}")

            df_filtered_for_county = df_level_full[df_level_full[parent_col] == parent_id].copy()
            if df_filtered_for_county.empty:
                continue

            # --- NEW: Generate Histograms for specific metrics in this county ---
            hist_dir = os.path.join(county_report_dir, 'histograms')
            os.makedirs(hist_dir, exist_ok=True)
            
            try:
                current_level_idx = HIERARCHY_LEVELS.index(level)
                parent_level_short_name = HIERARCHY_LEVELS[current_level_idx + 1]
            except (ValueError, IndexError):
                print(f"    Could not determine parent level for '{level}', skipping histograms.")
                continue

            pnc_suffix = f'_PNC_{parent_level_short_name}'
            ldr_suffix = '_LDR'

            cols_to_histogram = []
            
            for col in df_filtered_for_county.columns:
                is_sel_or_trans = col.startswith(('Sel_', 'Transmitted_'))
                
                # Check if the term is *for* the current level's units.
                # e.g., Sel_cm_... or Transmitted_..._to_cm
                is_for_level = f'_{level}_from_' in col or f'_to_{level}' in col

                if is_sel_or_trans and is_for_level:
                    # Check if it is a raw term, an LDR term, or a PNC term relative to immediate parent
                    is_raw_value = '_LDR' not in col and '_PNC' not in col
                    is_ldr_value = col.endswith(ldr_suffix)
                    is_pnc_value = col.endswith(pnc_suffix)
                    
                    if is_raw_value or is_ldr_value or is_pnc_value:
                        cols_to_histogram.append(col)
            
            cols_to_histogram = sorted(list(set(cols_to_histogram)))

            print(f"    Generating {len(cols_to_histogram)} focused histograms for {parent_level_name.title()} {safe_parent_id_str}...")
            # for col in cols_to_histogram:
            #     clean_name = clean_column_names(col)
            #     hist_title = f'Distribution of {clean_name}\n(Units in {parent_level_name.title()} {safe_parent_id_str})'
            #     hist_filename = os.path.join(hist_dir, f'hist_{col}.pdf')
            #     plot_histogram(df_filtered_for_county[col], hist_title, clean_name, hist_filename)
            # --- END NEW ---

            for path in ['_pop', '_inc']:
                path_name = "Population" if path == '_pop' else "Income"

                # Define the criteria for ranking units for different heatmaps
                ranking_criteria = {
                    'pop': f'PopInitial_{level}',
                    'pop_popwth': f'PopG_{level}',
                    'inc_popwth': f'AvgG_emp_{level}'
                }
                
                # Dynamically find selection and growth components to rank by
                child_level = HIERARCHY_LEVELS[HIERARCHY_LEVELS.index(level) - 1]
                own_sel_col = f'Sel_{level}_from_{child_level}{path}'
                if own_sel_col in df_filtered_for_county.columns:
                    ranking_criteria[f'sel_{level}'] = own_sel_col
                
                for col in df_filtered_for_county.columns:
                    if col.startswith('Transmitted_Sel_') and col.endswith(f'_to_{level}{path}'):
                        if not col.startswith(f'Transmitted_Sel_{BASE_ANALYSIS_LEVEL}_to_'):
                            short_name = re.search(r'Transmitted_Sel_(\w+)_to_', col).group(1)
                            ranking_criteria[f'trans_sel_{short_name}'] = col

                base_popwth_col = f'Transmitted_AvgG_{BASE_ANALYSIS_LEVEL}_to_{level}{path}'
                if base_popwth_col in df_filtered_for_county.columns:
                    ranking_criteria[f'trans_g_{BASE_ANALYSIS_LEVEL}'] = base_popwth_col
                
                # Loop through the criteria and generate a heatmap for each one
                for rank_key, rank_col_name in ranking_criteria.items():
                    if rank_col_name not in df_filtered_for_county.columns:
                        continue
                    
                    df_ranked = df_filtered_for_county.sort_values(by=rank_col_name, ascending=False)
                    df_top_10 = df_ranked.head(10)

                    # --- Plot 1: Local Dominance (Using LDR Columns) ---
                    ldr_heatmap_data = prepare_ldr_heatmap_data(df_top_10, path)
                    if ldr_heatmap_data is not None and not ldr_heatmap_data.empty:
                        title = f'Local Dominance (LDR) in {parent_level_name.title()} {safe_parent_id_str}\n(Top 10 by {rank_key.replace("_", " ").title()}, {path_name} Path)'
                        output_file = os.path.join(county_report_dir, f'ldr_heatmap_{path_name.lower()}_ranked_by_{rank_key}.pdf')
                        plot_heatmap(ldr_heatmap_data, title, output_file)

                    # --- Plot 2: Parent-Normalized Contribution (PNC) ---
                    pnc_heatmap_data = prepare_pnc_heatmap_data(df_top_10, level, path, HIERARCHY_LEVELS)
                    if pnc_heatmap_data is not None and not pnc_heatmap_data.empty:
                        title = f'Parent-Normalized Contribution in {parent_level_name.title()} {safe_parent_id_str}\n(Top 10 by {rank_key.replace("_", " ").title()}, {path_name} Path)'
                        output_file = os.path.join(county_report_dir, f'pnc_heatmap_{path_name.lower()}_ranked_by_{rank_key}.pdf')
                        plot_pnc_heatmap(pnc_heatmap_data, title, output_file)

    # --- NEW: Create the "Ultimate Plot" for selected Chicago Communities ---
    print("\n--- Generating Ultimate Plot for Selected Chicago Communities ---")

    target_communities = [
        'NEAR NORTH SIDE', 'NEAR SOUTH SIDE', 'NEAR WEST SIDE', 'LOOP', 
        'LOGAN SQUARE', 'WEST TOWN', 'AUSTIN', 'AVONDALE', 'HYDE PARK', 
        'KENWOOD', 'WOODLAWN', 'ENGLEWOOD', 'GREATER GRAND CROSSING'
    ]

    try:
        df_cm_full = decomposition_results[BASE_ANALYSIS_LEVEL]['cm']
        
        # Filter for the specific communities
        df_selected_communities = df_cm_full[df_cm_full['UnitName'].isin(target_communities)].copy()
        
        if df_selected_communities.empty:
            print("  Warning: None of the target communities were found in the dataset.")
        else:
            print(f"  Found {len(df_selected_communities)} of the {len(target_communities)} target communities.")
            
            # --- NEW: Generate Global Histograms for CM level ---
            global_hist_dir = os.path.join(OUTPUT_DIR, 'cm', 'global_histograms')
            os.makedirs(global_hist_dir, exist_ok=True)
            print(f"\n  Generating global histograms for key metrics at the Community level (saving to {global_hist_dir})...")
            
            level = 'cm'
            parent_level_short_name = 'ct' # Parent of cm is ct
            pnc_suffix = f'_PNC_{parent_level_short_name}'
            ldr_suffix = '_LDR'
            
            cols_to_histogram = []
            
            for col in df_cm_full.columns:
                is_sel_or_trans = col.startswith(('Sel_', 'Transmitted_'))
                is_for_level = f'_{level}_from_' in col or f'_to_{level}' in col

                if is_sel_or_trans and is_for_level:
                    is_raw_value = '_LDR' not in col and '_PNC' not in col
                    is_ldr_value = col.endswith(ldr_suffix)
                    is_pnc_value = col.endswith(pnc_suffix)
                    
                    if is_raw_value or is_ldr_value or is_pnc_value:
                        cols_to_histogram.append(col)

            cols_to_histogram = sorted(list(set(cols_to_histogram)))
            
            for col in cols_to_histogram:
                clean_name = clean_column_names(col)
                hist_title = f'Global Distribution of {clean_name}\n(All Communities)'
                hist_filename = os.path.join(global_hist_dir, f'hist_global_{col}.pdf')
                plot_histogram(df_cm_full[col], hist_title, clean_name, hist_filename)
            # --- END NEW ---

            # Preserve the user-defined order for the plot
            df_selected_communities['UnitName'] = pd.Categorical(
                df_selected_communities['UnitName'], 
                categories=target_communities, 
                ordered=True
            )
            df_selected_communities.sort_values('UnitName', inplace=True)

            for path in ['_pop', '_inc']:
                path_name = "Population" if path == '_pop' else "Income"
                
                # --- Dominance Plot (using LDR columns) ---
                ldr_data_ultimate = prepare_ldr_heatmap_data(
                    df_selected_communities, 
                    path
                )
                if ldr_data_ultimate is not None and not ldr_data_ultimate.empty:
                    title = f'Local Dominance (LDR) for Selected Chicago Communities\n({path_name} Path)'
                    output_file = os.path.join(OUTPUT_DIR, f'ultimate_plot_ldr_chicago_{path_name.lower()}.pdf')
                    plot_heatmap(ldr_data_ultimate, title, output_file)

                # --- Raw Values Plot is now PNC Plot ---
                pnc_data_ultimate = prepare_pnc_heatmap_data(
                    df_selected_communities,
                    'cm',
                    path,
                    HIERARCHY_LEVELS
                )
                if pnc_data_ultimate is not None and not pnc_data_ultimate.empty:
                    title = f'Parent-Normalized Contribution for Selected Chicago Communities\n({path_name} Path)'
                    output_file = os.path.join(OUTPUT_DIR, f'ultimate_plot_pnc_chicago_{path_name.lower()}.pdf')
                    plot_pnc_heatmap(pnc_data_ultimate, title, output_file)

    except (KeyError, TypeError):
        print("  Could not generate ultimate plot: Data for base 'bg' and level 'cm' not found.")
    # --- END of new section ---

    # --- State-level Summary Plots ---
    print("\n--- Generating State-level Summary Reports ---")
    try:
        # State-level plot
        df_st_full = decomposition_results[BASE_ANALYSIS_LEVEL]['st']
        if not df_st_full.empty:
            state_summary_dir = os.path.join(OUTPUT_DIR, 'st_summary')
            create_combined_summary_plot(
                df=df_st_full,
                level='st',
                output_dir=state_summary_dir,
                file_name='state_summary_combined.pdf',
                plot_title='State-Level Decomposition Summary'
            )
        else:
            print("  State-level data is empty. Skipping state summary plot.")

        # Cook County plot
        df_ct_full = decomposition_results[BASE_ANALYSIS_LEVEL]['ct']
        df_cook_county = df_ct_full[df_ct_full['UnitName'] == '17031'].copy()
        if not df_cook_county.empty:
            cook_county_summary_dir = os.path.join(OUTPUT_DIR, 'st_summary', 'cook_county')
            create_combined_summary_plot(
                df=df_cook_county,
                level='ct',
                output_dir=cook_county_summary_dir,
                file_name='cook_county_summary_combined.pdf',
                plot_title='Cook County Decomposition Summary'
            )
        else:
            print("  Could not find Cook County ('17031') in the data. Skipping specific plot.")

    except (KeyError, TypeError) as e:
        print(f"  Could not generate state-level summary reports: {e}")
    # --- END State-level Summary ---

    # --- NEW: Generating LDR Variation Summary Plots ---
    print("\n--- Generating LDR Variation Summary Plots ---")
    try:
        # State-level summary
        create_ldr_variation_summary_plot(
            decomposition_results=decomposition_results[BASE_ANALYSIS_LEVEL],
            output_dir=os.path.join(OUTPUT_DIR, 'st_summary'),
            file_name='state_ldr_variation_summary.pdf',
            plot_title='State-Level LDR Selection Dominance Variation',
            levels_to_plot=['tr', 'cm', 'ct', 'st'],
            cook_county_only=False
        )
        
        # Cook County summary
        create_ldr_variation_summary_plot(
            decomposition_results=decomposition_results[BASE_ANALYSIS_LEVEL],
            output_dir=os.path.join(OUTPUT_DIR, 'st_summary', 'cook_county'),
            file_name='cook_county_ldr_variation_summary.pdf',
            plot_title='Cook County LDR Selection Dominance Variation',
            levels_to_plot=['tr', 'cm', 'ct'],
            cook_county_only=True
        )
    except (KeyError, TypeError) as e:
        print(f"  Could not generate LDR variation summary reports: {e}")
    # --- END NEW ---

    print("\n--- Plot Generation Complete ---")


if __name__ == "__main__":
    main() 