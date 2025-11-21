'''
plotPrice.py

This script is used for visualizing the results from the Price equation decomposition
calculated in aggregatePriceV4.py and stored in all_decomposition_results.pkl.
'''

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from scipy.stats import linregress
import numpy as np

# --- Font Configuration for Computer Modern ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'] # Specify Computer Modern and fallbacks
plt.rcParams['mathtext.fontset'] = 'cm' # Use Computer Modern for math text
plt.rcParams['axes.unicode_minus'] = False # Ensure minus signs render correctly

custom_purple = '#633673'
custom_orange = '#E77429'

# --- Configuration ---
RESULTS_PICKLE_FILEPATH = "output_terms/all_decomposition_results.pkl"
# OUTPUT_PLOT_DIR is now set per run
# Define HIERARCHY_LEVELS here as it's used in multiple places
HIERARCHY_LEVELS = ['bg', 'tr', 'cm', 'ct', 'st']

# --- Plotting Style Configuration ---
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 10
AXIS_LABEL_FONTSIZE = 22
TICK_LABEL_FONTSIZE = 14
HISTOGRAM_X_RANGE_STD_MULTIPLIER = 3 # For mean +/- N * std_dev
SCATTER_OUTLIER_LABEL_STD_DEV_THRESHOLD = .33 # Std devs from fit to label points

# --- Plot Labels Dictionary ---
PLOT_LABELS = {
    # Histograms
    'hist_title_suffix': 'Distribution of {metric_name}\nLevel: {agg_level_name} (Base: {base_level_name})',
    'hist_initial_log_income_metric_name': 'Initial Log Average Income',
    'hist_initial_log_income_xlabel_fmt': 'Initial Log Average Income ({col_name})',
    'hist_lhs_g_metric_name': 'LHS Log Income Growth Rate (gro path)',
    'hist_lhs_g_xlabel_fmt': 'LHS Log Income Growth Rate ({col_name})',
    'hist_pop_g_metric_name': 'Population Growth Rate (PopG)',
    'hist_pop_g_xlabel_fmt': 'Population Growth Rate ({col_name})',
    'hist_pop_g_title_suffix_truncated': ' (Displaying PopG <= 1.0)',
    'hist_density_ylabel': "Density",
    'hist_unit_level_summary_title_fmt': '{title_prefix}\nAgg. Level: {agg_level_name} (Base: {base_level_name})',
    'hist_unit_level_summary_title_truncated_suffix': ' (Truncated to +/-1 Display)',

    # Scatter Plots (General - specific instances will format these)
    'scatter_title_fmt': '{y_metric} vs. {x_metric}, Colored by {color_metric}\nLevel: {agg_level_name} (Base: {base_level_name})',
    'scatter_xlabel_fmt': '{label}',
    'scatter_ylabel_fmt': '{label}',
    'scatter_colorlabel_fmt': '{label} ',

    # Specific Scatter Plot Metrics (to be used in format strings above)
    'metric_lhs_g': r'$\Delta_{\bar\gamma_{cm}}$', # Represents , the parent P's own direct empirical growth. General: $\bar{\gamma}_P$
    'metric_pop_g': r'$\gamma_{p_{cm}}$', # Represents PopG_{agg_level_name}. General: $\gamma_{p,P}$ or $\Delta \ln p_P / \Delta t$
    'metric_initial_log_income': r'$\log\bar y_{cm,0}$', # Represents LogAvgIncInitial_{agg_level_name}. General: $\ln \bar{y}_{P,0}$
    'metric_transmission_g': 'One-Level Transmission (gro)', # Represents the transmission term in the _gro path Price equation (e.g., E[w_k * gamma_k]/W_K). LaTeX: $\frac{1}{W_P} E_P[w_c \bar{\gamma}_c]$
    'metric_selection_g': 'Selection Term (gro)', # Represents the selection term in the _gro path Price equation (e.g., Cov(w_k, ln y_k)/W_K). LaTeX: $\frac{1}{W_P} \text{Cov}_P(w_c, \ln \bar{y}_{c,0})$
    'metric_agg_growth_g':  r'$\bar\gamma_{cm}$', # Represents LHS_SingleLevel_{agg_level_name}_gro. General: $\Delta E_P[\ln \bar{y}_c] / \Delta t$
    'metric_avg_child_growth_g': r'$\text{E}_{cm,tr} [f_{bg}\bar\gamma_{bg} ]$', # Represents TransmissionDirectChildGrowth_{agg_level_name}_gro. General: $E_P[\frac{p_c w_c}{P W_P} \bar{\gamma}_c]$ (approx) or more simply the weighted average child growth contributing to LHS_SingleLevel
    'metric_sel_inc_path': 'Selection Term (Income Path)', # Represents Sel_{P_from_c}_inc. LaTeX: $\frac{1}{E_P[\bar{y}_{c,0}]} \text{Cov}_P(\bar{\gamma}_c, \bar{y}_{c,0})$
    'metric_sel_gro_path': 'Selection Term (Growth Path)', # Same as metric_selection_g. LaTeX: $\frac{1}{W_P} \text{Cov}_P(w_c, \ln \bar{y}_{c,0})$

    # New entries for _inc path scatter
    'metric_emper_g': r'$\bar\gamma_{ \text{E}[\bar y_{cm}]}$', # Represents TotalEmpiricalGrowth_{agg_level_name}_inc. General: $\Delta \ln E_P[\bar{y}_c] / \Delta t$
    'metric_expected_child_direct_growth_inc': r'$\text{E}_{cm,tr} [\bar\gamma_{bg}]$', # Represents ExpectedChildDirectGrowth_{agg_level_name}_inc. General: $E_P[\Delta \ln \bar{y}_c] / \Delta t$

    # Entries for Aggregate Growth vs Cumulative Selection plots
    'metric_cum_sel_inc': r'$\omega_{cm}^{\bar y}$', # Represents cum_sel_inc. LaTeX: $\sum \text{Sel}_{inc}$
    'metric_cum_sel_gro': r'$\omega_{cm}^{p}$', # Represents cum_sel_gro. LaTeX: $\sum \text{Sel}_{gro}$
    'metric_rel_initial_log_income': r'$\ln \bar{y}_{cm,0}-\langle\ln \bar{y}_{cm,0}\rangle_{ct}$', # Represents LogAvgIncInitial - mean(LogAvgIncInitial). LaTeX: $\ln \bar{y}_{P,0} - \mathrm{E}[\ln \bar{y}_{P,0}]$

    # Entries for de-meaned aggregate growth plots
    'metric_rel_lhs_g': r'$\gamma_{ \text{E}[\bar y_{cm}]}-\langle\gamma_{ \text{E}[\bar y_{cm}]}\rangle_{ct}$', # Represents TotalEmpiricalGrowth_gro - mean(TotalEmpiricalGrowth_gro). LaTeX: $\bar{\gamma}_P - \mathrm{E}[\bar{\gamma}_P]$
    'metric_rel_total_emp_growth_inc': r'$\bar\gamma_{cm}-\langle\bar\gamma_{cm}\rangle_{ct}$', # Represents TotalEmpiricalGrowth_inc - mean(TotalEmpiricalGrowth_inc). LaTeX: $\Delta \ln E_P[\bar{y}_c] - \mathrm{E}[\Delta \ln E_P[\bar{y}_c]]$

    # Heatmaps - Focused Transmitted
    'heatmap_focused_transmitted_sel_title_fmt': 'Transmitted/Direct Selection as Fraction of Unit Growth ("S:{s_origin_level} P:{p_target_level}" base)', # Modified to be more generic for _gro & _inc if needed
    'heatmap_focused_sel_gro_title_fmt': 'Transmitted/Direct Selection as Fraction of Unit Growth ("{base_level_name}" base)',
    'heatmap_focused_sel_inc_title_fmt': 'Transmitted/Direct Selection (Income Path) as Fraction of Unit Growth ("S:{s_origin_level} P:{p_target_level}" base)',
    'heatmap_focused_sel_inc_title_fmt_main': 'Transmitted/Direct Selection (Income Path) as Fraction of Unit Growth (\\"{base_level_name}\\" base)',
    'heatmap_focused_s_origin_xlabel': "S_origin: True Origin Level of Selection Pressure / Summary Type",
    'heatmap_focused_s_origin_inc_xlabel': "S_origin: True Origin Level of Selection Pressure / Summary Type (Income Path)",
    'heatmap_focused_p_target_ylabel': "P_target: Unit/Level Receiving Selection Term",
    'heatmap_focused_cbarlabel': "Fraction of Unit's Growth (Shared Scale)",
    
    # Heatmaps - Aggregated Summary
    'heatmap_aggregated_summary_title_fmt': 'Aggregated Variation & Magnitude Summary ("{base_level_name}" base)',
    'heatmap_aggregated_summary_xlabel': "Aggregation Level",
    'heatmap_aggregated_summary_ylabel': "Metric",
    'heatmap_aggregated_summary_cbarlabel': "Avg. Normalized Value (across units in level)",

    # Selection Term Scatters (Inc vs Gro)
    'sel_scatter_title_fmt': '{desc_for_title}\n(Base: {base_level_name}, P_target: {p_target_level})',
    'sel_scatter_inc_xlabel_fmt': 'Selection Term ({col_name}, Income Path)',
    'sel_scatter_gro_ylabel_fmt': 'Selection Term ({col_name}, Growth Path)',
}

# --- FIPS to Name Mapping ---
# User can expand this map as needed
FIPS_TO_NAME_MAP = {
    '17031': 'Cook County', # Example for Cook County
    '17': 'Chicago MSA (IL)',
    '17043': 'DuPage County',
    '17089': 'Kane County',
    '17097': 'Lake County',
    '17111': 'McHenry County',
    '17197': 'Will County',
    '17093': 'Kendall County',
}

# Specific communities for focused heatmaps - can be global or passed
SPECIFIC_COMMUNITY_NAMES_FOR_HEATMAP = [
    "NEAR WEST SIDE", "NEAR SOUTH SIDE", "HYDE PARK", "LOOP", 
    "NEAR NORTH SIDE", "AUBURN GRESHAM", "ENGLEWOOD", "LOGAN SQUARE"
]

# --- Helper Functions ---

def load_decomposition_results(filepath):
    """Loads the decomposition results from a pickle file."""
    if not os.path.exists(filepath):
        print(f"Error: Results file not found at {filepath}")
        return None
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded decomposition results from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading decomposition results: {e}")
        return None

def create_output_directory(dir_path):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            print(f"Created output directory: {dir_path}")
        except Exception as e:
            print(f"Error creating output directory {dir_path}: {e}")
            return False
    return True

def plot_histogram_kde(data_series, title, xlabel, output_filename, stats_data_series=None, abs_x_axis=False):
    """Generates and saves a histogram with a KDE overlay for a given data series.
    Optionally, calculates mean/std from a different series for display.
    """
    if data_series is None or data_series.empty or data_series.isnull().all():
        print(f"  Skipping plot: {title} (no valid data for histogram/KDE)." )
        return

    sns.set_theme(style="whitegrid") # Apply a seaborn theme for prettier plots
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)) # Slightly larger figure size
    ax = sns.histplot(data_series, kde=True, stat="density", bins=30, common_norm=False,color=custom_purple)
    
    # plt.title(title, fontsize=AXIS_LABEL_FONTSIZE + 2) # Title slightly larger
    plt.xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Density", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)
    
    series_for_stats = data_series
    if stats_data_series is not None and not stats_data_series.empty and not stats_data_series.isnull().all():
        series_for_stats = stats_data_series
    elif stats_data_series is not None and (stats_data_series.empty or stats_data_series.isnull().all()):
        print(f"  Note for plot '{title}': Provided stats_data_series is empty/all NaN. Using main data_series for stats.")
        
    mean_val_stats = series_for_stats.mean() # For text box
    std_val_stats = series_for_stats.std()   # For text box

    # For xlim, use the data actually being plotted (data_series)
    mean_val_plot = data_series.mean()
    std_val_plot = data_series.std()

    if pd.notna(mean_val_plot) and pd.notna(std_val_plot) and std_val_plot > 0 and HISTOGRAM_X_RANGE_STD_MULTIPLIER > 0:
        x_min_limit = mean_val_plot - HISTOGRAM_X_RANGE_STD_MULTIPLIER * std_val_plot
        x_max_limit = mean_val_plot + HISTOGRAM_X_RANGE_STD_MULTIPLIER * std_val_plot
        plt.xlim(x_min_limit, x_max_limit)
        print(f"    Set x-axis limit for '{title}' to [{x_min_limit:.2f}, {x_max_limit:.2f}] (mean +/- {HISTOGRAM_X_RANGE_STD_MULTIPLIER}*std).")

    stats_text = f"Mean: {mean_val_stats:.3f}\\nStd Dev: {std_val_stats:.3f}"
    
    if pd.isna(mean_val_stats) and pd.isna(std_val_stats):
        stats_text = "Mean: N/A\nStd Dev: N/A"
    elif pd.isna(mean_val_stats):
        stats_text = f"Mean: N/A"+f"\n"+f"Std Dev: {std_val_stats:.3f}"
    elif pd.isna(std_val_stats):
        stats_text = f"Mean: {mean_val_stats:.3f} \nStd Dev: N/A"

    plt.text(0.97, 0.97, stats_text,
             transform=ax.transAxes,
             fontsize=TICK_LABEL_FONTSIZE - 2,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.grid(visible=False)
    plt.axvline(0,color="#3C3C3C",linewidth=1)
    plt.tight_layout()
    if abs_x_axis:
        plt.xlim(left=0)
    try:
        plt.savefig(output_filename, format='pdf')
        print(f"  Successfully saved plot: {output_filename}")
    except Exception as e:
        print(f"  Error saving plot {output_filename}: {e}")
    plt.close()

def plot_scatter_colored(df, x_col, y_col, color_col, title, xlabel, ylabel, color_label, output_filename,\
                         special=False,override=False,diff=False,weighted=False,pop=None,city_lbl=False,fit=False):
    """Generates and saves a scatter plot with points colored by a third variable."""
    
    unique_required_cols = list(set([x_col, y_col, color_col, 'UnitName']))
    if weighted:
        unique_required_cols = list(set([x_col, y_col, color_col, pop, 'UnitName']))

    
    if not all(col in df.columns for col in unique_required_cols):
        missing = [col for col in unique_required_cols if col not in df.columns]
        print(f"  Skipping plot: {title} (missing one or more required columns: {missing}).")
        return
    
    df_plotting = df[unique_required_cols].dropna()

    if df_plotting.empty:
        print(f"  Skipping plot: {title} (no valid data after dropping NaNs from essential columns)." )
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    r_squared = np.nan # Initialize r_squared

    x_vals = df_plotting[x_col]
    y_vals = df_plotting[y_col]
    if diff: 
        y_vals = y_vals-x_vals
        df_plotting["diff"] = y_vals
    elif weighted:
        y_vals = y_vals * df_plotting[pop]/df_plotting[pop].mean()
        df_plotting["wght"] = y_vals
    if len(x_vals.unique()) > 1 and len(df_plotting) >= 2:

        slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        r_squared = r_value**2
        y_fit = slope * x_vals + intercept
        if fit:
            ax.set_title(title, fontsize=AXIS_LABEL_FONTSIZE + 2) # Title slightly larger

            ax.plot(x_vals, y_fit, color='k', linewidth=1.25, label=f'Linear Fit (R²={r_squared:.2f})', alpha=0.5)
        fit_text = f"Slope: {slope:.3f}\nIntercept: {intercept:.3f}\nR-squared: {r_squared:.3f}"
        ax.legend(loc='lower right', fontsize=TICK_LABEL_FONTSIZE - 2)

        if 'UnitName' in df_plotting.columns:
            residuals = y_vals - y_fit
            std_residuals = np.std(residuals)
            label_points_flag = False
            labeling_threshold_std_multi = 0.0 # This will be overridden by global
            if pd.notna(std_residuals) and std_residuals > 1e-9 and SCATTER_OUTLIER_LABEL_STD_DEV_THRESHOLD > 0:
                label_points_flag = True
                # Use the global threshold directly
                actual_labeling_threshold = SCATTER_OUTLIER_LABEL_STD_DEV_THRESHOLD * std_residuals
                points_labeled_count = 0
                for index, row_data in df_plotting.iterrows():
                    residual_val = residuals.loc[index] 
                    if abs(residual_val) > actual_labeling_threshold:
                        # Determine horizontal and vertical alignment based on residual
                        if residual_val > 0: # Point is above the fit line
                            ha_label = 'right'
                            va_label = 'center'
                        else: # Point is below or on the fit line
                            ha_label = 'left'
                            va_label = 'bottom' # Current default
                        if city_lbl:
                            if diff:
                                ax.text(row_data[x_col], row_data["diff"], str(row_data['UnitName']), 
                                        fontsize=TICK_LABEL_FONTSIZE - 7, alpha=0.6, ha=ha_label, va=va_label,
                                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.3, ec='none'))
                            else:
                                ax.text(row_data[x_col], row_data[y_col], str(row_data['UnitName']), 
                                        fontsize=TICK_LABEL_FONTSIZE - 7, alpha=0.6, ha=ha_label, va=va_label,
                                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.3, ec='none'))
                        points_labeled_count += 1
                if points_labeled_count > 0:
                    print(f"    Labeled {points_labeled_count} points deviating by > {SCATTER_OUTLIER_LABEL_STD_DEV_THRESHOLD:.1f} std dev from fit line for '{title}'.")
    else:
        fit_text = "Fit not calculated\n(insufficient data or\nidentical x-values)"

    color_data = df_plotting[color_col]
    data_min_abs = color_data.min()
    data_max_abs = color_data.max()
    mean_val = color_data.mean()
    std_val = color_data.std()
    chosen_palette = None
    chosen_hue_norm = None
    if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
        vmin_norm = mean_val - 3 * std_val
        vmax_norm = mean_val + 3 * std_val
    else:
        vmin_norm = data_min_abs
        vmax_norm = data_max_abs
    if pd.isna(data_min_abs) or pd.isna(data_max_abs):
        chosen_palette = "viridis"
        chosen_hue_norm = Normalize(vmin=0, vmax=1)
    # elif data_min_abs > 0 and data_max_abs > 0:
    #     chosen_palette = "inferno"
    #     eff_inferno_vmin = max(1e-9, vmin_norm)
    #     eff_inferno_vmax = vmax_norm
    #     if eff_inferno_vmin >= eff_inferno_vmax:
    #          eff_inferno_vmin = data_min_abs
    #          eff_inferno_vmax = data_max_abs
    #     chosen_hue_norm = Normalize(vmin=eff_inferno_vmin, vmax=eff_inferno_vmax)
    else:

        chosen_palette = LinearSegmentedColormap.from_list("custom_diverging", [custom_purple, "white", custom_orange])
        if vmin_norm == vmax_norm:
            chosen_hue_norm = Normalize(vmin=vmin_norm - 0.1, vmax=vmax_norm + 0.1)
        elif vmin_norm < 0 < vmax_norm:
            chosen_hue_norm = TwoSlopeNorm(vmin=vmin_norm, vcenter=0, vmax=vmax_norm)
        else:
            print(f"  Note for plot '{title}': Data range [{vmin_norm:.2f}, {vmax_norm:.2f}] for color '{color_col}' does not strictly bracket 0 for TwoSlopeNorm. Using standard Normalize.")
            chosen_hue_norm = Normalize(vmin=vmin_norm, vmax=vmax_norm)
    if diff:
        scatter_plot = sns.scatterplot(
            ax=ax, data=df_plotting, x=x_col, y="diff", hue=color_col, edgecolor='#3C3C3C',
            palette=chosen_palette, hue_norm=chosen_hue_norm, s=50, alpha=.95)
    elif weighted:
        scatter_plot = sns.scatterplot(
            ax=ax, data=df_plotting, x=x_col, y="wght", hue=color_col, edgecolor='#3C3C3C',
            palette=chosen_palette, hue_norm=chosen_hue_norm, s=50, alpha=.95)
    else:
        scatter_plot = sns.scatterplot(
            ax=ax, data=df_plotting, x=x_col, y=y_col, hue=color_col, edgecolor='#3C3C3C',
            palette=chosen_palette, hue_norm=chosen_hue_norm, s=50, alpha=.95)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE); ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)
    # ax.set_ylim(top=.021)
    if fit:
        ax.text(0.68, 0.95, fit_text, transform=ax.transAxes, fontsize=TICK_LABEL_FONTSIZE - 3.5, verticalalignment='top', horizontalalignment='left', color='#3C3C3C', bbox=dict(boxstyle='round,pad=0.5', fc='#FBF4EE', alpha=1.0))
    plt.axhline(y=0, color='#3C3C3C',  linewidth=.5,zorder=0)
    plt.axvline(x=0, color='#3C3C3C',  linewidth=.5,zorder=0)


    if special:
        # plt.axvline(x=0, color='#3C3C3C',  linewidth=.5,zorder=0)
        if xlabel ==  PLOT_LABELS['metric_cum_sel_inc']:
            plt.xlim(-45,10)
            plt.ylim(bottom=-.035)
        elif xlabel == PLOT_LABELS['metric_cum_sel_gro']:
            plt.xlim(-40,25)
    if ax.get_legend() is not None: ax.get_legend().remove()
    if chosen_hue_norm is not None:
        sm = plt.cm.ScalarMappable(cmap=chosen_palette, norm=chosen_hue_norm); sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label(color_label, fontsize=AXIS_LABEL_FONTSIZE*.8); cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE*.8)
    else: print(f"  Skipping colorbar for plot '{title}' as color normalization was not applied.")
    fig.tight_layout()
    plt.grid(False)
    try: fig.savefig(output_filename, format='pdf'); print(f"  Successfully saved plot: {output_filename}")
    except Exception as e: print(f"  Error saving plot {output_filename}: {e}")
    plt.close(fig) # Ensure the specific figure is closed

# --- Helper function to prepare normalized heatmap data --- 
def _prepare_normalized_heatmap_data(decomposition_data_at_base_level, base_level_name_str, path_suffix, 
                                     specific_community_names_list, fips_to_name_map_dict, hierarchy_levels_list):
    print(f"\n--- Preparing Normalized Heatmap Data for Base '{base_level_name_str}', Path: '{path_suffix}' ---")

    target_community_unitnames = [] 
    df_CM_full = None
    if 'cm' in decomposition_data_at_base_level and isinstance(decomposition_data_at_base_level['cm'], pd.DataFrame) and not decomposition_data_at_base_level['cm'].empty:
        df_CM_full = decomposition_data_at_base_level['cm']
        if 'UnitName' in df_CM_full.columns:
            for name in specific_community_names_list:
                if name in df_CM_full['UnitName'].values:
                    target_community_unitnames.append(name)
        pop_initial_cm_col = 'PopInitial_cm' 
        if 'UnitName' in df_CM_full.columns and pop_initial_cm_col in df_CM_full.columns:
            df_CM_full_sorted = df_CM_full.dropna(subset=[pop_initial_cm_col])
            if not df_CM_full_sorted.empty: # Ensure not empty after dropna
                df_CM_full_sorted[pop_initial_cm_col] = pd.to_numeric(df_CM_full_sorted[pop_initial_cm_col], errors='coerce')
                df_CM_full_sorted = df_CM_full_sorted.dropna(subset=[pop_initial_cm_col])
                if not df_CM_full_sorted.empty:
                    top_10_communities = df_CM_full_sorted.nlargest(10, pop_initial_cm_col)['UnitName'].tolist()
                    for name in top_10_communities:
                        if name not in target_community_unitnames: 
                            target_community_unitnames.append(name)
        target_community_unitnames = sorted(list(set(target_community_unitnames)))
    # else: print(f"  Warning: 'cm' level data not found/empty for heatmap data prep (Path: {path_suffix}).")

    target_county_labels = [] 
    df_CT_full = None
    if 'ct' in decomposition_data_at_base_level and isinstance(decomposition_data_at_base_level['ct'], pd.DataFrame) and not decomposition_data_at_base_level['ct'].empty:
        df_CT_full = decomposition_data_at_base_level['ct']
        if 'UnitName' in df_CT_full.columns:
            raw_county_fips = sorted(list(df_CT_full['UnitName'].unique()))
            for fips_code in raw_county_fips:
                target_county_labels.append(fips_to_name_map_dict.get(str(fips_code), str(fips_code))) 
    # else: print(f"  Warning: 'ct' level data not found/empty for heatmap data prep (Path: {path_suffix}).")

    target_state_labels = []
    df_ST_full = None
    if 'st' in decomposition_data_at_base_level and isinstance(decomposition_data_at_base_level['st'], pd.DataFrame) and not decomposition_data_at_base_level['st'].empty:
        df_ST_full = decomposition_data_at_base_level['st']
        if 'UnitName' in df_ST_full.columns:
            raw_state_fips = sorted(list(df_ST_full['UnitName'].unique()))
            for fips_code in raw_state_fips:
                target_state_labels.append(fips_to_name_map_dict.get(str(fips_code), str(fips_code)))
    # else: print(f"  Warning: 'st' level data not found/empty for heatmap data prep (Path: {path_suffix}).")

    y_axis_labels_dynamic = target_community_unitnames + target_county_labels + target_state_labels
    if not y_axis_labels_dynamic:
        print(f"  No target units for Y-axis (Path: {path_suffix}). Returning empty DataFrame.")
        return pd.DataFrame(), False, [], [] # Added empty list for x_axis_s_origin_levels
        
    x_axis_s_origin_levels = ['bg', 'tr', 'cm', 'ct'] 
    
    normalized_heatmap_data = pd.DataFrame(index=y_axis_labels_dynamic, columns=x_axis_s_origin_levels, dtype=float)
    found_any_data = False

    for p_target_label in y_axis_labels_dynamic: 
        p_level = None
        current_df_for_lookup = None
        original_fips_for_lookup = p_target_label

        if p_target_label in target_community_unitnames: 
            p_level = 'cm'; current_df_for_lookup = df_CM_full
        elif p_target_label in target_county_labels: 
            p_level = 'ct'; current_df_for_lookup = df_CT_full
            for fips, name in fips_to_name_map_dict.items():
                if name == p_target_label and df_CT_full is not None and 'UnitName' in df_CT_full.columns and str(fips) in df_CT_full['UnitName'].astype(str).values:
                    original_fips_for_lookup = str(fips); break
        elif p_target_label in target_state_labels: 
            p_level = 'st'; current_df_for_lookup = df_ST_full
            for fips, name in fips_to_name_map_dict.items():
                if name == p_target_label and df_ST_full is not None and 'UnitName' in df_ST_full.columns and str(fips) in df_ST_full['UnitName'].astype(str).values:
                    original_fips_for_lookup = str(fips); break
        else: 
            if df_CT_full is not None and 'UnitName' in df_CT_full.columns and str(p_target_label) in df_CT_full['UnitName'].astype(str).values:
                p_level = 'ct'; current_df_for_lookup = df_CT_full
            elif df_ST_full is not None and 'UnitName' in df_ST_full.columns and str(p_target_label) in df_ST_full['UnitName'].astype(str).values:
                p_level = 'st'; current_df_for_lookup = df_ST_full
            else: continue 

        if current_df_for_lookup is None or current_df_for_lookup.empty: continue
        if 'UnitName' not in current_df_for_lookup.columns: continue

        unit_specific_row = current_df_for_lookup[current_df_for_lookup['UnitName'].astype(str) == str(original_fips_for_lookup)]
        if unit_specific_row.empty:
            for s_col_to_nan in x_axis_s_origin_levels:
                normalized_heatmap_data.loc[p_target_label, s_col_to_nan] = np.nan
            continue

        overall_growth_rate = np.nan
        if p_level: # Ensure p_level was determined
            if path_suffix == '_gro':
                effective_growth_col_norm = f'EffectiveGrowth_{p_level}_gro'
                avgg_col_norm = f'AvgG_{p_level}' 
                if effective_growth_col_norm in unit_specific_row.columns and pd.notna(unit_specific_row[effective_growth_col_norm].iloc[0]) and unit_specific_row[effective_growth_col_norm].iloc[0] != 0:
                    overall_growth_rate = unit_specific_row[effective_growth_col_norm].iloc[0]
                elif avgg_col_norm in unit_specific_row.columns and pd.notna(unit_specific_row[avgg_col_norm].iloc[0]) and unit_specific_row[avgg_col_norm].iloc[0] != 0:
                    overall_growth_rate = unit_specific_row[avgg_col_norm].iloc[0]
            elif path_suffix == '_inc':
                normalization_col_inc = f'TotalEmpiricalGrowth_{p_level}_inc'
                if normalization_col_inc in unit_specific_row.columns and pd.notna(unit_specific_row[normalization_col_inc].iloc[0]) and unit_specific_row[normalization_col_inc].iloc[0] != 0:
                    overall_growth_rate = unit_specific_row[normalization_col_inc].iloc[0]

        for s_origin_level in x_axis_s_origin_levels:
            raw_value_for_cell = np.nan
            col_name_for_raw_value = None
            if p_level: # Ensure p_level was determined for indexing hierarchy
                try:
                    s_origin_idx = hierarchy_levels_list.index(s_origin_level)
                    p_level_idx = hierarchy_levels_list.index(p_level)
                except ValueError: continue 

                if s_origin_idx < p_level_idx: 
                    direct_child_of_p_level_idx = p_level_idx - 1
                    if direct_child_of_p_level_idx < 0: continue 
                    direct_child_of_p_level = hierarchy_levels_list[direct_child_of_p_level_idx]

                    if s_origin_level == direct_child_of_p_level:
                        col_name_for_raw_value = f"Sel_{p_level}_from_{s_origin_level}{path_suffix}"
                    elif s_origin_idx < direct_child_of_p_level_idx: 
                        transmitting_level_for_sel_term_idx = s_origin_idx + 1
                        if transmitting_level_for_sel_term_idx < p_level_idx: 
                            transmitting_level_for_sel_term = hierarchy_levels_list[transmitting_level_for_sel_term_idx]
                            col_name_for_raw_value = f"Transmitted_Sel_{transmitting_level_for_sel_term}_to_{p_level}{path_suffix}"
                    
                    if col_name_for_raw_value and col_name_for_raw_value in unit_specific_row.columns:
                        raw_value_for_cell = unit_specific_row[col_name_for_raw_value].iloc[0]
            
            if pd.notna(raw_value_for_cell) and pd.notna(overall_growth_rate) and overall_growth_rate != 0:
                normalized_heatmap_data.loc[p_target_label, s_origin_level] = raw_value_for_cell / overall_growth_rate
                found_any_data = True
            else:
                normalized_heatmap_data.loc[p_target_label, s_origin_level] = np.nan
    
    return normalized_heatmap_data, found_any_data, y_axis_labels_dynamic, x_axis_s_origin_levels

def plot_variation_data(decomposition_data_at_base_level,hierarchy_levels_list,base_level_name_str,metric_plot_details,summary_heatmap_data):
    data_dict = {}
    # Iterate through the levels that will actually be columns in this heatmap
    for agg_level_col in ['tr','cm']: 
        if agg_level_col not in decomposition_data_at_base_level or not isinstance(decomposition_data_at_base_level[agg_level_col], pd.DataFrame) or decomposition_data_at_base_level[agg_level_col].empty:
            print(f"  Skipping data processing for column '{agg_level_col}' in summary heatmap: No data available in input.")
            continue # This column in summary_heatmap_data will remain NaN
        data_dict[agg_level_col] = {}
        df_level = decomposition_data_at_base_level[agg_level_col].copy()
        
        # Store per-unit calculated total variation and avg magnitude before averaging them
        unit_total_variations_gro = []
        unit_avg_magnitudes_gro = []
        unit_total_variations_inc = []
        unit_avg_magnitudes_inc = []

        for _, unit_row in df_level.iterrows():
            # --- Growth Path (_gro) ---
            sel_terms_gro_for_unit = []
            # Own selection
            child_level_idx_gro = -1
            try: child_level_idx_gro = hierarchy_levels_list.index(agg_level_col) -1
            except ValueError: pass
            if child_level_idx_gro >=0:
                child_level_gro = hierarchy_levels_list[child_level_idx_gro]
                own_sel_col_gro = f"Sel_{agg_level_col}_from_{child_level_gro}_gro"
                if own_sel_col_gro in unit_row and pd.notna(unit_row[own_sel_col_gro]):
                    sel_terms_gro_for_unit.append(unit_row[own_sel_col_gro])
            # Transmitted selection
            for col_name in unit_row.index:
                if col_name.startswith("Transmitted_Sel_") and col_name.endswith(f"_to_{agg_level_col}_gro"):
                    if pd.notna(unit_row[col_name]):
                        sel_terms_gro_for_unit.append(unit_row[col_name])
            
            # Normalizer for _gro path
            normalizer_gro = np.nan
            eff_growth_col = f"EffectiveGrowth_{agg_level_col}_gro"
            avg_g_col = f"AvgG_{agg_level_col}"
            if eff_growth_col in unit_row and pd.notna(unit_row[eff_growth_col]) and unit_row[eff_growth_col] != 0:
                normalizer_gro = unit_row[eff_growth_col]
            elif avg_g_col in unit_row and pd.notna(unit_row[avg_g_col]) and unit_row[avg_g_col] != 0:
                normalizer_gro = unit_row[avg_g_col]

            # Calculate Total Growth Variation and Growth Variation Magnitude (on-the-fly)
            if sel_terms_gro_for_unit and pd.notna(normalizer_gro):
                normalized_terms_gro = [term / normalizer_gro for term in sel_terms_gro_for_unit]
                unit_total_variations_gro.append(sum(normalized_terms_gro))
                unit_avg_magnitudes_gro.append(np.mean([abs(term) for term in normalized_terms_gro])) # Old way: avg of abs of normalized terms
            
            # --- Income Path (_inc) ---
            sel_terms_inc_for_unit = []
            # Own selection
            child_level_idx_inc = -1
            try: child_level_idx_inc = hierarchy_levels_list.index(agg_level_col) -1
            except ValueError: pass
            if child_level_idx_inc >=0:
                child_level_inc = hierarchy_levels_list[child_level_idx_inc]
                own_sel_col_inc = f"Sel_{agg_level_col}_from_{child_level_inc}_inc"
                if own_sel_col_inc in unit_row and pd.notna(unit_row[own_sel_col_inc]):
                    sel_terms_inc_for_unit.append(unit_row[own_sel_col_inc])
            # Transmitted selection
            for col_name in unit_row.index:
                if col_name.startswith("Transmitted_Sel_") and col_name.endswith(f"_to_{agg_level_col}_inc"):
                    if pd.notna(unit_row[col_name]):
                       sel_terms_inc_for_unit.append(unit_row[col_name])

            # Normalizer for _inc path
            normalizer_inc = np.nan
            total_emp_growth_inc_col = f"TotalEmpiricalGrowth_{agg_level_col}_inc"
            if total_emp_growth_inc_col in unit_row and pd.notna(unit_row[total_emp_growth_inc_col]) and unit_row[total_emp_growth_inc_col] != 0:
                normalizer_inc = unit_row[total_emp_growth_inc_col]

            # Calculate Total Income Variation and Income Variation Magnitude (on-the-fly)
            if sel_terms_inc_for_unit and pd.notna(normalizer_inc):
                normalized_terms_inc = [term / normalizer_inc for term in sel_terms_inc_for_unit]
                unit_total_variations_inc.append(sum(normalized_terms_inc))
                unit_avg_magnitudes_inc.append(np.mean([abs(term) for term in normalized_terms_inc])) # Old way: avg of abs of normalized terms

        # Calculate unweighted average for the agg_level_col and store in summary_heatmap_data
        if unit_total_variations_gro:
            summary_heatmap_data.loc["Total Growth Variation", agg_level_col] = np.mean(unit_total_variations_gro)
        if unit_avg_magnitudes_gro:
            summary_heatmap_data.loc["Growth Variation Magnitude", agg_level_col] = np.mean(unit_avg_magnitudes_gro)
        if unit_total_variations_inc:
            summary_heatmap_data.loc["Total Income Variation", agg_level_col] = np.mean(unit_total_variations_inc)
        if unit_avg_magnitudes_inc:
            summary_heatmap_data.loc["Income Variation Magnitude", agg_level_col] = np.mean(unit_avg_magnitudes_inc)

        # --- Plot Histograms for the collected per-unit values for this agg_level_col ---
        # These lists (unit_total_variations_gro, etc.) now contain all per-unit values for the current agg_level_col
        
        # Helper to avoid repetitive calls
        all_per_unit_lists_for_hist = {
            "Total Growth Variation": unit_total_variations_gro,
            "Growth Variation Magnitude": unit_avg_magnitudes_gro,
            "Total Income Variation": unit_total_variations_inc,
            "Income Variation Magnitude": unit_avg_magnitudes_inc
        }

        for metric_name, per_unit_values_list in all_per_unit_lists_for_hist.items():
            if per_unit_values_list: # Check if the list is not empty
                details = metric_plot_details[metric_name]
                
                original_series = pd.Series(per_unit_values_list)
                truncated_series = original_series[(original_series >= -1) & (original_series <= 1)]
                
                if truncated_series.empty and not original_series.empty:
                    print(f"  Note for histogram {metric_name} at Agg. Level: {agg_level_col} (Base: {base_level_name_str}): All data is outside +/-1 range. Plotting full range instead.")
                    series_to_plot = original_series
                    plot_title = f'{details["title_prefix"]}\nAgg. Level: {agg_level_col} (Base: {base_level_name_str})'
                elif truncated_series.empty and original_series.empty:
                     print(f"  Skipping histogram for {metric_name} at Agg. Level: {agg_level_col} (Base: {base_level_name_str}) - no per-unit data collected (original series also empty).")
                     continue # Skip to next metric if original data is also empty
                else:
                    series_to_plot = truncated_series
                    plot_title = f'{details["title_prefix"]} (Truncated to +/-1 Display)\nAgg. Level: {agg_level_col} (Base: {base_level_name_str})'
                if "Magnitude" in metric_name:
                    plot_histogram_kde(
                        data_series=series_to_plot, # Plot truncated data
                        stats_data_series=original_series, # Calculate stats on original data
                        title=plot_title,
                        xlabel=details["xlabel_suffix"],
                        output_filename=os.path.join(details["subdir"], f'{base_level_name_str}_{agg_level_col}_{metric_name}_hist.pdf'),
                        abs_x_axis=True
                    )
                    data_dict[agg_level_col][metric_name] = series_to_plot
                else:
                    plot_histogram_kde(
                        data_series=series_to_plot, # Plot truncated data
                        stats_data_series=original_series, # Calculate stats on original data
                        title=plot_title,
                        xlabel=details["xlabel_suffix"],
                        output_filename=os.path.join(details["subdir"], f'{base_level_name_str}_{agg_level_col}_{metric_name}_hist.pdf'),
                ) 
                    data_dict[agg_level_col][metric_name] = series_to_plot
            else:
                print(f"  Skipping histogram for {metric_name} at Agg. Level: {agg_level_col} (Base: {base_level_name_str}) - no per-unit data collected.")
        # --- End of Histogram Plotting for this agg_level_col ---

    return data_dict
def plot_transmitted_covariance_heatmap(normalized_data_input, shared_norm_object, 
                                        y_axis_labels_for_plot, x_axis_s_origin_levels_for_plot,
                                        base_level_name_str, output_plot_dir_name_prefix,
                                        title=None):
    print(f"\n--- Plotting Focused Transmitted Covariance Heatmap for Base '{base_level_name_str}' (using pre-normalized data) ---")

    heatmap_data_gro = normalized_data_input.copy()

    if heatmap_data_gro.empty or heatmap_data_gro.isnull().all().all():
        print(f"  Pre-normalized heatmap data for _gro path is empty or all NaN. Skipping plot for base '{base_level_name_str}'.")
        return

    sum_col_gro = 'Sum_Sel_Contrib_Frac'
    avg_mag_col_gro = 'Avg_Norm_Sel_Mag'

    heatmap_data_gro[sum_col_gro] = heatmap_data_gro[x_axis_s_origin_levels_for_plot].sum(axis=1, skipna=True)
    heatmap_data_gro[avg_mag_col_gro] = heatmap_data_gro[x_axis_s_origin_levels_for_plot].abs().mean(axis=1, skipna=True)
    # print(f"  Added '{sum_col_gro}' and '{avg_mag_col_gro}' columns to _gro heatmap data.")
    
    all_heatmap_cols_gro = x_axis_s_origin_levels_for_plot + [sum_col_gro, avg_mag_col_gro]
    plottable_heatmap_data_gro = heatmap_data_gro[[col for col in all_heatmap_cols_gro if col in heatmap_data_gro.columns]]

    if plottable_heatmap_data_gro.isnull().all().all():
        print(f"  Heatmap data for _gro path (after adding summary) is all NaN. Skipping plot for base '{base_level_name_str}'.")
        return

    # Prepare annotations as percentages
    annot_data_gro_percent = plottable_heatmap_data_gro.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "")

    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    x_col_to_label_map = {
        'bg': 'Block Groups', 'tr': 'Tracts', 'cm': 'Community Areas', 'ct': 'Counties',
        sum_col_gro: 'Total Variation', avg_mag_col_gro: 'Average Magnitude'
    }
    x_tick_labels = [x_col_to_label_map.get(col, col) for col in plottable_heatmap_data_gro.columns]

    cmap = LinearSegmentedColormap.from_list("custom_diverging", ["#633673", "white", "#E77429"])
    cmap.set_bad(color='lightgrey')
    
    sns.heatmap(plottable_heatmap_data_gro, annot=annot_data_gro_percent, fmt="s", cmap=cmap, norm=shared_norm_object, linewidths=.5, 
                cbar_kws={'label': PLOT_LABELS['heatmap_focused_cbarlabel']}, 
                xticklabels=x_tick_labels, yticklabels=y_axis_labels_for_plot) 
    plt.title(title, fontsize=AXIS_LABEL_FONTSIZE) # Title is passed in
    plt.xlabel(PLOT_LABELS['heatmap_focused_s_origin_xlabel'], fontsize=AXIS_LABEL_FONTSIZE - 2)
    plt.ylabel(PLOT_LABELS['heatmap_focused_p_target_ylabel'], fontsize=AXIS_LABEL_FONTSIZE - 2)
    plt.xticks(rotation=45, ha='right', fontsize=TICK_LABEL_FONTSIZE - 2); plt.yticks(rotation=0, fontsize=TICK_LABEL_FONTSIZE -2)
    
    # Adjust colorbar label and tick font sizes
    cbar = plt.gcf().axes[-1]
    # cbar.set_label(PLOT_LABELS['heatmap_focused_cbarlabel']) # Label set via cbar_kws
    cbar.yaxis.label.set_fontsize(AXIS_LABEL_FONTSIZE - 2)
    cbar.tick_params(labelsize=TICK_LABEL_FONTSIZE -2)

    output_filename = os.path.join(output_plot_dir_name_prefix, f"focused_heatmap_transmitted_sel_cov_base_{base_level_name_str}.pdf")
    plt.tight_layout()
    try:
        plt.savefig(output_filename, format='pdf')
        print(f"  Successfully saved focused heatmap (_gro path): {output_filename}")
    except Exception as e:
        print(f"  Error saving focused heatmap (_gro path) {output_filename}: {e}")
    plt.close()

def plot_transmitted_income_covariance_heatmap(normalized_data_input, shared_norm_object,
                                               y_axis_labels_for_plot, x_axis_s_origin_levels_for_plot,
                                               base_level_name_str, output_plot_dir_name_prefix,
                                               title=None):
    print(f"\n--- Plotting Focused Transmitted Income Covariance Heatmap for Base '{base_level_name_str}' (using pre-normalized data) ---")
    
    heatmap_data_inc = normalized_data_input.copy()

    if heatmap_data_inc.empty or heatmap_data_inc.isnull().all().all():
        print(f"  Pre-normalized heatmap data for _inc path is empty or all NaN. Skipping plot for base '{base_level_name_str}'.")
        return

    sum_col_inc = 'Sum_Sel_Contrib_Frac_Inc'
    avg_mag_col_inc = 'Avg_Norm_Sel_Mag_Inc'

    heatmap_data_inc[sum_col_inc] = heatmap_data_inc[x_axis_s_origin_levels_for_plot].sum(axis=1, skipna=True)
    heatmap_data_inc[avg_mag_col_inc] = heatmap_data_inc[x_axis_s_origin_levels_for_plot].abs().mean(axis=1, skipna=True)
    # print(f"  Added '{sum_col_inc}' and '{avg_mag_col_inc}' columns to _inc heatmap data.")

    all_heatmap_cols_inc = x_axis_s_origin_levels_for_plot + [sum_col_inc, avg_mag_col_inc]
    plottable_heatmap_data_inc = heatmap_data_inc[[col for col in all_heatmap_cols_inc if col in heatmap_data_inc.columns]]

    if plottable_heatmap_data_inc.isnull().all().all():
        print(f"  Heatmap data for _inc path (after adding summary) is all NaN. Skipping plot for base '{base_level_name_str}'.")
        return
        
    # Prepare annotations as percentages
    annot_data_inc_percent = plottable_heatmap_data_inc.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "")

    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    x_col_to_label_map = {
        'bg': 'Block Groups', 'tr': 'Tracts', 'cm': 'Community Areas', 'ct': 'Counties',
        sum_col_inc: 'Total Variation (Inc)', avg_mag_col_inc: 'Avg Magnitude (Inc)'
    }
    x_tick_labels = [x_col_to_label_map.get(col, col) for col in plottable_heatmap_data_inc.columns]

    cmap = LinearSegmentedColormap.from_list("custom_diverging", ["#633673", "white", "#E77429"])
    cmap.set_bad(color='lightgrey')
    
    sns.heatmap(plottable_heatmap_data_inc, annot=annot_data_inc_percent, fmt="s", cmap=cmap, norm=shared_norm_object, linewidths=.5,
                cbar_kws={'label': PLOT_LABELS['heatmap_focused_cbarlabel']}, 
                xticklabels=x_tick_labels, yticklabels=y_axis_labels_for_plot) 
    plt.title(title, fontsize=AXIS_LABEL_FONTSIZE) # Title is passed in
    plt.xlabel(PLOT_LABELS['heatmap_focused_s_origin_inc_xlabel'], fontsize=AXIS_LABEL_FONTSIZE -2)
    plt.ylabel(PLOT_LABELS['heatmap_focused_p_target_ylabel'], fontsize=AXIS_LABEL_FONTSIZE -2)
    plt.xticks(rotation=45, ha='right', fontsize=TICK_LABEL_FONTSIZE -2); plt.yticks(rotation=0, fontsize=TICK_LABEL_FONTSIZE -2)

    # Adjust colorbar label and tick font sizes
    cbar = plt.gcf().axes[-1]
    # cbar.set_label(PLOT_LABELS['heatmap_focused_cbarlabel']) # Label set via cbar_kws
    cbar.yaxis.label.set_fontsize(AXIS_LABEL_FONTSIZE - 2)
    cbar.tick_params(labelsize=TICK_LABEL_FONTSIZE -2)
    
    output_filename = os.path.join(output_plot_dir_name_prefix, f"focused_heatmap_transmitted_sel_cov_inc_base_{base_level_name_str}.pdf")
    plt.tight_layout()
    try:
        plt.savefig(output_filename, format='pdf')
        print(f"  Successfully saved focused income path heatmap: {output_filename}")
    except Exception as e:
        print(f"  Error saving focused income path heatmap {output_filename}: {e}")
    plt.close()

def bi_hist(data_dict):

    plot_labels = list(data_dict['tr'].keys())
    for label in plot_labels:
        sns.set_theme(style="whitegrid") # Apply a seaborn theme for prettier plots
        plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)) # Slightly larger figure size
        for agg in ['tr','cm']:
            if agg == 'tr':
                ax = sns.histplot(data_dict[agg][label], kde=True, stat="density", bins=30, common_norm=False,color=custom_orange)
            else:
                ax = sns.histplot(data_dict[agg][label], kde=True, stat="density", bins=30, common_norm=False,color=custom_purple)
            
            # plt.title(title, fontsize=AXIS_LABEL_FONTSIZE + 2) # Title slightly larger
            plt.xlabel(label, fontsize=AXIS_LABEL_FONTSIZE)
            plt.ylabel("Density", fontsize=AXIS_LABEL_FONTSIZE)
            plt.xticks(fontsize=TICK_LABEL_FONTSIZE)
            plt.yticks(fontsize=TICK_LABEL_FONTSIZE)
            m=np.mean(data_dict[agg][label]);s=np.std(data_dict[agg][label])
            print(label,m,s)
            if "Magnitude" in label:
                plt.xlim(0,m+HISTOGRAM_X_RANGE_STD_MULTIPLIER*s)
            elif "Income" in label:
                plt.xlim(-m-HISTOGRAM_X_RANGE_STD_MULTIPLIER*(1.5)*s,m+HISTOGRAM_X_RANGE_STD_MULTIPLIER*s)
            else:
                plt.xlim(-m-HISTOGRAM_X_RANGE_STD_MULTIPLIER*s,m+HISTOGRAM_X_RANGE_STD_MULTIPLIER*s)
        plt.tight_layout()
        plt.grid(False)
        plt.axvline(x=0,color='#3C3C3C',linewidth=.5)
        plt.savefig(os.path.join(f"VARIATIONHISTS/Variation_Metrics_Distributions_Combined/bi_hist_{label}.pdf"))
    input()


def plot_aggregated_variation_summary_heatmap(decomposition_data_at_base_level, hierarchy_levels_list, base_level_name_str, output_plot_dir_name_prefix,
                                              title=None):
    """
    Plots a summary heatmap of aggregated variation components for a specific parent unit (e.g., Cook County).
    This version is specifically designed to replicate the provided reference image.
    """
    print("\n--- Generating Aggregated Variation Summary Heatmap for Cook County ---")

    # --- 1. Configuration ---
    target_county_fips_end = '031' # FIPS code for Cook County
    target_county_name = "Cook County"
    
    # Define the rows and columns for the heatmap based on the reference image
    heatmap_rows = [
        'Cumulative population variation', 
        'Population variation magnitude', 
        'Cumulative income variation', 
        'Income variation magnitude'
    ]
    heatmap_cols = ['tracts', 'community areas', 'county']

    # Map the heatmap cells to the specific column names in the results DataFrame
    # (Row, Column) -> (Column Name in DataFrame, Path Suffix)
    cell_to_col_map = {
        ('Cumulative population variation', 'tracts'): f'Transmitted_Sel_tr_to_ct_gro_Ratio',
        ('Cumulative population variation', 'community areas'): f'Transmitted_Sel_cm_to_ct_gro_Ratio',
        ('Cumulative population variation', 'county'): f'Sel_ct_from_cm_gro_Ratio',
        
        ('Cumulative income variation', 'tracts'): f'Transmitted_Sel_tr_to_ct_inc_Ratio',
        ('Cumulative income variation', 'community areas'): f'Transmitted_Sel_cm_to_ct_inc_Ratio',
        ('Cumulative income variation', 'county'): f'Sel_ct_from_cm_inc_Ratio',
    }
    
    # --- 2. Data Extraction ---
    # The required data is in the county-level ('ct') DataFrame
    df_ct = decomposition_data_at_base_level.get(base_level_name_str, {}).get('ct')
    
    if df_ct is None or df_ct.empty:
        print("  Error: County-level ('ct') data not found. Cannot generate summary heatmap.")
        return None

    # Find the specific row for Cook County
    cook_county_row = df_ct[df_ct['UnitName'].str.endswith(target_county_fips_end)].copy()
    
    if cook_county_row.empty:
        print(f"  Error: Data for {target_county_name} (FIPS ending in {target_county_fips_end}) not found in 'ct' level data.")
        return None
    
    # --- 3. Heatmap Data Preparation ---
    summary_heatmap_data = pd.DataFrame(index=heatmap_rows, columns=heatmap_cols, dtype=float)

    for (row_name, col_name), source_col in cell_to_col_map.items():
        if source_col in cook_county_row.columns:
            value = cook_county_row[source_col].iloc[0]
            summary_heatmap_data.loc[row_name, col_name] = value
        else:
            print(f"  Warning: Source column '{source_col}' not found for heatmap cell ('{row_name}', '{col_name}').")
            
    # Calculate the 'magnitude' rows by taking the absolute value of the 'cumulative' rows
    summary_heatmap_data.loc['Population variation magnitude'] = summary_heatmap_data.loc['Cumulative population variation'].abs()
    summary_heatmap_data.loc['Income variation magnitude'] = summary_heatmap_data.loc['Cumulative income variation'].abs()

    print("  Successfully prepared data for summary heatmap:")
    print(summary_heatmap_data)

    # --- 4. Plotting ---
    if summary_heatmap_data.isnull().all().all():
        print("  Error: All data for heatmap is NaN. Skipping plot generation.")
        return summary_heatmap_data

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create annotation strings with '%' sign
    annot_data = summary_heatmap_data.applymap(lambda x: f'{x:.1f}%' if pd.notna(x) else '')

    # We will use two separate heatmaps with masks to achieve the two different color schemes
    mask_pop = np.zeros_like(summary_heatmap_data, dtype=bool)
    mask_pop[2:, :] = True # Mask out the bottom two rows
    
    mask_inc = np.zeros_like(summary_heatmap_data, dtype=bool)
    mask_inc[:2, :] = True # Mask out the top two rows

    # Plot Population Variation (top two rows) with a purple-based colormap
    sns.heatmap(summary_heatmap_data, mask=mask_pop, cmap="PuOr_r", annot=annot_data, fmt='',
                linewidths=.5, linecolor='white', cbar=False, ax=ax, center=0)

    # Plot Income Variation (bottom two rows) with an orange-based colormap
    sns.heatmap(summary_heatmap_data, mask=mask_inc, cmap="Oranges", annot=annot_data, fmt='',
                linewidths=.5, linecolor='white', cbar=False, ax=ax, vmin=0)

    # --- Formatting ---
    ax.set_title(title if title else target_county_name, fontsize=TITLE_FONTSIZE*1.5, pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=AXIS_LABEL_FONTSIZE, va='center')
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)

    # Manually add the mathematical symbols to the y-axis labels
    y_labels = ax.get_yticklabels()
    y_labels[0].set_text(f"{y_labels[0].get_text()}\n$\\omega_{{cm}}^p$")
    y_labels[1].set_text(f"{y_labels[1].get_text()}\n$|\\omega_{{cm}}^p|$")
    y_labels[2].set_text(f"{y_labels[2].get_text()}\n$\\omega_{{cm}}^y$")
    y_labels[3].set_text(f"{y_labels[3].get_text()}\n$|\\omega_{{cm}}^y|$")
    ax.set_yticklabels(y_labels)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_filename = os.path.join(output_plot_dir_name_prefix, "summary_heatmap.pdf")
    create_output_directory(output_plot_dir_name_prefix)
    try:
        fig.savefig(output_filename, format='pdf', bbox_inches='tight')
        print(f"  Successfully saved plot: {output_filename}")
    except Exception as e:
        print(f"  Error saving plot {output_filename}: {e}")
    plt.close(fig)

    return summary_heatmap_data

def _generate_plots_for_dataset(decomposition_data_subset, output_plot_dir_name):
    """
    Main plotting loop. Iterates through a filtered dataset to generate a standard set of plots.
    """
    print(f"\n--- Generating Plots for Dataset into: {output_plot_dir_name} ---")
    if not create_output_directory(output_plot_dir_name):
        print(f"Halting plotting for {output_plot_dir_name} as output directory could not be created.")
        return

    print("\nLoaded data keys (base analysis levels) for this run:")
    if isinstance(decomposition_data_subset, dict):
        for base_name in decomposition_data_subset.keys():
            print(f"  - {base_name}")
            # ... (rest of the print block for keys, can be kept or removed for brevity)
    else:
        print("  Decomposition data for this run is not in the expected dictionary format.")

    print("\n--- Generating and Saving Plots for this run --- ")
    if isinstance(decomposition_data_subset, dict):
        for base_level_name, levels_data in decomposition_data_subset.items():
            if not isinstance(levels_data, dict):
                print(f"Skipping base level {base_level_name}, data is not in expected format.")
                continue

            print(f"Processing base level: {base_level_name}")

            # --- Prepare data and shared norm for focused heatmaps ---
            normalized_data_gro, found_gro, y_labels_gro, x_origins_gro = _prepare_normalized_heatmap_data(
                levels_data, base_level_name, '_gro', 
                SPECIFIC_COMMUNITY_NAMES_FOR_HEATMAP, FIPS_TO_NAME_MAP, HIERARCHY_LEVELS
            )
            normalized_data_inc, found_inc, y_labels_inc, x_origins_inc = _prepare_normalized_heatmap_data(
                levels_data, base_level_name, '_inc',
                SPECIFIC_COMMUNITY_NAMES_FOR_HEATMAP, FIPS_TO_NAME_MAP, HIERARCHY_LEVELS
            )

            shared_norm = None
            all_norm_values = []
            if found_gro and not normalized_data_gro.empty:
                all_norm_values.extend(normalized_data_gro[x_origins_gro].stack().dropna().values)
            if found_inc and not normalized_data_inc.empty:
                all_norm_values.extend(normalized_data_inc[x_origins_inc].stack().dropna().values)

            if all_norm_values:
                val_min, val_max = np.min(all_norm_values), np.max(all_norm_values)
                if val_min < 0 < val_max:
                    shared_norm = TwoSlopeNorm(vmin=val_min, vcenter=0, vmax=val_max)
                elif val_min == 0 and val_max == 0:
                    shared_norm = Normalize(vmin=-0.01, vmax=0.01) 
                else:
                    if val_min >=0: shared_norm = Normalize(vmin=0, vmax=val_max if val_max > 0 else 0.01)
                    elif val_max <=0: shared_norm = Normalize(vmin=val_min if val_min < 0 else -0.01, vmax=0)
                    else: shared_norm = Normalize(vmin=val_min, vmax=val_max)
            else:
                shared_norm = Normalize(vmin=0, vmax=1) # Fallback
            
            # Call focused heatmaps with prepared data and shared norm
            if found_gro:
                plot_transmitted_covariance_heatmap(
                    normalized_data_gro, shared_norm, y_labels_gro, x_origins_gro,
                    base_level_name, output_plot_dir_name,
                    title=PLOT_LABELS['heatmap_focused_sel_gro_title_fmt'].format(base_level_name=base_level_name) # Pass title
                )
            if found_inc:
                plot_transmitted_income_covariance_heatmap(
                    normalized_data_inc, shared_norm, y_labels_inc, x_origins_inc,
                    base_level_name, output_plot_dir_name,
                     title=PLOT_LABELS['heatmap_focused_sel_inc_title_fmt_main'].format(base_level_name=base_level_name) # Pass title
                )
            
            # Call the new summary heatmap
            plot_aggregated_variation_summary_heatmap(
                levels_data, base_level_name, output_plot_dir_name, HIERARCHY_LEVELS,
                title=PLOT_LABELS['heatmap_aggregated_summary_title_fmt'].format(base_level_name=base_level_name) # Pass title
            )

            # --- NEW: Call Selection Term Scatter Plots ---
            plot_selection_term_scatters(
                levels_data, base_level_name, output_plot_dir_name, HIERARCHY_LEVELS
            )
            # --- END NEW ---

            # --- Existing Histograms and Scatter Plots ---
            for agg_level_name, df_level in levels_data.items():
                if not isinstance(df_level, pd.DataFrame) or df_level.empty:
                    # print(f"  Skipping aggregation level {agg_level_name} for base {base_level_name} (no data or not a DataFrame).")
                    continue
                
                # print(f"  Processing aggregation level for other plots: {agg_level_name}")

                # Define all necessary column names based on the NEW PKL structure
                log_avg_inc_initial_col = f'LogAvgIncInitial_{agg_level_name}'
                pop_g_col = f'PopG_{agg_level_name}'
                pop_col = f'PopInitial_{agg_level_name}'
                
                # For _gro path plots / Population Price Eqn LHS
                avg_g_col_pop_path = f'AvgG_pop_{agg_level_name}' 
                trans_direct_child_growth_col_gro = f'TransmissionDirectChildGrowth_{agg_level_name}_gro' 
                rel_avg_g_col_pop_path = f'AvgG_pop_{agg_level_name}' # Pre-calculated in new PKL
                
                 # For _inc path plots / Population Price Eqn LHS
                avg_gi_col_pop_path = f'AvgG_inc_{agg_level_name}' 
                rel_avg_gi_col_pop_path = f'AvgG_inc_{agg_level_name}' # Pre-calculated in new PKL
                

                # For agg path plots / Income Price Eqn LHS
                agg_g_col = f'AvgG_emp_{agg_level_name}' 
                expected_child_growth_inc_col = f'ExpectedChildDirectGrowth_{agg_level_name}_inc' 
                rel_agg_g_col = f'RelAvgG_emp_{agg_level_name}' # Pre-calculated in new PKL

                # Columns for retained relative plots (still needed for X and Color)
                cum_sel_gro_col = 'cum_sel_gro_Ratio'
                cum_sel_inc_col = 'cum_sel_inc_Ratio'
                # RelLogAvgIncInitial is calculated on the fly in plotPrice.py
                rel_log_avg_inc_initial_col = f'RelLogAvgIncInitial_{agg_level_name}' 


                # if log_avg_inc_initial_col in df_level.columns:
                #     plot_subdir = os.path.join(output_plot_dir_name, "InitialIncome_Distributions")
                #     create_output_directory(plot_subdir)
                #     plot_title = PLOT_LABELS['hist_title_suffix'].format(
                #         metric_name=PLOT_LABELS['hist_initial_log_income_metric_name'],
                #         agg_level_name=agg_level_name,
                #         base_level_name=base_level_name
                #     )
                #     plot_xlabel = PLOT_LABELS['hist_initial_log_income_xlabel_fmt'].format(col_name=log_avg_inc_initial_col)
                #     plot_filename = os.path.join(plot_subdir, f'{base_level_name}_{agg_level_name}_InitialLogAvgInc_dist.pdf')
                #     plot_histogram_kde(df_level[log_avg_inc_initial_col], plot_title, plot_xlabel, plot_filename)

                # # Histogram for the new AvgG (LHS of Population Price Eqn)
                # if avg_g_col_pop_path in df_level.columns: 
                #     plot_subdir = os.path.join(output_plot_dir_name, "IncomeGrowth_Distributions") 
                #     create_output_directory(plot_subdir)
                #     plot_title = PLOT_LABELS['hist_title_suffix'].format(
                #         metric_name=PLOT_LABELS['hist_lhs_g_metric_name'], # Label "LHS Log Income Growth Rate (gro path)" now refers to Pop Eqn LHS
                #         agg_level_name=agg_level_name,
                #         base_level_name=base_level_name
                #     )
                #     plot_xlabel = PLOT_LABELS['hist_lhs_g_xlabel_fmt'].format(col_name=avg_g_col_pop_path) 
                #     plot_filename = os.path.join(plot_subdir, f'{base_level_name}_{agg_level_name}_AvgG_PopPathLHS_dist.pdf') # Updated filename for clarity
                #     plot_histogram_kde(df_level[avg_g_col_pop_path], plot_title, plot_xlabel, plot_filename) 

                # if pop_g_col in df_level.columns:
                #     full_pop_g_series = df_level[pop_g_col]
                #     display_pop_g_series = full_pop_g_series[full_pop_g_series <= 1.0].copy() 
                #     plot_subdir = os.path.join(output_plot_dir_name, "PopulationGrowth_Distributions")
                #     create_output_directory(plot_subdir)
                #     plot_title = PLOT_LABELS['hist_title_suffix'].format(
                #         metric_name=PLOT_LABELS['hist_pop_g_metric_name'],
                #         agg_level_name=agg_level_name,
                #         base_level_name=base_level_name
                #     ) + (PLOT_LABELS['hist_pop_g_title_suffix_truncated'] if display_pop_g_series.max() > 1.0 else "") # Suffix for truncation
                #     plot_xlabel = PLOT_LABELS['hist_pop_g_xlabel_fmt'].format(col_name=pop_g_col)
                #     plot_filename = os.path.join(plot_subdir, f'{base_level_name}_{agg_level_name}_PopG_dist.pdf')
                #     plot_histogram_kde(display_pop_g_series, 'plot_title', plot_xlabel, plot_filename, stats_data_series=full_pop_g_series)

                # # --- Start of New Scatter Plot Implementations ---

                # # Plot 1: x: transmitted growth rate (gro), y: Agg_G (gro), color: PopG
                # if (trans_direct_child_growth_col_gro in df_level.columns and
                #     avg_g_col_pop_path in df_level.columns and
                #     pop_g_col in df_level.columns):
                #     plot_subdir_1 = os.path.join(output_plot_dir_name, "Scatter_AvgG_PopPathLHS_vs_ExpChildDirectGro_by_PopG") # CORRECTED
                #     create_output_directory(plot_subdir_1)
                #     title_1 = PLOT_LABELS['scatter_title_fmt'].format(
                #         y_metric=r'$\Delta_{avg}$', # metric_lhs_g represents AvgG
                #         x_metric=PLOT_LABELS['metric_avg_child_growth_g'], # Transmission
                #         color_metric=PLOT_LABELS['metric_pop_g'].split('(')[0].strip(),
                #         agg_level_name=agg_level_name, base_level_name=base_level_name)
                #     xlabel_1 = PLOT_LABELS['scatter_xlabel_fmt'].format(label=PLOT_LABELS['metric_avg_child_growth_g'])
                #     ylabel_1 = r'$\Delta_{\bar\gamma}^{avg}$'
                #     colorlabel_1 = PLOT_LABELS['scatter_colorlabel_fmt'].format(label=PLOT_LABELS['metric_pop_g'])
                #     filename_1 = os.path.join(plot_subdir_1, f'{base_level_name}_{agg_level_name}_AvgG_PopPathLHS_vs_ExpChildDirectGro_by_PopG.pdf') # CORRECTED
                #     plot_scatter_colored(df_level, expected_child_growth_inc_col, avg_g_col_pop_path, pop_g_col,
                #                          ' ', xlabel_1, ylabel_1, colorlabel_1, filename_1,diff=True,special=True,city_lbl=True)

                # # Plot 2: x: transmitted growth rate (inc), y: EmpiricalGrowth (inc), color: PopG
                # if (expected_child_growth_inc_col in df_level.columns and
                #     agg_g_col in df_level.columns and
                #     pop_g_col in df_level.columns):
                #     plot_subdir_2 = os.path.join(output_plot_dir_name, "Scatter_AggG_IncPathLHS_vs_ExpChildDirectInc_by_PopG") # CORRECTED
                #     create_output_directory(plot_subdir_2)
                #     title_2 = PLOT_LABELS['scatter_title_fmt'].format(
                #         y_metric=r'$\Delta_{agg}$', # EmpiricalGrowth (inc)
                #         x_metric=PLOT_LABELS['metric_expected_child_direct_growth_inc'], # Transmission (inc)
                #         color_metric=PLOT_LABELS['metric_pop_g'].split('(')[0].strip(),
                #         agg_level_name=agg_level_name, base_level_name=base_level_name)
                #     xlabel_2 = PLOT_LABELS['scatter_xlabel_fmt'].format(label=PLOT_LABELS['metric_expected_child_direct_growth_inc'])
                #     ylabel_2 = r'$\Delta_{\bar\gamma}^{agg}$'
                #     colorlabel_2 = PLOT_LABELS['scatter_colorlabel_fmt'].format(label=PLOT_LABELS['metric_pop_g'])
                #     filename_2 = os.path.join(plot_subdir_2, f'{base_level_name}_{agg_level_name}_AggG_IncPathLHS_vs_ExpChildDirectInc_by_PopG.pdf') # CORRECTED
                #     plot_scatter_colored(df_level, expected_child_growth_inc_col, agg_g_col, pop_g_col,
                #                          ' ', xlabel_2, ylabel_2, colorlabel_2, filename_2,diff=True,special=True)

                # # Plot 3: x: initial income, y: EmpiricalGrowth (inc), color: PopG
                # if (log_avg_inc_initial_col in df_level.columns and
                #     agg_g_col in df_level.columns and
                #     pop_g_col in df_level.columns):
                #     plot_subdir_3 = os.path.join(output_plot_dir_name, "Scatter_AggG_IncPathLHS_vs_InitInc_by_PopG") # CORRECTED
                #     create_output_directory(plot_subdir_3)
                #     title_3 = PLOT_LABELS['scatter_title_fmt'].format(
                #         y_metric=PLOT_LABELS['metric_lhs_i'], # EmpiricalGrowth (inc)
                #         x_metric=PLOT_LABELS['metric_initial_log_income'], # Initial Income
                #         color_metric=PLOT_LABELS['metric_pop_g'].split('(')[0].strip(),
                #         agg_level_name=agg_level_name, base_level_name=base_level_name)
                #     xlabel_3 = PLOT_LABELS['scatter_xlabel_fmt'].format(label=PLOT_LABELS['metric_initial_log_income'])
                #     ylabel_3 = PLOT_LABELS['scatter_ylabel_fmt'].format(label=PLOT_LABELS['metric_lhs_i'])
                #     colorlabel_3 = PLOT_LABELS['scatter_colorlabel_fmt'].format(label=PLOT_LABELS['metric_pop_g'])
                #     filename_3 = os.path.join(plot_subdir_3, f'{base_level_name}_{agg_level_name}_AggG_IncPathLHS_vs_InitInc_by_PopG.pdf') # CORRECTED
                #     plot_scatter_colored(df_level, avg_g_col_pop_path, agg_g_col, pop_g_col,
                #                         ' ', xlabel_3, ylabel_3, colorlabel_3, filename_3,special=True,)
                
                # # Plot 4: x: initial income, y: Agg_G (gro), color: PopG
                # if (log_avg_inc_initial_col in df_level.columns and
                #     avg_g_col_pop_path in df_level.columns and # Was just changed to avg_g_col in a previous step
                #     pop_g_col in df_level.columns):
                #     plot_subdir_4 = os.path.join(output_plot_dir_name, "Scatter_AvgG_PopPathLHS_vs_InitInc_by_PopG") # CORRECTED
                #     create_output_directory(plot_subdir_4)
                #     title_4 = PLOT_LABELS['scatter_title_fmt'].format(
                #         y_metric=PLOT_LABELS['metric_lhs_g'], # metric_lhs_g represents AvgG
                #         x_metric=PLOT_LABELS['metric_initial_log_income'], # Initial Income
                #         color_metric=PLOT_LABELS['metric_pop_g'].split('(')[0].strip(),
                #         agg_level_name=agg_level_name, base_level_name=base_level_name)
                #     xlabel_4 = PLOT_LABELS['scatter_xlabel_fmt'].format(label=PLOT_LABELS['metric_initial_log_income'])
                #     ylabel_4 = PLOT_LABELS['scatter_ylabel_fmt'].format(label=PLOT_LABELS['metric_lhs_g'])
                #     colorlabel_4 = PLOT_LABELS['scatter_colorlabel_fmt'].format(label=PLOT_LABELS['metric_pop_g'])
                #     filename_4 = os.path.join(plot_subdir_4, f'{base_level_name}_{agg_level_name}_AvgG_PopPathLHS_vs_InitInc_by_PopG.pdf') # CORRECTED
                #     plot_scatter_colored(df_level, log_avg_inc_initial_col, avg_g_col_pop_path, pop_g_col,
                #                          ' ', xlabel_4, ylabel_4, colorlabel_4, filename_4, special=True,override=True, fit=True,city_lbl=False)
                    
                # # Plot 4: x: initial income, y: Agg_G (gro), color: PopG
                # if (log_avg_inc_initial_col in df_level.columns and
                #     avg_g_col_pop_path in df_level.columns and # Was just changed to avg_g_col in a previous step
                #     pop_g_col in df_level.columns):
                #     plot_subdir_4 = os.path.join(output_plot_dir_name, "Scatter_AvgG_PopPathLHS_vs_InitInc_by_PopG_weighted") # CORRECTED
                #     create_output_directory(plot_subdir_4)
                #     title_4 = PLOT_LABELS['scatter_title_fmt'].format(
                #         y_metric=PLOT_LABELS['metric_lhs_g'], # metric_lhs_g represents AvgG
                #         x_metric=PLOT_LABELS['metric_initial_log_income'], # Initial Income
                #         color_metric=PLOT_LABELS['metric_pop_g'].split('(')[0].strip(),
                #         agg_level_name=agg_level_name, base_level_name=base_level_name)
                #     xlabel_4 = PLOT_LABELS['scatter_xlabel_fmt'].format(label=PLOT_LABELS['metric_initial_log_income'])
                #     ylabel_4 = PLOT_LABELS['scatter_ylabel_fmt'].format(label=PLOT_LABELS['metric_lhs_g'])
                #     colorlabel_4 = PLOT_LABELS['scatter_colorlabel_fmt'].format(label=PLOT_LABELS['metric_pop_g'])
                #     filename_4 = os.path.join(plot_subdir_4, f'{base_level_name}_{agg_level_name}_AvgG_PopPathLHS_vs_InitInc_by_PopG.pdf') # CORRECTED
                #     plot_scatter_colored(df_level, log_avg_inc_initial_col, avg_g_col_pop_path, pop_g_col,
                #                          title_4, xlabel_4, ylabel_4, colorlabel_4, filename_4, special=False,override=False,weighted=True,pop=pop_col)
                
                # # Plot 4: x: initial income, y: Agg_G (gro), color: PopG
                # if (log_avg_inc_initial_col in df_level.columns and
                #     avg_g_col_pop_path in df_level.columns and # Was just changed to avg_g_col in a previous step
                #     pop_g_col in df_level.columns):
                #     plot_subdir_4 = os.path.join(output_plot_dir_name, "Scatter_AggG_IncPathLHS_vs_InitInc_by_PopG") # CORRECTED
                #     create_output_directory(plot_subdir_4)
                #     title_4 = PLOT_LABELS['scatter_title_fmt'].format(
                #         y_metric=PLOT_LABELS['metric_lhs_i'], # metric_lhs_g represents AvgG
                #         x_metric=PLOT_LABELS['metric_initial_log_income'], # Initial Income
                #         color_metric=PLOT_LABELS['metric_pop_g'].split('(')[0].strip(),
                #         agg_level_name=agg_level_name, base_level_name=base_level_name)
                #     xlabel_4 = PLOT_LABELS['scatter_xlabel_fmt'].format(label=PLOT_LABELS['metric_initial_log_income'])
                #     ylabel_4 = PLOT_LABELS['scatter_ylabel_fmt'].format(label=PLOT_LABELS['metric_lhs_i'])
                #     colorlabel_4 = PLOT_LABELS['scatter_colorlabel_fmt'].format(label=PLOT_LABELS['metric_pop_g'])
                #     filename_4 = os.path.join(plot_subdir_4, f'{base_level_name}_{agg_level_name}_AvgG_IncPathLHS_vs_InitInc_by_PopG.pdf') # CORRECTED
                #     plot_scatter_colored(df_level, log_avg_inc_initial_col, agg_g_col, pop_g_col,
                #                          title_4, xlabel_4, ylabel_4, colorlabel_4, filename_4, special=True,override=False)
                
                # # Plot 4: x: initial income, y: Agg_G (gro), color: PopG
                # if (log_avg_inc_initial_col in df_level.columns and
                #     avg_g_col_pop_path in df_level.columns and # Was just changed to avg_g_col in a previous step
                #     pop_g_col in df_level.columns):
                #     plot_subdir_4 = os.path.join(output_plot_dir_name, "Scatter_AggG_IncPathLHS_vs_InitInc_by_PopG_weighted") # CORRECTED
                #     create_output_directory(plot_subdir_4)
                #     title_4 = PLOT_LABELS['scatter_title_fmt'].format(
                #         y_metric=PLOT_LABELS['metric_lhs_i'], # metric_lhs_g represents AvgG
                #         x_metric=PLOT_LABELS['metric_initial_log_income'], # Initial Income
                #         color_metric=PLOT_LABELS['metric_pop_g'].split('(')[0].strip(),
                #         agg_level_name=agg_level_name, base_level_name=base_level_name)
                #     xlabel_4 = PLOT_LABELS['scatter_xlabel_fmt'].format(label=PLOT_LABELS['metric_initial_log_income'])
                #     ylabel_4 = PLOT_LABELS['scatter_ylabel_fmt'].format(label=PLOT_LABELS['metric_lhs_i'])
                #     colorlabel_4 = PLOT_LABELS['scatter_colorlabel_fmt'].format(label=PLOT_LABELS['metric_pop_g'])
                #     filename_4 = os.path.join(plot_subdir_4, f'{base_level_name}_{agg_level_name}_AvgG_IncPathLHS_vs_InitInc_by_PopG.pdf') # CORRECTED
                #     plot_scatter_colored(df_level, log_avg_inc_initial_col, agg_g_col, pop_g_col,
                #                          title_4, xlabel_4, ylabel_4, colorlabel_4, filename_4, special=False,override=False,weighted=True,pop=pop_col)
                


                # # Plot 4: x: empirical growth rate, y: Diff empirical/population-avgd growth rate, color: PopG
                if (avg_g_col_pop_path in df_level.columns and
                    agg_g_col in df_level.columns and # Was just changed to avg_g_col in a previous step
                    pop_g_col in df_level.columns):
                    plot_subdir_4 = os.path.join(output_plot_dir_name, "Scatter_AvgG_vs_AggGby_PopG") # CORRECTED
                    create_output_directory(plot_subdir_4)
                    title_4 = PLOT_LABELS['scatter_title_fmt'].format(
                        y_metric=PLOT_LABELS['metric_lhs_g'], # metric_lhs_g represents AvgG
                        x_metric=PLOT_LABELS['metric_emper_g'], # Empirical Growth Rate
                        color_metric=PLOT_LABELS['metric_pop_g'].split('(')[0].strip(),
                        agg_level_name=agg_level_name, base_level_name=base_level_name)
                    xlabel_4 = PLOT_LABELS['scatter_xlabel_fmt'].format(label=PLOT_LABELS['metric_emper_g'])
                    ylabel_4 = PLOT_LABELS['scatter_ylabel_fmt'].format(label=PLOT_LABELS['metric_lhs_g'])
                    colorlabel_4 = PLOT_LABELS['scatter_colorlabel_fmt'].format(label=PLOT_LABELS['metric_pop_g'])
                    filename_4 = os.path.join(plot_subdir_4, f'{base_level_name}_{agg_level_name}_AvgG_vs_AggG_by_PopG.pdf') # CORRECTED
                    plot_scatter_colored(df_level, agg_g_col, avg_g_col_pop_path, pop_g_col,
                                         title_4, xlabel_4, ylabel_4, colorlabel_4, filename_4, special=False,override=False,weighted=False,fit=False,diff=True,city_lbl=True)



                # Plot 4: x: initial income, y: Agg_G (gro), color: PopG
                # if (avg_g_col_pop_path in df_level.columns and
                #     agg_g_col in df_level.columns and # Was just changed to avg_g_col in a previous step
                #     pop_g_col in df_level.columns):
                #     plot_subdir_4 = os.path.join(output_plot_dir_name, "Scatter_AvgG_vs_PopGby_InitInc") # CORRECTED
                #     create_output_directory(plot_subdir_4)
                #     title_4 = PLOT_LABELS['scatter_title_fmt'].format(
                #         y_metric=PLOT_LABELS['metric_lhs_g'], # metric_lhs_g represents AvgG
                #         x_metric=PLOT_LABELS['metric_pop_g'], # Initial Income
                #         color_metric=PLOT_LABELS['metric_initial_log_income'].split('(')[0].strip(),
                #         agg_level_name=agg_level_name, base_level_name=base_level_name)
                #     xlabel_4 = PLOT_LABELS['scatter_xlabel_fmt'].format(label=PLOT_LABELS['metric_lhs_g'])
                #     ylabel_4 = PLOT_LABELS['scatter_ylabel_fmt'].format(label=PLOT_LABELS['metric_pop_g'])
                #     colorlabel_4 = PLOT_LABELS['scatter_colorlabel_fmt'].format(label=PLOT_LABELS['metric_initial_log_income'])
                #     filename_4 = os.path.join(plot_subdir_4, f'{base_level_name}_{agg_level_name}_AvgG_vs_PopG_by_InitInc.pdf') # CORRECTED
                #     plot_scatter_colored(df_level, avg_g_col_pop_path, pop_g_col, log_avg_inc_initial_col,
                #                          ' ', xlabel_4, ylabel_4, colorlabel_4, filename_4, special=True,override=True)



                # Plot 4: x: initial income, y: Agg_G (gro), color: PopG
                # if (avg_g_col_pop_path in df_level.columns and
                #     agg_g_col in df_level.columns and # Was just changed to avg_g_col in a previous step
                #     pop_g_col in df_level.columns):
                #     plot_subdir_4 = os.path.join(output_plot_dir_name, "Scatter_InitInc_vs_PopGby_AvgG") # CORRECTED
                #     create_output_directory(plot_subdir_4)
                #     title_4 = PLOT_LABELS['scatter_title_fmt'].format(
                #         y_metric=PLOT_LABELS['metric_initial_log_income'], # metric_lhs_g represents AvgG
                #         x_metric=PLOT_LABELS['metric_pop_g'], # Initial Income
                #         color_metric=PLOT_LABELS['metric_lhs_g'].split('(')[0].strip(),
                #         agg_level_name=agg_level_name, base_level_name=base_level_name)
                #     xlabel_4 = PLOT_LABELS['scatter_xlabel_fmt'].format(label=PLOT_LABELS['metric_initial_log_income'])
                #     ylabel_4 = PLOT_LABELS['scatter_ylabel_fmt'].format(label=PLOT_LABELS['metric_pop_g'])
                #     colorlabel_4 = PLOT_LABELS['scatter_colorlabel_fmt'].format(label=PLOT_LABELS['metric_lhs_g'])
                #     filename_4 = os.path.join(plot_subdir_4, f'{base_level_name}_{agg_level_name}_Inc_vs_PopG_by_AvgG.pdf') # CORRECTED
                #     plot_scatter_colored(df_level, log_avg_inc_initial_col, pop_g_col, avg_g_col_pop_path,
                #                          ' ', xlabel_4, ylabel_4, colorlabel_4, filename_4, special=True,override=True,fit=True,city_lbl=False)

                # --- Kept Scatter Plots (Plots 5 & 6) ---
                # Calculate Relative Initial Log Income for coloring new plots
                if log_avg_inc_initial_col in df_level.columns:
                    mean_log_inc = df_level[log_avg_inc_initial_col].mean()
                    df_level[rel_log_avg_inc_initial_col] = df_level[log_avg_inc_initial_col] - mean_log_inc
                else:
                    df_level[rel_log_avg_inc_initial_col] = np.nan # Ensure column exists even if it's all NaN

                # Plot 5: Community Selection (Population Decomposition) vs. Community-Parent Growth Rate Difference (Population Path)
                # X-axis: Sel_cm_from_tr_gro (community selection by population decomposition)
                # Y-axis: AvgG_pop_cm - AvgG_pop_ct (community minus parent population-averaged growth rate)
                sel_cm_pop_col = f'Sel_cm_from_tr_gro'
                avg_g_pop_cm_col = f'AvgG_pop_cm'
                
                if (sel_cm_pop_col in df_level.columns and 
                    avg_g_pop_cm_col in df_level.columns and
                    'ParentState' in df_level.columns and
                    'ParentCounty' in df_level.columns and
                    'ct' in levels_data and
                    rel_log_avg_inc_initial_col in df_level.columns and 
                    not df_level[rel_log_avg_inc_initial_col].isnull().all()):
                    
                    # Get county-level DataFrame and create lookup for parent county growth rates
                    df_county = levels_data['ct']
                    if not df_county.empty and 'AvgG_pop_ct' in df_county.columns and 'UnitName' in df_county.columns:
                        
                        # Create parent county identifier for lookup
                        df_level['parent_county_id'] = df_level['ParentState'].astype(str) + df_level['ParentCounty'].astype(str)
                        
                        # Create lookup dictionary for county growth rates
                        county_growth_lookup = dict(zip(df_county['UnitName'].astype(str), df_county['AvgG_pop_ct']))
                        
                        # Map parent county growth rates to communities
                        df_level['parent_growth_pop'] = df_level['parent_county_id'].map(county_growth_lookup)
                        
                        # Calculate the growth rate difference: community minus parent
                        df_level['growth_diff_pop'] = df_level[avg_g_pop_cm_col] #- df_level['parent_growth_pop']
                        
                        # Only proceed if we have valid growth differences
                        if not df_level['growth_diff_pop'].isnull().all():
                            scatter_subdir_5 = os.path.join(output_plot_dir_name, "Scatter_CommunitySelectionPop_vs_GrowthDiffPop_by_RelInitInc")
                            create_output_directory(scatter_subdir_5)
                            
                            title_5 = PLOT_LABELS['scatter_title_fmt'].format(
                                y_metric="Community-Parent Growth Rate Difference (Population)",
                                x_metric="Community Selection (Population Decomposition)",
                                color_metric=PLOT_LABELS['metric_rel_initial_log_income'].split('(')[0].strip(),
                                agg_level_name=agg_level_name,
                                base_level_name=base_level_name
                            )
                            xlabel_5 = "Community Selection Term (Population Decomposition)"
                            ylabel_5 = "Community - Parent Growth Rate (Population Path)" 
                            colorlabel_5 = PLOT_LABELS['scatter_colorlabel_fmt'].format(label=PLOT_LABELS['metric_rel_initial_log_income'])
                            
                            filename_5 = os.path.join(scatter_subdir_5, f'{base_level_name}_{agg_level_name}_CommunitySelectionPop_vs_GrowthDiffPop_by_RelInitInc.pdf')
                            
                            plot_scatter_colored(df_level,
                                                 x_col=sel_cm_pop_col,
                                                 y_col='growth_diff_pop',
                                                 color_col=rel_log_avg_inc_initial_col,
                                                 title=title_5,
                                                 xlabel=xlabel_5,
                                                 ylabel=ylabel_5,
                                                 color_label=colorlabel_5,
                                                 output_filename=filename_5,
                                                 special=True,
                                                 city_lbl=True)

                # Plot 6: Community Selection (Income Decomposition) vs. Community-Parent Growth Rate Difference (Income Path)
                # X-axis: Sel_cm_from_tr_inc (community selection by income decomposition)
                # Y-axis: AvgG_inc_cm - AvgG_inc_ct (community minus parent income-averaged growth rate)
                sel_cm_inc_col = f'Sel_cm_from_tr_inc'
                avg_g_inc_cm_col = f'AvgG_inc_cm'
                
                if (sel_cm_inc_col in df_level.columns and 
                    avg_g_inc_cm_col in df_level.columns and
                    'ParentState' in df_level.columns and
                    'ParentCounty' in df_level.columns and
                    'ct' in levels_data and
                    rel_log_avg_inc_initial_col in df_level.columns and
                    not df_level[rel_log_avg_inc_initial_col].isnull().all()):
                    
                    # Get county-level DataFrame and create lookup for parent county growth rates
                    df_county = levels_data['ct']
                    if not df_county.empty and 'AvgG_inc_ct' in df_county.columns and 'UnitName' in df_county.columns:
                        
                        # Create parent county identifier for lookup (reuse if already created)
                        if 'parent_county_id' not in df_level.columns:
                            df_level['parent_county_id'] = df_level['ParentState'].astype(str) + df_level['ParentCounty'].astype(str)
                        
                        # Create lookup dictionary for county income growth rates
                        county_income_growth_lookup = dict(zip(df_county['UnitName'].astype(str), df_county['AvgG_inc_ct']))
                        
                        # Map parent county income growth rates to communities
                        df_level['parent_growth_inc'] = df_level['parent_county_id'].map(county_income_growth_lookup)
                        
                        # Calculate the growth rate difference: community minus parent
                        df_level['growth_diff_inc'] = df_level[avg_g_inc_cm_col]# - df_level['parent_growth_inc']
                        
                        # Only proceed if we have valid growth differences
                        if not df_level['growth_diff_inc'].isnull().all():
                            scatter_subdir_6 = os.path.join(output_plot_dir_name, "Scatter_CommunitySelectionInc_vs_GrowthDiffInc_by_RelInitInc")
                            create_output_directory(scatter_subdir_6)
                            
                            title_6 = PLOT_LABELS['scatter_title_fmt'].format(
                                y_metric="Community-Parent Growth Rate Difference (Income)",
                                x_metric="Community Selection (Income Decomposition)",
                                color_metric=PLOT_LABELS['metric_rel_initial_log_income'].split('(')[0].strip(),
                                agg_level_name=agg_level_name,
                                base_level_name=base_level_name
                            )
                            xlabel_6 = "Community Selection Term (Income Decomposition)"
                            ylabel_6 = "Community - Parent Growth Rate (Income Path)" 
                            colorlabel_6 = PLOT_LABELS['scatter_colorlabel_fmt'].format(label=PLOT_LABELS['metric_rel_initial_log_income'])
                            
                            filename_6 = os.path.join(scatter_subdir_6, f'{base_level_name}_{agg_level_name}_CommunitySelectionInc_vs_GrowthDiffInc_by_RelInitInc.pdf')

                            plot_scatter_colored(df_level,
                                                 x_col=sel_cm_inc_col,
                                                 y_col='growth_diff_inc',
                                                 color_col=rel_log_avg_inc_initial_col,
                                                 title=title_6,
                                                 xlabel=xlabel_6,
                                                 ylabel=ylabel_6,
                                                 color_label=colorlabel_6,
                                                 output_filename=filename_6,
                                                 special=True,
                                                 city_lbl=True)
                # --- End of Kept Scatter Plots ---

    print(f"\n--- Finished Generating Plots for Dataset in: {output_plot_dir_name} ---")

def _filter_data_for_cook_county(full_data_dict, target_county_fips_3digit='031'):
    print(f"\n--- Filtering Data for Cook County (ID: {target_county_fips_3digit}) ---")
    filtered_major_dict = {}
    if not isinstance(full_data_dict, dict):
        print("  Error: Input data for filtering is not a dictionary. Cannot filter.")
        return filtered_major_dict

    for base_name, levels_data_dict in full_data_dict.items():
        print(f"  Processing base level for filtering: {base_name}")
        filtered_major_dict[base_name] = {}
        current_cook_state_fips = None

        if 'ct' in levels_data_dict and isinstance(levels_data_dict['ct'], pd.DataFrame) and not levels_data_dict['ct'].empty:
            df_ct = levels_data_dict['ct']
            if 'UnitName' in df_ct.columns:
                cook_ct_rows = df_ct[df_ct['UnitName'].astype(str).str.endswith(target_county_fips_3digit)] # More robust check
                if not cook_ct_rows.empty:
                    if 'ParentState' in cook_ct_rows.columns:
                        current_cook_state_fips = cook_ct_rows['ParentState'].astype(str).iloc[0]
                        print(f"    Identified Cook County's state as: {current_cook_state_fips} for base {base_name}")
                    else:
                        print(f"    Warning: Cook County found at 'ct' level for base {base_name}, but 'ParentState' column missing.")
                # else: print(f"    Warning: Cook County ('{target_county_fips_3digit}') not found at 'ct' level for base {base_name}.")
            # else: print(f"    Warning: 'UnitName' column missing in 'ct' level for base {base_name}.")
        # else: print(f"    Warning: 'ct' level data not available or empty for base {base_name}.")

        for agg_level, df_orig in levels_data_dict.items():
            if not isinstance(df_orig, pd.DataFrame) or df_orig.empty:
                filtered_major_dict[base_name][agg_level] = pd.DataFrame() 
                # print(f"    Skipping filter for '{agg_level}' in base '{base_name}': Original data empty/not DataFrame.")
                continue

            df_filtered_for_level = pd.DataFrame() 
            if 'UnitName' not in df_orig.columns:
                # print(f"    Warning: 'UnitName' missing in '{agg_level}' for base '{base_name}'. Result empty.")
                filtered_major_dict[base_name][agg_level] = df_filtered_for_level
                continue
            
            df_orig_unitname_str = df_orig['UnitName'].astype(str)

            if agg_level == 'bg': # SSCCCTTTTTTBB
                df_filtered_for_level = df_orig[df_orig_unitname_str.str[2:5] == target_county_fips_3digit]
            elif agg_level == 'tr': # SSCCCTTTTTT
                df_filtered_for_level = df_orig[df_orig_unitname_str.str[2:5] == target_county_fips_3digit]
            elif agg_level == 'cm':
                if 'ParentCounty' in df_orig.columns: # ParentCounty is expected to be 3-digit FIPS
                    df_filtered_for_level = df_orig[df_orig['ParentCounty'].astype(str) == target_county_fips_3digit]
                # else: print(f"    Warning: 'ParentCounty' not in '{agg_level}'. Cook filter empty.")
            elif agg_level == 'ct': # UnitName is SSCCC or CCC
                 df_filtered_for_level = df_orig[df_orig_unitname_str.str.endswith(target_county_fips_3digit)]
            elif agg_level == 'st': # UnitName is SS
                if current_cook_state_fips:
                    df_filtered_for_level = df_orig[df_orig_unitname_str == current_cook_state_fips]
                # else: print(f"    Warning: Cook County's state FIPS unknown. Filter for 'st' empty.")
            # else: print(f"    Notice: No Cook County filter for '{agg_level}'. This level empty in Cook run.")
            
            filtered_major_dict[base_name][agg_level] = df_filtered_for_level.copy() 
            # print(f"    Filtered '{agg_level}' for Cook County (base: {base_name}): {len(df_orig)} -> {len(df_filtered_for_level)} rows")
    
    print(f"--- Finished Filtering Data for Cook County ---")
    return filtered_major_dict

# --- NEW FUNCTION: plot_selection_term_scatters ---
def plot_selection_term_scatters(levels_data, base_level_name, output_plot_dir_name_prefix, hierarchy_levels_list):
    print(f"\n--- Generating Selection Term Scatter Plots for Base '{base_level_name}' ---")
    
    output_subdir_base = os.path.join(output_plot_dir_name_prefix, "Selection_Term_Scatters", f"Base_{base_level_name}")

    for p_target_level, df_p_target in levels_data.items():
        if not isinstance(df_p_target, pd.DataFrame) or df_p_target.empty:
            continue
        
        # print(f"  Processing P_target level for scatters: {p_target_level}")
        output_subdir_ptarget = os.path.join(output_subdir_base, f"Ptarget_{p_target_level}")
        # create_output_directory is called by plot_scatter_colored if needed for its specific subdir,
        # but let's create the Ptarget level directory here.
        if not create_output_directory(output_subdir_ptarget): # Ensure this utility is available or defined
            print(f"    Could not create/access directory: {output_subdir_ptarget}. Skipping scatters for this P_target.")
            continue


        try:
            p_target_idx = hierarchy_levels_list.index(p_target_level)
        except ValueError:
            # print(f"    Skipping P_target {p_target_level}, not in hierarchy.")
            continue

        for s_origin_idx, s_origin_level in enumerate(hierarchy_levels_list):
            if s_origin_idx >= p_target_idx: # S_origin must be strictly below P_target
                continue

            col_stem = ""
            desc_for_title = ""
            
            child_of_p_target_idx = p_target_idx - 1
            # This check ensures P_target is not the lowest level (e.g. 'bg')
            # because it cannot receive selection from a level below it if it's already the lowest.
            if child_of_p_target_idx < 0 : 
                continue 

            if s_origin_idx == child_of_p_target_idx: # Direct selection
                col_stem = f"Sel_{p_target_level}_from_{s_origin_level}"
                desc_for_title = f"Direct Selection on {p_target_level} from {s_origin_level}"
            else: # Transmitted selection (s_origin_idx < child_of_p_target_idx)
                col_stem = f"Transmitted_Sel_{s_origin_level}_to_{p_target_level}"
                desc_for_title = f"Selection on {p_target_level} (originated from {s_origin_level})"

            if not col_stem:
                continue

            x_col_name = f"{col_stem}_inc"
            y_col_name = f"{col_stem}_gro"

            if x_col_name not in df_p_target.columns or y_col_name not in df_p_target.columns:
                # print(f"    Skipping scatter for P_target={p_target_level}, S_origin={s_origin_level}: Columns {x_col_name} or {y_col_name} not found.")
                continue

            # Define color-by variables
            # The value in the map is used for the colorbar label suffix.
            color_by_vars_map = {
                f"PopG_{p_target_level}": f"Pop. Growth ({p_target_level})",
                f"LogAvgIncInitial_{p_target_level}": f"Initial Log Avg Inc ({p_target_level})",
                # Example to add another:
                # f"TotalEmpiricalGrowth_{p_target_level}_gro": f"Total LHS Growth ({p_target_level}, gro path)"
            }
            
            for color_var_col, color_var_label_text in color_by_vars_map.items():
                if color_var_col not in df_p_target.columns:
                    # print(f"    Skipping color-by {color_var_col} for {desc_for_title}: Column not found in {p_target_level} data.")
                    continue

                scatter_title = f"{desc_for_title}\n(Base: {base_level_name}, P_target: {p_target_level})"
                scatter_xlabel = f"Selection Term ({x_col_name}, Income Path)"
                scatter_ylabel = f"Selection Term ({y_col_name}, Growth Path)"
                scatter_colorlabel = f"{color_var_label_text}" # Use the mapped text
                
                # Sanitize parts of the filename
                clean_s_origin = s_origin_level.replace(" ", "_")
                clean_color_var = color_var_col.replace(" ", "_").replace(f"_{p_target_level}","") # Make it shorter
                
                scatter_filename_leaf = f"Scatter_IncVsGro_{p_target_level}_from_{clean_s_origin}_by_{clean_color_var}.pdf"
                # The plot_scatter_colored function will create its own subdirectories based on its logic,
                # so we pass the Ptarget level directory as its base.
                # However, plot_scatter_colored does NOT create arbitrary subdirs, it creates specific ones.
                # So, we need to construct the full path here.

                # Create specific subdirectory for these types of plots if plot_scatter_colored doesn't handle it
                # For now, let's assume output_subdir_ptarget is sufficient and filename is descriptive.
                scatter_filename_full = os.path.join(output_subdir_ptarget, scatter_filename_leaf)

                # print(f"    Generating scatter: {scatter_filename_leaf} into {output_subdir_ptarget}")
                plot_scatter_colored(
                    df_p_target, 
                    x_col=x_col_name, 
                    y_col=y_col_name, 
                    color_col=color_var_col,
                    title=scatter_title, 
                    xlabel=scatter_xlabel, 
                    ylabel=scatter_ylabel, 
                    color_label=scatter_colorlabel,
                    output_filename=scatter_filename_full # Pass the full path directly
                )
# --- END NEW FUNCTION ---

def main():
    print("--- Starting plotPrice.py ---")

    full_decomposition_data = load_decomposition_results(RESULTS_PICKLE_FILEPATH)
    if full_decomposition_data is None:
        print("Halting script as main decomposition data could not be loaded.")
        return

    # Run 1: Full dataset
    print("\n>>> Processing: Full Dataset")
    _generate_plots_for_dataset(full_decomposition_data, "output_plots")

    # Run 2: Cook County only
    print("\n>>> Processing: Cook County (031) Only")
    cook_county_data = _filter_data_for_cook_county(full_decomposition_data, target_county_fips_3digit='031')
    
    is_cook_data_substantially_empty = True
    if isinstance(cook_county_data, dict):
        for base_key in cook_county_data:
            if isinstance(cook_county_data[base_key], dict):
                for agg_key in cook_county_data[base_key]:
                    if isinstance(cook_county_data[base_key][agg_key], pd.DataFrame) and not cook_county_data[base_key][agg_key].empty:
                        is_cook_data_substantially_empty = False
                        break
            if not is_cook_data_substantially_empty: break
            
    if is_cook_data_substantially_empty:
        print("  Cook County data appears to be empty or invalid after filtering. Skipping Cook County plot generation.")
    else:
        _generate_plots_for_dataset(cook_county_data, "output_plots_cook")

    print("\n--- plotPrice.py finished all processing ---")

if __name__ == "__main__":
    main()