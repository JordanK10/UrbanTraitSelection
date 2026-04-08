#!/usr/bin/env python3
"""
specialty_histogram_with_loglog.py

Plots two histograms (income PNC_st, population PNC_st) with skew-t fits,
plus log-log plots of the fitted t-distribution curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import sys
from scipy.stats import t
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Custom colors
custom_purple = '#633673'
custom_orange = '#E77429'

# Check for 'null' argument to switch directories
if 'null' in sys.argv:
    INPUT_DIR = 'output_terms_null'
    BASE_OUTPUT_DIR = 'plots_null'
else:
    INPUT_DIR = 'output_terms'
    BASE_OUTPUT_DIR = 'plots'

# Output directory
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'specialty_histograms_with_loglog')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Data ---
def load_data(path):
    try:
        df = pd.read_csv(path)
        print(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

cm_data = load_data(os.path.join(INPUT_DIR, 'bg_cm_exported_terms.csv'))
tr_data = load_data(os.path.join(INPUT_DIR, 'bg_tr_exported_terms.csv'))

# --- Extraction helpers ---
def extract_tr(df, sel_col, scale=None):
    if sel_col in df.columns:
        vals = pd.to_numeric(df[sel_col], errors='coerce')
        if scale:
            vals = vals / scale
        return vals
    else:
        return pd.Series([np.nan]*len(df))

# --- Fit helpers ---

# Custom skew-t implementation
def skewt_pdf(x, df, loc, scale, alpha):
    """
    Azzalini's skew-t PDF implementation
    """
    # Handle edge cases
    if scale <= 0 or df <= 0:
        return np.full_like(x, np.nan)
    
    z = (x - loc) / scale
    t_pdf = t.pdf(z, df)
    t_cdf = t.cdf(alpha * z * np.sqrt((df + 1) / (df + z**2)), df + 1)
    
    return 2 * t_pdf * t_cdf / scale

def fit_skewt(data):
    """
    Fit skew-t distribution to data using curve_fit
    Returns: (df, loc, scale, alpha) or None if fit fails
    """
    clean_data = data[np.isfinite(data) & ~np.isnan(data) & (data != 0)]
    
    if len(clean_data) < 10:
        return None
    
    # Create histogram for fitting
    hist, bin_edges = np.histogram(clean_data, bins=min(30, len(clean_data)//5), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initial parameter guesses
    p0 = [5, np.mean(clean_data), np.std(clean_data), 0]
    
    # Parameter bounds: df > 1, scale > 0, alpha can be negative or positive
    bounds = ([1, np.min(clean_data), 1e-6, -10], 
              [50, np.max(clean_data), np.std(clean_data)*5, 10])
    
    try:
        popt, _ = curve_fit(skewt_pdf, bin_centers, hist, p0=p0, bounds=bounds, maxfev=5000)
        return tuple(popt)  # Convert numpy array to tuple: (df, loc, scale, alpha)
    except Exception as e:
        print(f"Skew-t fit failed: {e}")
        return None

# --- Plotting ---
def plot_hist(cm, tr, name):
    # Remove NaN, inf, and exactly zero values (0 means calculation failed)
    cm_clean = cm[np.isfinite(cm) & ~np.isnan(cm) & (cm != 0)]
    tr_clean = tr[np.isfinite(tr) & ~np.isnan(tr) & (tr != 0)]
    all_data = np.concatenate([cm_clean, tr_clean])
    
    print(f"\n{'='*60}")
    print(f"PLOT: {name} (SKEWT distribution)")
    print(f"{'='*60}")
    
    # Print comprehensive data statistics
    print(f"\nDATA STATISTICS:")
    print(f"Community data:")
    print(f"  - Count: {len(cm_clean)}")
    if len(cm_clean) > 0:
        cm_mean = cm_clean.mean()
        cm_std = cm_clean.std()
        cm_skew = ((cm_clean - cm_mean) / cm_std)**3
        cm_kurt = ((cm_clean - cm_mean) / cm_std)**4
        print(f"  - Mean: {cm_mean:.6f}")
        print(f"  - Std: {cm_std:.6f}")
        print(f"  - Min: {cm_clean.min():.6f}")
        print(f"  - Max: {cm_clean.max():.6f}")
        print(f"  - Median: {np.median(cm_clean):.6f}")
        print(f"  - Skewness: {cm_skew.mean():.6f}")
        print(f"  - Kurtosis: {cm_kurt.mean():.6f}")
    
    print(f"\nTract data:")
    print(f"  - Count: {len(tr_clean)}")
    if len(tr_clean) > 0:
        tr_mean = tr_clean.mean()
        tr_std = tr_clean.std()
        tr_skew = ((tr_clean - tr_mean) / tr_std)**3
        tr_kurt = ((tr_clean - tr_mean) / tr_std)**4
        print(f"  - Mean: {tr_mean:.6f}")
        print(f"  - Std: {tr_std:.6f}")
        print(f"  - Min: {tr_clean.min():.6f}")
        print(f"  - Max: {tr_clean.max():.6f}")
        print(f"  - Median: {np.median(tr_clean):.6f}")
        print(f"  - Skewness: {tr_skew.mean():.6f}")
        print(f"  - Kurtosis: {tr_kurt.mean():.6f}")
    
    if len(all_data) == 0:
        print("No valid data to plot!")
        return None, None
    
    # --- Create figure and axis ---
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Define colors
    dark_purple = custom_purple
    dark_orange = custom_orange
    
    # --- Plot histograms ---
    y_max = 0
    
    # Community histogram
    if len(cm_clean) > 0:
        n_bins_cm = 2 * (min(30, max(10, int(np.sqrt(len(cm_clean))))) if len(cm_clean) > 0 else 10)
        n_cm, bins_cm, patches_cm = ax.hist(cm_clean, bins=n_bins_cm, alpha=0.7, 
                                           color=dark_purple, density=True, 
                                           edgecolor='white', linewidth=0.8)
        y_max = max(y_max, n_cm.max())
        print(f"\nCommunity histogram: {n_bins_cm} bins, max density = {n_cm.max():.6f}")
    
    # Tract histogram
    if len(tr_clean) > 0:
        n_bins_tr = 2 * (min(75, max(20, int(np.sqrt(len(tr_clean))))) if len(tr_clean) > 0 else 20)
        n_tr, bins_tr, patches_tr = ax.hist(tr_clean, bins=n_bins_tr, alpha=0.7, 
                                           color=dark_orange, density=True, 
                                           edgecolor='white', linewidth=0.8)
        y_max = max(y_max, n_tr.max())
        print(f"Tract histogram: {n_bins_tr} bins, max density = {n_tr.max():.6f}")
    
    # --- Fit and plot distributions ---
    print(f"\nFITTING RESULTS:")
    
    # Community fit
    cm_stats_text = None
    cm_fit = None
    if len(cm_clean) >= 2:
        print(f"\nCommunity SKEWT fit:")
        cm_fit = fit_skewt(cm_clean)
        if cm_fit:
            df_param, loc, scale, a = cm_fit
            print(f"  - SUCCESS: df={df_param:.6f}, loc={loc:.6f}, scale={scale:.6f}, skew={a:.6f}")
            x = np.linspace(cm_clean.min(), cm_clean.max(), 200)
            y_fit = skewt_pdf(x, df_param, loc, scale, a)
            y_fit = np.clip(y_fit, 0, y_max)
            ax.plot(x, y_fit, color=dark_purple, linewidth=2.5)
            cm_stats_text = f'df = {df_param:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\nskew = {a:.3f}\nmean = {cm_clean.mean():.3f}'
            print(f"  - Fit curve: min={y_fit.min():.6f}, max={y_fit.max():.6f}")
        else:
            print(f"  - FAILED: Could not fit skew-t distribution")
    
    # Tract fit
    tr_stats_text = None
    tr_fit = None
    if len(tr_clean) >= 2:
        print(f"\nTract SKEWT fit:")
        tr_fit = fit_skewt(tr_clean)
        if tr_fit:
            df_param, loc, scale, a = tr_fit
            print(f"  - SUCCESS: df={df_param:.6f}, loc={loc:.6f}, scale={scale:.6f}, skew={a:.6f}")
            x = np.linspace(tr_clean.min(), tr_clean.max(), 200)
            y_fit = skewt_pdf(x, df_param, loc, scale, a)
            y_fit = np.clip(y_fit, 0, y_max)
            ax.plot(x, y_fit, color=dark_orange, linewidth=2.5)
            tr_stats_text = f'df = {df_param:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\nskew = {a:.3f}\nmean = {tr_clean.mean():.3f}'
            print(f"  - Fit curve: min={y_fit.min():.6f}, max={y_fit.max():.6f}")
        else:
            print(f"  - FAILED: Could not fit skew-t distribution")
    
    # --- Mean and reference lines ---
    if len(cm_clean) > 0:
        ax.axvline(cm_clean.mean(), color=dark_purple, linestyle='-', linewidth=1.5, alpha=0.75)
    if len(tr_clean) > 0:
        ax.axvline(tr_clean.mean(), color=dark_orange, linestyle='-', linewidth=1.5, alpha=0.75)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.75)
    
    # --- Statistics boxes ---
    if cm_stats_text:
        ax.text(0.7, 0.95, cm_stats_text, transform=ax.transAxes, fontsize=5,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=dark_purple))
    if tr_stats_text:
        ax.text(0.70, 0.80, tr_stats_text, transform=ax.transAxes, fontsize=5,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=dark_orange))
    
    # --- Axis, ticks, and title formatting ---
    mean_value = all_data.mean()
    std_value = all_data.std()
    x_min_limit = mean_value - 2 * std_value
    x_max_limit = mean_value + 2 * std_value
    ax.set_xlim(x_min_limit, x_max_limit)
    ax.set_ylim(0, y_max * 1.05)
    
    # Spines
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CCCCCC')
    # Ticks
    ax.tick_params(axis='x', which='major', labelsize=18, colors='#333333', bottom=True, top=False)
    ax.tick_params(axis='y', which='major', labelsize=18, colors='#333333', left=True, right=False, labelleft=True)
    # Y-axis scientific notation
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # No legend
    ax.grid(False)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"specialty_histogram_{name.lower().replace(' ', '_').replace('/', '_')}.pdf")
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fname}")
    
    return cm_fit, tr_fit


def plot_loglog_distribution(cm_fit, tr_fit, cm_clean, tr_clean, name):
    """
    Plot the fitted t-distributions on a log-log scale
    """
    print(f"\n{'='*60}")
    print(f"LOG-LOG PLOT: {name}")
    print(f"{'='*60}")
    
    if cm_fit is None and tr_fit is None:
        print("No fitted distributions to plot!")
        return
    
    # --- Create figure and axis ---
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    dark_purple = custom_purple
    dark_orange = custom_orange
    
    # --- Plot community fit on log-log scale ---
    if cm_fit is not None and len(cm_clean) > 0:
        df_param, loc, scale, alpha = cm_fit
        
        # Generate x values (only positive values for log-log plot)
        x_min = max(1e-6, cm_clean[cm_clean > 0].min() if np.any(cm_clean > 0) else 1e-6)
        x_max = cm_clean.max()
        
        # Create x values on a log scale
        x = np.logspace(np.log10(x_min), np.log10(x_max), 500)
        
        # Calculate PDF values
        y_fit = skewt_pdf(x, df_param, loc, scale, alpha)
        
        # Filter out invalid values (zero or negative) for log scale
        valid_mask = (y_fit > 0) & np.isfinite(y_fit)
        x_valid = x[valid_mask]
        y_valid = y_fit[valid_mask]
        
        if len(x_valid) > 0:
            ax.loglog(x_valid, y_valid, color=dark_purple, linewidth=2.5, label='Community')
            print(f"Community fit plotted: {len(x_valid)} points")
    
    # --- Plot tract fit on log-log scale ---
    if tr_fit is not None and len(tr_clean) > 0:
        df_param, loc, scale, alpha = tr_fit
        
        # Generate x values (only positive values for log-log plot)
        x_min = max(1e-6, tr_clean[tr_clean > 0].min() if np.any(tr_clean > 0) else 1e-6)
        x_max = tr_clean.max()
        
        # Create x values on a log scale
        x = np.logspace(np.log10(x_min), np.log10(x_max), 500)
        
        # Calculate PDF values
        y_fit = skewt_pdf(x, df_param, loc, scale, alpha)
        
        # Filter out invalid values (zero or negative) for log scale
        valid_mask = (y_fit > 0) & np.isfinite(y_fit)
        x_valid = x[valid_mask]
        y_valid = y_fit[valid_mask]
        
        if len(x_valid) > 0:
            ax.loglog(x_valid, y_valid, color=dark_orange, linewidth=2.5, label='Tract')
            print(f"Tract fit plotted: {len(x_valid)} points")
    
    # --- Formatting ---
    ax.set_xlabel('Selection Value', fontsize=14, color='#333333')
    ax.set_ylabel('Probability Density', fontsize=14, color='#333333')
    
    # Spines
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CCCCCC')
    
    # Ticks
    ax.tick_params(axis='both', which='major', labelsize=12, colors='#333333')
    ax.tick_params(axis='both', which='minor', labelsize=10, colors='#999999')
    
    # Legend
    ax.legend(loc='best', fontsize=10)
    
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    
    fname = os.path.join(OUTPUT_DIR, f"loglog_{name.lower().replace(' ', '_').replace('/', '_')}.pdf")
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved log-log plot: {fname}")


# --- Main logic ---
if cm_data is not None and tr_data is not None:
    # Income PNC_st
    print("\n" + "="*60)
    print("PROCESSING: Income PNC_st")
    print("="*60)
    cm_inc_pnc = extract_tr(cm_data, 'Sel_cm_from_tr_inc_PNC_st', scale=1)
    tr_inc_pnc = extract_tr(tr_data, 'Sel_tr_from_bg_inc_PNC_st', scale=1)
    
    # Clean data for log-log plot
    cm_inc_pnc_clean = cm_inc_pnc[np.isfinite(cm_inc_pnc) & ~np.isnan(cm_inc_pnc) & (cm_inc_pnc != 0)]
    tr_inc_pnc_clean = tr_inc_pnc[np.isfinite(tr_inc_pnc) & ~np.isnan(tr_inc_pnc) & (tr_inc_pnc != 0)]
    
    cm_fit_inc, tr_fit_inc = plot_hist(cm_inc_pnc, tr_inc_pnc, 'Income PNC_st')
    if cm_fit_inc or tr_fit_inc:
        plot_loglog_distribution(cm_fit_inc, tr_fit_inc, cm_inc_pnc_clean, tr_inc_pnc_clean, 'Income PNC_st')
    
    # Population PNC_st
    print("\n" + "="*60)
    print("PROCESSING: Population PNC_st")
    print("="*60)
    cm_pop_pnc = extract_tr(cm_data, 'Sel_cm_from_tr_pop_PNC_st', scale=1)
    tr_pop_pnc = extract_tr(tr_data, 'Sel_tr_from_bg_pop_PNC_st', scale=1)
    
    # Clean data for log-log plot
    cm_pop_pnc_clean = cm_pop_pnc[np.isfinite(cm_pop_pnc) & ~np.isnan(cm_pop_pnc) & (cm_pop_pnc != 0)]
    tr_pop_pnc_clean = tr_pop_pnc[np.isfinite(tr_pop_pnc) & ~np.isnan(tr_pop_pnc) & (tr_pop_pnc != 0)]
    
    cm_fit_pop, tr_fit_pop = plot_hist(cm_pop_pnc, tr_pop_pnc, 'Population PNC_st')
    if cm_fit_pop or tr_fit_pop:
        plot_loglog_distribution(cm_fit_pop, tr_fit_pop, cm_pop_pnc_clean, tr_pop_pnc_clean, 'Population PNC_st')
    
    print("\n" + "="*60)
    print("ALL PLOTS COMPLETED")
    print("="*60)
else:
    print("Error: Could not load required data files.")
