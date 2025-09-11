#!/usr/bin/env python3
"""
specialty_histogram2.py

Plots four histograms (inc PNC, inc LDR, pop PNC, pop LDR):
- For community: sum transmitted and selection terms for each variable
- For tract: use the selection term for each variable
- Overlay both on the same axes, with the correct fit (Student t for PNC_st, Beta for LDR)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import t, beta, gaussian_kde, chi2, gamma
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Custom colors
custom_purple = '#633673'
custom_orange = '#E77429'

# Output directory
os.makedirs('specialty_histograms2', exist_ok=True)

# --- Load Data ---
def load_data(path):
    try:
        df = pd.read_csv(path)
        print(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

cm_data = load_data('output_terms/bg_cm_exported_terms.csv')
tr_data = load_data('output_terms/bg_tr_exported_terms.csv')

# --- Extraction helpers ---
def extract_sum_cm(df, trans_col, sel_col, scale=None):
    vals = pd.Series([np.nan]*len(df))
    if trans_col in df.columns and sel_col in df.columns:
        trans = pd.to_numeric(df[trans_col], errors='coerce')
        sel = pd.to_numeric(df[sel_col], errors='coerce')
        vals = trans + sel
        if scale:
            vals = vals / scale
    return vals

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
    from scipy.stats import t
    
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

# Custom gamma implementation from scratch
def gamma_pdf(x, shape, loc, scale):
    """
    Gamma PDF implementation from scratch
    """
    # Ensure parameters are valid
    if shape <= 0 or scale <= 0:
        return np.full_like(x, np.nan)
    
    # Shift by location parameter
    x_shifted = x - loc
    
    # Only compute for positive shifted values
    result = np.zeros_like(x)
    valid_mask = x_shifted > 0
    
    if np.any(valid_mask):
        x_valid = x_shifted[valid_mask]
        
        # Gamma PDF formula: (x^(a-1) * exp(-x/b)) / (b^a * Gamma(a))
        # where a = shape, b = scale
        log_pdf = (shape - 1) * np.log(x_valid) - x_valid / scale - shape * np.log(scale) - np.log(math.gamma(shape))
        result[valid_mask] = np.exp(log_pdf)
    
    return result

def fit_gamma(data):
    """
    Fit gamma distribution to data from scratch
    Returns: (shape, loc, scale) or None if fit fails
    """
    clean_data = data[np.isfinite(data) & ~np.isnan(data) & (data > 0)]
    
    if len(clean_data) < 10:
        return None
    
    # Create histogram for fitting
    hist, bin_edges = np.histogram(clean_data, bins=min(30, len(clean_data)//5), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initial parameter guesses
    data_mean = np.mean(clean_data)
    data_var = np.var(clean_data)
    data_min = np.min(clean_data)
    
    # Method of moments initial guesses
    initial_scale = data_var / data_mean if data_mean > 0 else 1.0
    initial_shape = data_mean / initial_scale if initial_scale > 0 else 1.0
    initial_loc = max(0, data_min - 0.1)
    
    p0 = [initial_shape, initial_loc, initial_scale]
    
    # Parameter bounds: shape > 0, loc >= 0, scale > 0
    bounds = ([0.1, 0, 0.01], 
              [50, data_min, data_var * 10])
    
    try:
        popt, _ = curve_fit(gamma_pdf, bin_centers, hist, p0=p0, bounds=bounds, maxfev=5000)
        return tuple(popt)  # (shape, loc, scale)
    except Exception as e:
        print(f"Gamma fit failed: {e}")
        return None

# --- Plotting ---
def plot_hist(cm, tr, name, dist_type):
    # Remove NaN, inf, and exactly zero values (0 means calculation failed)
    cm_clean = cm[np.isfinite(cm) & ~np.isnan(cm) & (cm != 0)]
    tr_clean = tr[np.isfinite(tr) & ~np.isnan(tr) & (tr != 0)]
    all_data = np.concatenate([cm_clean, tr_clean])
    
    print(f"\n{'='*60}")
    print(f"PLOT: {name} ({dist_type.upper()} distribution)")
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
    
    print(f"\nCombined data:")
    print(f"  - Count: {len(all_data)}")
    if len(all_data) > 0:
        print(f"  - Mean: {all_data.mean():.6f}")
        print(f"  - Std: {all_data.std():.6f}")
        print(f"  - Min: {all_data.min():.6f}")
        print(f"  - Max: {all_data.max():.6f}")
        print(f"  - Range: {all_data.max() - all_data.min():.6f}")
    
    if len(all_data) == 0:
        print("No valid data to plot!")
        return
    
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
    if len(cm_clean) >= 2:
        print(f"\nCommunity {dist_type.upper()} fit:")
        if dist_type == 'skewt':
            cm_fit = fit_skewt(cm_clean)
            if cm_fit:
                df, loc, scale, a = cm_fit
                print(f"  - SUCCESS: df={df:.6f}, loc={loc:.6f}, scale={scale:.6f}, skew={a:.6f}")
                x = np.linspace(cm_clean.min(), cm_clean.max(), 200)
                y_fit = skewt_pdf(x, df, loc, scale, a)
                y_fit = np.clip(y_fit, 0, y_max)
                ax.plot(x, y_fit, color=dark_purple, linewidth=2.5)
                cm_stats_text = f'df = {df:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\nskew = {a:.3f}\nmean = {cm_clean.mean():.3f}'
                print(f"  - Fit curve: min={y_fit.min():.6f}, max={y_fit.max():.6f}")
            else:
                print(f"  - FAILED: Could not fit skew-t distribution")
        elif dist_type == 'gamma':
            cm_fit = fit_gamma(cm_clean)
            if cm_fit:
                shape, loc, scale = cm_fit
                print(f"  - SUCCESS: shape={shape:.6f}, loc={loc:.6f}, scale={scale:.6f}")
                x = np.linspace(cm_clean.min(), cm_clean.max(), 200)
                y_fit = gamma_pdf(x, shape, loc, scale)
                y_fit = np.clip(y_fit, 0, y_max)
                ax.plot(x, y_fit, color=dark_purple, linewidth=2.5)
                cm_stats_text = f'shape = {shape:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\nmean = {cm_clean.mean():.3f}'
                print(f"  - Fit curve: min={y_fit.min():.6f}, max={y_fit.max():.6f}")
            else:
                print(f"  - FAILED: Could not fit gamma distribution")
    
    # Tract fit
    tr_stats_text = None
    if len(tr_clean) >= 2:
        print(f"\nTract {dist_type.upper()} fit:")
        if dist_type == 'skewt':
            tr_fit = fit_skewt(tr_clean)
            if tr_fit:
                df, loc, scale, a = tr_fit
                print(f"  - SUCCESS: df={df:.6f}, loc={loc:.6f}, scale={scale:.6f}, skew={a:.6f}")
                x = np.linspace(tr_clean.min(), tr_clean.max(), 200)
                y_fit = skewt_pdf(x, df, loc, scale, a)
                y_fit = np.clip(y_fit, 0, y_max)
                ax.plot(x, y_fit, color=dark_orange, linewidth=2.5)
                tr_stats_text = f'df = {df:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\nskew = {a:.3f}\nmean = {tr_clean.mean():.3f}'
                print(f"  - Fit curve: min={y_fit.min():.6f}, max={y_fit.max():.6f}")
            else:
                print(f"  - FAILED: Could not fit skew-t distribution")
        elif dist_type == 'gamma':
            tr_fit = fit_gamma(tr_clean)
            if tr_fit:
                shape, loc, scale = tr_fit
                print(f"  - SUCCESS: shape={shape:.6f}, loc={loc:.6f}, scale={scale:.6f}")
                x = np.linspace(tr_clean.min(), tr_clean.max(), 200)
                y_fit = gamma_pdf(x, shape, loc, scale)
                y_fit = np.clip(y_fit, 0, y_max)
                ax.plot(x, y_fit, color=dark_orange, linewidth=2.5)
                tr_stats_text = f'shape = {shape:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\nmean = {tr_clean.mean():.3f}'
                print(f"  - Fit curve: min={y_fit.min():.6f}, max={y_fit.max():.6f}")
            else:
                print(f"  - FAILED: Could not fit gamma distribution")
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
    if dist_type == 'gamma':
        x_min_limit = 0
    else:
        x_min_limit = mean_value - 2 * std_value
    x_max_limit = mean_value + 2 * std_value
    ax.set_xlim(x_min_limit, x_max_limit)
    
    # Set y-axis limit - force max to 9.9 for Income LDR
    if 'income ldr' in name.lower():
        ax.set_ylim(0, .099)
    else:
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
    fname = f"specialty_histograms2/specialty_histogram_{name.lower().replace(' ', '_').replace('/', '_')}.pdf"
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fname}")

# --- Main logic ---
if cm_data is not None and tr_data is not None:
    # inc PNC_st
    cm_inc_pnc = extract_sum_cm(cm_data, 'Transmitted_Sel_tr_to_cm_inc_PNC_st', 'Sel_cm_from_tr_inc_PNC_st',scale=1)
    tr_inc_pnc = extract_tr(tr_data, 'Sel_tr_from_bg_inc_PNC_st',scale=1)
    plot_hist(cm_inc_pnc, tr_inc_pnc, 'Income PNC_st', 'skewt')

    # inc LDR
    cm_inc_ldr = extract_sum_cm(cm_data, 'Transmitted_Sel_tr_to_cm_inc_LDR', 'Sel_cm_from_tr_inc_LDR', scale=1)
    tr_inc_ldr = extract_tr(tr_data, 'Sel_tr_from_bg_inc_LDR', scale=1)
    plot_hist(cm_inc_ldr, tr_inc_ldr, 'Income LDR', 'gamma')

    # pop PNC_st
    cm_pop_pnc = extract_sum_cm(cm_data, 'Transmitted_Sel_tr_to_cm_pop_PNC_st', 'Sel_cm_from_tr_pop_PNC_st',scale=1)
    tr_pop_pnc = extract_tr(tr_data, 'Sel_tr_from_bg_pop_PNC_st',scale=1  )
    plot_hist(cm_pop_pnc, tr_pop_pnc, 'Population PNC_st', 'skewt')

    # pop LDR
    cm_pop_ldr = extract_sum_cm(cm_data, 'Transmitted_Sel_tr_to_cm_pop_LDR', 'Sel_cm_from_tr_pop_LDR', scale=1)
    tr_pop_ldr = extract_tr(tr_data, 'Sel_tr_from_bg_pop_LDR', scale=1)
    plot_hist(cm_pop_ldr, tr_pop_ldr, 'Population LDR', 'gamma')
else:
    print("Error: Could not load required data files.") 