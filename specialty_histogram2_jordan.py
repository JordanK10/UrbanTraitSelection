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
import sys
from scipy.stats import t, beta, gaussian_kde, chi2, gamma
from scipy.optimize import minimize
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
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'specialty_histograms2')
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

def _skewt_neg_log_likelihood(params, data):
    """Negative log-likelihood for the skew-t distribution."""
    df, loc, scale, alpha = params
    if scale <= 0 or df <= 0:
        return 1e12
    pdf_vals = skewt_pdf(data, df, loc, scale, alpha)
    pdf_vals = np.maximum(pdf_vals, 1e-300)
    return -np.sum(np.log(pdf_vals))

def fit_skewt(data):
    """
    Fit skew-t distribution via MLE on raw observations.
    df >= 3 ensures finite mean and variance.
    Returns: (df, loc, scale, alpha) or None if fit fails.
    """
    # Zeros represent upstream data unsuitability; remove them along with NaN/inf
    clean_data = data[np.isfinite(data) & ~np.isnan(data) & (data != 0)]
    n_removed = len(data) - len(clean_data)
    if n_removed > 0:
        print(f"    [fit_skewt] Removed {n_removed} values (NaN/inf/zero) from {len(data)} observations")

    if len(clean_data) < 10:
        return None

    x0 = [5.0, np.median(clean_data), np.std(clean_data), 0.0]
    bounds = [(3.0, 50.0),
              (np.min(clean_data), np.max(clean_data)),
              (1e-6, np.std(clean_data) * 5),
              (-10.0, 10.0)]

    best_result = None
    best_nll = np.inf

    for df_init in [3.0, 5.0, 10.0]:
        for alpha_init in [-1.0, 0.0, 1.0]:
            x0_trial = [df_init, np.median(clean_data), np.std(clean_data), alpha_init]
            try:
                result = minimize(_skewt_neg_log_likelihood, x0_trial, args=(clean_data,),
                                  method='L-BFGS-B', bounds=bounds, options={'maxiter': 5000})
                if result.success and result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except Exception:
                continue

    if best_result is not None and best_result.success:
        return tuple(best_result.x)

    # Fallback: try Nelder-Mead (unbounded) with clamping
    try:
        result = minimize(_skewt_neg_log_likelihood, x0, args=(clean_data,),
                          method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-8})
        if result.success:
            df, loc, scale, alpha = result.x
            df = max(3.0, min(df, 50.0))
            scale = max(1e-6, scale)
            return (df, loc, scale, alpha)
    except Exception as e:
        print(f"Skew-t MLE fit failed: {e}")

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
    Fit gamma distribution via MLE on raw observations using scipy.stats.gamma.fit.
    Returns: (shape, loc, scale) or None if fit fails.
    """
    # Zeros represent upstream data unsuitability; remove them along with NaN/inf
    clean_data = data[np.isfinite(data) & ~np.isnan(data) & (data > 0)]
    n_removed = len(data) - len(clean_data)
    if n_removed > 0:
        print(f"    [fit_gamma] Removed {n_removed} values (NaN/inf/non-positive) from {len(data)} observations")

    if len(clean_data) < 10:
        return None

    try:
        shape, loc, scale = gamma.fit(clean_data, floc=0)
        return (shape, loc, scale)
    except Exception as e:
        print(f"Gamma MLE fit failed: {e}")
        return None

# --- Plotting ---
def plot_hist(cm, tr, name, dist_type):
    # Remove NaN, inf, and exactly zero values.
    # Zeros indicate upstream data unsuitability (missing census data, unmatched units),
    # not genuine zero growth/selection. Removing them is a data-quality filter.
    cm_clean = cm[np.isfinite(cm) & ~np.isnan(cm) & (cm != 0)]
    tr_clean = tr[np.isfinite(tr) & ~np.isnan(tr) & (tr != 0)]
    cm_removed = len(cm) - len(cm_clean)
    tr_removed = len(tr) - len(tr_clean)
    all_data = np.concatenate([cm_clean, tr_clean])
    
    print(f"\n{'='*60}")
    print(f"PLOT: {name} ({dist_type.upper()} distribution)")
    print(f"{'='*60}")
    if cm_removed > 0:
        print(f"  Community: removed {cm_removed}/{len(cm)} values (NaN/inf/zero)")
    if tr_removed > 0:
        print(f"  Tract: removed {tr_removed}/{len(tr)} values (NaN/inf/zero)")
    
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
                ax.plot(x, y_fit, color=dark_purple, linewidth=2.5)
                cm_stats_text = f'df = {df:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\nskew = {a:.3f}\nmean = {cm_clean.mean():.3f}\nmed = {np.median(cm_clean):.3f}'
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
                ax.plot(x, y_fit, color=dark_purple, linewidth=2.5)
                cm_stats_text = f'shape = {shape:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\nmean = {cm_clean.mean():.3f}\nmed = {np.median(cm_clean):.3f}'
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
                ax.plot(x, y_fit, color=dark_orange, linewidth=2.5)
                tr_stats_text = f'df = {df:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\nskew = {a:.3f}\nmean = {tr_clean.mean():.3f}\nmed = {np.median(tr_clean):.3f}'
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
                ax.plot(x, y_fit, color=dark_orange, linewidth=2.5)
                tr_stats_text = f'shape = {shape:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\nmean = {tr_clean.mean():.3f}\nmed = {np.median(tr_clean):.3f}'
                print(f"  - Fit curve: min={y_fit.min():.6f}, max={y_fit.max():.6f}")
            else:
                print(f"  - FAILED: Could not fit gamma distribution")
    # --- Mean (solid), median (dashed), and zero reference lines ---
    if len(cm_clean) > 0:
        ax.axvline(cm_clean.mean(), color=dark_purple, linestyle='-', linewidth=1.5, alpha=0.75)
        ax.axvline(np.median(cm_clean), color=dark_purple, linestyle='--', linewidth=1.0, alpha=0.6)
    if len(tr_clean) > 0:
        ax.axvline(tr_clean.mean(), color=dark_orange, linestyle='-', linewidth=1.5, alpha=0.75)
        ax.axvline(np.median(tr_clean), color=dark_orange, linestyle='--', linewidth=1.0, alpha=0.6)
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
    
    # Let y-axis auto-scale so the fitted PDF is shown honestly
    ax.set_ylim(bottom=0)
    
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


def plot_hist_log_scale(cm, tr, name, dist_type):
    # Log-scale y-axis variant. Same zero-removal rationale as plot_hist.
    cm_clean = cm[np.isfinite(cm) & ~np.isnan(cm) & (cm != 0)]
    tr_clean = tr[np.isfinite(tr) & ~np.isnan(tr) & (tr != 0)]
    all_data = np.concatenate([cm_clean, tr_clean])
    
    if len(all_data) == 0:
        print(f"No valid data to plot for log-scale {name}!")
        return
    
    # --- Create figure and axis ---
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    dark_purple = custom_purple
    dark_orange = custom_orange
    
    # --- Plot histograms ---
    y_max = 0
    
    if len(tr_clean) > 0:
        n_bins_tr = 2 * (min(75, max(20, int(np.sqrt(len(tr_clean))))) if len(tr_clean) > 0 else 20)
        n_tr, _, _ = ax.hist(tr_clean, bins=n_bins_tr, alpha=0.7, color=dark_orange, density=True, edgecolor='white', linewidth=0.8)
        y_max = max(y_max, n_tr.max())

    if len(cm_clean) > 0:
        n_bins_cm = 2 * (min(30, max(10, int(np.sqrt(len(cm_clean))))) if len(cm_clean) > 0 else 10)
        n_cm, _, _ = ax.hist(cm_clean, bins=n_bins_cm, alpha=0.7, color=dark_purple, density=True, edgecolor='white', linewidth=0.8)
        y_max = max(y_max, n_cm.max())
    

        
    # --- Set Y-axis to log scale ---
    ax.set_yscale('log')
    
    # Set a lower bound for the y-axis to avoid issues with log(0)
    # A small positive number is appropriate for density plots.
    ax.set_ylim(bottom=1e-4) # Bounded below by a small number instead of -5

    # Fit and plot distributions
    # Note: Fitting is done on linear scale data, but plotted on log scale.
    if len(cm_clean) >= 2:
        if dist_type == 'skewt':
            cm_fit = fit_skewt(cm_clean)
            if cm_fit:
                x = np.linspace(cm_clean.min(), cm_clean.max(), 200)
                y_fit = skewt_pdf(x, *cm_fit)
                ax.plot(x, y_fit, color=dark_purple, linewidth=2.5)
        elif dist_type == 'gamma':
            cm_fit = fit_gamma(cm_clean)
            if cm_fit:
                x = np.linspace(cm_clean.min(), cm_clean.max(), 200)
                y_fit = gamma_pdf(x, *cm_fit)
                ax.plot(x, y_fit, color=dark_purple, linewidth=2.5)

    if len(tr_clean) >= 2:
        if dist_type == 'skewt':
            tr_fit = fit_skewt(tr_clean)
            if tr_fit:
                x = np.linspace(tr_clean.min(), tr_clean.max(), 200)
                y_fit = skewt_pdf(x, *tr_fit)
                ax.plot(x, y_fit, color=dark_orange, linewidth=2.5)
        elif dist_type == 'gamma':
            tr_fit = fit_gamma(tr_clean)
            if tr_fit:
                x = np.linspace(tr_clean.min(), tr_clean.max(), 200)
                y_fit = gamma_pdf(x, *tr_fit)
                ax.plot(x, y_fit, color=dark_orange, linewidth=2.5)

    # --- Mean (solid), median (dashed), and zero reference lines ---
    if len(cm_clean) > 0:
        ax.axvline(cm_clean.mean(), color=dark_purple, linestyle='-', linewidth=1.5, alpha=0.75)
        ax.axvline(np.median(cm_clean), color=dark_purple, linestyle='--', linewidth=1.0, alpha=0.6)
    if len(tr_clean) > 0:
        ax.axvline(tr_clean.mean(), color=dark_orange, linestyle='-', linewidth=1.5, alpha=0.75)
        ax.axvline(np.median(tr_clean), color=dark_orange, linestyle='--', linewidth=1.0, alpha=0.6)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.75)
    
    # --- Aesthetics ---
    mean_value = all_data.mean()
    std_value = all_data.std()
    ax.set_xlim(mean_value - 2 * std_value, mean_value + 2 * std_value)
    
    for spine in ['top', 'right', 'left', 'bottom']: ax.spines[spine].set_color('#CCCCCC')
    ax.tick_params(axis='x', which='major', labelsize=18, colors='#333333', bottom=True, top=False)
    ax.tick_params(axis='y', which='major', labelsize=18, colors='#333333', left=True, right=False, labelleft=True)
    # ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0)) # This line causes an error with log scale
    ax.grid(False)
    
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"specialty_histogram_{name.lower().replace(' ', '_').replace('/', '_')}_log_scale.pdf")
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved log-scale plot: {fname}")


# --- Main logic ---
if cm_data is not None and tr_data is not None:
    # inc PNC_st
    cm_inc_pnc = extract_tr(cm_data, 'Sel_cm_from_tr_inc_PNC_st', scale=1)
    tr_inc_pnc = extract_tr(tr_data, 'Sel_tr_from_bg_inc_PNC_st',scale=1)
    plot_hist(cm_inc_pnc, tr_inc_pnc, 'Income PNC_st', 'skewt')
    plot_hist_log_scale(cm_inc_pnc, tr_inc_pnc, 'Income PNC_st', 'skewt')

    # inc LDR
    cm_inc_ldr = extract_tr(cm_data, 'Sel_cm_from_tr_inc_LDR', scale=1)
    tr_inc_ldr = extract_tr(tr_data, 'Sel_tr_from_bg_inc_LDR', scale=1)
    plot_hist(cm_inc_ldr, tr_inc_ldr, 'Income LDR', 'gamma')
    plot_hist_log_scale(cm_inc_ldr, tr_inc_ldr, 'Income LDR', 'gamma')

    # pop PNC_st
    cm_pop_pnc = extract_tr(cm_data, 'Sel_cm_from_tr_pop_PNC_st', scale=1)
    tr_pop_pnc = extract_tr(tr_data, 'Sel_tr_from_bg_pop_PNC_st',scale=1  )
    plot_hist(cm_pop_pnc, tr_pop_pnc, 'Population PNC_st', 'skewt')
    plot_hist_log_scale(cm_pop_pnc, tr_pop_pnc, 'Population PNC_st', 'skewt')

    # pop LDR
    cm_pop_ldr = extract_tr(cm_data, 'Sel_cm_from_tr_pop_LDR', scale=1)
    tr_pop_ldr = extract_tr(tr_data, 'Sel_tr_from_bg_pop_LDR', scale=1)
    plot_hist(cm_pop_ldr, tr_pop_ldr, 'Population LDR', 'gamma')
    plot_hist_log_scale(cm_pop_ldr, tr_pop_ldr, 'Population LDR', 'gamma')
else:
    print("Error: Could not load required data files.") 