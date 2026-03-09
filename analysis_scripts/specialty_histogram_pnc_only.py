#!/usr/bin/env python3
"""
Calculate skew-t distribution parameters for PNC_st data without plotting.
Returns df, loc, scale, skew, and mean for community and tract levels.
"""

import pandas as pd
import numpy as np
import sys
from scipy.stats import t
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Check for 'null' argument
if 'null' in sys.argv:
    INPUT_DIR = 'output_terms_null'
else:
    INPUT_DIR = 'output_terms'

# Loads final pipeline results as a dataframe
def load_data(path):
    """Load CSV data"""
    try:
        df = pd.read_csv(path)
        print(f"Loaded {path}: {df.shape[0]} rows")
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# Extracts a specific column from the dataframe
def extract_tr(df, sel_col, scale=None):
    """Extract selection column"""
    if sel_col in df.columns:
        vals = pd.to_numeric(df[sel_col], errors='coerce')
        if scale:
            vals = vals / scale
        return vals
    else:
        return pd.Series([np.nan]*len(df))


def skewt_pdf(x, df, loc, scale, alpha):
    """Azzalini's skew-t PDF"""
    if scale <= 0 or df <= 0:
        return np.full_like(x, np.nan)
    
    z = (x - loc) / scale
    t_pdf = t.pdf(z, df)
    t_cdf = t.cdf(alpha * z * np.sqrt((df + 1) / (df + z**2)), df + 1)
    
    return 2 * t_pdf * t_cdf / scale


def fit_skewt(data):
    """Fit skew-t distribution, returns (df, loc, scale, alpha) or None"""
    clean_data = data[np.isfinite(data) & ~np.isnan(data) & (data != 0)]
    
    if len(clean_data) < 10:
        return None
    
    hist, bin_edges = np.histogram(clean_data, bins=min(30, len(clean_data)//5), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    p0 = [5, np.mean(clean_data), np.std(clean_data), 0]
    bounds = ([1, np.min(clean_data), 1e-6, -10], 
              [50, np.max(clean_data), np.std(clean_data)*5, 10])
    
    try:
        popt, _ = curve_fit(skewt_pdf, bin_centers, hist, p0=p0, bounds=bounds, maxfev=5000)
        return tuple(popt)
    except Exception as e:
        print(f"  Fit failed: {e}")
        return None


def calculate_parameters(cm, tr, name):
    """
    Calculate skew-t parameters for community and tract data.
    Returns a dictionary with parameters for both levels.
    """
    # Clean data
    cm_clean = cm[np.isfinite(cm) & ~np.isnan(cm) & (cm != 0)]
    tr_clean = tr[np.isfinite(tr) & ~np.isnan(tr) & (tr != 0)]
    
    print(f"\n{'='*60}")
    print(f"Calculating parameters for: {name}")
    print(f"{'='*60}")
    
    results = {
        'metric': name,
        'community': None,
        'tract': None
    }
    
    # Community parameters
    if len(cm_clean) >= 10:
        print(f"\nCommunity data: {len(cm_clean)} observations")
        cm_fit = fit_skewt(cm_clean)
        if cm_fit:
            df_val, loc, scale, skew = cm_fit
            mean_val = cm_clean.mean()
            results['community'] = {
                'df': df_val,
                'loc': loc,
                'scale': scale,
                'skew': skew,
                'mean': mean_val,
                'n_obs': len(cm_clean)
            }
            print(f"  ✓ df={df_val:.6f}, loc={loc:.6f}, scale={scale:.6f}, skew={skew:.6f}, mean={mean_val:.6f}")
        else:
            print(f"  ✗ Fit failed")
    else:
        print(f"\nCommunity data: insufficient observations ({len(cm_clean)})")
    
    # Tract parameters
    if len(tr_clean) >= 10:
        print(f"\nTract data: {len(tr_clean)} observations")
        tr_fit = fit_skewt(tr_clean)
        if tr_fit:
            df_val, loc, scale, skew = tr_fit
            mean_val = tr_clean.mean()
            results['tract'] = {
                'df': df_val,
                'loc': loc,
                'scale': scale,
                'skew': skew,
                'mean': mean_val,
                'n_obs': len(tr_clean)
            }
            print(f"  ✓ df={df_val:.6f}, loc={loc:.6f}, scale={scale:.6f}, skew={skew:.6f}, mean={mean_val:.6f}")
        else:
            print(f"  ✗ Fit failed")
    else:
        print(f"\nTract data: insufficient observations ({len(tr_clean)})")
    
    return results


def main():
    print("\n" + "="*60)
    print("Calculating Skew-t Parameters for PNC_st Metrics")
    print("="*60)
    
    # Load data
    cm_data = load_data(f'{INPUT_DIR}/bg_cm_exported_terms.csv')
    tr_data = load_data(f'{INPUT_DIR}/bg_tr_exported_terms.csv')
    
    if cm_data is None or tr_data is None:
        print("Error: Could not load data files.")
        return []
    
    all_results = []
    
    # Income PNC_st
    cm_inc_pnc = extract_tr(cm_data, 'Sel_cm_from_tr_inc_PNC_st', scale=1)
    tr_inc_pnc = extract_tr(tr_data, 'Sel_tr_from_bg_inc_PNC_st', scale=1)
    income_results = calculate_parameters(cm_inc_pnc, tr_inc_pnc, 'Income PNC_st')
    all_results.append(income_results)
    
    # Population PNC_st
    cm_pop_pnc = extract_tr(cm_data, 'Sel_cm_from_tr_pop_PNC_st', scale=1)
    tr_pop_pnc = extract_tr(tr_data, 'Sel_tr_from_bg_pop_PNC_st', scale=1)
    pop_results = calculate_parameters(cm_pop_pnc, tr_pop_pnc, 'Population PNC_st')
    all_results.append(pop_results)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL PARAMETERS")
    print("="*60)
    for result in all_results:
        print(f"\n{result['metric']}:")
        print(f"  Community: {result['community']}")
        print(f"  Tract:     {result['tract']}")
    
    print("\n" + "="*60)
    print("✓ Calculation complete!")
    print("="*60)
    
    return all_results


if __name__ == '__main__':
    results = main()
