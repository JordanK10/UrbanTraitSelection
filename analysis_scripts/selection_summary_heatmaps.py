"""
selection_summary_heatmaps.py

Two summary heatmaps (population path and income path), each 3 rows × 5 columns.

Columns (canonical order, matching state_summary_combined):
    TR | CM | CT | ST | BG

Rows:
    Weighted Mean MSS  — population-weighted mean of |Sel_level| / |AvgG_msa| × 100
                         across all units at each level.  BG empty.
                         (= msa_normalized_selection_summary row 1)

    Weighted Std MSS   — population-weighted std dev of the same.  BG empty.
                         (= msa_normalized_selection_summary row 2)

    NS                 — each column's Price-decomposition term / AvgG_msa × 100.
                         These five values sum to 100%.
                         (= state_summary_combined NS row)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman']})

# ── Paths ─────────────────────────────────────────────────────────────────────

if 'null' in sys.argv:
    INPUT_DIR       = 'output_terms_null'
    BASE_OUTPUT_DIR = 'plots_null'
else:
    INPUT_DIR       = 'output_terms'
    BASE_OUTPUT_DIR = 'plots'

PICKLE_FILE_PATH = os.path.join(INPUT_DIR, 'all_decomposition_results.pkl')
OUTPUT_DIR       = os.path.join(BASE_OUTPUT_DIR, 'heatmaps_local_dominance', 'st_summary')

BASE_ANALYSIS_LEVEL = 'bg'

# ── Style (identical to createHeatmaps.py / state_summary_combined) ───────────

CUSTOM_PURPLE = '#633673'
CUSTOM_ORANGE = '#E77429'
CUSTOM_GREY   = '#3C3C3C'

# Canonical column order matches state_summary_combined: selection terms first, BG last
CANONICAL_COLS   = ['TR', 'CM', 'CT', 'ST', 'BG']
TRANSMITTED_GROWTH_COL = 'BG'   # rendered last; excluded from colormap bounds

# ── Column definitions ────────────────────────────────────────────────────────

# For MSS rows: own-level selection col and level key, per display column and path.
MSS_SEL_COL = {
    'TR': {'_pop': ('tr', 'Sel_tr_from_bg_pop'),  '_inc': ('tr', 'Sel_tr_from_bg_inc')},
    'CM': {'_pop': ('cm', 'Sel_cm_from_tr_pop'),  '_inc': ('cm', 'Sel_cm_from_tr_inc')},
    'CT': {'_pop': ('ct', 'Sel_ct_from_cm_pop'),  '_inc': ('ct', 'Sel_ct_from_cm_inc')},
    'ST': {'_pop': ('st', 'Sel_st_from_ct_pop'),  '_inc': ('st', 'Sel_st_from_ct_inc')},
}

# For NS row: state-level Price decomposition terms (sum / AvgG_msa = 100%).
NS_COLS = {
    '_pop': {
        'TR': 'Transmitted_Sel_tr_to_st_pop',
        'CM': 'Transmitted_Sel_cm_to_st_pop',
        'CT': 'Transmitted_Sel_ct_to_st_pop',
        'ST': 'Sel_st_from_ct_pop',
        'BG': 'Transmitted_AvgG_bg_to_st_pop',
    },
    '_inc': {
        'TR': 'Transmitted_Sel_tr_to_st_inc',
        'CM': 'Transmitted_Sel_cm_to_st_inc',
        'CT': 'Transmitted_Sel_ct_to_st_inc',
        'ST': 'Sel_st_from_ct_inc',
        'BG': 'Transmitted_AvgG_bg_to_st_inc',
    },
}

MSA_GROWTH_COL = {'_pop': 'AvgG_pop_st', '_inc': 'AvgG_inc_st'}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(path):
    if not os.path.exists(path):
        print(f"Error: file not found at {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

# ── Row calculations ──────────────────────────────────────────────────────────

def _weighted_mean_std(series, weights):
    if weights is None or weights.isnull().all() or weights.sum() == 0:
        return series.mean(), series.std()
    if len(series) == 1:
        return series.iloc[0], 0.0
    mean = np.average(series, weights=weights)
    var  = np.average((series - mean) ** 2, weights=weights)
    return mean, np.sqrt(var)


def _mss_mean_std(df_level, level_key, sel_col, msa_growth):
    if sel_col not in df_level.columns or pd.isna(msa_growth) or msa_growth == 0:
        return np.nan, np.nan
    mss = (df_level[sel_col].abs() / abs(msa_growth)) * 100
    pop_col = f'PopInitial_{level_key}'
    weights = df_level[pop_col] if pop_col in df_level.columns else None
    return _weighted_mean_std(mss, weights)


# ── Build 3×5 DataFrame ───────────────────────────────────────────────────────

def build_plot_df(bg_results, path_suffix):
    df_st = bg_results.get('st', pd.DataFrame())
    msa_col = MSA_GROWTH_COL[path_suffix]
    msa_growth = df_st[msa_col].iloc[0] \
        if not df_st.empty and msa_col in df_st.columns else np.nan

    mean_row = {}
    std_row  = {}
    ns_row   = {}

    for col_label in CANONICAL_COLS:
        # NS row
        ns_col = NS_COLS[path_suffix].get(col_label)
        if ns_col and not df_st.empty and ns_col in df_st.columns \
                and not pd.isna(msa_growth) and msa_growth != 0:
            ns_row[col_label] = (df_st[ns_col].iloc[0] / msa_growth) * 100
        else:
            ns_row[col_label] = 0.0

        # MSS rows — BG has no own-level selection term
        if col_label == TRANSMITTED_GROWTH_COL:
            mean_row[col_label] = 0.0
            std_row[col_label]  = 0.0
            continue

        level_key, sel_col = MSS_SEL_COL[col_label][path_suffix]
        df_level = bg_results.get(level_key, pd.DataFrame())
        if df_level.empty:
            mean_row[col_label] = 0.0
            std_row[col_label]  = 0.0
            continue

        m, s = _mss_mean_std(df_level, level_key, sel_col, msa_growth)
        mean_row[col_label] = m if not pd.isna(m) else 0.0
        std_row[col_label]  = s if not pd.isna(s) else 0.0

    df = pd.DataFrame(
        [mean_row, std_row, ns_row],
        index=['Weighted Mean MSS', 'Weighted Std MSS', 'NS'],
        columns=CANONICAL_COLS,
    )
    return df

# ── Colormap (matches get_custom_colormap_and_bounds in createHeatmaps.py) ────

def _colormap_and_bounds(df):
    # Bounds from selection columns only — exclude the transmitted growth (BG) column
    sel_cols = [c for c in df.columns if c != TRANSMITTED_GROWTH_COL]
    v_bound = df[sel_cols].abs().max().max() if sel_cols else df.abs().max().max()
    if pd.isna(v_bound) or v_bound == 0:
        v_bound = 1.0
    cmap = LinearSegmentedColormap.from_list(
        'custom_purple_orange_grey', [CUSTOM_PURPLE, 'white', CUSTOM_ORANGE]
    )
    cmap.set_over(CUSTOM_GREY)
    cmap.set_under(CUSTOM_GREY)
    return cmap, -v_bound, v_bound

# ── Plot (matches _plot_summary_heatmap in createHeatmaps.py exactly) ─────────

COL_LABELS = ['tr', 'cm', 'ct', 'm', ' ']

ROW_LABELS = {
    '_pop': [r'$\langle\rho^p_j\rangle$', r'$\sigma(\rho^p_j)$', r'$\omega^p_j$'],
    '_inc': [r'$\langle\rho^i_j\rangle$', r'$\sigma(\rho^i_j)$', r'$\omega^i_j$'],
}


def plot_summary_heatmap(df, title, output_path, path_suffix='_pop'):
    n_rows, n_cols = df.shape
    figsize = (1.5 * n_cols, n_rows)

    cmap, vmin, vmax = _colormap_and_bounds(df)

    annotations = np.array(
        [['{:.1f}%'.format(val) for val in row] for row in df.values]
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df,
        annot=annotations,
        fmt='',
        cmap=cmap,
        linewidths=0.5,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        annot_kws={'size': 25},
    )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(COL_LABELS, fontsize=16)
    ax.set_yticklabels(ROW_LABELS[path_suffix], fontsize=16, rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"  Saved: {output_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = load_results(PICKLE_FILE_PATH)
    if results is None:
        return

    bg_results = results.get(BASE_ANALYSIS_LEVEL, {})

    for path_suffix, label in [('_pop', 'Population'), ('_inc', 'Income')]:
        df_plot = build_plot_df(bg_results, path_suffix)

        print(f"\n{label} path:")
        print(df_plot.to_string())
        print(f"  NS row sum: {df_plot.loc['NS'].sum():.4f}%")

        fname = f'selection_summary_{label.lower()}.pdf'
        plot_summary_heatmap(
            df_plot,
            title=f'Selection Summary — {label} Path',
            output_path=os.path.join(OUTPUT_DIR, fname),
            path_suffix=path_suffix,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
