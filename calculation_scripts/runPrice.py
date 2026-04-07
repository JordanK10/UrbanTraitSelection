"""
runPrice.py  —  Combined census-to-decomposition pipeline.

Replaces procPrice.py + aggregatePriceV5.py.

Pipeline
--------
1. Load calculation_scripts/data/census_data1.pkl
2. Assign population columns; compute LogAvgIncome = log(E[y]) and
   AvgLogInc = E[log(y)]; recursively correct AvgLogInc upward through
   the geographic hierarchy.
3. Join initial-year (14) and final-year (19) DataFrames per level —
   no intermediate analysis_Price2.pkl needed.
4. prepare_level_data  →  calculate_multilevel_price_decomposition
5. calculate_pnc_and_ldr_metrics  →  export CSVs + all_decomposition_results.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings

from procHelpers import (
    load_data,
    assign_population_column,
    calculate_log_avg_income,
    calculate_avg_log_income,
    income_vars, comparison_data, xbins_values,
)

warnings.simplefilter(action='ignore', category=FutureWarning)

# ── Constants ─────────────────────────────────────────────────────────────────

INPUT_DATA_PATH      = "data/census_data1.pkl"
OUTPUT_DIR           = "output_terms"
TARGET_N             = 5
INITIAL_YEAR         = 14
FINAL_YEAR           = INITIAL_YEAR + TARGET_N   # 19
HIERARCHY_LEVELS     = ['bg', 'tr', 'cm', 'ct', 'st']
BASE_ANALYSIS_LEVELS = ['bg']

# Which column in the child DataFrame identifies the child's parent unit.
CHILD_TO_PARENT_ID_COL_MAP = {
    'bg': 'ParentTract',
    'tr': 'ParentCommunity',
    'cm': 'ParentCounty',
    'ct': 'ParentState',
}

# ── Phase 1: Census data preparation ─────────────────────────────────────────

def _recursively_calculate_avg_log_income(incomes_data):
    """Overwrite each parent's AvgLogInc with the pop-weighted avg of its children."""
    CHILD_PARENT_LINK = {
        'bg': ('tr', 'temp_tract_id'),
        'tr': ('cm', 'community'),
        'cm': ('ct', 'county'),
        'ct': ('st', 'state'),
    }
    for yr in sorted(incomes_data.keys()):
        for child_level, (parent_level, link_col) in CHILD_PARENT_LINK.items():
            if child_level not in incomes_data[yr] or parent_level not in incomes_data[yr]:
                continue
            df_child  = incomes_data[yr][child_level]
            df_parent = incomes_data[yr][parent_level]
            if df_child.empty or 'AvgLogInc' not in df_child.columns \
                    or 'population' not in df_child.columns:
                continue
            df_child = df_child.copy()
            df_child['_wt'] = df_child['population'] * df_child['AvgLogInc']
            grouped = df_child.groupby(link_col).agg(
                _pop=('population', 'sum'),
                _wt=('_wt', 'sum'),
            )
            grouped['corrected'] = grouped['_wt'] / grouped['_pop']
            df_parent = df_parent.copy()
            df_parent['AvgLogInc'] = df_parent.index.map(grouped['corrected'])
            incomes_data[yr][parent_level] = df_parent
    return incomes_data


def prepare_census_data(input_path):
    """
    Load census_data1.pkl, standardise population columns, compute
    LogAvgIncome and AvgLogInc for every unit/year/level, and recursively
    correct AvgLogInc upward through the hierarchy.

    Returns
    -------
    incomes_data : dict  {year: {level: DataFrame}}
    """
    incomes_data = load_data(input_path)
    if incomes_data is None:
        return None

    all_years = sorted(incomes_data.keys())
    agg_levels = HIERARCHY_LEVELS

    # Household count → 'population'; real population → 'population_r'
    assign_population_column(incomes_data, all_years, agg_levels, 'B19001_001E')
    for yr in all_years:
        if not isinstance(incomes_data[yr], dict):
            continue
        for agg in list(incomes_data[yr].keys()):
            df = incomes_data[yr][agg]
            if isinstance(df, pd.DataFrame) and 'B01003_001E' in df.columns:
                incomes_data[yr][agg] = df.rename(columns={'B01003_001E': 'population_r'})

    # Compute income metrics from the raw bin distributions
    print("Computing LogAvgIncome and AvgLogInc for all levels/years...")
    for yr in all_years:
        for agg in agg_levels:
            if yr in incomes_data and agg in incomes_data[yr]:
                df = incomes_data[yr][agg]
                if not df.empty:
                    df['LogAvgIncome'] = df.apply(calculate_log_avg_income, axis=1)
                    df['AvgLogInc']    = df.apply(calculate_avg_log_income,  axis=1)
                    incomes_data[yr][agg] = df

    # Recursively correct AvgLogInc so each parent level equals the
    # population-weighted average of its children's AvgLogInc.
    incomes_data = _recursively_calculate_avg_log_income(incomes_data)
    print("Census data preparation complete.")
    return incomes_data


# ── Phase 2: Build per-level initial/final DataFrames ────────────────────────

def build_level_panel(incomes_data, level, initial_year, final_year):
    """
    Join the initial- and final-year census DataFrames for *level* on their
    common unit indices, producing a single wide DataFrame ready for
    prepare_level_data.

    Returns a DataFrame whose column names match what aggregatePriceV5 expected
    from analysis_Price2.pkl (PopInitialN, Population, LogAvgIncInitialN, …).
    """
    missing = []
    if initial_year not in incomes_data or level not in incomes_data[initial_year]:
        missing.append(f"year {initial_year}")
    if final_year not in incomes_data or level not in incomes_data[final_year]:
        missing.append(f"year {final_year}")
    if missing:
        print(f"  Missing census data for level '{level}': {', '.join(missing)}.")
        return pd.DataFrame()

    df_yn = incomes_data[initial_year][level]
    df_y  = incomes_data[final_year][level]

    common_idx = df_yn.index.intersection(df_y.index)
    if common_idx.empty:
        print(f"  No common units for '{level}' between years {initial_year} and {final_year}.")
        return pd.DataFrame()

    def _col(df, name):
        return df.loc[common_idx, name].values if name in df.columns else np.nan

    df_out = pd.DataFrame(index=range(len(common_idx)))
    df_out['UnitName']           = common_idx.astype(str).tolist()
    df_out['PopInitialN']        = _col(df_yn, 'population')
    df_out['Population']         = _col(df_y,  'population')
    df_out['LogAvgIncInitialN']  = _col(df_yn, 'LogAvgIncome')
    df_out['LogAvgIncome']       = _col(df_y,  'LogAvgIncome')
    df_out['AvgLogIncInitialN']  = _col(df_yn, 'AvgLogInc')
    df_out['AvgLogInc']          = _col(df_y,  'AvgLogInc')
    df_out['PopInitialNR']       = _col(df_yn, 'population_r')
    df_out['PopulationR']        = _col(df_y,  'population_r')

    # Parent identifiers — taken from the final-year row
    for src, dst in [('state',     'ParentState'),
                     ('county',    'ParentCounty'),
                     ('community', 'ParentCommunity')]:
        if src in df_y.columns:
            df_out[dst] = df_y.loc[common_idx, src].values

    # ParentTract for block groups: first 11 chars of the FIPS index
    if level == 'bg':
        df_out['ParentTract'] = [idx[:11] for idx in common_idx.astype(str)]

    df_out['Agg']       = level
    df_out['FirstYear'] = initial_year
    df_out['Year']      = final_year

    print(f"  Built panel for level '{level}': {len(df_out)} units.")
    return df_out


# ── Phase 3: Data preparation helpers (from aggregatePriceV5.py) ─────────────

def _calculate_empirical_growth_rate(df, level_name, target_n_years,
                                     log_avg_inc_initial_col,
                                     log_avg_inc_final_col, emp_growth_col):
    if log_avg_inc_initial_col in df.columns and \
       log_avg_inc_final_col in df.columns and target_n_years > 0:
        df[emp_growth_col] = \
            (df[log_avg_inc_final_col] - df[log_avg_inc_initial_col]) / target_n_years
    else:
        df[emp_growth_col] = np.nan
    return df


def _calculate_actual_average_incomes(df, log_avg_inc_initial_col,
                                      log_avg_inc_final_col,
                                      actual_avg_inc_initial_col,
                                      actual_avg_inc_final_col):
    df[actual_avg_inc_initial_col] = (np.exp(df[log_avg_inc_initial_col].astype(float))
                                      if log_avg_inc_initial_col in df.columns
                                      else np.nan)
    df[actual_avg_inc_final_col]   = (np.exp(df[log_avg_inc_final_col].astype(float))
                                      if log_avg_inc_final_col in df.columns
                                      else np.nan)
    return df


def _ensure_parent_identifier(df, level_name, child_to_parent_id_map,
                               parent_id_col):
    if not parent_id_col:
        return df
    if level_name == 'bg' and parent_id_col == 'ParentTract':
        if 'UnitName' in df.columns:
            df[parent_id_col] = df['UnitName'].astype(str).str[:11]
        else:
            df[parent_id_col] = pd.NA
    elif parent_id_col not in df.columns:
        df[parent_id_col] = pd.NA
    else:
        df[parent_id_col] = df[parent_id_col].astype(str)
    return df


def _perform_final_checks_and_filter(df, level_name, pop_initial_col,
                                     log_avg_inc_initial_col):
    if log_avg_inc_initial_col in df.columns:
        nan_count = df[log_avg_inc_initial_col].isna().sum()
        if nan_count > 0:
            df[log_avg_inc_initial_col].fillna(0, inplace=True)
    if pop_initial_col in df.columns:
        df.dropna(subset=[pop_initial_col], inplace=True)
        df = df[df[pop_initial_col] > 0]
    return df


def prepare_level_data(df_level_raw, level_name, target_n_years,
                       child_to_parent_id_map):
    """
    Rename columns, compute growth rates and relative metrics, ensure parent
    identifiers.  Mirrors the aggregatePriceV5 function of the same name.
    """
    if df_level_raw.empty:
        return pd.DataFrame()

    df = df_level_raw.copy()

    # 1. Rename core columns
    rename_map = {
        'PopInitialN':      f'PopInitial_{level_name}',
        'LogAvgIncInitialN': f'LogAvgIncInitial_{level_name}',
        'Population':       f'PopFinal_{level_name}',
        'LogAvgIncome':     f'LogAvgIncFinal_{level_name}',
        'AvgLogIncInitialN': f'AvgLogIncInitial_{level_name}',
        'AvgLogInc':        f'AvgLogIncFinal_{level_name}',
        'PopInitialNR':     f'PopInitialR_{level_name}',
        'PopulationR':      f'PopFinalR_{level_name}',
    }
    df.rename(columns=rename_map, inplace=True)

    pop_initial_col          = f'PopInitial_{level_name}'
    pop_final_col            = f'PopFinal_{level_name}'
    log_avg_inc_initial_col  = f'LogAvgIncInitial_{level_name}'
    log_avg_inc_final_col    = f'LogAvgIncFinal_{level_name}'
    actual_avg_inc_initial_col = f'ActualAvgIncInitial_{level_name}'
    actual_avg_inc_final_col   = f'ActualAvgIncFinal_{level_name}'
    emp_growth_col           = f'AvgG_emp_{level_name}'
    g_avg_log_inc_col        = f'G_avg_log_inc_{level_name}'
    avg_log_inc_initial_col  = f'AvgLogIncInitial_{level_name}'
    avg_log_inc_final_col    = f'AvgLogIncFinal_{level_name}'
    pop_g_col                = f'PopG_{level_name}'
    rel_avg_g_emp_col        = f'RelAvgG_emp_{level_name}'
    rel_pop_g_col            = f'RelPopG_{level_name}'
    pop_initial_r_col        = f'PopInitialR_{level_name}'
    pop_final_r_col          = f'PopFinalR_{level_name}'
    parent_id_col            = child_to_parent_id_map.get(level_name)

    # 2. Empirical growth rate: (log(E[y])_final - log(E[y])_initial) / N
    df = _calculate_empirical_growth_rate(
        df, level_name, target_n_years,
        log_avg_inc_initial_col, log_avg_inc_final_col, emp_growth_col)

    # 2b. Growth rate of E[log(y)]
    if avg_log_inc_initial_col in df.columns and avg_log_inc_final_col in df.columns \
            and target_n_years > 0:
        df[g_avg_log_inc_col] = \
            (df[avg_log_inc_final_col] - df[avg_log_inc_initial_col]) / target_n_years
    else:
        df[g_avg_log_inc_col] = np.nan

    # 3. Actual average incomes (exp of log versions)
    df = _calculate_actual_average_incomes(
        df, log_avg_inc_initial_col, log_avg_inc_final_col,
        actual_avg_inc_initial_col, actual_avg_inc_final_col)

    # 4. Ensure parent identifier column
    df = _ensure_parent_identifier(df, level_name, child_to_parent_id_map, parent_id_col)

    # 5. Drop units with invalid / zero initial population
    df = _perform_final_checks_and_filter(
        df, level_name, pop_initial_col, log_avg_inc_initial_col)

    # 6. Population growth rate and relative metrics
    if pop_initial_col in df.columns and pop_final_col in df.columns and target_n_years > 0:
        safe_ini = df[pop_initial_col].apply(lambda x: x if x > 0 else np.nan)
        safe_fin = df[pop_final_col].apply(lambda x: x if x > 0 else np.nan)
        df[pop_g_col] = (np.log(safe_fin) - np.log(safe_ini)) / target_n_years
    else:
        df[pop_g_col] = np.nan

    if parent_id_col and parent_id_col in df.columns:
        grp = df.groupby(parent_id_col)

        # Pop-weighted sibling mean helper
        def _rel(df, val_col, out_col):
            w_col = f'_w_{val_col}'
            s_col = f'_sw_{val_col}'
            n_col = f'_sn_{val_col}'
            df[w_col] = df[pop_initial_col] * df[val_col]
            df[s_col] = df.groupby(parent_id_col)[w_col].transform('sum')
            df[n_col] = df.groupby(parent_id_col)[pop_initial_col].transform('sum').replace(0, np.nan)
            df[out_col] = df[val_col] - df[s_col] / df[n_col]
            df.drop(columns=[w_col, s_col, n_col], inplace=True)
            return df

        df = _rel(df, emp_growth_col, rel_avg_g_emp_col)
        df = _rel(df, pop_g_col,      rel_pop_g_col)
    else:
        df[rel_avg_g_emp_col] = 0.0
        df[rel_pop_g_col]     = 0.0

    # 7. Select essential output columns
    essential = [
        'UnitName',
        pop_initial_col, pop_final_col,
        log_avg_inc_initial_col, log_avg_inc_final_col,
        avg_log_inc_initial_col, avg_log_inc_final_col,
        g_avg_log_inc_col,
        actual_avg_inc_initial_col, actual_avg_inc_final_col,
        emp_growth_col, pop_g_col,
        rel_avg_g_emp_col, rel_pop_g_col,
        pop_initial_r_col, pop_final_r_col,
    ]
    for p in CHILD_TO_PARENT_ID_COL_MAP.values():
        if p in df.columns and p not in essential:
            essential.append(p)
    if 'Agg' not in df.columns:
        df['Agg'] = level_name
    final_cols = [c for c in essential if c in df.columns]
    if 'Agg' in df.columns and 'Agg' not in final_cols:
        final_cols.append('Agg')

    print(f"  Prepared level '{level_name}': {len(df)} units.")
    return df[final_cols].copy()


# ── Decomposition helpers (from aggregatePriceV5.py, unchanged) ───────────────

def _calculate_sanity_check(df_output, lhs_col, primary_rhs_col,
                            other_rhs_cols, sanity_col,
                            level='', path='', check_type=''):
    cols_present = df_output.columns.tolist()
    actual_other = [c for c in other_rhs_cols if c in cols_present]
    missing = [c for c in ([lhs_col, primary_rhs_col] + other_rhs_cols)
               if c not in cols_present]
    if not missing:
        rhs_other = df_output[actual_other].sum(axis=1, skipna=False) \
            if actual_other else 0.0
        df_output[sanity_col] = df_output[lhs_col] - \
            (df_output[primary_rhs_col] + rhs_other)
    else:
        df_output[sanity_col] = np.nan
    return df_output


def _calculate_de_meaned_column(df, source_col, target_col):
    if source_col in df.columns:
        m = df[source_col].mean()
        df[target_col] = df[source_col] - m if pd.notna(m) else np.nan
    else:
        df[target_col] = np.nan
    return df


def _aggregate_split_tracts_v5(df_child, df_parent_y, df_parent_yn,
                                child_level_name, parent_level_name):
    """
    Remap child ParentTract IDs so that block groups whose tract split between
    initial and final year are grouped under the original (Y-N) tract ID.
    """
    if parent_level_name != 'tr':
        return df_child
    if df_parent_yn is None or df_parent_yn.empty:
        return df_child

    parent_y_idx  = df_parent_y.index.astype(str)  if not df_parent_y.empty  else pd.Index([])
    parent_yn_idx = df_parent_yn.index.astype(str) if not df_parent_yn.empty else pd.Index([])
    split_ids = parent_yn_idx.difference(parent_y_idx)

    if split_ids.empty:
        return df_child
    if 'ParentTract' not in df_child.columns:
        return df_child

    print(f"      Detected {len(split_ids)} split tracts; remapping child ParentTract IDs.")
    df_out = df_child.copy()
    df_out['_prefix9'] = df_out['ParentTract'].astype(str).str[:9]
    for split_id in split_ids:
        mask = df_out['_prefix9'] == split_id[:9]
        if mask.any():
            df_out.loc[mask, 'ParentTract'] = split_id
    return df_out.drop(columns=['_prefix9'])


def _part_a_initial_child_preparation(df_child, child_level_name):
    df = df_child.copy()
    names = {
        'pop_initial':           f'PopInitial_{child_level_name}',
        'pop_final':             f'PopFinal_{child_level_name}',
        'log_avg_inc_initial':   f'LogAvgIncInitial_{child_level_name}',
        'actual_avg_inc_initial': f'ActualAvgIncInitial_{child_level_name}',
        'avg_g_emp':             f'AvgG_emp_{child_level_name}',
        'log_avg_inc_final':     f'LogAvgIncFinal_{child_level_name}',
        'pop_final_for_shares':  f'PopFinal_{child_level_name}',
        'avg_log_inc_initial':   f'AvgLogIncInitial_{child_level_name}',
        'g_avg_log_inc':         f'G_avg_log_inc_{child_level_name}',
    }
    for col in names.values():
        if col not in df.columns:
            print(f"      Error: essential child column '{col}' missing for '{child_level_name}'.")
            return None, None
    df['c_w_c'] = np.divide(
        df[names['pop_final']].astype(float),
        df[names['pop_initial']].astype(float))
    df.loc[df[names['pop_initial']] == 0, 'c_w_c'] = 0.0
    return df, names


def _part_b_parent_group_aggregates(grp, names):
    pop_sum = grp[names['pop_initial']].sum()
    grp['c_p_c_P_grp'] = grp[names['pop_initial']] / pop_sum if pop_sum != 0 else 0.0
    W_P  = (grp['c_p_c_P_grp'] * grp['c_w_c']).sum()
    Z_P  = (grp['c_p_c_P_grp'] * grp[names['avg_log_inc_initial']]).sum()
    mu_P = (grp['c_p_c_P_grp'] * grp[names['actual_avg_inc_initial']]).sum()
    return W_P, Z_P, mu_P, grp


def _part_d_calculate_selection_terms(grp, names, W_P, Z_P, mu_P,
                                      target_n, child_growth_col):
    sel_pop = np.nan
    sel_inc = np.nan
    avg_child_growth_inc = np.nan

    if pd.notna(W_P) and W_P != 0 and target_n > 0:
        cov_pop = (grp['c_p_c_P_grp'] *
                   (grp['c_w_c'] / W_P - 1) *
                   (grp[names['avg_log_inc_initial']] - Z_P)).sum()
        sel_pop = cov_pop / target_n

    if pd.notna(mu_P) and mu_P != 0 and child_growth_col in grp.columns:
        avg_g = (grp['c_p_c_P_grp'] * grp[child_growth_col]).sum()
        cov_inc = (grp['c_p_c_P_grp'] *
                   (grp[child_growth_col] - avg_g) *
                   (grp[names['actual_avg_inc_initial']] - mu_P)).sum()
        sel_inc = cov_inc / mu_P
        avg_child_growth_inc = avg_g

    return sel_pop, sel_inc, avg_child_growth_inc


def _part_e_calculate_transmitted_terms(grp, names, child_level, parent_level,
                                        base_level, full_hierarchy,
                                        W_P, mu_P):
    out = {}
    df_t = grp.copy()

    for path in ['_pop', '_inc']:
        wt_col = f'_wf{path}'
        if path == '_pop':
            df_t[wt_col] = (df_t['c_p_c_P_grp'] * df_t['c_w_c']) / W_P \
                if pd.notna(W_P) and W_P != 0 else 0.0
        else:
            df_t[wt_col] = df_t['c_p_c_P_grp']

        # Transmit base growth
        if child_level == base_level:
            base_col = names['g_avg_log_inc'] if path == '_pop' else names['avg_g_emp']
        else:
            base_col = f'Transmitted_AvgG_{base_level}_to_{child_level}{path}'

        tgt = f'Transmitted_AvgG_{base_level}_to_{parent_level}{path}'
        out[tgt] = (df_t[wt_col] * df_t[base_col]).sum() \
            if base_col in df_t.columns else np.nan

        # Transmit selection terms for intermediate levels
        if child_level != base_level:
            try:
                c_idx = full_hierarchy.index(child_level)
                if c_idx > 0:
                    grand = full_hierarchy[c_idx - 1]
                    own_sel_col = f'Sel_{child_level}_from_{grand}{path}'
                    tgt_sel = f'Transmitted_Sel_{child_level}_to_{parent_level}{path}'
                    out[tgt_sel] = (df_t[wt_col] * df_t[own_sel_col]).sum() \
                        if own_sel_col in df_t.columns else np.nan

                    base_idx = full_hierarchy.index(base_level)
                    for k in range(base_idx + 1, c_idx):
                        prev = full_hierarchy[k]
                        src = f'Transmitted_Sel_{prev}_to_{child_level}{path}'
                        dst = f'Transmitted_Sel_{prev}_to_{parent_level}{path}'
                        if src in df_t.columns:
                            out[dst] = (df_t[wt_col] * df_t[src]).sum()
                        else:
                            out[dst] = 0.0 if prev == base_level else np.nan
            except ValueError:
                pass

    return out


def _sanity_check_handler(df_parent, child_level, parent_level, base_level):
    for path in ['_pop', '_inc']:
        # Multi-level check
        lhs_multi = f'AvgG_pop_{parent_level}' if path == '_pop' \
            else f'AvgG_inc_{parent_level}'
        primary_multi = f'Sel_{parent_level}_from_{child_level}{path}'
        other_multi   = [f'Transmitted_AvgG_{base_level}_to_{parent_level}{path}'] + \
                        [c for c in df_parent.columns
                         if c.startswith('Transmitted_Sel_') and
                         c.endswith(f'_to_{parent_level}{path}')]
        df_parent = _calculate_sanity_check(
            df_parent, lhs_multi, primary_multi, other_multi,
            f'SanChk_MultiLevel_{path[1:]}_{base_level}_to_{parent_level}')

        # Single-level check
        if path == '_pop':
            lhs_s  = f'LHS_AvgG_pop_{parent_level}'
            prim_s = f'TransmissionDirectChildGrowth_{parent_level}_pop'
            oth_s  = [f'Sel_{parent_level}_from_{child_level}_pop']
        else:
            lhs_s  = f'AvgG_inc_{parent_level}'
            prim_s = f'ExpectedChildEmpiricalGrowth_{parent_level}_inc'
            oth_s  = [f'Sel_{parent_level}_from_{child_level}_inc']
        df_parent = _calculate_sanity_check(
            df_parent, lhs_s, prim_s, oth_s,
            f'SanChk_SingleLevel_{path[1:]}_{child_level}_to_{parent_level}')

    return df_parent


# ── Core decomposition ────────────────────────────────────────────────────────

def calculate_single_parent_level_terms(df_child, child_level, parent_level,
                                        df_parent_intrinsic,
                                        child_to_parent_id_map,
                                        target_n, full_hierarchy, base_level,
                                        df_parent_y, df_parent_yn):
    """
    Compute all selection and transmission terms for one child→parent step.
    """
    print(f"    {child_level} → {parent_level}")

    # Remap split tracts if needed
    if parent_level == 'tr':
        df_child = _aggregate_split_tracts_v5(
            df_child, df_parent_y, df_parent_yn, child_level, parent_level)

    parent_id_col = child_to_parent_id_map.get(child_level)
    if not parent_id_col or parent_id_col not in df_child.columns:
        print(f"      Error: parent ID col '{parent_id_col}' not in child data.")
        return df_parent_intrinsic.copy() if df_parent_intrinsic is not None else pd.DataFrame()

    df_child_prep, names = _part_a_initial_child_preparation(df_child, child_level)
    if df_child_prep is None:
        return df_parent_intrinsic.copy() if df_parent_intrinsic is not None else pd.DataFrame()

    df_parent_out = df_parent_intrinsic.copy() \
        if df_parent_intrinsic is not None and not df_parent_intrinsic.empty \
        else pd.DataFrame()

    g_avg_log_inc_col   = names['g_avg_log_inc']
    log_avg_inc_fin_col = names['log_avg_inc_final']
    pop_fin_col         = names['pop_final_for_shares']

    parent_rows = []

    for parent_id, grp in df_child_prep.groupby(parent_id_col):
        grp = grp.copy()
        W_P, Z_P, mu_P, grp = _part_b_parent_group_aggregates(grp, names)

        child_growth_col = (f'AvgG_emp_{child_level}' if child_level == base_level
                            else f'AvgG_inc_{child_level}')

        sel_pop, sel_inc, avg_child_g_inc = _part_d_calculate_selection_terms(
            grp, names, W_P, Z_P, mu_P, target_n, child_growth_col)

        transmitted = _part_e_calculate_transmitted_terms(
            grp, names, child_level, parent_level, base_level,
            full_hierarchy, W_P, mu_P)

        # LHS AvgG_pop (single-level sanity check)
        avg_g_pop_val = np.nan
        if log_avg_inc_fin_col in grp.columns and pop_fin_col in grp.columns \
                and target_n > 0:
            pop_fin_sum = grp[pop_fin_col].sum()
            if pop_fin_sum > 0:
                grp['c_p_prime'] = grp[pop_fin_col] / pop_fin_sum
                avg_log_fin_col = names['avg_log_inc_initial'].replace('Initial', 'Final')
                if avg_log_fin_col in grp.columns:
                    Z_P_prime = (grp['c_p_prime'] * grp[avg_log_fin_col]).sum()
                    if pd.notna(Z_P_prime) and pd.notna(Z_P):
                        avg_g_pop_val = (Z_P_prime - Z_P) / target_n

        # Fitness-weighted transmission of child direct growth (single-level RHS)
        trans_direct_gro = np.nan
        if g_avg_log_inc_col in grp.columns and pd.notna(W_P) and W_P != 0:
            trans_direct_gro = (grp['c_p_c_P_grp'] * grp['c_w_c'] *
                                grp[g_avg_log_inc_col]).sum() / W_P

        # Recursive AvgG_pop (multi-level)
        g_pop_child_col = f'AvgG_pop_{child_level}'
        if g_pop_child_col in grp.columns:
            trans_child_pop = ((grp['c_p_c_P_grp'] * grp['c_w_c'] *
                                grp[g_pop_child_col]).sum() / W_P
                               if pd.notna(W_P) and W_P != 0 else np.nan)
        else:
            g_direct = names['g_avg_log_inc']
            trans_child_pop = ((grp['c_p_c_P_grp'] * grp['c_w_c'] *
                                grp[g_direct]).sum() / W_P
                               if g_direct in grp.columns and
                               pd.notna(W_P) and W_P != 0 else np.nan)

        g_pop_rec = ((sel_pop if pd.notna(sel_pop) else 0.0) +
                     (trans_child_pop if pd.notna(trans_child_pop) else 0.0))

        # AvgG_inc
        avg_g_inc_val = np.nan
        if (child_growth_col in grp.columns and
                names['pop_initial'] in grp.columns and
                names['actual_avg_inc_initial'] in grp.columns and
                pd.notna(mu_P) and mu_P != 0):
            num = (grp['c_p_c_P_grp'] * grp[child_growth_col] *
                   grp[names['actual_avg_inc_initial']]).sum()
            avg_g_inc_val = num / mu_P

        row = {
            'UnitName': parent_id,
            f'Sel_{parent_level}_from_{child_level}_pop': sel_pop,
            f'Sel_{parent_level}_from_{child_level}_inc': sel_inc,
            f'AvgG_pop_{parent_level}': g_pop_rec,
            f'AvgG_inc_{parent_level}': avg_g_inc_val,
            f'LHS_AvgG_pop_{parent_level}': avg_g_pop_val,
            f'TransmissionDirectChildGrowth_{parent_level}_pop': trans_direct_gro,
            f'ExpectedChildEmpiricalGrowth_{parent_level}_inc': avg_child_g_inc,
        }

        # Transmitted aggregated growth (multi-level sanity check)
        g_pop_c_col = f'AvgG_pop_{child_level}'
        g_inc_c_col = f'AvgG_inc_{child_level}'
        if g_pop_c_col in grp.columns and pd.notna(W_P) and W_P != 0:
            row[f'Transmitted_AggG_pop_{child_level}_to_{parent_level}'] = \
                (grp['c_p_c_P_grp'] * grp['c_w_c'] * grp[g_pop_c_col]).sum() / W_P
        else:
            row[f'Transmitted_AggG_pop_{child_level}_to_{parent_level}'] = np.nan

        if g_inc_c_col in grp.columns:
            row[f'Transmitted_AggG_inc_{child_level}_to_{parent_level}'] = \
                (grp['c_p_c_P_grp'] * grp[g_inc_c_col]).sum()
        else:
            row[f'Transmitted_AggG_inc_{child_level}_to_{parent_level}'] = np.nan

        row.update(transmitted)
        parent_rows.append(row)

    if parent_rows:
        df_terms = pd.DataFrame(parent_rows)
        if not df_parent_out.empty:
            df_parent_out = pd.merge(df_parent_out, df_terms, on='UnitName', how='left')
        else:
            df_parent_out = df_terms

    df_parent_out = _sanity_check_handler(df_parent_out, child_level, parent_level, base_level)
    return df_parent_out


def calculate_multilevel_price_decomposition(prepared_level_data, base_level,
                                             full_hierarchy,
                                             child_to_parent_id_map,
                                             target_n, incomes_data):
    """
    Walk up the hierarchy from *base_level*, computing selection and
    transmission terms at each step.

    *incomes_data* is the raw census dict {year: {level: DataFrame}},
    used only to supply the year-14 and year-19 tract panels for the
    split-tract remapping step.
    """
    print(f"  Decomposition starting from '{base_level}'")

    if base_level not in prepared_level_data or \
            prepared_level_data[base_level].empty:
        print(f"  Error: no prepared data for base level '{base_level}'.")
        return {}

    results = {base_level: prepared_level_data[base_level].copy()}

    try:
        start = full_hierarchy.index(base_level)
    except ValueError:
        print(f"  Error: '{base_level}' not in hierarchy.")
        return {}

    # Pre-fetch census panels for both years (used for tract split detection)
    census_y  = incomes_data.get(FINAL_YEAR,   {})
    census_yn = incomes_data.get(INITIAL_YEAR, {})

    for i in range(start, len(full_hierarchy) - 1):
        child_level  = full_hierarchy[i]
        parent_level = full_hierarchy[i + 1]

        df_child = results.get(child_level)
        if df_child is None or df_child.empty:
            print(f"      Skipping {child_level} → {parent_level}: no child data.")
            results[parent_level] = pd.DataFrame()
            continue

        df_parent_intrinsic = prepared_level_data.get(parent_level, pd.DataFrame())

        # Raw census DataFrames for the tract-split logic
        df_parent_census_y  = census_y.get(parent_level,  pd.DataFrame())
        df_parent_census_yn = census_yn.get(parent_level, pd.DataFrame())

        results[parent_level] = calculate_single_parent_level_terms(
            df_child, child_level, parent_level,
            df_parent_intrinsic, child_to_parent_id_map,
            target_n, full_hierarchy, base_level,
            df_parent_y=df_parent_census_y,
            df_parent_yn=df_parent_census_yn,
        )

    # Summary columns: de-meaned growth and cumulative selection
    for level_name, df_lvl in results.items():
        if df_lvl is None or df_lvl.empty:
            continue
        df_lvl = _calculate_de_meaned_column(df_lvl, f'AvgG_pop_{level_name}', f'RelAvgG_pop_{level_name}')
        df_lvl = _calculate_de_meaned_column(df_lvl, f'AvgG_inc_{level_name}', f'RelAvgG_inc_{level_name}')

        for path in ['_gro', '_inc']:
            cum_col = f'cum_sel{path}'
            df_lvl[cum_col] = 0.0
            try:
                idx = full_hierarchy.index(level_name)
                if idx > 0:
                    child = full_hierarchy[idx - 1]
                    own = f'Sel_{level_name}_from_{child}{path}'
                    if own in df_lvl.columns:
                        df_lvl[cum_col] += df_lvl[own].fillna(0)
            except ValueError:
                pass
            for c in df_lvl.columns:
                if c.startswith('Transmitted_Sel_') and \
                        c.endswith(f'_to_{level_name}{path}'):
                    df_lvl[cum_col] += df_lvl[c].fillna(0)

        results[level_name] = df_lvl

    return results


def calculate_pnc_and_ldr_metrics(decomp_results, full_hierarchy,
                                  child_to_parent_id_map, base_level):
    """
    Post-processing: Local Dominance Ratio (LDR) and
    Parent-Normalized Contribution (PNC) for every component at every level.
    """
    print("  Calculating LDR and PNC metrics...")

    # ── LDR ──────────────────────────────────────────────────────────────────
    for i, level_name in enumerate(full_hierarchy):
        df_lvl = decomp_results.get(level_name)
        if df_lvl is None or df_lvl.empty or i < 1:
            continue
        child = full_hierarchy[i - 1]
        for path in ['_pop', '_inc']:
            own_sel = f'Sel_{level_name}_from_{child}{path}'
            # Identify all component columns reaching this level
            comp_cols = ([own_sel] if own_sel in df_lvl.columns else []) + [
                c for c in df_lvl.columns
                if (c.startswith('Transmitted_Sel_') or
                    c.startswith('Transmitted_AvgG_') or
                    c.startswith('Transmitted_AggG_'))
                and c.endswith(f'_to_{level_name}{path}')
                and not c.endswith('_LDR') and '_PNC' not in c
            ]
            comp_cols = sorted(set(comp_cols))
            if not comp_cols:
                continue
            denom = df_lvl[comp_cols].abs().sum(axis=1).replace(0, np.nan)
            for cc in comp_cols:
                df_lvl[f'{cc}_LDR'] = (df_lvl[cc].abs() / denom) * 100
        decomp_results[level_name] = df_lvl

    # ── PNC ──────────────────────────────────────────────────────────────────
    ancestor_growth = {}
    for level_name in full_hierarchy:
        df_anc = decomp_results.get(level_name)
        if df_anc is None or df_anc.empty or 'UnitName' not in df_anc.columns:
            continue
        g_cols = [c for c in df_anc.columns if c.startswith('AvgG_')]
        ancestor_growth[level_name] = df_anc[['UnitName'] + g_cols].copy()

    all_out = {}
    for i, level_name in enumerate(full_hierarchy):
        df_lvl = decomp_results.get(level_name)
        if df_lvl is None or df_lvl.empty:
            all_out[level_name] = pd.DataFrame()
            continue

        df_aug = df_lvl.copy()

        # Merge ancestor growth rates into this level's DataFrame
        for j in range(i + 1, len(full_hierarchy)):
            anc_name = full_hierarchy[j]
            df_anc   = ancestor_growth.get(anc_name)
            if df_anc is None:
                continue
            link_level  = full_hierarchy[j - 1]
            merge_key   = child_to_parent_id_map.get(link_level)
            if not merge_key or merge_key not in df_aug.columns:
                continue
            df_aug = pd.merge(df_aug, df_anc, left_on=merge_key,
                              right_on='UnitName', how='left',
                              suffixes=('', f'_{anc_name}_y'))

        # Component columns
        comp_cols = [c for c in df_lvl.columns
                     if (c.startswith('Sel_') or c.startswith('Transmitted_'))
                     and ('_pop' in c or '_inc' in c)
                     and not c.endswith('_LDR') and '_PNC' not in c]

        for j in range(i, len(full_hierarchy)):
            anc_name = full_hierarchy[j]
            pop_d = f'AvgG_pop_{anc_name}'
            inc_d = f'AvgG_inc_{anc_name}'
            if pop_d not in df_aug.columns or inc_d not in df_aug.columns:
                continue
            denoms = {
                '_pop': df_aug[pop_d].replace(0, np.nan),
                '_inc': df_aug[inc_d].replace(0, np.nan),
            }
            pnc_sfx = '_PNC' if anc_name == level_name else f'_PNC_{anc_name}'
            for cc in comp_cols:
                path = '_pop' if '_pop' in cc else '_inc'
                if cc in df_aug.columns:
                    df_aug[f'{cc}{pnc_sfx}'] = \
                        (df_aug[cc] / denoms[path]) * 100

        # Cumulative selection + its PNC
        for path in ['_pop', '_inc']:
            sel_cols = [c for c in df_aug.columns
                        if (c.startswith(f'Sel_{level_name}_from_') or
                            c.startswith('Transmitted_Sel_'))
                        and c.endswith(f'_to_{level_name}{path}' if
                                       c.startswith('Transmitted_') else path)
                        and not c.endswith('_LDR') and '_PNC' not in c]
            # Direct own-level sel cols don't end with '_to_...'
            own_sel_cols = [c for c in df_aug.columns
                            if c.startswith(f'Sel_{level_name}_from_')
                            and c.endswith(path)
                            and not c.endswith('_LDR') and '_PNC' not in c]
            trans_sel_cols = [c for c in df_aug.columns
                              if c.startswith('Transmitted_Sel_')
                              and c.endswith(f'_to_{level_name}{path}')
                              and not c.endswith('_LDR') and '_PNC' not in c]
            all_sel = sorted(set(own_sel_cols + trans_sel_cols))
            if not all_sel:
                continue
            cum_name = f'Cumulative_Sel_{path[1:]}_{level_name}'
            if cum_name not in df_aug.columns:
                df_aug[cum_name] = df_aug[all_sel].sum(axis=1)
                for j in range(i, len(full_hierarchy)):
                    anc_name = full_hierarchy[j]
                    d_col = f'AvgG{path}_{anc_name}'
                    if d_col in df_aug.columns:
                        pnc_sfx = '_PNC' if anc_name == level_name \
                            else f'_PNC_{anc_name}'
                        pnc_col = f'{cum_name}{pnc_sfx}'
                        if pnc_col not in df_aug.columns:
                            denom = df_aug[d_col].replace(0, np.nan)
                            df_aug[pnc_col] = (df_aug[cum_name] / denom) * 100

        # Drop merge artifacts
        df_aug.drop(columns=[c for c in df_aug.columns
                              if c.endswith('_y') or c == 'UnitName_y'],
                    inplace=True, errors='ignore')
        all_out[level_name] = df_aug

    return all_out


# ── Export ────────────────────────────────────────────────────────────────────

def _export_results(all_results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for base_name, results_dict in all_results.items():
        for level_name, df_lvl in results_dict.items():
            if df_lvl is None or df_lvl.empty:
                print(f"    Skipping export for {base_name}_{level_name}: no data.")
                continue

            # Build ordered column list
            potential = ['UnitName'] + list(CHILD_TO_PARENT_ID_COL_MAP.values())
            potential += [
                f'PopInitial_{level_name}', f'PopFinal_{level_name}',
                f'LogAvgIncInitial_{level_name}', f'LogAvgIncFinal_{level_name}',
                f'PopG_{level_name}', f'RelPopG_{level_name}',
                f'AvgG_emp_{level_name}', f'RelAvgG_emp_{level_name}',
                f'AvgG_pop_{level_name}', f'RelAvgG_pop_{level_name}',
                f'LHS_AvgG_pop_{level_name}',
                f'AvgG_inc_{level_name}', f'RelAvgG_inc_{level_name}',
                f'PopInitialR_{level_name}', f'PopFinalR_{level_name}',
            ]
            for c in df_lvl.columns:
                if c.startswith(('Sel_', 'Transmitted_')) and \
                        ('_pop' in c or '_inc' in c):
                    potential.append(c)
            for c in df_lvl.columns:
                if c.startswith('Cumulative_Sel_'):
                    potential.append(c)

            existing = list(dict.fromkeys(c for c in potential if c in df_lvl.columns))
            raw_cols = [c for c in existing if not c.endswith('_LDR') and '_PNC' not in c]
            ldr_cols = sorted(c for c in existing if c.endswith('_LDR'))
            pnc_cols = sorted(c for c in existing if '_PNC' in c)
            final_cols = raw_cols + ldr_cols + pnc_cols

            if not final_cols:
                continue

            csv_path = os.path.join(OUTPUT_DIR, f"{base_name}_{level_name}_exported_terms.csv")
            df_lvl[final_cols].to_csv(csv_path, index=False)
            print(f"    Exported {base_name}_{level_name} → {csv_path}")

    pkl_path = os.path.join(OUTPUT_DIR, "all_decomposition_results.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"  Saved full results → {pkl_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== runPrice.py ===")

    # Phase 1: load and prepare census data
    incomes_data = prepare_census_data(INPUT_DATA_PATH)
    if incomes_data is None:
        return

    # Phase 2: build initial/final panels per level
    print("\n--- Building per-level panels ---")
    prepared_level_data = {}
    for level in HIERARCHY_LEVELS:
        df_raw = build_level_panel(incomes_data, level, INITIAL_YEAR, FINAL_YEAR)
        prepared_level_data[level] = (
            prepare_level_data(df_raw, level, TARGET_N, CHILD_TO_PARENT_ID_COL_MAP)
            if not df_raw.empty else pd.DataFrame()
        )

    # Phase 3: hierarchical decomposition
    print("\n--- Running hierarchical Price decomposition ---")
    all_results = {}
    for base in BASE_ANALYSIS_LEVELS:
        print(f"\n{'='*10} Base level: {base.upper()} {'='*10}")
        results = calculate_multilevel_price_decomposition(
            prepared_level_data, base, HIERARCHY_LEVELS,
            CHILD_TO_PARENT_ID_COL_MAP, TARGET_N, incomes_data,
        )
        all_results[base] = calculate_pnc_and_ldr_metrics(
            results, HIERARCHY_LEVELS, CHILD_TO_PARENT_ID_COL_MAP, base,
        )

    # Export
    print("\n--- Exporting results ---")
    _export_results(all_results)
    print("\n=== runPrice.py complete ===")


if __name__ == '__main__':
    main()
