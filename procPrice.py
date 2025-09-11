# procPrice2.py

import pandas as pd
import pickle
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Import helpers
from procHelpers import (
    load_data,
    pre_calculate_first_years,
    assign_population_column,
    calculate_log_avg_income,
    calculate_avg_log_income,
    # Also import necessary constants if not handled within helpers
    # income_vars, aggs, years, comparison_data, xbins_values
    years, aggs, income_vars, comparison_data, xbins_values
)

# --- NEW HELPER FUNCTION for Tract Aggregation ---
def _aggregate_split_tracts(df_tracts_yn, df_tracts_y, income_vars, population_col):
    """Aggregates tracts in year Y that split from a single tract in year Y-N.

    Args:
        df_tracts_yn: DataFrame of tracts for year Y-N (index = 11-digit tract ID).
        df_tracts_y: DataFrame of tracts for year Y (index = 11-digit tract ID).
        income_vars: List of income variable column names.
        population_col: Name of the population column.

    Returns:
        pd.DataFrame: Modified df_tracts_y with split tracts aggregated.
    """

    # Work on copies to avoid modifying original dict data directly if not intended
    df_y_modified = df_tracts_y.copy()
    processed_y_indices = set() # Keep track of year Y indices already handled

    # Ensure index is string for slicing
    try:
        df_tracts_yn.index = df_tracts_yn.index.astype(str)
        df_y_modified.index = df_y_modified.index.astype(str)
    except Exception as e:
        print(f"      Warning: Could not convert tract index to string for aggregation: {e}")
        return df_tracts_y # Return original if conversion fails

    # Add 9-digit prefix column
    df_tracts_yn['prefix9'] = df_tracts_yn.index.str[:9]
    df_y_modified['prefix9'] = df_y_modified.index.str[:9]

    prefixes_yn = df_tracts_yn['prefix9'].unique()

    for prefix in prefixes_yn:
        # Find tracts matching this prefix in both years
        tracts_yn_matching = df_tracts_yn[df_tracts_yn['prefix9'] == prefix]
        tracts_y_matching = df_y_modified[df_y_modified['prefix9'] == prefix]

        # --- Handle 1-to-Many Split --- 
        if len(tracts_yn_matching) == 1 and len(tracts_y_matching) > 1:
            # Check if these year Y tracts have already been processed (e.g., part of another aggregation)
            if any(idx in processed_y_indices for idx in tracts_y_matching.index):
                continue

            original_index_yn = tracts_yn_matching.index[0]
            indices_to_aggregate_y = tracts_y_matching.index
            
            # Aggregate data from year Y tracts
            aggregated_data = {} 
            # Sum population and income vars
            numeric_cols_to_sum = [population_col] + income_vars
            # Filter to columns that actually exist
            numeric_cols_to_sum = [col for col in numeric_cols_to_sum if col in tracts_y_matching.columns]
            aggregated_data.update(tracts_y_matching[numeric_cols_to_sum].sum(axis=0))

            # Keep other relevant info (e.g., from the first tract in the split set)
            # Adjust this logic based on which columns are essential and how to handle them
            first_tract_y_row = tracts_y_matching.iloc[0]
            other_cols = [col for col in df_y_modified.columns if col not in numeric_cols_to_sum and col != 'prefix9']
            for col in other_cols:
                if col in first_tract_y_row:
                    aggregated_data[col] = first_tract_y_row[col]
                    
            # Create the new aggregated row Series
            new_row = pd.Series(aggregated_data, name=original_index_yn) # Set name to original Y-N index

            # Remove old rows and add the new one
            df_y_modified = df_y_modified.drop(indices_to_aggregate_y)
            # Use pd.concat instead of direct assignment for potentially better type handling
            new_row_df = pd.DataFrame([new_row]) # Convert Series to DataFrame for concat
            df_y_modified = pd.concat([df_y_modified, new_row_df])
            
            # Mark these indices as processed
            processed_y_indices.update(indices_to_aggregate_y)
            
            # print(f"      Aggregated tracts {list(indices_to_aggregate_y)} in year Y to match {original_index_yn} from year Y-N.")

        # --- Optional: Handle Many-to-1 Merge (less likely based on description) --- 
        # if len(tracts_yn_matching) > 1 and len(tracts_y_matching) == 1:
            # Similar logic, but aggregate tracts_yn_matching and replace in df_tracts_yn
            # This requires modifying df_tracts_yn, which might complicate things.
            # Focusing on splits for now.
            # pass
            
    # Clean up prefix column before returning
    df_y_modified = df_y_modified.drop(columns=['prefix9'], errors='ignore')

    return df_y_modified
# --- END HELPER FUNCTION ---

# Map child to member level for recursive calculation
# If parent_level_for_decomposition = aggs[i], then member_level_of_decomposition = aggs[i+1]
# Corrected member_agg_map:
member_agg_map = {aggs[i]: aggs[i+1] for i in range(len(aggs) - 1)}
print(f"Corrected Member Aggregation Map used: {member_agg_map}")


def _find_common_members_for_parent(parent_unit_index, parent_agg_level, member_agg_level, member_df_yn, member_df_y):
    """
    Filters member DataFrames to find common members for a given parent unit.

    Args:
        parent_unit_index: The index of the parent unit.
        parent_agg_level: The aggregation level of the parent unit (e.g., 'ct').
        member_agg_level: The aggregation level of the member units (e.g., 'cm').
        member_df_yn: DataFrame of member units for year Y-N.
        member_df_y: DataFrame of member units for year Y.

    Returns:
        pd.Index: An index of common member units. Returns empty if error or no match.
    """
    common_member_indices = pd.Index([]) # Default to empty

    # Determine the column in the MEMBER dataframe that links to the PARENT
    link_col = None
    if parent_agg_level == 'st': link_col = 'state'    # County (member) df should have a 'state' col
    elif parent_agg_level == 'ct': link_col = 'county'   # Community (member) df should have a 'county' col
    elif parent_agg_level == 'cm': link_col = 'community' # Tract (member) df should have a 'community' col
    elif parent_agg_level == 'tr': link_col = 'temp_tract_id' # BG (member) df should have a 'tract' col
    # Add more specific handling if your BG data uses 'tract' instead of 'temp_tract_id'

    if link_col is None:
        # print(f"Warning: No link_col defined for parent {parent_agg_level} -> member {member_agg_level}.")
        return common_member_indices
    
    if link_col not in member_df_yn.columns or link_col not in member_df_y.columns:
        # print(f"Warning: Link column '{link_col}' not found in member DFs ({member_agg_level}) for parent {parent_agg_level}.")
        return common_member_indices

    # Extract the identifier of the parent unit to filter members
    parent_filter_value = None
    try:
        if isinstance(parent_unit_index, tuple):
            if len(parent_unit_index) > 0: parent_filter_value = parent_unit_index[0]
            else: raise ValueError("Parent unit index is an empty tuple")
        else:
            parent_filter_value = parent_unit_index
        
        if pd.isna(parent_filter_value):
            return common_member_indices

        # Filter members belonging to this parent
        members_yn_for_parent = member_df_yn[member_df_yn[link_col] == parent_filter_value]
        members_y_for_parent = member_df_y[member_df_y[link_col] == parent_filter_value]

        common_member_indices = members_yn_for_parent.index.intersection(members_y_for_parent.index)

    except Exception as e:
        # print(f"Error during member filtering for parent {parent_unit_index} ({parent_agg_level}) -> members ({member_agg_level}): {e}")
        # common_member_indices remains empty
        pass

    return common_member_indices, len(members_yn_for_parent.index.to_list())-len(members_y_for_parent.index.to_list()),len(common_member_indices)

# Placeholder for the recursive Price component calculation function
def computePriceComponentsRecursive(df_yn, df_y, common_member_indices, income_variables, comparison_data, n_years):
    """Calculates N-year Price Equation terms based on common members.

    Args:
        df_yn: DataFrame containing data ONLY for common members in year Y-N.
        df_y: DataFrame containing data ONLY for common members in year Y.
        common_member_indices: List/Index of the common members.
        income_variables: List of income column names.
        comparison_data: Log(income) values for bins.
        n_years: The number of years in the period (N).

    Returns:
        tuple: (delta_z_n, cov_term_n, exp_term_n, num_members_n)
               Returns (NaN, NaN, NaN, initial_num_members) if calculation fails or not enough valid members.
    """
    initial_num_members = len(common_member_indices)
    # Initialize all terms to NaN or 0 for num_members
    delta_z_n = np.nan
    cov_pop_term_n = np.nan
    exp_pop_term_n = np.nan # New expectation term for population decomposition
    income_weighted_growth_n = np.nan

    if initial_num_members == 0:
        return delta_z_n, cov_pop_term_n, exp_pop_term_n, income_weighted_growth_n, 0

    try:
        # Ensure inputs are aligned to common_member_indices (should be already)
        df_yn = df_yn.loc[common_member_indices]
        df_y = df_y.loc[common_member_indices]

        # 1. Extract Populations (N_i, N'x_i)
        pop_yn = df_yn['population'].astype(np.longdouble)
        pop_y = df_y['population'].astype(np.longdouble)

        # 2. Calculate Member Avg Log Incomes (z_i, z'_i) - Vectorized
        hh_counts_yn = df_yn[income_variables].sum(axis=1).astype(np.longdouble)
        hh_counts_y = df_y[income_variables].sum(axis=1).astype(np.longdouble)

        # Ensure comparison_data aligns with income_variables columns
        comp_data_aligned = comparison_data.loc[income_variables]

        weighted_sum_yn = (df_yn[income_variables].multiply(comp_data_aligned, axis=1)).sum(axis=1).astype(np.longdouble)
        weighted_sum_y = (df_y[income_variables].multiply(comp_data_aligned, axis=1)).sum(axis=1).astype(np.longdouble)

        # Use np.divide to handle division by zero -> NaN
        z_i = pd.Series(np.nan, index=common_member_indices, dtype=np.longdouble)
        z_prime_i = pd.Series(np.nan, index=common_member_indices, dtype=np.longdouble)

        mask_hh_yn_gt0 = hh_counts_yn > 0
        mask_hh_y_gt0 = hh_counts_y > 0

        z_i[mask_hh_yn_gt0] = weighted_sum_yn[mask_hh_yn_gt0] / hh_counts_yn[mask_hh_yn_gt0]
        z_prime_i[mask_hh_y_gt0] = weighted_sum_y[mask_hh_y_gt0] / hh_counts_y[mask_hh_y_gt0]

        # 3. Filter Mask: require N_i > 0 and valid z_i, z'_i
        valid_mask = (pop_yn > 0) & pop_yn.notna() & pop_y.notna() & z_i.notna() & z_prime_i.notna()

        if not valid_mask.any():
            # print("      No valid members after filtering for Price components.")
            return delta_z_n, cov_pop_term_n, exp_pop_term_n, income_weighted_growth_n, initial_num_members

        # 4. Apply mask
        N_i = pop_yn[valid_mask]
        N_prime_i = pop_y[valid_mask]
        z_i_f = z_i[valid_mask]
        z_prime_i_f = z_prime_i[valid_mask]
        # num_valid_members = len(N_i)

        # --- START NEW CALCULATION for Income-Weighted Growth ---
        if n_years > 0:
            # y_i_0 (actual average income at start)
            y_i_0 = np.exp(z_i_f)
            # gamma_i (direct log-income growth rate)
            gamma_i = (z_prime_i_f - z_i_f) / n_years
            # Numerator of income-weighted growth
            numerator = (N_i * gamma_i * y_i_0).sum()
            # Denominator of income-weighted growth
            denominator = (N_i * y_i_0).sum()
            if denominator != 0 and pd.notna(denominator):
                income_weighted_growth_n = numerator / denominator
        # --- END NEW CALCULATION ---

        # 5. Calculate Totals & Weights
        N_total = N_i.sum()
        if N_total <= 0: # Should not happen due to filter N_i > 0, but safe check
             # print("      N_total is zero or less after filtering.")
             return delta_z_n, cov_pop_term_n, exp_pop_term_n, income_weighted_growth_n, initial_num_members

        N_prime_total = N_prime_i.sum()

        p_i = N_i / N_total
        w_i = N_prime_i / N_i # Safe due to N_i > 0 filter
        bar_w = N_prime_total / N_total

        # 6. Calculate Aggregate Averages
        bar_z = (p_i * z_i_f).sum()
        bar_z_prime = np.nan
        if N_prime_total > 0:
            p_prime_i = N_prime_i / N_prime_total
            bar_z_prime = (p_prime_i * z_prime_i_f).sum()
        # else: bar_z_prime remains NaN

        # 7. Calculate Decomposition Terms
        # delta_z_n was initialized to np.nan
        if pd.notna(bar_z_prime) and pd.notna(bar_z):
             delta_z_n = bar_z_prime - bar_z

        # cov_pop_term_n (formerly term1_covariance)
        # All initialized to np.nan

        if bar_w != 0 and pd.notna(bar_w):
            # --- Population-related covariance term ---
            cov_w_z = (p_i * (w_i - bar_w) * (z_i_f - bar_z)).sum()
            cov_pop_term_n = cov_w_z / bar_w
            # -------------------------------------------
            
        # Calculate separate expectation terms for each decomposition
        if pd.notna(delta_z_n) and pd.notna(cov_pop_term_n):
            exp_pop_term_n = delta_z_n - cov_pop_term_n
        # else exp_pop_term_n remains NaN
        
        # Final check for NaN results (optional, but good practice)
        # All terms are initialized to NaN and only overwritten if calculation succeeds.

        return delta_z_n, cov_pop_term_n, exp_pop_term_n, income_weighted_growth_n, initial_num_members

    except Exception as e:
        print(f"      Error in computePriceComponentsRecursive: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        # Return initial NaN values on error
        return delta_z_n, cov_pop_term_n, exp_pop_term_n, income_weighted_growth_n, initial_num_members


# --- NEW: Recursive Calculation of E[ln(y)] ---
def _recursively_calculate_avg_log_income(incomes_data, population_col):
    """
    Recursively calculates the AvgLogInc (E[ln(y)]) for all levels.
    Starts from the base 'bg' level and aggregates upwards, ensuring that
    each parent's E[ln(y)] is the population-weighted average of its children's E[ln(y)].
    """
    print("\n--- Recursively recalculating AvgLogInc (E[ln(y)]) for all levels ---")
    
    # This map defines which column in the child dataframe identifies the parent.
    # The parent dataframe is assumed to be indexed by this identifier.
    CHILD_TO_PARENT_LINK_MAP = {
        'bg': {'parent_level': 'tr', 'link_on': 'temp_tract_id'},
        'tr': {'parent_level': 'cm', 'link_on': 'community'},
        'cm': {'parent_level': 'ct', 'link_on': ['state', 'county']},
        'ct': {'parent_level': 'st', 'link_on': 'state'}
    }

    sorted_years = sorted(incomes_data.keys())

    for year in sorted_years:
        print(f"  Processing year: {year}")
        for child_level, parent_info in CHILD_TO_PARENT_LINK_MAP.items():
            parent_level = parent_info['parent_level']
            link_on = parent_info['link_on']
            
            if child_level not in incomes_data[year] or parent_level not in incomes_data[year]:
                print(f"    Skipping {child_level} -> {parent_level}: level not in data for year {year}.")
                continue

            df_child = incomes_data[year][child_level]
            df_parent = incomes_data[year][parent_level]

            if df_child.empty or 'AvgLogInc' not in df_child.columns or population_col not in df_child.columns:
                print(f"    Skipping {child_level} -> {parent_level}: child data is empty or missing required columns ('AvgLogInc', '{population_col}').")
                continue
            
            # Create the weighted value: population * E[ln(y)]_child
            df_child['weighted_avg_log_inc'] = df_child[population_col] * df_child['AvgLogInc']
            
            # Group by the parent identifier and sum the populations and weighted values
            grouped = df_child.groupby(link_on).agg(
                total_child_pop=('population', 'sum'),
                total_weighted_avg_log_inc=('weighted_avg_log_inc', 'sum')
            )
            
            # Calculate the corrected E[ln(y)] for the parent
            grouped['corrected_AvgLogInc'] = grouped['total_weighted_avg_log_inc'] / grouped['total_child_pop']
            
            # Map the corrected values back to the parent DataFrame
            # The parent's index should match the `link_on` grouping key from the child
            df_parent['AvgLogInc'] = df_parent.index.map(grouped['corrected_AvgLogInc'])
            
            # Replace the original parent dataframe with the corrected one
            incomes_data[year][parent_level] = df_parent

            print(f"    Successfully recalculated 'AvgLogInc' for level '{parent_level}' from '{child_level}'.")

    return incomes_data
# --- END NEW ---


# --- Refactored Processing Function for a Single Unit/Year ---
def process_unit_year_price(unit_index, unit_row_y, year_y, agg, unit_first_year_map, incomes_data, member_agg_map, analysis_period_years):
    """Calculates all direct and recursive metrics for a single unit for a given end year Y."""

    # --- Basic Unit Info ---
    if isinstance(unit_index, tuple):
        unit_key = (agg,) + unit_index
    else:
        unit_key = (agg, unit_index)

    first_year = unit_first_year_map.get(unit_key, np.inf)
    if first_year == np.inf:
        return None

    pop_y = unit_row_y.get('population', np.nan)
    # --- MODIFIED: Use pre-calculated values ---
    log_avg_inc_y = unit_row_y.get('LogAvgIncome', np.nan)
    avg_log_inc_y = unit_row_y.get('AvgLogInc', np.nan)
    # ----------------------------------------

    # --- Get 'real' population for year Y ---
    pop_r_y = unit_row_y.get('population_r', np.nan)
    # ----------------------------------------

    # --- Prepare Parent Identifiers ---
    parent_state = unit_row_y.get('state', np.nan)
    parent_county = unit_row_y.get('county', np.nan)
    parent_community = unit_row_y.get('community', np.nan)

    # Max lookback based on the unit's own life
    unit_max_lookback_n = year_y - first_year 

    if unit_max_lookback_n <= 0:
        # No valid N-year periods for this unit in this year
        return None

    # --- Collect result rows for each valid N-year period ---
    result_rows = []

    # Iterate through potential start years (year_yn_candidate) from the defined analysis_period_years
    for year_yn_candidate in analysis_period_years:
        if year_yn_candidate >= year_y:
            continue
        n = year_y - year_yn_candidate
        if n <= 0:
            continue
        if n > unit_max_lookback_n:
            continue

        # Now 'n' is a valid lookback period for this unit
        # --- Prepare Result Row for this N-year period ---
        result_row = {
            'original_index': unit_index,
            'Agg': agg,
            'Year': year_y,
            'FirstYear': year_yn_candidate,  # PATCH: Set to start year of N-year period
            'Population': pop_y,
            'LogAvgIncome': log_avg_inc_y,
            'AvgLogInc': avg_log_inc_y,
            'ParentState': parent_state,
            'ParentCounty': parent_county,
            'ParentCommunity': parent_community
        }
        
        # --- Add 'real' population for year Y to result_row ---
        result_row['PopulationR'] = pop_r_y
        # ----------------------------------------------------
        
        # --- ADD ParentTract ---
        parent_tract_id_val = np.nan
        if agg == 'bg':
            if isinstance(unit_index, str) and len(unit_index) >= 11: # Assuming unit_index is FIPS for BG
                parent_tract_id_val = unit_index[:11]
            # else: # unit_index is not a string or too short, parent_tract_id_val remains np.nan
        result_row['ParentTract'] = parent_tract_id_val
        # --- END ADD ---

        # === 1. Direct Metrics Calculation (reused from procCov2) ===
        cov_dist_shift_n = np.nan # Initialize, though we might remove its direct use
        pop_yn = np.nan # Initialize pop_yn
        log_avg_inc_yn = np.nan # Initialize log_avg_inc_yn
        # avg_log_inc_yn is already initialized/calculated
        
        # --- Initialize 'real' population for year Y-N ---
        pop_r_yn = np.nan
        # ------------------------------------------------

        try:
            if year_yn_candidate not in incomes_data or agg not in incomes_data[year_yn_candidate]:
                # Store NaNs for initial values if data for year_yn_candidate is missing
                result_row[f'PopInitialN'] = np.nan
                result_row[f'LogAvgIncInitialN'] = np.nan
                continue # Skip to next N-year period if no initial year data
            
            unit_row_yn = incomes_data[year_yn_candidate][agg].loc[unit_index]
            pop_yn = unit_row_yn.get('population', np.nan)
            # --- MODIFIED: Use pre-calculated values ---
            log_avg_inc_yn = unit_row_yn.get('LogAvgIncome', np.nan)
            avg_log_inc_yn = unit_row_yn.get('AvgLogInc', np.nan)
            # ----------------------------------------
            
            # --- Get 'real' population for year Y-N ---
            pop_r_yn = unit_row_yn.get('population_r', np.nan)
            # -------------------------------------------

            result_row[f'PopInitialN'] = pop_yn # Store Pop at Year Y-N
            result_row[f'LogAvgIncInitialN'] = log_avg_inc_yn # Store LogAvgInc at Year Y-N
            result_row[f'AvgLogIncInitialN'] = avg_log_inc_yn
            
            # --- Store 'real' Pop at Year Y-N ---
            result_row[f'PopInitialNR'] = pop_r_yn
            # -------------------------------------
            
            # AvgLogInc for Y-N is used for AvgLogIncGrowth but not stored separately with 'N' suffix for now unless requested.

            if pd.notna(pop_y) and pd.notna(pop_yn) and pd.notna(log_avg_inc_y) and pd.notna(log_avg_inc_yn):
                if pop_yn > 0: pop_frac_change_n = (pop_y - pop_yn) / pop_yn
                elif pop_y > 0: pop_frac_change_n = np.inf
                else: pop_frac_change_n = np.nan
                result_row[f'PopFracChange_{n}'] = pop_frac_change_n
                
                # --- Calculate and store 'real' PopFracChangeR_{n} ---
                if pd.notna(pop_r_y) and pd.notna(pop_r_yn):
                    if pop_r_yn > 0: pop_frac_change_r_n = (pop_r_y - pop_r_yn) / pop_r_yn
                    elif pop_r_y > 0: pop_frac_change_r_n = np.inf
                    else: pop_frac_change_r_n = np.nan
                    result_row[f'PopFracChangeR_{n}'] = pop_frac_change_r_n
                else:
                    result_row[f'PopFracChangeR_{n}'] = np.nan
                # ----------------------------------------------------
                
                if n > 0: log_avg_income_growth_n = (log_avg_inc_y - log_avg_inc_yn) / n
                else: log_avg_income_growth_n = np.nan
                result_row[f'LogAvgIncomeGrowth_{n}'] = log_avg_income_growth_n
                if pd.notna(avg_log_inc_y) and pd.notna(avg_log_inc_yn) and n > 0:
                    avg_log_income_growth_n = (avg_log_inc_y - avg_log_inc_yn) / n
                else:
                    avg_log_income_growth_n = np.nan
                result_row[f'AvgLogIncGrowth_{n}'] = avg_log_income_growth_n
                # cov_dist_shift_n = calculate_distribution_covariance(
                #     unit_row_yn, unit_row_y, income_vars, comparison_data
                # )
                # result_row[f'CovDistShift_{n}'] = cov_dist_shift_n # REMOVED
                # cov_frac_pop_chg_n = calculate_frac_pop_change_covariance(
                #     unit_row_yn, unit_row_y, comparison_data
                # )
                # result_row[f'CovFracPopChg_{n}'] = cov_frac_pop_chg_n # REMOVED
        except KeyError:
            log_avg_inc_yn = np.nan
            pass
        except Exception as e:
            log_avg_inc_yn = np.nan
            print(f"Error calculating direct metrics (N={n}) for unit {unit_key}, year {year_y}: {e}")
            pass

        # === 2. Recursive Price Components Calculation ===
        agg_parent = agg
        agg_member = member_agg_map.get(agg_parent)
        perform_tract_distributional_decomposition = False
        if agg_parent == 'tr':
            if not (year_yn_candidate in incomes_data and 'bg' in incomes_data[year_yn_candidate] and \
                    year_y in incomes_data and 'bg' in incomes_data[year_y] and \
                    not incomes_data[year_yn_candidate]['bg'].empty and not incomes_data[year_y]['bg'].empty):
                perform_tract_distributional_decomposition = True
        if agg_member is None or perform_tract_distributional_decomposition:
            delta_z_n_dist = np.nan
            cov_term_n_dist = np.nan
            exp_term_n_dist = np.nan
            num_members_n_dist = 0 # Changed from len(income_vars)
            if pd.notna(log_avg_inc_y) and pd.notna(log_avg_inc_yn):
                delta_z_n_dist = log_avg_inc_y - log_avg_inc_yn
                # Original logic using cov_dist_shift_n removed
                # Now, CovTerm and ExpPopTerm will be NaN for this path
                # as Price recursive decomposition is not performed.
            # num_members_n_dist = len(income_vars) # Original, changed to 0
            result_row[f'DeltaZbar_{n}'] = delta_z_n_dist
            result_row[f'CovTerm_{n}'] = np.nan # Was cov_term_n_dist (i.e. cov_dist_shift_n)
            result_row[f'ExpPopTerm_{n}'] = np.nan # Was exp_term_n_dist
            result_row[f'NumMembers_{n}'] = num_members_n_dist # Set to 0
            if pd.notna(delta_z_n_dist) and pd.notna(result_row.get(f'CovTerm_{n}', np.nan)) and pd.notna(result_row.get(f'ExpPopTerm_{n}', np.nan)):
                result_row[f'SanityCheckPop_{n}'] = delta_z_n_dist - (result_row[f'CovTerm_{n}'] + result_row[f'ExpPopTerm_{n}'])
            else:
                result_row[f'SanityCheckPop_{n}'] = np.nan
        else:
            try:
                 if not (year_yn_candidate in incomes_data and agg_member in incomes_data[year_yn_candidate] and \
                         year_y in incomes_data and agg_member in incomes_data[year_y]):
                      result_row[f'DeltaZbar_{n}'] = np.nan
                      result_row[f'CovTerm_{n}'] = np.nan
                      result_row[f'ExpPopTerm_{n}'] = np.nan
                      result_row[f'NumMembers_{n}'] = 0
                      continue 
                 member_df_yn = incomes_data[year_yn_candidate][agg_member]
                 member_df_y = incomes_data[year_y][agg_member]
                 if agg_member == 'tr':
                     pop_col_name = 'population'
                     if 'population' not in member_df_y.columns or 'population' not in member_df_yn.columns:
                         print(f"      Warning: Population column '{pop_col_name}' not found in tract DFs for {year_yn_candidate}/{year_y}. Skipping tract aggregation.")
                     else:
                         member_df_y = _aggregate_split_tracts(member_df_yn, member_df_y, income_vars, pop_col_name)
                 common_member_indices, num_members_diff, num_members_total = _find_common_members_for_parent(
                     parent_unit_index=unit_index,
                     parent_agg_level=agg_parent,
                     member_agg_level=agg_member,
                     member_df_yn=member_df_yn,
                     member_df_y=member_df_y
                 )
                 if len(common_member_indices) == 1 and num_members_diff == 0:
                     delta_z_n_direct = np.nan
                     cov_term_n_direct = np.nan
                     exp_term_n_direct = np.nan
                     num_members_n_direct = 0
                     if pd.notna(log_avg_inc_y) and pd.notna(log_avg_inc_yn):
                         delta_z_n_direct = log_avg_inc_y - log_avg_inc_yn
                         if pd.notna(cov_dist_shift_n):
                             cov_term_n_direct = cov_dist_shift_n
                             exp_term_n_direct = delta_z_n_direct - cov_term_n_direct
                     num_members_n_direct = len(income_vars)
                     result_row[f'DeltaZbar_{n}'] = delta_z_n_direct
                     result_row[f'CovTerm_{n}'] = cov_term_n_direct
                     result_row[f'ExpPopTerm_{n}'] = exp_term_n_direct
                     result_row[f'NumMembers_{n}'] = num_members_n_direct
                     if pd.notna(delta_z_n_direct) and pd.notna(cov_term_n_direct) and pd.notna(exp_term_n_direct):
                         result_row[f'SanityCheckPop_{n}'] = delta_z_n_direct - (cov_term_n_direct + exp_term_n_direct)
                     else:
                         result_row[f'SanityCheckPop_{n}'] = np.nan
                 else:
                     min_members_needed = 1
                     if len(common_member_indices) >= min_members_needed:
                         try:
                             df_common_yn = member_df_yn.loc[common_member_indices].copy()
                             df_common_y = member_df_y.loc[common_member_indices].copy()
                             required_cols_rec = set(income_vars + ['population'])
                             if not required_cols_rec.issubset(df_common_yn.columns) or \
                                not required_cols_rec.issubset(df_common_y.columns):
                                    result_row[f'DeltaZbar_{n}'] = np.nan
                                    result_row[f'CovTerm_{n}'] = np.nan
                                    result_row[f'ExpPopTerm_{n}'] = np.nan
                                    result_row[f'NumMembers_{n}'] = 0
                                    continue
                         except Exception as slice_e:
                             result_row[f'DeltaZbar_{n}'] = np.nan
                             result_row[f'CovTerm_{n}'] = np.nan
                             result_row[f'ExpPopTerm_{n}'] = np.nan
                             result_row[f'NumMembers_{n}'] = 0
                             continue
                         delta_z_n, cov_pop_term_n, exp_pop_term_n, income_weighted_growth_n, num_members_n_actual = computePriceComponentsRecursive(
                             df_common_yn, df_common_y, common_member_indices, income_vars, comparison_data, n
                         )
                         result_row[f'DeltaZbar_{n}'] = delta_z_n
                         result_row[f'CovTerm_{n}'] = cov_pop_term_n
                         result_row[f'ExpPopTerm_{n}'] = exp_pop_term_n
                         result_row[f'IncWeightedGrowth_{n}'] = income_weighted_growth_n
                         result_row[f'NumMembers_{n}'] = len(common_member_indices)
                     else:
                         result_row[f'DeltaZbar_{n}'] = np.nan
                         result_row[f'CovTerm_{n}'] = np.nan
                         result_row[f'ExpPopTerm_{n}'] = np.nan
                         result_row[f'NumMembers_{n}'] = len(common_member_indices)
                     dz_check = result_row.get(f'DeltaZbar_{n}', np.nan)
                     ct_pop_check = result_row.get(f'CovTerm_{n}', np.nan)
                     et_pop_check = result_row.get(f'ExpPopTerm_{n}', np.nan)
                     if pd.notna(dz_check) and pd.notna(ct_pop_check) and pd.notna(et_pop_check):
                         result_row[f'SanityCheckPop_{n}'] = dz_check - (ct_pop_check + et_pop_check)
                     else:
                         result_row[f'SanityCheckPop_{n}'] = np.nan
            except KeyError as ke:
                 result_row[f'DeltaZbar_{n}'] = np.nan
                 result_row[f'CovTerm_{n}'] = np.nan
                 result_row[f'ExpPopTerm_{n}'] = np.nan
                 result_row[f'NumMembers_{n}'] = 0
                 pass
            except Exception as e:
                 result_row[f'DeltaZbar_{n}'] = np.nan
                 result_row[f'CovTerm_{n}'] = np.nan
                 result_row[f'ExpPopTerm_{n}'] = np.nan
                 result_row[f'NumMembers_{n}'] = 0
                 pass
        # --- End Branch ---

        # Only add result_row if Year > FirstYear (i.e., n > 0)
        if year_y > year_yn_candidate:
            result_rows.append(result_row)

    # If any valid N-year rows were created, return them (as a list)
    if result_rows:
        return result_rows
    else:
        return None


def main():
    """Main execution function for procPrice2."""
    print("--- Running procPrice2.py ---")

    # --- Configuration ---
    input_data_path = "data/census_data1.pkl"
    output_data_path = "results/analysis_Price2.pkl"
    population_col_name = 'B19001_001E'

    global years # Make global years accessible
    defined_input_years = sorted(years) # These are the years defined in procHelpers.py e.g. [14, 23]
    print(f"Using DEFINED input years for analysis period: {defined_input_years}")

    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

    # --- Load Data ---
    incomes_data = load_data(input_data_path)
    if incomes_data is None: return

    # Filter the defined_input_years to those ACTUALLY available in incomes_data.
    # This becomes the list of years we can actually work with for pairing.
    analysis_years_in_data = sorted([y for y in defined_input_years if y in incomes_data])
    
    if not analysis_years_in_data:
        print(f"Error: None of the defined input years ({defined_input_years}) found in the loaded data ({incomes_data.keys()}).")
        return
    
    if len(analysis_years_in_data) < len(defined_input_years):
        print(f"Warning: Not all defined input years ({defined_input_years}) are present in loaded data.")
        print(f"Actual years available for processing from defined list: {analysis_years_in_data}")
    else:
        print(f"All defined input years are available in loaded data. Processing years: {analysis_years_in_data}")


    # --- Pre-calculations ---
    # unit_first_year_map should use all available_data_years to accurately find the true first year of a unit
    all_available_data_years = sorted(incomes_data.keys())
    unit_first_year_map = pre_calculate_first_years(incomes_data, all_available_data_years, aggs)
    assign_population_column(incomes_data, all_available_data_years, aggs, population_col_name)

    # --- Add step to standardize B01003_001ER to 'population_r' ---
    real_population_input_col = 'B01003_001E' # Assuming this name from saveCensusDataV2.py
    real_population_internal_col = 'population_r'
    print(f"Standardizing '{real_population_input_col}' to '{real_population_internal_col}'...")
    for year_val_key in incomes_data.keys(): # Iterate through years
        if isinstance(incomes_data[year_val_key], dict):
            for agg_key in incomes_data[year_val_key].keys(): # Iterate through agg levels
                agg_df = incomes_data[year_val_key][agg_key]
                if isinstance(agg_df, pd.DataFrame) and real_population_input_col in agg_df.columns:
                    # Use .rename() and assign back to ensure modification
                    incomes_data[year_val_key][agg_key] = agg_df.rename(columns={real_population_input_col: real_population_internal_col})
                # else:
                    # print(f"  Warning: Column '{real_population_input_col}' not found in DataFrame for year {year_val_key}, agg {agg_key} for renaming.")
    # --------------------------------------------------------------

    # --- NEW: Pre-calculate all income metrics from distributions ---
    print("\n--- Pre-calculating LogAvgIncome and AvgLogInc for all levels/years ---")
    for year in all_available_data_years:
        for agg_level in aggs:
            if year in incomes_data and agg_level in incomes_data[year]:
                df = incomes_data[year][agg_level]
                if not df.empty:
                    df['LogAvgIncome'] = df.apply(calculate_log_avg_income, axis=1)
                    df['AvgLogInc'] = df.apply(calculate_avg_log_income, axis=1)
                    incomes_data[year][agg_level] = df
    print("--- Pre-calculation complete ---")
    # --- END NEW ---

    # --- NEW: Recursively correct AvgLogInc ---
    incomes_data = _recursively_calculate_avg_log_income(incomes_data, 'population')
    # --- END NEW ---

    # --- DEBUG: Print head of each DataFrame in incomes_data ---
    print("\n--- DEBUG: Inspecting loaded incomes_data DataFrames ---")
    for year_debug, year_data_debug in incomes_data.items():
        if isinstance(year_data_debug, dict):
            print(f"  Year: {year_debug}")
            for agg_level_debug, df_debug in year_data_debug.items():
                if isinstance(df_debug, pd.DataFrame) and not df_debug.empty:
                    print(f"    Aggregation Level: {agg_level_debug}, Shape: {df_debug.shape}")
                    print(df_debug.head())
                    print("      --- Columns ---")
                    print(f"      {df_debug.columns.tolist()}")
                    print("      ---------------")

                elif isinstance(df_debug, pd.DataFrame) and df_debug.empty:
                    print(f"    Aggregation Level: {agg_level_debug} is an empty DataFrame.")
                else:
                    print(f"    Aggregation Level: {agg_level_debug} is not a DataFrame or is None.")
        else:
            print(f"  Year: {year_debug} data is not a dictionary.")
    print("--- END DEBUG: Inspecting loaded incomes_data DataFrames ---\n")
    # --- END DEBUG --- 

    # --- Main Processing Loop ---
    print("\nStarting main processing loop...")
    all_results = []

    # The outer loop iterates through each year in analysis_years_in_data as a potential 'end year' (year_y)
    for agg in aggs:
        print(f"\nProcessing Aggregation Level: {agg}...")
        for year_y_idx, year_y in enumerate(tqdm(analysis_years_in_data, desc=f"Years ({agg})")):
            # Skip if year_y is the very first year in the defined list, as it cannot be an end-year for any pair.
            if year_y == analysis_years_in_data[0] and len(analysis_years_in_data) > 1: # only skip if there's more than one year
                 # Or more simply, if no year_yn_candidate < year_y exists, process_unit_year_price won't do N-calcs.
                 # Let's allow all years to be year_y, the inner logic in process_unit_year_price will handle pair formation.
                 pass


            if year_y not in incomes_data or agg not in incomes_data[year_y]:
                # This check might be redundant if analysis_years_in_data is correctly sourced
                continue

            df_year_y = incomes_data[year_y][agg]

            for unit_index, unit_row_y in df_year_y.iterrows():
                result = process_unit_year_price(
                    unit_index, unit_row_y, year_y, agg,
                    unit_first_year_map, incomes_data, member_agg_map,
                    analysis_years_in_data # Pass the list of years to form pairs from
                )
                if result is not None:
                    all_results.extend(result)

    print("\nFinished main processing loop.")

    # --- Combine, Add Identifiers, Save ---
    if not all_results:
        print("\nWarning: No results were generated.")
        df_final = pd.DataFrame()
    else:
        print(f"\nGenerated {len(all_results)} result rows.")
        print("Combining results into DataFrame...")
        df_final = pd.DataFrame(all_results)

        # --- Add Parent Identifiers (Placeholder - Needs proper implementation) ---
        print("Populating Parent identifiers from input data...") # Updated message
        df_final['UnitName'] = df_final['original_index'].astype(str) # Basic placeholder
        # df_final['ParentState'] = np.nan # REMOVE - Now populated in process_unit_year_price
        # df_final['ParentCounty'] = np.nan # REMOVE
        # df_final['ParentCommunity'] = np.nan # REMOVE
        # -----------------------------------------------------------------------

        # Reorder columns (similar to procCov2)
        base_cols = ['UnitName', 'Agg', 'Year', 'FirstYear', 
                     'ParentState', 'ParentCounty', 'ParentCommunity', 'ParentTract', # ADDED ParentTract
                     'Population', 'PopulationR', # ADDED PopulationR
                     'LogAvgIncome', 'AvgLogInc',
                     'PopInitialN', 'PopInitialNR', # ADDED PopInitialNR
                     'LogAvgIncInitialN', 'AvgLogIncInitialN'] 
        metric_cols = [col for col in df_final.columns if col not in base_cols and col != 'original_index']
        final_cols = base_cols + sorted(metric_cols)
        final_cols = [col for col in final_cols if col in df_final.columns] # Ensure existence
        df_final = df_final[final_cols]

        # --- ADD INSPECTION --- 
        print("\n--- Inspecting df_final just before saving ---")
        df_final.info()
        print("\n--- df_final Head (First 20 rows) ---")
        print(df_final)
        print(df_final.columns.tolist())
        print("-------------------------------------------\n")
        # --- END INSPECTION --- 

        # --- Drop columns that are all NaN before saving ---
        df_final = df_final.dropna(axis=1, how='all')
        print(f"Shape of df_final after dropping all-NaN columns: {df_final.shape}")
        print(f"Columns remaining: {df_final.columns.tolist()}")
        # ---------------------------------------------------

        # --- Save ---
        print(f"Saving final DataFrame to {output_data_path}...")
        df_final.to_pickle(output_data_path)
        print("Save complete.")

    print("--- procPrice2.py finished ---")

if __name__ == "__main__":
    main()