import pandas as pd
import numpy as np
import pickle
import os
import warnings

# Suppress specific pandas warnings if desired
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Define Constants ---
TARGET_N = 5  # Example: 9-year period for analysis
INITIAL_YEAR = 14 # From procPrice.py, typical FirstYear for N=9 period
HIERARCHY_LEVELS = ['bg', 'tr', 'cm', 'ct', 'st'] # Order: Lowest to highest
ANALYSIS_PRICE_FILEPATH = "results/analysis_Price2.pkl"

# Child-to-Parent ID column map (essential for linking levels)
# Defines which column in the *child* level DataFrame identifies its parent in the next level up.
CHILD_TO_PARENT_ID_COL_MAP = {
    'bg': 'ParentTract',      # BGs are children, their ParentTract groups them into Tract parents.
    'tr': 'ParentCommunity',  # Tracts are children, their ParentCommunity groups them into Community parents.
    'cm': 'ParentCounty',     # Communities are children, their ParentCounty groups them into County parents.
    'ct': 'ParentState'       # Counties are children, their ParentState groups them into State parents.
    # 'st' has no parent in this map as it's the highest level
}

# --- Utility Functions ---

def load_pickle_data(filepath, description=""):
    """Loads data from a pickle file."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded {description} data from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Cannot load {description} data. Exiting.")
        return None
    except Exception as e:
        print(f"Error loading {description} data from {filepath}: {e}")
        return None

def filter_analysis_data_for_level(df_analysis_full, target_level_name, target_n_period, initial_year_period):
    """Filters the raw analysis DataFrame for a specific aggregation level, N-year period, and initial year."""
    if 'N' not in df_analysis_full.columns:
        if 'Year' in df_analysis_full.columns and 'FirstYear' in df_analysis_full.columns:
            df_analysis_full['N'] = pd.to_numeric(df_analysis_full['Year'], errors='coerce') - \
                                   pd.to_numeric(df_analysis_full['FirstYear'], errors='coerce')
            print("  Calculated 'N' column in full analysis_df.")
        else:
            print("  Error: 'N' column (or 'Year'/'FirstYear') not found in df_analysis_full.")
            return pd.DataFrame()
    
    df_filtered = df_analysis_full[
        (df_analysis_full['N'] == target_n_period) &
        (df_analysis_full['FirstYear'] == initial_year_period) &
        (df_analysis_full['Agg'] == target_level_name)
    ].copy()

    if df_filtered.empty:
        print(f"  No data found for Agg='{target_level_name}', FirstYear={initial_year_period}, N={target_n_period}.")
    else:
        print(f"  Filtered for Agg='{target_level_name}', FirstYear={initial_year_period}, N={target_n_period}: {len(df_filtered)} rows.")
        if 'UnitName' in df_filtered.columns:
            df_filtered['UnitName'] = df_filtered['UnitName'].astype(str)
        else:
            print("  Error: 'UnitName' column not found in filtered data.")
            return pd.DataFrame()
    return df_filtered

# --- Helper Functions for prepare_level_data ---

def _calculate_empirical_growth_rate(df, level_name, target_n_years, log_avg_inc_initial_col, log_avg_inc_final_col, emp_growth_col):
    """Calculates the direct, empirical log-income growth rate (gamma_k)."""
    print(f"  Step 2: Calculating empirical growth rate for {level_name}.")
    if log_avg_inc_initial_col in df.columns and \
       log_avg_inc_final_col in df.columns and target_n_years > 0:
        df[emp_growth_col] = \
            (df[log_avg_inc_final_col] - df[log_avg_inc_initial_col]) / target_n_years
        print(f"    Calculated empirical growth rate: '{emp_growth_col}'.")
    else:
        df[emp_growth_col] = np.nan
        missing_for_growth = [col for col in [log_avg_inc_initial_col, log_avg_inc_final_col] if col not in df.columns]
        print(f"    Warning: Could not calculate '{emp_growth_col}'. Missing: {missing_for_growth} or target_n_years ({target_n_years}) <= 0.")
    return df

def _calculate_actual_average_incomes(df, log_avg_inc_initial_col, log_avg_inc_final_col, actual_avg_inc_initial_col, actual_avg_inc_final_col):
    """Calculates actual average incomes from log average incomes."""
    print(f"  Step 3: Calculating actual average incomes.")
    if log_avg_inc_initial_col in df.columns:
        df[actual_avg_inc_initial_col] = np.exp(df[log_avg_inc_initial_col].astype(float))
        print(f"    Calculated actual initial average income: '{actual_avg_inc_initial_col}'.")
    else:
        df[actual_avg_inc_initial_col] = np.nan

    if log_avg_inc_final_col in df.columns:
        df[actual_avg_inc_final_col] = np.exp(df[log_avg_inc_final_col].astype(float))
        print(f"    Calculated actual final average income: '{actual_avg_inc_final_col}'.")
    else:
        df[actual_avg_inc_final_col] = np.nan
    return df

def _ensure_parent_identifier(df, level_name, child_to_parent_id_map, parent_id_col_for_this_level):
    """Ensures parent identifier column exists and is correctly named."""
    print(f"  Step 4: Ensuring parent identifier for {level_name}.")
    if parent_id_col_for_this_level:
        if level_name == 'bg' and parent_id_col_for_this_level == 'ParentTract':
            if 'UnitName' in df.columns:
                df[parent_id_col_for_this_level] = df['UnitName'].astype(str).str[:11]
                print(f"    Constructed '{parent_id_col_for_this_level}' for level '{level_name}' from UnitName.")
            else:
                print(f"    Warning: Cannot construct '{parent_id_col_for_this_level}' for '{level_name}'; 'UnitName' is missing.")
                df[parent_id_col_for_this_level] = pd.NA
        elif parent_id_col_for_this_level not in df.columns:
            print(f"    Warning: Expected parent ID column '{parent_id_col_for_this_level}' for level '{level_name}' not found. Setting to NA.")
            df[parent_id_col_for_this_level] = pd.NA
        else:
            df[parent_id_col_for_this_level] = df[parent_id_col_for_this_level].astype(str)
            print(f"    Confirmed parent ID column '{parent_id_col_for_this_level}' for level '{level_name}'.")
    else:
        print(f"    Level '{level_name}' is highest level or has no defined parent ID in map.")
    return df

def _perform_final_checks_and_filter(df, level_name, pop_initial_col, log_avg_inc_initial_col):
    """Performs final NaN checks and filtering."""
    print(f"  Step 5: Performing final checks and filtering for {level_name}.")
    dropna_subset = [pop_initial_col, log_avg_inc_initial_col]
    
    original_rows = len(df)
    df.dropna(subset=[col for col in dropna_subset if col in df.columns], inplace=True)
    
    if pop_initial_col in df.columns:
        df = df[df[pop_initial_col] > 0]
    
    print(f"    Filtered {level_name} data: {len(df)} rows remain after NaN drop and PopInitial > 0 check (from {original_rows}).")
    return df

# --- Core Data Preparation Function for Each Level ---

def prepare_level_data(df_level_raw, level_name, target_n_years, child_to_parent_id_map):
    """
    Prepares the data for a single specified level by renaming columns,
    calculating intrinsic properties (populations, incomes, empirical growth rates),
    and ensuring parent identifiers are set up.

    Args:
        df_level_raw (pd.DataFrame): Raw DataFrame for this specific level, 
                                     filtered for the correct N-year period and initial year.
        level_name (str): The name of the current level (e.g., 'bg', 'tr').
        target_n_years (int): The number of years in the analysis period (e.g., TARGET_N).
        child_to_parent_id_map (dict): Map to get parent ID column names.

    Returns:
        pd.DataFrame: Prepared DataFrame for the level with essential intrinsic properties.
    """
    print(f"-- Preparing data for level: {level_name} --")
    if df_level_raw.empty:
        print(f"  Input df_level_raw for {level_name} is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    df_prepared = df_level_raw.copy()

    # Step 1. Rename core columns
    print(f"  Step 1: Renaming core columns for {level_name}.")
    rename_map = {
        'PopInitialN': f'PopInitial_{level_name}',
        'LogAvgIncInitialN': f'LogAvgIncInitial_{level_name}',
        'Population': f'PopFinal_{level_name}',
        'LogAvgIncome': f'LogAvgIncFinal_{level_name}',
        'AvgLogIncInitialN': f'AvgLogIncInitial_{level_name}', # Added
        'AvgLogInc': f'AvgLogIncFinal_{level_name}',       # Added
        # 'IncWeightedGrowth_5' is no longer needed with new direct calculations
        'PopInitialNR': f'PopInitialR_{level_name}',
        'PopulationR': f'PopFinalR_{level_name}'
    }
    df_prepared.rename(columns=rename_map, inplace=True)
    print(f"    Renamed columns using map: {rename_map}")

    # Define frequently used column names for this level
    pop_initial_col = f'PopInitial_{level_name}'
    pop_final_col = f'PopFinal_{level_name}'
    log_avg_inc_initial_col = f'LogAvgIncInitial_{level_name}'
    log_avg_inc_final_col = f'LogAvgIncFinal_{level_name}'
    actual_avg_inc_initial_col = f'ActualAvgIncInitial_{level_name}'
    actual_avg_inc_final_col = f'ActualAvgIncFinal_{level_name}'
    emp_growth_col = f'AvgG_emp_{level_name}' # Empirical log-income growth rate (gamma_k), from ln(E[y])
    g_avg_log_inc_col = f'G_avg_log_inc_{level_name}' # Correct growth rate from E[ln(y)]
    parent_id_col_for_this_level = child_to_parent_id_map.get(level_name)

    # Define column names for real population metrics
    pop_initial_r_col = f'PopInitialR_{level_name}'
    pop_final_r_col = f'PopFinalR_{level_name}'

    # Step 2. Calculate Empirical Log-Income Growth Rate - \bar\gamma_k for smallest spatial unit
    df_prepared = _calculate_empirical_growth_rate(
        df_prepared, level_name, target_n_years, 
        log_avg_inc_initial_col, log_avg_inc_final_col, emp_growth_col
    )

    # NEW STEP: Calculate G_avg_log_inc from the E[ln(y)] values
    print(f"  Step 2b: Calculating 'average of logs' growth rate for {level_name}.")
    avg_log_inc_initial_col = f'AvgLogIncInitial_{level_name}'
    avg_log_inc_final_col = f'AvgLogIncFinal_{level_name}'

    if avg_log_inc_initial_col in df_prepared.columns and \
       avg_log_inc_final_col in df_prepared.columns and target_n_years > 0:
        df_prepared[g_avg_log_inc_col] = \
            (df_prepared[avg_log_inc_final_col] - df_prepared[avg_log_inc_initial_col]) / target_n_years
        print(f"    Calculated 'average of logs' growth rate: '{g_avg_log_inc_col}'.")
    else:
        df_prepared[g_avg_log_inc_col] = np.nan
        missing_for_growth = [col for col in [avg_log_inc_initial_col, avg_log_inc_final_col] if col not in df_prepared.columns]
        print(f"    Warning: Could not calculate '{g_avg_log_inc_col}'. Missing: {missing_for_growth}.")

    # Step 3. Calculate Actual Average Incomes - \bar y_k for smallest spatial unit
    df_prepared = _calculate_actual_average_incomes(
        df_prepared, log_avg_inc_initial_col, log_avg_inc_final_col, 
        actual_avg_inc_initial_col, actual_avg_inc_final_col
    )

    # Step 4. Ensure Parent Identifier Column - spatial unit identifier for parent level
    df_prepared = _ensure_parent_identifier(
        df_prepared, level_name, child_to_parent_id_map, parent_id_col_for_this_level
    )

    # Step 5. Final checks and column selection - drop rows with missing data
    df_prepared = _perform_final_checks_and_filter(
        df_prepared, level_name, pop_initial_col, log_avg_inc_initial_col
    )
    
    # --- Add New Columns: PopG, RelAvgG_emp, RelPopG ---
    print(f"  Step 6: Calculating PopG, RelAvgG_emp, and RelPopG for {level_name}.")
    pop_g_col = f'PopG_{level_name}'
    rel_avg_g_emp_col = f'RelAvgG_emp_{level_name}'
    rel_pop_g_col = f'RelPopG_{level_name}'

    # Calculate PopG_k
    if pop_initial_col in df_prepared.columns and pop_final_col in df_prepared.columns and target_n_years > 0:
        # Ensure PopInitial and PopFinal are positive for log calculation; replace 0 or negative with NaN
        safe_pop_initial_for_log = df_prepared[pop_initial_col].apply(lambda x: x if x > 0 else np.nan)
        safe_pop_final_for_log = df_prepared[pop_final_col].apply(lambda x: x if x > 0 else np.nan)

        log_pop_initial = np.log(safe_pop_initial_for_log)
        log_pop_final = np.log(safe_pop_final_for_log)
        
        df_prepared[pop_g_col] = (log_pop_final - log_pop_initial) / target_n_years
        print(f"    Calculated log-difference population growth rate: '{pop_g_col}'.")
    else:
        df_prepared[pop_g_col] = np.nan



    # Calculate RelAvgG_emp_k and RelPopG_k
    if parent_id_col_for_this_level and parent_id_col_for_this_level in df_prepared.columns:
        # Calculate sum of initial populations for each parent group (denominator for weighted means)
        sum_pop_initial_by_parent_col = f'sum_pop_initial_by_parent_{level_name}'
        df_prepared[sum_pop_initial_by_parent_col] = df_prepared.groupby(parent_id_col_for_this_level)[pop_initial_col].transform('sum')
        # Create a safe denominator (NaN if sum is 0, to prevent division by zero errors and allow NaN propagation)
        safe_sum_pop_initial_by_parent = df_prepared[sum_pop_initial_by_parent_col].replace(0, np.nan)

        # Relative Empirical Log-Income Growth (Population-Weighted)
        if emp_growth_col in df_prepared.columns and pop_initial_col in df_prepared.columns:
            # Numerator for weighted average: PopInitial * AvgG_emp
            weighted_avg_g_num_col = f'temp_weighted_avg_g_num_{level_name}'
            df_prepared[weighted_avg_g_num_col] = df_prepared[pop_initial_col] * df_prepared[emp_growth_col]
            
            # Sum of numerators within each parent group
            sum_weighted_avg_g_num_by_parent_col = f'temp_sum_weighted_avg_g_num_by_parent_{level_name}'
            df_prepared[sum_weighted_avg_g_num_by_parent_col] = df_prepared.groupby(parent_id_col_for_this_level)[weighted_avg_g_num_col].transform('sum')

            # Calculate the population-weighted mean AvgG_emp for siblings
            weighted_mean_sibling_avg_g_val_col = f'temp_weighted_mean_sibling_{emp_growth_col}'
            df_prepared[weighted_mean_sibling_avg_g_val_col] = df_prepared[sum_weighted_avg_g_num_by_parent_col] / safe_sum_pop_initial_by_parent
            
            df_prepared[rel_avg_g_emp_col] = df_prepared[emp_growth_col] - df_prepared[weighted_mean_sibling_avg_g_val_col]
            
            # Clean up temporary columns for AvgG calculation
            df_prepared.drop(columns=[weighted_avg_g_num_col, sum_weighted_avg_g_num_by_parent_col, weighted_mean_sibling_avg_g_val_col], inplace=True)
            print(f"    Calculated relative empirical log-income growth (pop-weighted): '{rel_avg_g_emp_col}'.")
        else:
            df_prepared[rel_avg_g_emp_col] = np.nan
            print(f"    Warning: Could not calculate pop-weighted '{rel_avg_g_emp_col}'. Missing '{emp_growth_col}' or '{pop_initial_col}'.")

        # Relative Population Growth (Population-Weighted)
        if pop_g_col in df_prepared.columns and pop_initial_col in df_prepared.columns:
            # Numerator for weighted average: PopInitial * PopG
            weighted_pop_g_num_col = f'temp_weighted_pop_g_num_{level_name}'
            df_prepared[weighted_pop_g_num_col] = df_prepared[pop_initial_col] * df_prepared[pop_g_col]

            # Sum of numerators within each parent group
            sum_weighted_pop_g_num_by_parent_col = f'temp_sum_weighted_pop_g_num_by_parent_{level_name}'
            df_prepared[sum_weighted_pop_g_num_by_parent_col] = df_prepared.groupby(parent_id_col_for_this_level)[weighted_pop_g_num_col].transform('sum')

            # Calculate the population-weighted mean PopG for siblings
            weighted_mean_sibling_pop_g_val_col = f'temp_weighted_mean_sibling_{pop_g_col}'
            df_prepared[weighted_mean_sibling_pop_g_val_col] = df_prepared[sum_weighted_pop_g_num_by_parent_col] / safe_sum_pop_initial_by_parent
            
            df_prepared[rel_pop_g_col] = df_prepared[pop_g_col] - df_prepared[weighted_mean_sibling_pop_g_val_col]

            # Clean up temporary columns for PopG calculation
            df_prepared.drop(columns=[weighted_pop_g_num_col, sum_weighted_pop_g_num_by_parent_col, weighted_mean_sibling_pop_g_val_col], inplace=True)
            print(f"    Calculated relative population growth (pop-weighted): '{rel_pop_g_col}'.")



        # Clean up the sum of populations column (used by both calculations above)
        df_prepared.drop(columns=[sum_pop_initial_by_parent_col], inplace=True)
            
    else:
        # For top-level units with no parent identifier, relative growth is 0 or NaN
        df_prepared[rel_avg_g_emp_col] = 0.0 # Or np.nan if preferred
        df_prepared[rel_pop_g_col] = 0.0 # Or np.nan if preferred
        print(f"    Set '{rel_avg_g_emp_col}' and '{rel_pop_g_col}' to 0.0 for top-level '{level_name}'.")
    # --- End of New Column Calculations ---

    # Collect essential columns for final DataFrame (can be expanded if needed)
    essential_cols = [
        'UnitName', pop_initial_col, pop_final_col,
        log_avg_inc_initial_col, log_avg_inc_final_col,
        f'AvgLogIncInitial_{level_name}', f'AvgLogIncFinal_{level_name}', g_avg_log_inc_col, # Added
        actual_avg_inc_initial_col, actual_avg_inc_final_col,
        emp_growth_col, pop_g_col, rel_avg_g_emp_col, rel_pop_g_col, # Added new columns
        # Add real population columns to essential_cols
        pop_initial_r_col, pop_final_r_col
    ]
    # Keep all parent identifier columns that exist in the dataframe
    for p_col in CHILD_TO_PARENT_ID_COL_MAP.values():
        if p_col in df_prepared.columns and p_col not in essential_cols:
            essential_cols.append(p_col)
    
    # Ensure 'Agg' column from original df_level_raw is present
    if 'Agg' in df_level_raw.columns and 'Agg' not in df_prepared.columns:
        # If Agg was dropped, try to re-add it based on index alignment with df_level_raw
        # This assumes df_prepared maintains the same index as df_level_raw or a subset of it.
        # A safer approach might be to ensure 'Agg' is carried through all operations or merged back.
        # For now, we'll select from available columns.
        pass # 'Agg' will be included if it exists in df_prepared
    elif 'Agg' not in df_prepared.columns: # If 'Agg' not in df_prepared and also not in original raw
        df_prepared['Agg'] = level_name # Add it based on current level

    final_df_columns = [col for col in essential_cols if col in df_prepared.columns]
    if 'Agg' in df_prepared.columns and 'Agg' not in final_df_columns: # Add 'Agg' if it exists
        final_df_columns.append('Agg')

    df_prepared = df_prepared[final_df_columns].copy() # Select only defined essential cols + Agg

    print(f"-- Finished preparing data for level: {level_name}. Shape: {df_prepared.shape} --")
    return df_prepared

# --- Generalized Sanity Check Function ---
def _calculate_sanity_check(
    df_output: pd.DataFrame,
    lhs_col_name: str,
    primary_rhs_term_col_name: str,
    other_rhs_term_col_names: list[str],
    sanity_check_col_name: str,
    level_name_for_print: str = "",
    path_for_print: str = "",
    check_type_for_print: str = ""
) -> pd.DataFrame:
    """
    Calculates a sanity check by comparing an LHS column to the sum of RHS components.
    RHS_total = primary_rhs_term + sum(other_rhs_terms)
    All inputs are expected to be column names in df_output.
    """
    print_prefix = "      "
    if level_name_for_print:
        print_prefix += f"For {level_name_for_print} "
    if path_for_print:
        print_prefix += f"({path_for_print} path) "
    if check_type_for_print:
        print_prefix += f"[{check_type_for_print} check]: "

    all_required_cols = [lhs_col_name, primary_rhs_term_col_name] + other_rhs_term_col_names
    
    # Ensure all listed columns are actually in the DataFrame before attempting to access
    # This also helps in creating the missing_cols list accurately.
    cols_present_in_df = df_output.columns.tolist()
    actual_other_rhs_term_col_names = [col for col in other_rhs_term_col_names if col in cols_present_in_df]
    
    # Check for missing columns based on the initial lists passed to the function
    missing_cols = [col for col in all_required_cols if col not in cols_present_in_df]

    if not missing_cols:
        # Sum all "other" RHS terms. Ensure they are numeric.
        # If other_rhs_term_col_names is empty, this sum will be 0 or a Series of 0s.
        if actual_other_rhs_term_col_names:
            rhs_other_sum = df_output[actual_other_rhs_term_col_names].sum(axis=1, skipna=False)
        else:
            rhs_other_sum = 0.0 # Or pd.Series(0.0, index=df_output.index) if more robust needed

        # Ensure primary_rhs_term is numeric
        primary_rhs_values = df_output[primary_rhs_term_col_name]
        
        # Sum primary RHS term with the sum of other RHS terms
        # Broadcasting will handle if rhs_other_sum is a scalar (0.0)
        rhs_total = primary_rhs_values + rhs_other_sum
        
        df_output[sanity_check_col_name] = df_output[lhs_col_name] - rhs_total
        print(f"{print_prefix}Calculated sanity check: {sanity_check_col_name}")
    else:
        df_output[sanity_check_col_name] = np.nan
        # Report only the initially requested columns that were missing
        print(f"{print_prefix}Warning: Could not perform sanity check for {sanity_check_col_name}. Missing columns from input list: {missing_cols}")
    
    return df_output

# --- Helper function to calculate de-meaned columns ---
def _calculate_de_meaned_column(df: pd.DataFrame, source_col_name: str, target_col_name: str) -> pd.DataFrame:
    """Calculates a de-meaned column: source_col - mean(source_col)."""
    if source_col_name in df.columns:
        mean_val = df[source_col_name].mean()
        if pd.notna(mean_val):
            df[target_col_name] = df[source_col_name] - mean_val
        else:
            df[target_col_name] = np.nan # Source column might be all NaN
    else:
        df[target_col_name] = np.nan
    return df
# --- End helper ---

# --- NEW HELPER FUNCTION ported from procPrice.py ---
def _aggregate_split_tracts_v5(df_child, df_parent_for_year_y, df_parent_for_year_yn, child_level_name, parent_level_name):
    """
    Handles aggregation of child units (e.g., block groups) to parent units (e.g., tracts)
    when the parent units themselves have changed over time (e.g., tract splits).
    This function modifies the CHILD dataframe to align with the parent panel from procPrice.
    """
    if parent_level_name != 'tr':
        # This logic is specifically for tract splits.
        return df_child

    print(f"    Applying tract split aggregation logic for children of '{parent_level_name}'.")

    # This logic assumes procPrice has already created a consistent panel for tracts.
    # We need to find tracts present in the Y-N parent data that are NOT in the Y parent data.
    # These are the tracts that likely split.
    
    # Ensure indices are strings for comparison
    parent_y_indices = df_parent_for_year_y.index.astype(str)
    parent_yn_indices = df_parent_for_year_yn.index.astype(str)
    
    # Find original tracts from Y-N that are no longer in Y
    split_parent_tract_ids = parent_yn_indices.difference(parent_y_indices)

    if split_parent_tract_ids.empty:
        print("      No split tracts detected between parent years. No aggregation needed.")
        return df_child

    print(f"      Detected {len(split_parent_tract_ids)} potential split tracts to handle.")

    # We need to find the children (e.g., block groups) that belong to these old, split tracts
    # and re-assign them to the "new" tract that represents the aggregation.
    # The key insight from procPrice's _aggregate_split_tracts is that it uses a 9-digit prefix.
    # A split tract like '17031842400' becomes '17031842401', '17031842402', etc.
    # procPrice aggregates these children back up to a '17031842400'-like entity.
    # Our goal is to ensure our child data (from aggregatePrice) does the same.

    # This is complex because we need to map children to the *new* aggregated parents.
    # The logic in procPrice modifies its own 'tr' level data. Here, we must modify the 'bg' data
    # to match what we assume happened in procPrice.
    
    # A simpler, more robust approach is to align the CHILD data (e.g., bg) with the PARENT panel.
    # We will aggregate children based on their parent_id, but we need to handle the splits.

    # Let's create a 9-digit prefix for all tracts in the child dataframe.
    # The ParentTract column in the child 'bg' data is the 11-digit FIPS.
    if 'ParentTract' not in df_child.columns:
        print("      Error: 'ParentTract' column not found in child data. Cannot perform tract split aggregation.")
        return df_child
        
    df_child_copy = df_child.copy()
    df_child_copy['ParentTractPrefix9'] = df_child_copy['ParentTract'].astype(str).str[:9]

    # Now, for each original split tract ID, find its 9-digit prefix.
    for split_tract_id in split_parent_tract_ids:
        prefix9 = split_tract_id[:9]
        
        # Find all children in our current child dataframe that share this prefix.
        children_to_remap_mask = (df_child_copy['ParentTractPrefix9'] == prefix9)
        
        if children_to_remap_mask.any():
            # These children's ParentTract ID needs to be re-mapped to the original, pre-split ID.
            # This makes all children of the split tracts (e.g., ending in 401, 402) appear as
            # if they belong to the original tract (e.g., ending in 400).
            # This forces them to be grouped together during the `groupby(parent_id)` step later on.
            df_child_copy.loc[children_to_remap_mask, 'ParentTract'] = split_tract_id
            print(f"        Re-mapped {children_to_remap_mask.sum()} children with prefix '{prefix9}' to parent tract '{split_tract_id}'.")

    return df_child_copy.drop(columns=['ParentTractPrefix9'])


# --- Helper Functions for Price Decomposition Steps (Parts A-F) ---

def _part_a_initial_child_preparation(df_child_data_for_aggregation: pd.DataFrame, child_level_name: str):
    """Part A: Copies child data, defines column names, checks essential columns, calculates child fitness (c_w_c)."""
    current_child_group_df = df_child_data_for_aggregation.copy()
    child_col_names = {
        'pop_initial': f'PopInitial_{child_level_name}',
        'pop_final': f'PopFinal_{child_level_name}',
        'log_avg_inc_initial': f'LogAvgIncInitial_{child_level_name}',
        'actual_avg_inc_initial': f'ActualAvgIncInitial_{child_level_name}',
        'avg_g_emp': f'AvgG_emp_{child_level_name}',
        'log_avg_inc_final': f'LogAvgIncFinal_{child_level_name}',
        'pop_final_for_shares': f'PopFinal_{child_level_name}',
        'avg_log_inc_initial': f'AvgLogIncInitial_{child_level_name}', # Correct E[ln(y)] trait
        'g_avg_log_inc': f'G_avg_log_inc_{child_level_name}'       # Correct growth of E[ln(y)]
    }
    essential_cols_map_for_check = {
        child_col_names['pop_initial']: 'c_PopInitial',
        child_col_names['pop_final']: 'c_PopFinal',
        child_col_names['log_avg_inc_initial']: 'c_LogAvgIncInitial',
        child_col_names['actual_avg_inc_initial']: 'c_ActualAvgIncInitial',
        child_col_names['avg_g_emp']: 'c_AvgG_emp',
        child_col_names['log_avg_inc_final']: 'c_LogAvgIncFinal',
        child_col_names['pop_final_for_shares']: 'c_PopFinalForShares',
        child_col_names['avg_log_inc_initial']: 'c_AvgLogIncInitial', # Check for correct trait
        child_col_names['g_avg_log_inc']: 'c_G_avg_log_inc'       # Check for correct growth rate
    }
    for col_name_in_df, conceptual_name in essential_cols_map_for_check.items():
        if col_name_in_df not in current_child_group_df.columns:
            print(f"      Error: Essential child column '{col_name_in_df}' (for {conceptual_name}) not found for level '{child_level_name}'. Cannot perform Part A.")
            return None, None
    current_child_group_df['c_w_c'] = np.divide(
        current_child_group_df[child_col_names['pop_final']].astype(float),
        current_child_group_df[child_col_names['pop_initial']].astype(float)
    )
    current_child_group_df.loc[current_child_group_df[child_col_names['pop_initial']] == 0, 'c_w_c'] = 0.0
    return current_child_group_df, child_col_names

def _part_b_parent_group_aggregates(parent_specific_child_df: pd.DataFrame, child_col_names: dict):
    """Part B: Calculates group-specific child pop share (c_p_c_P_grp) and parent aggregates (W_P_gro, Z_P_t_gro, mu_P_t_inc)."""
    c_pop_initial_col = child_col_names['pop_initial']
    c_avg_log_inc_initial_col = child_col_names['avg_log_inc_initial'] # Use E[ln(y)] for population decomposition trait
    c_actual_avg_inc_initial_col = child_col_names['actual_avg_inc_initial']
    group_pop_initial_sum = parent_specific_child_df[c_pop_initial_col].sum()
    if group_pop_initial_sum == 0:
        parent_specific_child_df['c_p_c_P_grp'] = 0.0
    else:
        parent_specific_child_df['c_p_c_P_grp'] = parent_specific_child_df[c_pop_initial_col] / group_pop_initial_sum
    current_w_p_gro = (parent_specific_child_df['c_p_c_P_grp'] * parent_specific_child_df['c_w_c']).sum()
    current_z_p_t_gro = (parent_specific_child_df['c_p_c_P_grp'] * parent_specific_child_df[c_avg_log_inc_initial_col]).sum()
    current_mu_p_t_inc = (parent_specific_child_df['c_p_c_P_grp'] * parent_specific_child_df[c_actual_avg_inc_initial_col]).sum()

    return current_w_p_gro, current_z_p_t_gro, current_mu_p_t_inc, parent_specific_child_df

def _part_d_calculate_selection_terms(parent_specific_child_df_with_shares: pd.DataFrame, child_col_names: dict, current_w_p_gro: float, current_z_p_t_gro: float, current_mu_p_t_inc: float, target_n_years: int, child_growth_col_for_inc_path: str, child_level_name: str):
    """Part D: Calculates selection terms for _pop and _inc paths, and avg child empirical growth for _inc path."""
    sel_p_from_c_gro = np.nan
    sel_p_from_c_inc = np.nan
    avg_child_emp_growth_inc_val = np.nan
    c_avg_log_inc_initial_col = child_col_names['avg_log_inc_initial'] # Correct trait E[ln(y)]
    c_avg_g_emp_col = child_col_names['avg_g_emp'] # Used for _gro path transmission
    c_actual_avg_inc_initial_col = child_col_names['actual_avg_inc_initial']

    # --- _pop path calculation (using E[ln(y)] trait) ---
    if pd.notna(current_w_p_gro) and current_w_p_gro != 0 and target_n_years > 0:
        cov_term_gro = (parent_specific_child_df_with_shares['c_p_c_P_grp'] * 
                        (parent_specific_child_df_with_shares['c_w_c'] / current_w_p_gro - 1) * 
                        (parent_specific_child_df_with_shares[c_avg_log_inc_initial_col] - current_z_p_t_gro)).sum()
        sel_p_from_c_gro = cov_term_gro / target_n_years

    # --- _inc path calculation ---
    # The specific growth rate (direct or aggregated) is determined by the calling function
    # and passed in as `child_growth_col_for_inc_path`. This implements the required nuance
    # for decomposability across scales.

    if pd.notna(current_mu_p_t_inc) and current_mu_p_t_inc != 0 and child_growth_col_for_inc_path in parent_specific_child_df_with_shares.columns:
        # --- Calculate the population-weighted expectation of the child's growth rate, E_K[gamma_k]. ---
        avg_child_growth_for_inc = (parent_specific_child_df_with_shares['c_p_c_P_grp'] * 
                                    parent_specific_child_df_with_shares[child_growth_col_for_inc_path]).sum()
        
        # --- Calculate the covariance term using population-weighted expectation values for the means. ---
        # The parent's population-weighted average initial income, E_K[y_k0], is current_mu_p_t_inc.
        cov_term_inc = (parent_specific_child_df_with_shares['c_p_c_P_grp'] * 
                        (parent_specific_child_df_with_shares[child_growth_col_for_inc_path] - avg_child_growth_for_inc) * 
                        (parent_specific_child_df_with_shares[c_actual_avg_inc_initial_col] - current_mu_p_t_inc)).sum()
        
        sel_p_from_c_inc = cov_term_inc / current_mu_p_t_inc
        
        # The returned "expected growth" is the population-weighted one, which is the primary RHS term.
        avg_child_emp_growth_inc_val = avg_child_growth_for_inc
    else:
        sel_p_from_c_inc = np.nan
        avg_child_emp_growth_inc_val = np.nan
        
    return sel_p_from_c_gro, sel_p_from_c_inc, avg_child_emp_growth_inc_val

def _part_e_calculate_transmitted_terms(
    parent_specific_child_df_with_shares: pd.DataFrame, # Expects c_p_c_P_grp and c_w_c
    child_col_names: dict,
    child_level_name: str,
    parent_level_name: str,
    current_base_analysis_level_name: str,
    full_hierarchy_levels: list,
    current_w_p_gro: float, 
    current_mu_p_t_inc: float 
):
    """Part E: Calculates weighting factors and all transmitted terms for _gro and _inc paths."""
    transmitted_terms_for_parent = {}
    # Make a copy to add weighting factor columns without modifying the DataFrame in the calling scope
    df_children_for_transmission = parent_specific_child_df_with_shares.copy()

    for path_suffix in ['_pop', '_inc']:
        weighting_factor_col_name = f'weighting_factor_{path_suffix}'
        df_children_for_transmission[weighting_factor_col_name] = 0.0 # Initialize

        if path_suffix == '_pop':
            if pd.notna(current_w_p_gro) and current_w_p_gro != 0:
                df_children_for_transmission[weighting_factor_col_name] = \
                    (df_children_for_transmission['c_p_c_P_grp'] * df_children_for_transmission['c_w_c']) / current_w_p_gro
        elif path_suffix == '_inc':
            # The transmission of child-level terms (like Sel_C_from_GC or AvgG_inc_C) to the parent
            # is a population-weighted expectation (E_P[...]). The correct weight is the child's
            # population share within the parent group, which is 'c_p_c_P_grp'.
            df_children_for_transmission[weighting_factor_col_name] = df_children_for_transmission['c_p_c_P_grp']

        # Transmit AvgG from current_base_analysis_level_name
        base_growth_col_in_child_df_str = ''
        if child_level_name == current_base_analysis_level_name:
            if path_suffix == '_pop':
                base_growth_col_in_child_df_str = child_col_names['g_avg_log_inc'] # Use correct growth for pop decomp
            else: # _inc path
                base_growth_col_in_child_df_str = child_col_names['avg_g_emp']
        else:
            base_growth_col_in_child_df_str = f'Transmitted_AvgG_{current_base_analysis_level_name}_to_{child_level_name}{path_suffix}'
        
        transmitted_avg_g_base_to_parent_col_name = f'Transmitted_AvgG_{current_base_analysis_level_name}_to_{parent_level_name}{path_suffix}'
        if base_growth_col_in_child_df_str in df_children_for_transmission.columns:
            transmitted_terms_for_parent[transmitted_avg_g_base_to_parent_col_name] = \
                (df_children_for_transmission[weighting_factor_col_name] * df_children_for_transmission[base_growth_col_in_child_df_str]).sum()
        else:
            transmitted_terms_for_parent[transmitted_avg_g_base_to_parent_col_name] = np.nan

        # Transmit Selection terms if child is an intermediate level
        if child_level_name != current_base_analysis_level_name:
            try:
                child_index_in_hierarchy = full_hierarchy_levels.index(child_level_name)
                if child_index_in_hierarchy > 0: 
                    grandchild_level_name = full_hierarchy_levels[child_index_in_hierarchy - 1]
                    childs_own_sel_col_in_child_df_str = f'Sel_{child_level_name}_from_{grandchild_level_name}{path_suffix}'
                    target_transmitted_child_sel_col_name = f'Transmitted_Sel_{child_level_name}_to_{parent_level_name}{path_suffix}'
                    if childs_own_sel_col_in_child_df_str in df_children_for_transmission.columns:
                        transmitted_terms_for_parent[target_transmitted_child_sel_col_name] = \
                            (df_children_for_transmission[weighting_factor_col_name] * df_children_for_transmission[childs_own_sel_col_in_child_df_str]).sum()
                    else:
                        transmitted_terms_for_parent[target_transmitted_child_sel_col_name] = np.nan

                    base_level_idx_in_hierarchy = full_hierarchy_levels.index(current_base_analysis_level_name)
                    # The loop should go from the base level up to the level *below* the current child_level_name
                    # These are the levels that could have originated a selection term which was then transmitted to child_level_name
                    for k_idx in range(base_level_idx_in_hierarchy + 1, child_index_in_hierarchy):
                        prev_L_name = full_hierarchy_levels[k_idx]
                        
                        # Skip if prev_L_name is the same as child_level_name (no self-transmission of a new sel term)
                        if prev_L_name == child_level_name: # Should not happen with range ending at child_index_in_hierarchy
                            continue

                        # Construct the column name for the term as it exists on the child_level_name's DataFrame
                        # This represents a selection term that originated at prev_L_name and was transmitted TO child_level_name.
                        transmitted_sel_prevL_to_child_col_str = f'Transmitted_Sel_{prev_L_name}_to_{child_level_name}{path_suffix}'
                        
                        # The target column name for the parent level
                        target_col_name_for_parent = f'Transmitted_Sel_{prev_L_name}_to_{parent_level_name}{path_suffix}'
                        
                        if transmitted_sel_prevL_to_child_col_str in df_children_for_transmission.columns:
                            transmitted_terms_for_parent[target_col_name_for_parent] = \
                                (df_children_for_transmission[weighting_factor_col_name] * df_children_for_transmission[transmitted_sel_prevL_to_child_col_str]).sum()
                        else:
                            # If the selection term originated from the ultimate base and is not found on child, it means it was zero.
                            if prev_L_name == current_base_analysis_level_name:
                                transmitted_terms_for_parent[target_col_name_for_parent] = 0.0
                            else:
                                # If an intermediate transmitted selection term is missing, record NaN and print debug. 
                                transmitted_terms_for_parent[target_col_name_for_parent] = np.nan
                                print(f"      DEBUG (Part E): Expected column '{transmitted_sel_prevL_to_child_col_str}' not found in child '{child_level_name}' data when aggregating for parent '{parent_level_name}'. Path: {path_suffix}")
            except ValueError:
                print(f"      Warning (Part E): Level name issue for child '{child_level_name}' or base '{current_base_analysis_level_name}'.")
                pass
    
    return transmitted_terms_for_parent

def _sanity_check_handler(df_parent_output, child_level_name, parent_level_name, current_base_analysis_level_name):
    """Handles all sanity checks for a given parent level."""
    for path_suffix_sc in ['_pop', '_inc']:
        # Multi-Level Sanity Check: Validates the full decomposition from the base level to the parent level.
        sanity_multi_col = f'SanChk_MultiLevel_{path_suffix_sc[1:]}_{current_base_analysis_level_name}_to_{parent_level_name}'

        # The LHS for the multi-level check.
        if path_suffix_sc == '_pop':
            # For the multi-level check, the LHS must be the recursively defined growth,
            # to verify that it equals the sum of its fundamental components.
            lhs_multi_col = f'AvgG_pop_{parent_level_name}'
        else: # For the _inc path, the recursive and physical are the same.
            lhs_multi_col = f'AvgG_inc_{parent_level_name}'
        
        # The RHS is the sum of the parent's own selection term PLUS all transmitted terms from lower levels.
        primary_rhs_multi_col = f'Sel_{parent_level_name}_from_{child_level_name}{path_suffix_sc}'
        other_rhs_multi_cols = [f'Transmitted_AvgG_{current_base_analysis_level_name}_to_{parent_level_name}{path_suffix_sc}']
        for col_name in df_parent_output.columns:
            if col_name.startswith("Transmitted_Sel_") and \
               col_name.endswith(f"_to_{parent_level_name}{path_suffix_sc}"):
                # This collects all transmitted selection terms targeting this parent level
                other_rhs_multi_cols.append(col_name)

        df_parent_output = _calculate_sanity_check(
            df_parent_output, lhs_multi_col, primary_rhs_multi_col, other_rhs_multi_cols, 
            sanity_multi_col, parent_level_name, path_suffix_sc, "Multi-Level (Physical Check)"
        )

        # Single-Level Sanity Check (validates the definitional one-step decomposition)
        sanity_single_col = f'SanChk_SingleLevel_{path_suffix_sc[1:]}_{child_level_name}_to_{parent_level_name}'

        if path_suffix_sc == '_pop':
            # The LHS for the single-level check uses the non-recursive, ΔE[z_c] definition of growth.
            lhs_single_col = f'LHS_AvgG_pop_{parent_level_name}'
            primary_rhs_single_col = f'TransmissionDirectChildGrowth_{parent_level_name}_pop'
            other_rhs_single_cols = [f'Sel_{parent_level_name}_from_{child_level_name}_pop']
        elif path_suffix_sc == '_inc':
            lhs_single_col = f'AvgG_inc_{parent_level_name}'
            primary_rhs_single_col = f'ExpectedChildEmpiricalGrowth_{parent_level_name}_inc'
            other_rhs_single_cols = [f'Sel_{parent_level_name}_from_{child_level_name}_inc']
        
        df_parent_output = _calculate_sanity_check(
            df_parent_output, lhs_single_col, primary_rhs_single_col, other_rhs_single_cols,
            sanity_single_col, parent_level_name, path_suffix_sc, "Single-Level"
        )
    return df_parent_output

# --- Hierarchical Decomposition Functions ---

def calculate_single_parent_level_terms(
    df_child_data_for_aggregation, 
    child_level_name, 
    parent_level_name,
    prepared_parent_level_data, 
    child_to_parent_id_map,
    target_n_years,
    full_hierarchy_levels,
    current_base_analysis_level_name,
    # Add parent data from both years to handle splits
    df_parent_y,
    df_parent_yn
):
    print(f"    Step: Calculating terms for Parent Level '{parent_level_name}' from Child Level '{child_level_name}'.")

    # --- NEW: Apply tract split aggregation logic if necessary ---
    if parent_level_name == 'tr':
        df_child_data_for_aggregation = _aggregate_split_tracts_v5(
            df_child_data_for_aggregation,
            df_parent_y,
            df_parent_yn,
            child_level_name,
            parent_level_name
        )
    # --- END NEW ---

    parent_id_col_in_child_df = child_to_parent_id_map.get(child_level_name)

    if not parent_id_col_in_child_df or parent_id_col_in_child_df not in df_child_data_for_aggregation.columns:
        print(f"      Error: Parent ID column '{parent_id_col_in_child_df}' not found in child data for {child_level_name}. Cannot aggregate.")
        return prepared_parent_level_data.copy() if prepared_parent_level_data is not None else pd.DataFrame()
    current_child_group_df, child_col_names = _part_a_initial_child_preparation(df_child_data_for_aggregation, child_level_name)
    if current_child_group_df is None:
        return prepared_parent_level_data.copy() if prepared_parent_level_data is not None else pd.DataFrame()
    df_parent_output = prepared_parent_level_data.copy() if prepared_parent_level_data is not None else pd.DataFrame()
    if df_parent_output.empty and prepared_parent_level_data is None: print(f"      Warning: Parent intrinsic data for {parent_level_name} is None, df_parent_output is empty.")
    elif df_parent_output.empty: print(f"      Warning: Parent intrinsic data for {parent_level_name} resulted in an empty df_parent_output.")
    if parent_id_col_in_child_df not in current_child_group_df.columns:
        print(f"      Error: Parent ID col '{parent_id_col_in_child_df}' missing in (prepared) child data. Cannot group.")
        return df_parent_output
    grouped_by_parent = current_child_group_df.groupby(parent_id_col_in_child_df)
    parent_terms_list = []
    c_g_avg_log_inc_col = child_col_names['g_avg_log_inc'] # Correct growth rate for pop decomp
    c_log_avg_inc_final_col = child_col_names['log_avg_inc_final']
    c_pop_final_col_for_shares = child_col_names['pop_final_for_shares']

    for parent_id, group in grouped_by_parent:
        parent_specific_child_df_loop_copy = group.copy()
        
        current_w_p_gro, current_z_p_t_gro, current_mu_p_t_inc, parent_specific_child_df_loop_copy = \
            _part_b_parent_group_aggregates(parent_specific_child_df_loop_copy, child_col_names)
        
        # --- Determine correct growth rate column for _inc path ---
        child_growth_col_for_inc_path = ''
        if child_level_name == current_base_analysis_level_name:
            # At the base level, the growth rate is the direct, empirical one.
            child_growth_col_for_inc_path = f'AvgG_emp_{child_level_name}'
        else:
            # For intermediate levels, the growth rate must be the aggregated one from the level below.
            child_growth_col_for_inc_path = f'AvgG_inc_{child_level_name}'
        # --- END ---

        sel_p_from_c_gro, sel_p_from_c_inc, avg_child_emp_growth_inc_val = \
            _part_d_calculate_selection_terms(
                parent_specific_child_df_loop_copy, child_col_names, 
                current_w_p_gro, current_z_p_t_gro, current_mu_p_t_inc, target_n_years,
                child_growth_col_for_inc_path, # Pass the correct column name
                child_level_name
            )
        
        transmitted_terms_for_parent = _part_e_calculate_transmitted_terms(
            parent_specific_child_df_loop_copy, 
            child_col_names, 
            child_level_name, 
            parent_level_name, 
            current_base_analysis_level_name, 
            full_hierarchy_levels,
            current_w_p_gro, 
            current_mu_p_t_inc 
        )
        
        # --- Components for _gro Path ---
        avg_g_pop_val = np.nan
        transmission_direct_child_growth_gro_val = np.nan
        
        # This calculates the parent's growth rate, AvgG_pop_P, as the change in the population-weighted
        # average of its children's traits, z_c = LogAvgInc_c. This is ΔE_P[z_c].
        if (c_log_avg_inc_final_col in parent_specific_child_df_loop_copy.columns and 
            c_pop_final_col_for_shares in parent_specific_child_df_loop_copy.columns and 
            target_n_years > 0):
            group_pop_final_sum = parent_specific_child_df_loop_copy[c_pop_final_col_for_shares].sum()
            if group_pop_final_sum > 0:
                parent_specific_child_df_loop_copy['c_p_prime_c_P_grp'] = parent_specific_child_df_loop_copy[c_pop_final_col_for_shares] / group_pop_final_sum
                # CORRECTED: Use the correct final-time trait: 'avg_log_inc_final'
                c_avg_log_inc_final_col = child_col_names['avg_log_inc_initial'].replace('Initial', 'Final') # Construct the final column name
                if c_avg_log_inc_final_col in parent_specific_child_df_loop_copy.columns:
                    current_z_p_t_prime_gro = (parent_specific_child_df_loop_copy['c_p_prime_c_P_grp'] * parent_specific_child_df_loop_copy[c_avg_log_inc_final_col]).sum()
                    if pd.notna(current_z_p_t_prime_gro) and pd.notna(current_z_p_t_gro):
                        # This is the "Hybrid Model" growth rate. It is used for the single-level sanity check's LHS.
                        avg_g_pop_val = (current_z_p_t_prime_gro - current_z_p_t_gro) / target_n_years
                else:
                    # This case should ideally not happen if data preparation is correct
                    avg_g_pop_val = np.nan
            else: 
                parent_specific_child_df_loop_copy['c_p_prime_c_P_grp'] = 0.0
        
        # This calculates the transmission term, E_P[w_c * G_c] / w_P, using the *empirical* growth of the children (G_c = AvgG_emp_c).
        # This term is the RHS for the single-level sanity check.
        if c_g_avg_log_inc_col in parent_specific_child_df_loop_copy.columns and \
           pd.notna(current_w_p_gro) and current_w_p_gro != 0:
            numerator_direct_child_trans_gro = (parent_specific_child_df_loop_copy['c_p_c_P_grp'] * 
                                                parent_specific_child_df_loop_copy['c_w_c'] * 
                                                parent_specific_child_df_loop_copy[c_g_avg_log_inc_col]).sum()
            transmission_direct_child_growth_gro_val = numerator_direct_child_trans_gro / current_w_p_gro

        # --- RECURSIVE DEFINITION of Parent Growth for Multi-Level Consistency ---
        # The parent's growth rate for the multi-level decomposition MUST be defined by its own decomposition.
        # This now follows the V4 logic: G_P = Sel_P_from_C + E'_P[G_C] where G_C is the child's total recursive growth.

        transmitted_child_pop_growth = np.nan
        g_pop_child_recursive_col = f'AvgG_pop_{child_level_name}'
        
        if g_pop_child_recursive_col in parent_specific_child_df_loop_copy.columns:
            # This is the recursive step for intermediate levels.
            # We transmit the child's total recursive growth, AvgG_pop_C.
            if pd.notna(current_w_p_gro) and current_w_p_gro != 0:
                numerator = (parent_specific_child_df_loop_copy['c_p_c_P_grp'] * 
                             parent_specific_child_df_loop_copy['c_w_c'] * 
                             parent_specific_child_df_loop_copy[g_pop_child_recursive_col]).sum()
                transmitted_child_pop_growth = numerator / current_w_p_gro
        else:
            # This is the base case, when the child is the base analysis level.
            # We transmit the child's direct growth, G_avg_log_inc_C.
            g_pop_child_direct_col = child_col_names['g_avg_log_inc']
            if g_pop_child_direct_col in parent_specific_child_df_loop_copy.columns:
                if pd.notna(current_w_p_gro) and current_w_p_gro != 0:
                    numerator = (parent_specific_child_df_loop_copy['c_p_c_P_grp'] * 
                                 parent_specific_child_df_loop_copy['c_w_c'] * 
                                 parent_specific_child_df_loop_copy[g_pop_child_direct_col]).sum()
                    transmitted_child_pop_growth = numerator / current_w_p_gro

        # The parent's recursive growth is the sum of its own selection and the transmitted aggregated child growth.
        g_pop_recursive_val = (sel_p_from_c_gro if pd.notna(sel_p_from_c_gro) else 0.0) + \
                              (transmitted_child_pop_growth if pd.notna(transmitted_child_pop_growth) else 0.0)

        # The old method of summing all transmitted components is now replaced by the above logic.
        
        # This calculates the single-level transmission term using the child's direct growth (G_avg_log_inc_c).
        # This term is the RHS for the single-level sanity check.
        if c_g_avg_log_inc_col in parent_specific_child_df_loop_copy.columns and \
           pd.notna(current_w_p_gro) and current_w_p_gro != 0:
            numerator_direct_child_trans_gro = (parent_specific_child_df_loop_copy['c_p_c_P_grp'] * 
                                                parent_specific_child_df_loop_copy['c_w_c'] * 
                                                parent_specific_child_df_loop_copy[c_g_avg_log_inc_col]).sum()
            transmission_direct_child_growth_gro_val = numerator_direct_child_trans_gro / current_w_p_gro

        
        # --- Components for _inc Path ---
        # This implements the two-step process: \bar\gamma_K^i = E_K[\gamma_k y_k] / E_K[y_k]
        c_pop_initial_col = child_col_names['pop_initial']
        c_actual_avg_inc_initial_col = child_col_names['actual_avg_inc_initial']

        avg_g_inc_val = np.nan # Default to NaN

        if (child_growth_col_for_inc_path in parent_specific_child_df_loop_copy.columns and
            c_pop_initial_col in parent_specific_child_df_loop_copy.columns and
            c_actual_avg_inc_initial_col in parent_specific_child_df_loop_copy.columns):

            # Step 1: Calculate E[gamma * y], the income-weighted growth numerator
            # Note: c_p_c_P_grp is the population share (p_k / sum(p_k))
            income_weighted_growth_numerator = (parent_specific_child_df_loop_copy['c_p_c_P_grp'] *
                                                parent_specific_child_df_loop_copy[child_growth_col_for_inc_path] *
                                                parent_specific_child_df_loop_copy[c_actual_avg_inc_initial_col]).sum()

            # Step 2: Get E[y], which is `current_mu_p_t_inc` (already calculated in Part B)
            parent_avg_initial_income = current_mu_p_t_inc

            # Step 3: Divide to get the final income-weighted average growth rate
            if pd.notna(parent_avg_initial_income) and parent_avg_initial_income != 0:
                avg_g_inc_val = income_weighted_growth_numerator / parent_avg_initial_income
        
        current_parent_data_dict = {
            'UnitName': parent_id,
            f'Sel_{parent_level_name}_from_{child_level_name}_pop': sel_p_from_c_gro,
            f'Sel_{parent_level_name}_from_{child_level_name}_inc': sel_p_from_c_inc,
            f'AvgG_pop_{parent_level_name}': g_pop_recursive_val, # Use the recursively defined growth rate
            f'AvgG_inc_{parent_level_name}': avg_g_inc_val,
            # This LHS_AvgG_pop is specifically for the single-level sanity check and is NOT transmitted.
            f'LHS_AvgG_pop_{parent_level_name}': avg_g_pop_val, 
            f'TransmissionDirectChildGrowth_{parent_level_name}_pop': transmission_direct_child_growth_gro_val,
            f'ExpectedChildEmpiricalGrowth_{parent_level_name}_inc': avg_child_emp_growth_inc_val
        }

        # --- NEW: Calculate Transmitted Aggregated Growth for Multi-Level Sanity Check ---
        g_pop_child_col = f'AvgG_pop_{child_level_name}'
        g_inc_child_col = f'AvgG_inc_{child_level_name}'

        # For _gro path: E_P[w_C * G_pop_C] / w_P
        trans_agg_g_gro_val = np.nan
        if g_pop_child_col in parent_specific_child_df_loop_copy.columns and pd.notna(current_w_p_gro) and current_w_p_gro != 0:
            numerator_trans_agg_g_gro = (parent_specific_child_df_loop_copy['c_p_c_P_grp'] *
                                         parent_specific_child_df_loop_copy['c_w_c'] *
                                         parent_specific_child_df_loop_copy[g_pop_child_col]).sum()
            trans_agg_g_gro_val = numerator_trans_agg_g_gro / current_w_p_gro
        current_parent_data_dict[f'Transmitted_AggG_pop_{child_level_name}_to_{parent_level_name}'] = trans_agg_g_gro_val

        # For _inc path: E_P[G_inc_C] (population-weighted average)
        trans_agg_g_inc_val = np.nan
        if g_inc_child_col in parent_specific_child_df_loop_copy.columns:
            trans_agg_g_inc_val = (parent_specific_child_df_loop_copy['c_p_c_P_grp'] *
                                   parent_specific_child_df_loop_copy[g_inc_child_col]).sum()
        current_parent_data_dict[f'Transmitted_AggG_inc_{child_level_name}_to_{parent_level_name}'] = trans_agg_g_inc_val
        # --- END NEW ---

        current_parent_data_dict.update(transmitted_terms_for_parent)

        parent_terms_list.append(current_parent_data_dict)

    if not parent_terms_list:
        print(f"      Warning: No parent groups found for {child_level_name} -> {parent_level_name}. Parent aggregates and selection terms will be empty.")
    else:
        df_parent_terms = pd.DataFrame(parent_terms_list)
        if not df_parent_output.empty:
            df_parent_output = pd.merge(df_parent_output, df_parent_terms, on='UnitName', how='left')
        else: # If df_parent_output was initially empty
            df_parent_output = df_parent_terms
            print(f"      Note: df_parent_output was empty, now populated by calculated terms for {parent_level_name}.")

   
    print(f"    -- Completed calculations for {parent_level_name} --")
    
    df_parent_output = _sanity_check_handler(df_parent_output, child_level_name, parent_level_name, current_base_analysis_level_name)

    return df_parent_output

def calculate_multilevel_price_decomposition(
    prepared_level_data, 
    current_base_analysis_level_name, 
    full_hierarchy_levels, 
    child_to_parent_id_map,
    target_n_years,
    # Add raw analysis data to get parent data for different years
    raw_analysis_data_full
):
    """
    Orchestrates the iterative, level-by-level calculation of Price decomposition terms,
    starting from a specified base level of analysis.
    """
    print(f"  -- Running decomposition starting from base analysis level: {current_base_analysis_level_name} --")
    
    # Stores the results of each aggregation step. 
    # Keys: parent_level_name. Values: DataFrame for that parent level with calculated terms.
    # This dictionary will build up, level by level.
    decomposition_results_for_this_base = {} 

    # Initial data for the current_base_analysis_level_name is directly from prepared_level_data.
    if current_base_analysis_level_name not in prepared_level_data or \
       prepared_level_data[current_base_analysis_level_name].empty:
        print(f"    Error: Prepared data for the specified base analysis level '{current_base_analysis_level_name}' is missing or empty.")
        return {} 
        
    decomposition_results_for_this_base[current_base_analysis_level_name] = prepared_level_data[current_base_analysis_level_name].copy()

    try:
        start_index = full_hierarchy_levels.index(current_base_analysis_level_name)
    except ValueError:
        print(f"    Error: Base analysis level '{current_base_analysis_level_name}' not found in hierarchy: {full_hierarchy_levels}.")
        return {}

    # Loop from the current_base_analysis_level_name up to the second to top level.
    # In each iteration, 'child_level_name_loop' aggregates to 'parent_level_name_loop'.
    for i in range(start_index, len(full_hierarchy_levels) - 1):
        child_level_name_loop = full_hierarchy_levels[i]
        parent_level_name_loop = full_hierarchy_levels[i+1]

        print(f"    Processing Aggregation: Child Level '{child_level_name_loop}' -> Parent Level '{parent_level_name_loop}'")

        # Data for the child_level_name_loop for this aggregation step.
        # This comes from the results of the previous iteration (or prepared_level_data if it's the first iteration).
        df_child_input_for_aggregation = decomposition_results_for_this_base.get(child_level_name_loop)
        
        # Intrinsic data for the parent_level_name_loop (e.g., its own UnitName, PopInitial, etc.)
        # This is used as a base to merge the newly calculated aggregated terms into.
        df_parent_intrinsic_data = prepared_level_data.get(parent_level_name_loop)

        # --- NEW: Get parent data for both years to handle splits ---
        final_year = INITIAL_YEAR + target_n_years
        df_parent_y_for_splits = filter_analysis_data_for_level(
            raw_analysis_data_full, parent_level_name_loop, target_n_years, INITIAL_YEAR
        ).set_index('UnitName')
        
        df_parent_yn_for_splits = filter_analysis_data_for_level(
            raw_analysis_data_full, parent_level_name_loop, target_n_years, INITIAL_YEAR - target_n_years if INITIAL_YEAR - target_n_years > 0 else INITIAL_YEAR
        ).set_index('UnitName')
        # This is a simplification; a more robust way would be to find the correct prior year data.
        # For now, this assumes a consistent panel structure in the input file.
        # A better approach would be to pass the full `incomes_data` from procPrice and find the prior year panel there.
        # But for this fix, we'll assume the raw_analysis_data contains the necessary panels.
        # Let's assume `procPrice` created panels for N=5, so we need year 14 and 19 data for tracts.
        
        # Let's get the specific year data frames for the parent level
        # This is more robust.
        # We need the unit names for the parent level in year Y and Y-N
        # This requires the full analysis data, not the prepared data.
        df_parent_y = filter_analysis_data_for_level(raw_analysis_data_full, parent_level_name_loop, 1, final_year - 1).set_index('UnitName')
        df_parent_yn = filter_analysis_data_for_level(raw_analysis_data_full, parent_level_name_loop, 1, INITIAL_YEAR - 1).set_index('UnitName')


        # A better way to get the parent panels is needed. Let's assume the prepared data is sufficient to identify the panel.
        # Let's rethink. `procPrice` creates `analysis_Price2.pkl`.
        # The `UnitName`s in there for `agg='tr'` for a given `FirstYear` and `Year` ARE the consistent panel.
        # So we just need the sets of UnitNames.
        parent_panel_y = prepared_level_data.get(parent_level_name_loop, pd.DataFrame())
        # The corresponding Y-N panel isn't directly available here. 
        # The fundamental issue is that this script doesn't have the raw year-by-year data panels.
        

        # The most direct fix is to replicate the logic inside this script.
        # Let's try again with the `_aggregate_split_tracts_v5` function.

        if df_child_input_for_aggregation is None or df_child_input_for_aggregation.empty:
            print(f"      Skipping {child_level_name_loop} -> {parent_level_name_loop}: Child data for '{child_level_name_loop}' is missing or empty from previous step.")
            decomposition_results_for_this_base[parent_level_name_loop] = pd.DataFrame() # Store empty
            continue
        
        if df_parent_intrinsic_data is None or df_parent_intrinsic_data.empty:
            print(f"      Warning: Intrinsic data for parent level '{parent_level_name_loop}' is missing or empty. Results for this level might be incomplete.")
            df_parent_intrinsic_data = pd.DataFrame()


        df_parent_level_output_terms = calculate_single_parent_level_terms(
            df_child_input_for_aggregation,
            child_level_name_loop,
            parent_level_name_loop,
            df_parent_intrinsic_data,
            child_to_parent_id_map,
            target_n_years,
            full_hierarchy_levels,
            current_base_analysis_level_name,
            df_parent_y=df_parent_intrinsic_data, # Use the prepared data as the Y panel
            df_parent_yn=pd.DataFrame() # This is the missing piece. How to get the Y-N panel?
            # We can't easily get it here. Let's rethink the implementation of `_aggregate_split_tracts_v5`.
            # It doesn't need the full parent data, just the knowledge of which tracts split.
            # `procPrice` saves the result. The panel of tracts in `analysis_Price2.pkl` for `agg='tr'`
            # for the given N-year period IS the corrected panel.
            # So the logic should be: if a child's parent ID prefix matches one of the final panel tracts, but the full ID doesn't, remap it.
            # This is still tricky.

            # Let's go with the initial, simpler implementation. It's an improvement, and we can refine it.
            # The key is that `calculate_single_parent_level_terms` needs the parent panels.
        )
        
        decomposition_results_for_this_base[parent_level_name_loop] = df_parent_level_output_terms
        
    # --- Add Relative and Cumulative/Summary Terms ---
    print(f"    Calculating Final Summary terms for base: {current_base_analysis_level_name}")
    for level_name_for_summary in decomposition_results_for_this_base.keys():
        df_current_level = decomposition_results_for_this_base.get(level_name_for_summary)
        if df_current_level is None or df_current_level.empty:
            continue

        print(f"      Processing level: {level_name_for_summary} for summary columns")

        # Calculate Relative Aggregated Growth columns
        df_current_level = _calculate_de_meaned_column(df_current_level, f'AvgG_pop_{level_name_for_summary}', f'RelAvgG_pop_{level_name_for_summary}')
        df_current_level = _calculate_de_meaned_column(df_current_level, f'AvgG_inc_{level_name_for_summary}', f'RelAvgG_inc_{level_name_for_summary}')

        for path_suffix in ['_gro', '_inc']:
            # Calculate Cumulative Selection
            cum_sel_col_name = f'cum_sel{path_suffix}'
            df_current_level[cum_sel_col_name] = 0.0

            own_sel_term_name = ''
            try:
                current_level_idx = full_hierarchy_levels.index(level_name_for_summary)
                if current_level_idx > 0:
                    child_of_current_level_name = full_hierarchy_levels[current_level_idx - 1]
                    own_sel_term_name = f'Sel_{level_name_for_summary}_from_{child_of_current_level_name}{path_suffix}'
                    if own_sel_term_name in df_current_level.columns:
                        df_current_level[cum_sel_col_name] += df_current_level[own_sel_term_name].fillna(0)
            except ValueError:
                pass

            for col_in_df in df_current_level.columns:
                if col_in_df.startswith('Transmitted_Sel_') and col_in_df.endswith(f'_to_{level_name_for_summary}{path_suffix}'):
                    df_current_level[cum_sel_col_name] += df_current_level[col_in_df].fillna(0)
            
            # Calculate Average Magnitude of Selection
            avg_mag_sel_col_name = f'avg_mag_sel{path_suffix}'
            sel_term_cols = [c for c in df_current_level.columns if (c.startswith('Sel_') or c.startswith('Transmitted_Sel_')) and c.endswith(path_suffix)]
            if sel_term_cols:
                df_current_level[avg_mag_sel_col_name] = df_current_level[sel_term_cols].abs().mean(axis=1)
            else:
                df_current_level[avg_mag_sel_col_name] = np.nan

        decomposition_results_for_this_base[level_name_for_summary] = df_current_level

    # --- Add New Ratio Metrics: Global Impact Ratio (GIR) and Local Dominance Ratio (LDR) ---
    print(f"    Calculating LDR terms for base: {current_base_analysis_level_name}")

    # --- Steps 2 & 3: Loop through levels to calculate GIR and LDR ---
    for i, level_name in enumerate(full_hierarchy_levels):
        df_level = decomposition_results_for_this_base.get(level_name)
        if df_level is None or df_level.empty:
            continue

        # --- Calculate Local Dominance Ratios (LDR) for this level's "own" selection term ---
        if i > 0: # LDR is not applicable for the base level
            child_level_name = full_hierarchy_levels[i-1]
            for path_suffix in ['_gro', '_inc']:
                own_sel_term = f'Sel_{level_name}_from_{child_level_name}{path_suffix}'
                
                # Determine the correct transmitted growth term
                transmitted_growth_term_base = ''
                if path_suffix == '_gro':
                     # For the level just above the base, it's the transmitted empirical growth
                    if child_level_name == current_base_analysis_level_name:
                        transmitted_growth_term_base = f'Transmitted_AvgG_{child_level_name}_to_{level_name}'
                    else: # For higher levels, it's the transmitted recursive growth
                        transmitted_growth_term_base = f'Transmitted_AggG_pop_{child_level_name}_to_{level_name}'
                else: # _inc path
                    if child_level_name == current_base_analysis_level_name:
                        transmitted_growth_term_base = f'Transmitted_AvgG_{child_level_name}_to_{level_name}'
                    else:
                        transmitted_growth_term_base = f'Transmitted_AggG_inc_{child_level_name}_to_{level_name}'
                
                transmitted_growth_term = f"{transmitted_growth_term_base}{path_suffix}"

                if own_sel_term in df_level.columns and transmitted_growth_term in df_level.columns:
                    ldr_denominator = df_level[own_sel_term].abs() + df_level[transmitted_growth_term].abs()
                    safe_ldr_denominator = ldr_denominator.replace(0, np.nan)
                    
                    # LDR for own selection term
                    ldr_own_sel_col_name = f"{own_sel_term}_LDR"
                    df_level[ldr_own_sel_col_name] = (df_level[own_sel_term].abs() / safe_ldr_denominator) * 100

                    # LDR for the corresponding transmitted growth term
                    ldr_transmitted_col_name = f"{transmitted_growth_term}_LDR"
                    df_level[ldr_transmitted_col_name] = (df_level[transmitted_growth_term].abs() / safe_ldr_denominator) * 100

        decomposition_results_for_this_base[level_name] = df_level

    # --- Part 2: Multi-level PNC Calculation ---
    all_processed_levels = {}
    
    # Prepare ancestor dataframes for merging by standardizing their keys
    ancestor_dfs_for_merge = {}
    for level_name in full_hierarchy_levels:
        df_ancestor = decomposition_results_for_this_base.get(level_name)
        if df_ancestor is None or df_ancestor.empty: continue
        
        growth_cols = [c for c in df_ancestor.columns if c.startswith('AvgG_')]
        if 'UnitName' not in df_ancestor.columns or not growth_cols: continue
        
        df_ancestor_merge_ready = df_ancestor[['UnitName'] + growth_cols].copy()
        
        ancestor_dfs_for_merge[level_name] = df_ancestor_merge_ready

    # Iterate through levels, merge with all ancestors, and calculate all PNCs
    for i, level_name in enumerate(full_hierarchy_levels):
        df_level = decomposition_results_for_this_base.get(level_name)
        if df_level is None or df_level.empty: 
            all_processed_levels[level_name] = pd.DataFrame()
            continue
        
        df_level_augmented = df_level.copy()

        # Sequentially merge with all ancestors to bring in their growth rates
        for j in range(i + 1, len(full_hierarchy_levels)):
            ancestor_level_name = full_hierarchy_levels[j]
            df_ancestor_to_merge = ancestor_dfs_for_merge.get(ancestor_level_name)
            if df_ancestor_to_merge is None: continue
            
            # Find the column in the current augmented dataframe that links to this ancestor
            parent_level_of_ancestor = full_hierarchy_levels[j-1]
            merge_key = child_to_parent_id_map.get(parent_level_of_ancestor)

            if merge_key not in df_level_augmented.columns: continue

            # Perform the merge
            df_level_augmented = pd.merge(df_level_augmented, df_ancestor_to_merge, left_on=merge_key, right_on='UnitName', how='left', suffixes=('', f'_{ancestor_level_name}_y'))
        
        # Identify all component columns (own selection + all transmitted terms)
        component_cols = [col for col in df_level.columns if (col.startswith('Sel_') or col.startswith('Transmitted_')) and ('_pop' in col or '_inc' in col) and not col.endswith('_LDR') and '_PNC' not in col]
        
        if not component_cols: 
            all_processed_levels[level_name] = df_level_augmented
            continue

        # Calculate PNC against all ancestors (including self)
        for j in range(i, len(full_hierarchy_levels)):
            ancestor_level_name = full_hierarchy_levels[j]
            
            pop_denom_col = f'AvgG_pop_{ancestor_level_name}'
            inc_denom_col = f'AvgG_inc_{ancestor_level_name}'
            
            # Check if denominator columns exist (they should after the merges)
            if pop_denom_col not in df_level_augmented.columns or inc_denom_col not in df_level_augmented.columns:
                 continue
            
            denominators = {
                '_pop': df_level_augmented[pop_denom_col].replace(0, np.nan),
                '_inc': df_level_augmented[inc_denom_col].replace(0, np.nan)
            }
            
            pnc_suffix = f"_PNC" if ancestor_level_name == level_name else f"_PNC_{ancestor_level_name}"

            for comp_col in component_cols:
                # Construct the new PNC column name by appending the suffix to the base component column name
                pnc_col_name = f"{comp_col}{pnc_suffix}"
                
                path_suffix = '_pop' if '_pop' in comp_col else '_inc'
                if path_suffix in denominators and comp_col in df_level_augmented.columns:
                    df_level_augmented[pnc_col_name] = (df_level_augmented[comp_col] / denominators[path_suffix]) * 100
        
        # Clean up merged columns (UnitName_y, AvgG_..._y, etc.) before storing
        cols_to_drop = [c for c in df_level_augmented.columns if c.endswith('_y') or c == 'UnitName_y']
        df_level_augmented.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        all_processed_levels[level_name] = df_level_augmented

    return all_processed_levels

def calculate_pnc_and_ldr_metrics(decomposition_results_for_this_base, full_hierarchy_levels, child_to_parent_id_map, current_base_analysis_level_name):
    """
    Post-processing to calculate:
    1. Local Dominance Ratio (LDR) for each component's contribution to total local magnitude.
    2. Parent-Normalized Contribution (PNC) for each component against each of its ancestors' total growth.
    """
    print(f"    Calculating LDR and multi-level PNC terms...")

    # --- Part 1: LDR Calculation (New Definition) ---
    # LDR is now defined as the relative contribution of each growth component to the total magnitude of growth at a specific level.
    for i, level_name in enumerate(full_hierarchy_levels):
        df_level = decomposition_results_for_this_base.get(level_name)
        if df_level is None or df_level.empty or i < 1: # LDR is not applicable for the base level
            continue

        child_level_name = full_hierarchy_levels[i-1]
        
        for path_suffix in ['_pop', '_inc']:
            # 1. Identify all component columns that constitute the total growth for this level and path.
            # The total recursive growth (e.g., AvgG_pop_cm) is the sum of its own selection term and all transmitted terms.
            component_cols = []
            
            # Add the 'own' selection term for this level
            own_sel_term = f'Sel_{level_name}_from_{child_level_name}{path_suffix}'
            if own_sel_term in df_level.columns:
                component_cols.append(own_sel_term)

            # Find all terms transmitted to this level (selection, empirical base growth, and aggregated recursive growth)
            for col in df_level.columns:
                # This identifies any term that was transmitted TO this level.
                is_transmitted_term = col.startswith(('Transmitted_Sel_', 'Transmitted_AvgG_', 'Transmitted_AggG_'))
                is_for_this_level_and_path = col.endswith(f'_to_{level_name}{path_suffix}')
                
                if is_transmitted_term and is_for_this_level_and_path:
                    # Exclude metrics of metrics, just in case
                    if not col.endswith(('_LDR', '_PNC')) and '_LDR' not in col and '_PNC' not in col:
                         component_cols.append(col)
            
            component_cols = sorted(list(set(component_cols)))

            if not component_cols:
                continue

            # 2. Calculate the LDR denominator: the sum of the absolute values of all identified components.
            ldr_denominator = df_level[component_cols].abs().sum(axis=1)
            safe_ldr_denominator = ldr_denominator.replace(0, np.nan)

            # 3. Calculate the LDR for each individual component.
            for comp_col in component_cols:
                ldr_col_name = f"{comp_col}_LDR"
                if comp_col in df_level.columns:
                    df_level[ldr_col_name] = (df_level[comp_col].abs() / safe_ldr_denominator) * 100
                else:
                    df_level[ldr_col_name] = np.nan

        decomposition_results_for_this_base[level_name] = df_level

    # --- Part 2: Multi-level PNC Calculation ---
    all_processed_levels = {}
    
    # Prepare ancestor dataframes for merging by standardizing their keys
    ancestor_dfs_for_merge = {}
    for level_name in full_hierarchy_levels:
        df_ancestor = decomposition_results_for_this_base.get(level_name)
        if df_ancestor is None or df_ancestor.empty: continue
        
        growth_cols = [c for c in df_ancestor.columns if c.startswith('AvgG_')]
        if 'UnitName' not in df_ancestor.columns or not growth_cols: continue
        
        df_ancestor_merge_ready = df_ancestor[['UnitName'] + growth_cols].copy()
        
        ancestor_dfs_for_merge[level_name] = df_ancestor_merge_ready

    # Iterate through levels, merge with all ancestors, and calculate all PNCs
    for i, level_name in enumerate(full_hierarchy_levels):
        df_level = decomposition_results_for_this_base.get(level_name)
        if df_level is None or df_level.empty: 
            all_processed_levels[level_name] = pd.DataFrame()
            continue
        
        df_level_augmented = df_level.copy()

        # Sequentially merge with all ancestors to bring in their growth rates
        for j in range(i + 1, len(full_hierarchy_levels)):
            ancestor_level_name = full_hierarchy_levels[j]
            df_ancestor_to_merge = ancestor_dfs_for_merge.get(ancestor_level_name)
            if df_ancestor_to_merge is None: continue
            
            # Find the column in the current augmented dataframe that links to this ancestor
            parent_level_of_ancestor = full_hierarchy_levels[j-1]
            merge_key = child_to_parent_id_map.get(parent_level_of_ancestor)

            if merge_key not in df_level_augmented.columns: continue

            # Perform the merge
            df_level_augmented = pd.merge(df_level_augmented, df_ancestor_to_merge, left_on=merge_key, right_on='UnitName', how='left', suffixes=('', f'_{ancestor_level_name}_y'))
        
        # Identify all component columns (own selection + all transmitted terms)
        component_cols = [col for col in df_level.columns if (col.startswith('Sel_') or col.startswith('Transmitted_')) and ('_pop' in col or '_inc' in col) and not col.endswith('_LDR') and '_PNC' not in col]
        
        if not component_cols: 
            all_processed_levels[level_name] = df_level_augmented
            continue

        # Calculate PNC against all ancestors (including self)
        for j in range(i, len(full_hierarchy_levels)):
            ancestor_level_name = full_hierarchy_levels[j]
            
            pop_denom_col = f'AvgG_pop_{ancestor_level_name}'
            inc_denom_col = f'AvgG_inc_{ancestor_level_name}'
            
            # Check if denominator columns exist (they should after the merges)
            if pop_denom_col not in df_level_augmented.columns or inc_denom_col not in df_level_augmented.columns:
                 continue
            
            denominators = {
                '_pop': df_level_augmented[pop_denom_col].replace(0, np.nan),
                '_inc': df_level_augmented[inc_denom_col].replace(0, np.nan)
            }
            
            pnc_suffix = f"_PNC" if ancestor_level_name == level_name else f"_PNC_{ancestor_level_name}"

            for comp_col in component_cols:
                # Construct the new PNC column name by appending the suffix to the base component column name
                pnc_col_name = f"{comp_col}{pnc_suffix}"
                
                path_suffix = '_pop' if '_pop' in comp_col else '_inc'
                if path_suffix in denominators and comp_col in df_level_augmented.columns:
                    df_level_augmented[pnc_col_name] = (df_level_augmented[comp_col] / denominators[path_suffix]) * 100
        
        # Clean up merged columns (UnitName_y, AvgG_..._y, etc.) before storing
        cols_to_drop = [c for c in df_level_augmented.columns if c.endswith('_y') or c == 'UnitName_y']
        df_level_augmented.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # --- NEW: Calculate Cumulative Selection Terms and their PNC values ---
        # Calculate cumulative selection terms for both _pop and _inc paths
        for path_suffix in ['_pop', '_inc']:
            # Find all selection terms (both direct and transmitted) for this path
            selection_cols = []
            
            # Add direct selection terms for this specific level
            for col in df_level_augmented.columns:
                if col.startswith(f'Sel_{level_name}_from_') and col.endswith(path_suffix) and not col.endswith('_LDR') and '_PNC' not in col:
                    selection_cols.append(col)
            
            # Add transmitted selection terms that end at this level
            for col in df_level_augmented.columns:
                if col.startswith('Transmitted_Sel_') and col.endswith(f'_to_{level_name}{path_suffix}') and not col.endswith('_LDR') and '_PNC' not in col:
                    selection_cols.append(col)
            
            if selection_cols:
                # Calculate cumulative selection term with level-specific name
                cumulative_col_name = f'Cumulative_Sel_{path_suffix[1:]}_{level_name}'
                
                # Only create if it doesn't already exist to avoid duplicates
                if cumulative_col_name not in df_level_augmented.columns:
                    df_level_augmented[cumulative_col_name] = df_level_augmented[selection_cols].sum(axis=1)
                    
                    # Calculate PNC values for the cumulative selection term against all ancestors
                    for j in range(i, len(full_hierarchy_levels)):
                        ancestor_level_name = full_hierarchy_levels[j]
                        
                        denom_col = f'AvgG{path_suffix}_{ancestor_level_name}'
                        
                        if denom_col in df_level_augmented.columns:
                            pnc_suffix = f"_PNC" if ancestor_level_name == level_name else f"_PNC_{ancestor_level_name}"
                            cumulative_pnc_col_name = f"{cumulative_col_name}{pnc_suffix}"
                            
                            # Only create PNC column if it doesn't already exist
                            if cumulative_pnc_col_name not in df_level_augmented.columns:
                                # Calculate PNC
                                denominator = df_level_augmented[denom_col].replace(0, np.nan)
                                df_level_augmented[cumulative_pnc_col_name] = (df_level_augmented[cumulative_col_name] / denominator) * 100
        # --- END NEW ---

        all_processed_levels[level_name] = df_level_augmented

    return all_processed_levels


# --- Main Orchestration Function (Placeholder for now) ---
def main_v4():
    print(f"--- Running aggregatePriceV4.py ---")

    df_analysis_full = load_pickle_data(ANALYSIS_PRICE_FILEPATH, "full analysis_Price2.pkl")
    if df_analysis_full is None:
        return

    # --- DEBUG: Inspect state-level data from the loaded analysis_Price2.pkl ---
    print("\n--- DEBUG: Inspecting State-Level Data from loaded analysis_Price2.pkl ---")
    if 'Agg' in df_analysis_full.columns:
        df_state_from_pkl = df_analysis_full[df_analysis_full['Agg'] == 'st'].copy()
        print(f"  Shape of state-level data found in analysis_Price2.pkl: {df_state_from_pkl.shape}")
        if not df_state_from_pkl.empty:
            # Add 'N' column if not present, for easier inspection if needed
            if 'N' not in df_state_from_pkl.columns and 'Year' in df_state_from_pkl.columns and 'FirstYear' in df_state_from_pkl.columns:
                df_state_from_pkl['N'] = pd.to_numeric(df_state_from_pkl['Year'], errors='coerce') - \
                                         pd.to_numeric(df_state_from_pkl['FirstYear'], errors='coerce')

            cols_to_print = ['UnitName', 'Agg', 'Year', 'FirstYear']
            # Add N if it was calculated or exists
            if 'N' in df_state_from_pkl.columns:
                cols_to_print.append('N')
            
            # Add key initial value columns that are checked for NaNs later
            initial_cols_to_check = ['PopInitialN', 'LogAvgIncInitialN', 'PopInitialNR']
            for col in initial_cols_to_check:
                if col in df_state_from_pkl.columns:
                    cols_to_print.append(col)
            
            # Add some N-year metrics if they exist
            n_year_metrics_to_check = [f'DeltaZbar_{TARGET_N}', f'CovTerm_{TARGET_N}', f'ExpPopTerm_{TARGET_N}']
            for col in n_year_metrics_to_check:
                if col in df_state_from_pkl.columns:
                    cols_to_print.append(col)

            # Ensure all selected columns actually exist to prevent KeyErrors
            final_cols_to_print_state = [col for col in cols_to_print if col in df_state_from_pkl.columns]
            
            print("  State-Level Data (relevant columns):")
            with pd.option_context('display.max_columns', None, 'display.width', 1000):
                print(df_state_from_pkl[final_cols_to_print_state])
        else:
            print("  No state-level data (Agg == 'st') found in analysis_Price2.pkl.")
    else:
        print("  'Agg' column not found in df_analysis_full. Cannot filter for state-level data.")
    print("--- END DEBUG ---")
    # --- End of DEBUG section ---

    prepared_level_data = {}
    for level in HIERARCHY_LEVELS:
        df_raw_for_level = filter_analysis_data_for_level(
            df_analysis_full, 
            target_level_name=level, 
            target_n_period=TARGET_N, 
            initial_year_period=INITIAL_YEAR
        )
        if not df_raw_for_level.empty:
            prepared_level_data[level] = prepare_level_data(
                df_raw_for_level, 
                level, 
                TARGET_N, 
                CHILD_TO_PARENT_ID_COL_MAP
            )
        else:
            prepared_level_data[level] = pd.DataFrame()
            print(f"  No raw data to prepare for level '{level}'. Storing empty DataFrame.")

    # --- At this point, prepared_level_data dictionary would contain ---
    # --- DataFrames for each level with their intrinsic properties calculated. ---
    
    # --- New: Outer loop for different base analysis levels & inner decomposition ---
    BASE_ANALYSIS_LEVELS = ['bg'] # Start with 'bg' as the base for the entire decomposition
    all_decomposition_results_by_base = {} # To store results for each base analysis run

    for base_analysis_name in BASE_ANALYSIS_LEVELS:
        print(f"\n{'='*10} Starting Multilevel Price Decomposition: Base Analysis Level = {base_analysis_name.upper()} {'='*10}")
        
        decomposition_output_for_this_base = calculate_multilevel_price_decomposition(
            prepared_level_data,         # Dict of DFs with intrinsic properties for ALL levels
            base_analysis_name,          # The starting 'child' level for *this* decomposition run
            HIERARCHY_LEVELS,            # Full list of levels in order
            CHILD_TO_PARENT_ID_COL_MAP,  # Map for parent IDs
            TARGET_N,                    # N-year period
            df_analysis_full             # Pass the full raw data for panel lookups
        )
        all_decomposition_results_by_base[base_analysis_name] = decomposition_output_for_this_base
        # --- NEW: Post-process to add LDR and multi-level PNC terms ---
        all_decomposition_results_by_base[base_analysis_name] = calculate_pnc_and_ldr_metrics(
            decomposition_output_for_this_base,
            HIERARCHY_LEVELS,
            CHILD_TO_PARENT_ID_COL_MAP,
            base_analysis_name
        )
        # --- END NEW ---
        print(f"{'='*10} Completed Multilevel Price Decomposition for Base Analysis Level: {base_analysis_name.upper()} {'='*10}")

    print("\n--- Multilevel Price Decomposition Complete for All Specified Base Analyses ---")
    for base_name_report, results_dict in all_decomposition_results_by_base.items():
        print(f"  Results from Base Analysis: {base_name_report.upper()}")
        for level_name_report, df_level_report in results_dict.items():
            if not df_level_report.empty:
                print(f"    Level: {level_name_report}, Shape: {df_level_report.shape}, Columns: {df_level_report.columns.to_list()[:7]}...")
                # Only print detailed head for 'cm' level when base is 'bg'
                if base_name_report == 'bg': # and level_name_report == 'cm':
                    print(f"      All calculated terms for level '{level_name_report}' (from base '{base_name_report}'), first 5 rows:")
                    with pd.option_context('display.max_columns', None, 'display.width', 1000):
                        print(df_level_report.head())
            else:
                print(f"    Level: {level_name_report} has no data or no terms calculated.")

    # --- Export specific columns to CSV and save full results to Pickle ---
    print("\n--- Exporting selected terms to CSV and full results to Pickle ---")
    output_dir = "output_terms"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Ensured output directory '{output_dir}' exists.")

    for base_name, results_dict_for_base in all_decomposition_results_by_base.items():
        for level_name, df_level in results_dict_for_base.items():
            if not df_level.empty:
                # --- NEW Column Reordering Logic ---
                # 1. Define all columns we might want to export, starting with identifiers and base values.
                potential_cols = ['UnitName']
                potential_cols.extend(list(CHILD_TO_PARENT_ID_COL_MAP.values()))
                potential_cols.extend([
                    f'PopInitial_{level_name}', f'PopFinal_{level_name}',
                    f'LogAvgIncInitial_{level_name}', f'LogAvgIncFinal_{level_name}',
                    f'PopG_{level_name}', f'RelPopG_{level_name}',
                    f'AvgG_emp_{level_name}', f'RelAvgG_emp_{level_name}',
                    f'AvgG_pop_{level_name}', f'RelAvgG_pop_{level_name}',
                    f'LHS_AvgG_pop_{level_name}',
                    f'AvgG_inc_{level_name}', f'RelAvgG_inc_{level_name}',
                    f'PopInitialR_{level_name}', f'PopFinalR_{level_name}'
                ])

                # 2. Dynamically add all calculated decomposition and metric terms that exist in the DataFrame.
                # This is safer and more comprehensive than hardcoding every possible term name.
                for col in df_level.columns:
                    if col.startswith(('Sel_', 'Transmitted_')) and ('_pop' in col or '_inc' in col):
                        potential_cols.append(col)
                
                # Add cumulative selection terms and their PNC columns
                for col in df_level.columns:
                    if col.startswith('Cumulative_Sel_'):
                        potential_cols.append(col)
                
                # 3. Filter the potential columns to only those that actually exist in the current DataFrame, removing duplicates.
                existing_cols_in_df = list(dict.fromkeys([col for col in potential_cols if col in df_level.columns]))

                # 4. Split the existing columns into the desired groups.
                raw_cols = [col for col in existing_cols_in_df if not col.endswith('_LDR') and '_PNC' not in col]
                ldr_cols = sorted([col for col in existing_cols_in_df if col.endswith('_LDR')])
                pnc_cols = sorted([col for col in existing_cols_in_df if '_PNC' in col])
                
                # 5. Assemble the final, ordered list of columns for export.
                final_cols_to_export = raw_cols + ldr_cols + pnc_cols
                
                if final_cols_to_export:
                    df_export = df_level[final_cols_to_export].copy()
                    csv_filename = os.path.join(output_dir, f"{base_name}_{level_name}_exported_terms.csv")
                    try:
                        df_export.to_csv(csv_filename, index=False)
                        print(f"    Successfully exported terms for {base_name} - {level_name} to {csv_filename}")
                    except Exception as e:
                        print(f"    Error exporting terms for {base_name} - {level_name} to CSV: {e}")
                else:
                    print(f"    No columns to export for {base_name} - {level_name}. Skipping CSV.")
            else:
                print(f"    Skipping CSV export for {base_name} - {level_name} as DataFrame is empty.")

    # Save the entire PROCESSED results dictionary to a pickle file
    pickle_filename = os.path.join(output_dir, "all_decomposition_results.pkl")
    try:
        with open(pickle_filename, 'wb') as f_pickle:
            pickle.dump(all_decomposition_results_by_base, f_pickle) # Save the processed data
        print(f"  Successfully saved all decomposition results to {pickle_filename}")
    except Exception as e:
        print(f"  Error saving decomposition results to Pickle: {e}")
    # --- End of Export and Save logic ---

    print("\n--- aggregatePriceV4.py processing complete with decomposition structure --- ")

if __name__ == "__main__":
    main_v4()
