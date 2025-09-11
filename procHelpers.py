# procHelpers.py

import pandas as pd
import pickle
import os
import numpy as np
from collections import defaultdict

# === Define Constants ===
# Ideally, these might come from a central config or saveCensusDataV2

# years = np.arange(10, 24)        # Example: Years 10 to 23 inclusive
years = [14,19]
aggs = ['st', 'ct', 'cm', 'tr', 'bg'] # Example aggregation levels (Removed 'bg' for now as it's often a member level)
income_vars = [f'B19001_0{i:02d}E' for i in range(2, 18)] # Example income variables

# Corrected definition of xbins_values and comparison_data
# These are the standard ACS defined lower boundaries for the 16 income bins.
# Bin 1: <$10k
# Bin 2: $10k - $14,999
# ...
# Bin 15: $150k - $199,999
# Bin 16: $200k+
standard_bin_lower_bounds = np.array([
    0, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 
    60000, 75000, 100000, 125000, 150000, 200000 
], dtype=np.longdouble)

# Define the corresponding upper boundaries for the first 15 closed bins
standard_bin_upper_bounds = np.array([
    9999, 14999, 19999, 24999, 29999, 34999, 39999, 44999, 49999, 59999,
    74999, 99999, 124999, 149999, 199999
], dtype=np.longdouble) # Length 15

# Calculate midpoints for the first 15 bins
# For the first bin (<$10k), using (0 + 10000)/2 = 5000 as a representative midpoint.
# For other closed bins, use (lower + upper_approx)/2.
# A common approach for ACS bins is to use the textual description:
# e.g. $10,000 to $14,999 -> midpoint is ($10,000 + $14,999)/2 is not quite $12,500.
# Using (lower_bound_bin_i + lower_bound_bin_i+1)/2 is simpler for closed intervals before the top one.
midpoints_closed_bins = (standard_bin_lower_bounds[:-1] + standard_bin_lower_bounds[1:]) / 2.0 # Length 15

# For the last open-ended bin ($200,000+), use an estimated mean or representative value.
# Your original tail_est_income = 500e3 ($500,000) is a reasonable estimate for this.
representative_value_for_open_bin = np.longdouble(400000.0)

# Combine to form the 16 representative income values (midpoints or estimated mean) for the 16 income bins
xbin_midpoints_income_corrected = np.concatenate((midpoints_closed_bins, [representative_value_for_open_bin]))

# Create Series for calculations, ensuring it has 16 values matching income_vars
if len(xbin_midpoints_income_corrected) != len(income_vars):
    raise ValueError(f"Mismatch between number of income variables ({len(income_vars)}) and calculated midpoints ({len(xbin_midpoints_income_corrected)}).")

xbins_values = pd.Series(xbin_midpoints_income_corrected, index=income_vars).astype(np.longdouble)
# For log(E[Inc]) we need E[Inc] first (using xbins_values). 
# For E[log(Inc)], we need log(xbins_values):
comparison_data = np.log(xbins_values).astype(np.longdouble) # This is log(midpoint) for each bin

# === Helper Functions ===

def load_data(filename='data/census_data1.pkl'):
    """Load processed data from pickle file."""
    if not os.path.exists(filename):
        print(f"Error: Data file not found at {filename}")
        return None
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded data from {filename}")
        return data
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None

def pre_calculate_first_years(incomes_data, available_years, aggs):
    """Find the first year each unique unit appears in the data."""
    print("Pre-calculating first year of appearance for each unit...")
    unit_first_year_map = defaultdict(lambda: np.inf)
    
    for year in sorted(available_years):
        if year not in incomes_data:
            continue
        for agg in aggs:
            if agg not in incomes_data[year]:
                continue
                
            df_year_agg = incomes_data[year][agg]
            for unit_index in df_year_agg.index:
                 if isinstance(unit_index, tuple):
                     unit_key = (agg,) + unit_index
                 else:
                     unit_key = (agg, unit_index)
                 
                 current_first = unit_first_year_map[unit_key]
                 if year < current_first:
                     unit_first_year_map[unit_key] = year
                     
    print(f"Finished pre-calculation. Found first years for {len(unit_first_year_map)} unique unit-agg combinations.")
    return dict(unit_first_year_map)

def assign_population_column(incomes_data, available_years, aggs, pop_col_name):
    """Assigns a standardized 'population' column based on pop_col_name."""
    print(f"Assigning 'population' column from source '{pop_col_name}'...")
    years_processed_count = 0
    for year in available_years:
        if year not in incomes_data:
            continue
            
        aggs_processed_count = 0
        for agg in aggs:
            if agg not in incomes_data[year]:
                continue
                
            df = incomes_data[year][agg] # Modifies original dict
            if pop_col_name in df.columns:
                try:
                    numeric_pop = pd.to_numeric(df[pop_col_name], errors='coerce')
                    df['population'] = numeric_pop.astype(np.longdouble)
                except Exception as e:
                    print(f"    Error converting population column for {agg}, Year {year}: {e}")
                    df['population'] = np.nan
            else:
                df['population'] = np.nan
            aggs_processed_count += 1
            
        if aggs_processed_count > 0:
            years_processed_count += 1
            
    print(f"Finished assigning population column for {years_processed_count} years.")
    # Modifies incomes_data in place.

def calculate_log_avg_income(unit_row):
    """Calculates log(E[Income]) for a given data row (Series).
    
    Args:
        unit_row (pd.Series): A row containing income bin counts and 'population'.

    Returns:
        float: log(E[Income]), or np.nan if calculation fails.
    """
    try:
        # Ensure required columns and constants are available
        # Assuming income_vars and xbins_values are defined globally in this module
        if not all(v in unit_row.index for v in income_vars):
            # print(f"Warning: Missing some income_vars in row {unit_row.name}. Cannot calculate AvgLogInc.")
            return np.nan
            
        # Use total HOUSEHOLD count for normalization (sum of income bins)
        # If unit_row includes non-income columns, select only income_vars first
        household_counts = unit_row[income_vars].astype(np.longdouble)
        N_households = household_counts.sum()
        
        if N_households <= 0:
            return np.nan # Avoid division by zero
            
        # Calculate p_j (distribution of households across income bins)
        p_j = household_counts / N_households
        
        # Calculate E[Income] = Sum(p_j * x_j)
        # Ensure xbins_values aligns with p_j index (income_vars)
        expected_income = np.sum(p_j * xbins_values[income_vars])
        
        if expected_income <= 0:
            return np.nan # Avoid log(0) or log(negative)
            
        # Calculate log(E[Income])
        log_avg_income = np.log(expected_income)
        
        return log_avg_income.astype(float) # Cast back to standard float
        
    except KeyError as e:
        # print(f"KeyError calculating AvgLogInc for row {unit_row.name}: {e}")
        return np.nan
    except Exception as e:
        # print(f"Error calculating AvgLogInc for row {unit_row.name}: {e}")
        return np.nan
    
def calculate_avg_log_income(unit_row):
    """Calculates log(E[Income]) for a given data row (Series).
    
    Args:
        unit_row (pd.Series): A row containing income bin counts and 'population'.

    Returns:
        float: log(E[Income]), or np.nan if calculation fails.
    """
    try:
        # Ensure required columns and constants are available
        # Assuming income_vars and xbins_values are defined globally in this module
        if not all(v in unit_row.index for v in income_vars):
            # print(f"Warning: Missing some income_vars in row {unit_row.name}. Cannot calculate AvgLogInc.")
            return np.nan
            
        # Use total HOUSEHOLD count for normalization (sum of income bins)
        # If unit_row includes non-income columns, select only income_vars first
        household_counts = unit_row[income_vars].astype(np.longdouble)
        N_households = household_counts.sum()
        
        if N_households <= 0:
            return np.nan # Avoid division by zero
            
        # Calculate p_j (distribution of households across income bins)
        p_j = household_counts / N_households
        
        # Calculate E[Income] = Sum(p_j * x_j)
        # Ensure xbins_values aligns with p_j index (income_vars)
        expected_log_income = np.sum(p_j * np.log(xbins_values[income_vars]))
        
        if expected_log_income == np.nan:
            return np.nan # Avoid log(0) or log(negative)
            
        
        return expected_log_income.astype(float) # Cast back to standard float
        
    except KeyError as e:
        # print(f"KeyError calculating AvgLogInc for row {unit_row.name}: {e}")
        return np.nan
    except Exception as e:
        # print(f"Error calculating AvgLogInc for row {unit_row.name}: {e}")
        return np.nan

# Add other potential shared utilities here later, like:
# - calculate_distribution_covariance
# - calculate_frac_pop_change_covariance
# - get_unit_identifiers (for cleaner ParentState/County extraction)
# - etc. 