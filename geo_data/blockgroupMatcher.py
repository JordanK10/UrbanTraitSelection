# blockgroupMatcher.py
import pandas as pd
import argparse
import pickle
import os

# --- Configuration ---
# !!! PLEASE CONFIRM OR CHANGE THIS PATH !!!
# This is a placeholder for the actual data file outputted by saveCensusDataV2.py
DEFAULT_CENSUS_DATA_FILE = "data/census_data1.pkl" 

def load_data(filepath):
    """Loads data from a pickle file."""
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        return None
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded data from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

def find_blockgroups_in_df_for_year(year_df_bg, target_tract_id, current_year):
    """
    Filters a single year's block group DataFrame for a target tract ID and prints its block groups.
    """
    if not isinstance(year_df_bg, pd.DataFrame):
        print(f"Error for year {current_year}: Expected a DataFrame for block group data, got {type(year_df_bg)}.")
        return False # Indicate failure for this year

    if 'TempTractID' not in year_df_bg.columns:
        print(f"Error for year {current_year}: DataFrame must contain 'TempTractID' column to find block groups.")
        return False # Indicate failure

    # Ensure TempTractID is string for manipulation
    # Create a copy to avoid SettingWithCopyWarning if year_df_bg is a slice
    df_processed = year_df_bg.copy()
    df_processed['TempTractID_str'] = df_processed['TempTractID'].astype(str)

    # Derive TractID from BG TempTractID. BG TempTractID is typically 12 chars, Tract is 11.
    df_processed['DerivedTractID'] = df_processed['TempTractID_str'].str[:11]

    # Filter for the target tract
    tract_data_for_year = df_processed[df_processed['DerivedTractID'] == target_tract_id]

    if tract_data_for_year.empty:
        # This is not an error, just means no BGs for this tract in this specific year's BG data.
        # print(f"  Year: {current_year} - No block groups found for Tract ID: {target_tract_id}")
        return False # Indicate no BGs found for this year, but not an error in processing

    print(f"\n  Year: {current_year}")
    block_groups = sorted(tract_data_for_year['TempTractID_str'].unique()) # Sort for consistent output
    found_any_for_this_year = False
    if block_groups:
        for bg_id in block_groups:
            if bg_id[:11] == target_tract_id: # Double check derived tract ID
                print(f"    - {bg_id}")
                found_any_for_this_year = True
    
    if not found_any_for_this_year:
        # This case handles if tract_data_for_year was not empty initially, but no BGs passed the final check.
        # Typically, if tract_data_for_year is not empty, found_any_for_this_year should be true.
        print("    No block groups found for this year in the filtered tract data after final check.")
        return False
        
    return True # Indicate BGs were found and printed for this year

def main():
    parser = argparse.ArgumentParser(
        description="Find all block groups within a given TempTractID for each year, "
                    "reading data from the output of saveCensusDataV2.py."
    )

    parser.add_argument(
        "--data_file",
        default=DEFAULT_CENSUS_DATA_FILE,
        help=f"Path to the data file outputted by saveCensusDataV2.py. "
             f"Default: {DEFAULT_CENSUS_DATA_FILE}."
    )

    args = parser.parse_args()

    census_data_loaded = load_data(args.data_file)

    print(f"Attempting to load data from: {args.data_file}")
    for year in range(13, 24):
        d = census_data_loaded[year]['bg']
        print(year)
        input(d[d['temp_tract_id'] == '17031280400']['B19001_001E'])
    if census_data_loaded is None:
        print("Exiting due to data loading failure.")
        return

    if not isinstance(census_data_loaded, dict):
        print("Error: Loaded data is not a dictionary as expected (keyed by year). Please check the input file structure.")
        return

    print(f"\nBlock groups belonging to Tract ID: {args.temp_tract_id}")
    overall_found_any_bgs = False

    # Iterate through years (assuming keys are integer/string years)
    # Sorting years for consistent output order if keys are not already sorted (e.g. strings '10', '11', ... '22', '9')
    try:
        sorted_year_keys = sorted(census_data_loaded.keys(), key=lambda x: int(str(x)))
    except ValueError:
        print("Warning: Could not sort year keys numerically, using default order.")
        sorted_year_keys = census_data_loaded.keys()

    for year_key in sorted_year_keys:
        year_data_dict = census_data_loaded.get(year_key)

        if not isinstance(year_data_dict, dict):
            print(f"Warning: Data for year '{year_key}' is not a dictionary. Skipping this year.")
            continue

        df_bg_for_year = year_data_dict.get('bg')
        input(df_bg_for_year)
        if df_bg_for_year is None:
            # print(f"  Year: {year_key} - No 'bg' (block group) data found.")
            continue
        
        if not isinstance(df_bg_for_year, pd.DataFrame):
            print(f"Warning for year '{year_key}': 'bg' data is not a DataFrame. Skipping.")
            continue
            
        if df_bg_for_year.empty:
            # print(f"  Year: {year_key} - 'bg' DataFrame is empty.")
            continue

        # Call the function to find and print blockgroups for this specific year's BG DataFrame
        found_bgs_this_year = find_blockgroups_in_df_for_year(df_bg_for_year, args.temp_tract_id, year_key)
        if found_bgs_this_year:
            overall_found_any_bgs = True

    if not overall_found_any_bgs:
        print(f"No block groups found for Tract ID: {args.temp_tract_id} in any year or in 'bg' data across all processed years.")

if __name__ == "__main__":
    main() 