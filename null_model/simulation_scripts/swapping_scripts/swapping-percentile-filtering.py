#!/usr/bin/env python3
"""
Migration Null Model with Income Percentile Filtering
Fetches Chicago metro area census data, cleans it, converts households to population,
filters out population above a specified income percentile threshold,
and applies external and internal migration null model.
"""

import requests
import pandas as pd
import numpy as np

#sentinel value
sentinel_value = -666666666.0

# Set the API key
api_key = "35d314060d56f894db2f7621b0e5e5f7eca9af27"

# Counties in the Chicago metro area
counties = ["031", "043", "089", "093", "097", "111", "197"]
states = ["17", "17", "17", "17", "17", "17", "17"]
cty_name = ["Cook", "DuPg", "Kane", "Kndl", "Lke", "McHn", "Will"]

# Income bracket variables
income_vars = [
    "B19001_002E",  # Households with income less than $10,000
    "B19001_003E",  # Households with income between $10,000 and $14,999
    "B19001_004E",  # Households with income between $15,000 and $19,999
    "B19001_005E",  # Households with income between $20,000 and $24,999
    "B19001_006E",  # Households with income between $25,000 and $29,999
    "B19001_007E",  # Households with income between $30,000 and $34,999
    "B19001_008E",  # Households with income between $35,000 and $39,999
    "B19001_009E",  # Households with income between $40,000 and $44,999
    "B19001_010E",  # Households with income between $45,000 and $49,999
    "B19001_011E",  # Households with income between $50,000 and $59,999
    "B19001_012E",  # Households with income between $60,000 and $74,999
    "B19001_013E",  # Households with income between $75,000 and $99,999
    "B19001_014E",  # Households with income between $100,000 and $124,999
    "B19001_015E",  # Households with income between $125,000 and $149,999
    "B19001_016E",  # Households with income between $150,000 and $199,999
    "B19001_017E",  # Households with income $200,000 or more
]

# Population variables
pop_vars = [
    "B01003_001E",  # Total Population
    "B19001_001E",  # Total Households
    "B25010_001E"   # Average Household Size
]

# All variables to fetch
all_vars = income_vars + pop_vars

# ============================================================================
# INCOME PERCENTILE FILTERING CONFIGURATION
# ============================================================================

# Percentile threshold - remove population ABOVE this percentile
# E.g., 90 means keep bottom 90%, remove top 10%
INCOME_PERCENTILE_THRESHOLD = 80.0

# Midpoint income for each bin (in dollars) - used for percentile calculation
INCOME_BIN_MIDPOINTS = [
    5000,      # < $10k
    12500,     # $10k - $15k
    17500,     # $15k - $20k
    22500,     # $20k - $25k
    27500,     # $25k - $30k
    32500,     # $30k - $35k
    37500,     # $35k - $40k
    42500,     # $40k - $45k
    47500,     # $45k - $50k
    55000,     # $50k - $60k
    67500,     # $60k - $75k
    87500,     # $75k - $100k
    112500,    # $100k - $125k
    137500,    # $125k - $150k
    175000,    # $150k - $200k
    250000,    # $200k+ (estimate)
]

def fetch_data(year):
    
    print(f"\nData retrieval for 20{year} commencing...")
    print("-"*60)
    base_url = f"https://api.census.gov/data/20{year}/acs/acs5"
    all_dfs = []
    
    for state, county, cty in zip(states, counties, cty_name):
        print(f"Fetching {cty} County (State: {state}, County: {county})...", end=" ")
        # Construct the API URL for block group data
        bg_geography = f"block%20group:*&in=state:{state}&in=county:{county}"
        bg_url = f"{base_url}?get={','.join(all_vars)}&for={bg_geography}&key={api_key}"
        
        try:
            response = requests.get(bg_url, timeout=60)
            response.raise_for_status()
            data = response.json()
            if len(data) > 1:
                columns = data[0]
                data_rows = data[1:]
                df = pd.DataFrame(data_rows, columns=columns)
                print(f"✓ {len(df)} block groups")
                all_dfs.append(df)
            else:
                print(f"✗ No data")
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Error: {e}")
    
    if not all_dfs:
        print("\nERROR: No data fetched for any county")
        return None
    
    # Combine all county dataframes
    print(f"\nCombining data from all counties...")
    df_combined = pd.concat(all_dfs, ignore_index=True)
    
    # Create GEOID for block groups (state + county + tract + block group)
    df_combined['GEOID'] = (
        df_combined['state'] + 
        df_combined['county'] + 
        df_combined['tract'] + 
        df_combined['block group']
    )
    
    # Convert all numeric columns to numeric types
    print("Converting columns to numeric types...")
    for col in all_vars:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
    
    print(f"\n✓ Total block groups fetched: {len(df_combined)}")
    print(f"✓ Columns: {list(df_combined.columns)}")
    
    print("\n")
    print("*" * 60)
    print(f"Data retrieval for 20{year} successful!!")
    print("*" * 60)
    
    return df_combined    

def clean_dataframe(df_2014, df_2019):
    print(f"\nPreparing the dataframes for the migration null model.")
    print(f"First, we remove rows with sentinel values in the average household size column.")
    print("-"*60)
    
    # Remove rows with sentinel values in the average household size column for 2014
    print(f"\nInitial number of block groups in 2014 dataframe: {len(df_2014)}")
    df_2014_filtered = df_2014[df_2014['B25010_001E'] >= 0]
    print(f"Number of block groups in 2014 dataframe after removing sentinel value rows: {len(df_2014_filtered)}")
    
    # Remove rows with sentinel values in the average household size column for 2019
    print(f"Initial number of block groups in 2019 dataframe: {len(df_2019)}")
    df_2019_filtered = df_2019[df_2019['B25010_001E'] >= 0]
    print(f"Number of block groups in 2019 dataframe after removing sentinel value rows: {len(df_2019_filtered)}")

    print(f"\nNext, we remove block groups that are not present in both years.")
    print("-"*60)

    only_in_2014 = set(df_2014_filtered['GEOID'].values) - set(df_2019_filtered['GEOID'].values)
    only_in_2019 = set(df_2019_filtered['GEOID'].values) - set(df_2014_filtered['GEOID'].values)
    in_both = set(df_2014_filtered['GEOID'].values) & set(df_2019_filtered['GEOID'].values)

    print(f"Number of block groups that are only found in the 2014: {len(only_in_2014)}")
    print(f"Number of block groups that are only found in the 2019: {len(only_in_2019)}")

    # Keep only block groups that are in both years
    df_2014_aligned = df_2014_filtered[df_2014_filtered['GEOID'].isin(in_both)]
    df_2019_aligned = df_2019_filtered[df_2019_filtered['GEOID'].isin(in_both)]
    
    # CRITICAL FIX: Sort both dataframes by GEOID to ensure row-by-row alignment
    df_2014_aligned = df_2014_aligned.sort_values('GEOID').reset_index(drop=True)
    df_2019_aligned = df_2019_aligned.sort_values('GEOID').reset_index(drop=True)

    final_geoids_2014 = df_2014_aligned['GEOID'].values
    final_geoids_2019 = df_2019_aligned['GEOID'].values
    
    if set(final_geoids_2014) == set(final_geoids_2019):
        print(f"\n✓ SUCCESS: Both dataframes have identical GEOIDs ({len(final_geoids_2014)} block groups)")
        
        # Check if order matches too
        if np.array_equal(final_geoids_2014, final_geoids_2019):
            print(f"✓ SUCCESS: GEOIDs are in the SAME ORDER (row-by-row alignment verified)")
        else:
            print(f"✗ ERROR: GEOIDs match but are in DIFFERENT ORDER!")
    else:
        print(f"\n✗ ERROR: Dataframes still misaligned!")
    
    return df_2014_aligned, df_2019_aligned

def convert_households_to_population(df):
    print("-"*60)
    print(f"\nConverting Households to Population")
    print("-"*60)
    
    df_converted = df.copy()
    
    # Get average household size for each block group
    avg_household_size = df_converted['B25010_001E']
    # Convert each income bin from households to population
    for income_var in income_vars:
        df_converted[income_var] = df_converted[income_var] * avg_household_size
    print(f"✓ Income bins converted to population counts")
    # Round down income bins to remove decimals (floor operation)
    print(f"Rounding down income bins to integers...")
    for income_var in income_vars:
        df_converted[income_var] = np.floor(df_converted[income_var])
    print(f"✓ Income bins rounded down to integers")
    # Calculate total population by summing all income bin populations
    print(f"Calculating total population by summing income bins...")
    total_population = df_converted[income_vars].sum(axis=1)
        
    # Overwrite B19001_001E with the new population calculation
    df_converted['B19001_001E'] = total_population
    
    print(f"✓ Total population calculated and B19001_001E overwritten")
    
    # Drop B01003_001E (Total Population) and B25010_001E (Average Household Size)
    # These are no longer needed after conversion
    print(f"\nDropping B01003_001E and B25010_001E columns...")
    df_converted = df_converted.drop(columns=['B01003_001E', 'B25010_001E'])
    print(f"✓ Columns dropped. Final shape: {df_converted.shape}")
    
    return df_converted


def calculate_global_percentile_threshold(df, income_vars, bin_midpoints, percentile):
    """
    Calculate the income value at a given percentile across the entire metro area.
    
    Args:
        df: DataFrame with income bin columns
        income_vars: List of income variable column names
        bin_midpoints: List of income midpoints for each bin
        percentile: Target percentile (0-100)
    
    Returns:
        tuple: (threshold_income, threshold_bin_index, fraction_of_bin)
    """
    print(f"\n{'='*60}")
    print(f"Calculating Global {percentile}th Percentile Threshold")
    print(f"{'='*60}")
    
    # Aggregate population across all block groups for each income bin
    global_bin_populations = [df[var].sum() for var in income_vars]
    total_population = sum(global_bin_populations)
    
    print(f"Total metro area population: {total_population:,.0f}")
    
    # Calculate cumulative population and percentages
    cumulative_pop = 0
    cumulative_percentages = []
    
    for i, pop in enumerate(global_bin_populations):
        cumulative_pop += pop
        cumulative_pct = (cumulative_pop / total_population) * 100
        cumulative_percentages.append(cumulative_pct)
        print(f"  Bin {i+1} (${bin_midpoints[i]:,}): "
              f"{pop:>10,.0f} people ({cumulative_pct:>6.2f}% cumulative)")
    
    # Find which bin contains the threshold
    target_population = (percentile / 100.0) * total_population
    
    threshold_bin_index = None
    fraction_of_bin = 0.0
    
    cumulative = 0
    for i, pop in enumerate(global_bin_populations):
        if cumulative + pop >= target_population:
            threshold_bin_index = i
            # Calculate what fraction of this bin we need
            population_needed = target_population - cumulative
            fraction_of_bin = population_needed / pop if pop > 0 else 0.0
            break
        cumulative += pop
    
    if threshold_bin_index is None:
        threshold_bin_index = len(income_vars) - 1
        fraction_of_bin = 1.0
    
    threshold_income = bin_midpoints[threshold_bin_index]
    
    print(f"\n✓ {percentile}th Percentile Results:")
    print(f"  Threshold bin: {threshold_bin_index + 1} (income ≈ ${threshold_income:,})")
    print(f"  Fraction of threshold bin to keep: {fraction_of_bin:.2%}")
    print(f"  Target population: {target_population:,.0f}")
    
    return threshold_income, threshold_bin_index, fraction_of_bin


def filter_dataframe_by_percentile(df, income_vars, threshold_bin_index, fraction_of_bin, year_label=""):
    """
    Filter the dataframe to remove population above the income threshold.
    
    Args:
        df: DataFrame with income bin columns
        income_vars: List of income variable column names
        threshold_bin_index: Index of the bin containing the threshold
        fraction_of_bin: Fraction of the threshold bin to keep (0-1)
        year_label: Label for printing (e.g., "2014" or "2019")
    
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    print(f"\n{'='*60}")
    print(f"Filtering {year_label} Block Groups")
    print(f"{'='*60}")
    
    df_filtered = df.copy()
    
    original_total = df[income_vars].sum().sum()
    
    # Apply filtering to each block group
    for idx in df_filtered.index:
        # Keep all bins below threshold completely
        # Partially keep threshold bin
        # Remove all bins above threshold
        
        for i, var in enumerate(income_vars):
            if i < threshold_bin_index:
                # Keep this bin fully
                pass
            elif i == threshold_bin_index:
                # Keep only a fraction of this bin
                df_filtered.at[idx, var] = df_filtered.at[idx, var] * fraction_of_bin
            else:
                # Remove this bin entirely
                df_filtered.at[idx, var] = 0
    
    # Recalculate total
    df_filtered['B19001_001E'] = df_filtered[income_vars].sum(axis=1)
    
    filtered_total = df_filtered[income_vars].sum().sum()
    removed_total = original_total - filtered_total
    
    print(f"Original population: {original_total:,.0f}")
    print(f"Filtered population: {filtered_total:,.0f}")
    print(f"Removed population: {removed_total:,.0f} ({(removed_total/original_total)*100:.2f}%)")
    print(f"Block groups retained: {len(df_filtered)} (keeping all to maintain alignment)")
    
    return df_filtered


def apply_external_migration(df_2014, df_2019):
    print(f"Applying External Migration to 2014 Data")
    print(f"{'-'*60}")

    df_migrated = df_2014.copy()
    # Store net changes for each income bin
    net_changes = {}
    
    for income_var in income_vars:
        total_2014 = df_2014[income_var].sum()
        total_2019 = df_2019[income_var].sum()
        net_change = total_2019 - total_2014
        net_changes[income_var] = int(net_change)
    
    # Apply migration for each income bin
    for income_var in income_vars:
        net_change = net_changes[income_var]
        if net_change == 0:
            print(f"\n{income_var}: No migration needed (net change = 0)")
            continue
        
        if net_change < 0:
            # Departures: remove people
            num_departures = abs(net_change)
            print(f"\n{income_var}: Removing {num_departures} people (departures)...")
            
            for i in range(num_departures):
                # Find block groups with at least 1 person in this income bin
                eligible_indices = df_migrated[df_migrated[income_var] >= 1].index.tolist()
                
                if not eligible_indices:
                    print(f"  WARNING: No eligible block groups left at iteration {i+1}/{num_departures}")
                    break
                
                # Randomly select a block group
                selected_idx = np.random.choice(eligible_indices)
                
                # Remove 1 person from this income bin
                df_migrated.loc[selected_idx, income_var] -= 1
                df_migrated.loc[selected_idx, 'B19001_001E'] -= 1
                
                if (i + 1) % 1000 == 0 or (i + 1) == num_departures:
                    print(f"  Progress: {i+1}/{num_departures} departures applied")
        
        else:
            # Arrivals: add people
            num_arrivals = net_change
            print(f"\n{income_var}: Adding {num_arrivals} people (arrivals)...")
            
            for i in range(num_arrivals):
                # Randomly select any block group
                selected_idx = np.random.choice(df_migrated.index)
                
                # Add 1 person to this income bin
                df_migrated.loc[selected_idx, income_var] += 1
                df_migrated.loc[selected_idx, 'B19001_001E'] += 1
                
                if (i + 1) % 1000 == 0 or (i + 1) == num_arrivals:
                    print(f"  Progress: {i+1}/{num_arrivals} arrivals applied")
    
    # Summary
    print(f"\nExternal Migration Summary")
    print(f"{'-'*60}")
    print(f"Original 2014 total population: {df_2014['B19001_001E'].sum():.0f}")
    print(f"Modified 2014 total population: {df_migrated['B19001_001E'].sum():.0f}")
    print(f"Target 2019 total population: {df_2019['B19001_001E'].sum():.0f}")
    print(f"Difference: {df_migrated['B19001_001E'].sum() - df_2019['B19001_001E'].sum():.0f}")
    
    return df_migrated

def apply_internal_migration_swap(df_2014, df_2019):
    """
    Apply internal migration using SWAPS instead of random moves.
    
    A swap:
    1. Pick BG1 and income bin INC1 (with at least 1 person)
    2. Pick BG2 (different from BG1) and income bin INC2 (with at least 1 person)
    3. Move person from BG1[INC1] to BG2[INC1] - person keeps their income
    4. Move person from BG2[INC2] to BG1[INC2] - person keeps their income
    
    This preserves each block group's total population while shuffling income distributions.
    """
    print(f"\n{'='*60}")
    print("Applying Internal Migration (SWAP-BASED)")
    print(f"{'='*60}")
    
    df = df_2014.copy()
    
    # Calculate churn
    pop_2014 = df_2014['B19001_001E']
    pop_2019 = df_2019['B19001_001E']
    total_churn = (pop_2019 - pop_2014).abs().sum()
    num_swaps = int(total_churn / 2)
    
    print(f"\n  Total churn: {total_churn:.0f}")
    print(f"  Number of swaps to perform: {num_swaps}")
    
    print(f"\n{'='*60}")
    print(f"Performing {num_swaps} Swaps")
    print(f"{'='*60}")
    
    indices = df.index.tolist()
    
    for i in range(num_swaps):
        # Find BG1 with at least 1 person in some income bin
        bg1_idx = None
        inc1 = None
        for _ in range(100):
            bg1_idx = np.random.choice(indices)
            inc1 = np.random.choice(income_vars)
            if df.loc[bg1_idx, inc1] >= 1:
                break
        else:
            continue
        
        # Find BG2 (different from BG1) with at least 1 person in some income bin
        bg2_idx = None
        inc2 = None
        for _ in range(100):
            bg2_idx = np.random.choice(indices)
            if bg2_idx == bg1_idx:
                continue
            inc2 = np.random.choice(income_vars)
            if df.loc[bg2_idx, inc2] >= 1:
                break
        else:
            continue
        
        # Perform the swap:
        # Person from BG1[INC1] moves to BG2, keeping income INC1
        df.loc[bg1_idx, inc1] -= 1
        df.loc[bg2_idx, inc1] += 1
        
        # Person from BG2[INC2] moves to BG1, keeping income INC2
        df.loc[bg2_idx, inc2] -= 1
        df.loc[bg1_idx, inc2] += 1
        
        # Note: Total population per block group is preserved!
        # BG1: -1 (from inc1) + 1 (from inc2) = 0 net change
        # BG2: +1 (from inc1) - 1 (from inc2) = 0 net change
        
        if (i + 1) % 50000 == 0:
            print(f"  Progress: {i+1}/{num_swaps} swaps")
    
    print(f"\n{'='*60}")
    print("Swap Migration Summary")
    print(f"{'='*60}")
    print(f"  Original population: {df_2014['B19001_001E'].sum():.0f}")
    print(f"  Final population: {df['B19001_001E'].sum():.0f}")
    print(f"  Swaps performed: {num_swaps}")
    
    # Verify each block group's population is preserved
    pop_diff = (df['B19001_001E'] - df_2014['B19001_001E']).abs().sum()
    if pop_diff < 0.1:
        print(f"✓ Block group populations preserved (swaps only)")
    else:
        print(f"✗ WARNING: Block group populations changed by {pop_diff:.0f}")
    
    return df


def main():
    """
    Main execution function.
    Fetches data, cleans it, converts households to population, 
    filters by income percentile, and applies migration null model.
    """
    print("*" * 60)
    print("Running 'Migration with Percentile Filter' null model.")
    print("*" * 60)

    # Step 1: Fetch census data for 2014 and 2019
    df_2014_raw = fetch_data(14)
    df_2019_raw = fetch_data(19)

    if df_2014_raw is None or df_2019_raw is None:
        print("\nEither 2014 or 2019 data was unretrieved. Terminating program...")
        return None
    
    # Step 2: Clean dataframes (align GEOIDs, remove sentinel values)
    df_2014_clean, df_2019_clean = clean_dataframe(df_2014_raw, df_2019_raw)
    
    # Step 3: Convert households to population
    df_2014_converted = convert_households_to_population(df_2014_clean)
    df_2019_converted = convert_households_to_population(df_2019_clean)

    # Step 4: Apply income percentile filtering
    print("\n" + "="*60)
    print(f"APPLYING INCOME PERCENTILE FILTER")
    print(f"Threshold: Keep bottom {INCOME_PERCENTILE_THRESHOLD}%, remove top {100-INCOME_PERCENTILE_THRESHOLD}%")
    print("="*60)
    
    # Calculate global percentile threshold for 2014 data
    print("\n" + "="*60)
    print("YEAR 2014")
    print("="*60)
    threshold_income_2014, threshold_bin_index_2014, fraction_of_bin_2014 = calculate_global_percentile_threshold(
        df_2014_converted, income_vars, INCOME_BIN_MIDPOINTS, INCOME_PERCENTILE_THRESHOLD
    )
    
    # Calculate global percentile threshold for 2019 data
    print("\n" + "="*60)
    print("YEAR 2019")
    print("="*60)
    threshold_income_2019, threshold_bin_index_2019, fraction_of_bin_2019 = calculate_global_percentile_threshold(
        df_2019_converted, income_vars, INCOME_BIN_MIDPOINTS, INCOME_PERCENTILE_THRESHOLD
    )
    
    # Apply respective thresholds to each dataframe
    df_2014_filtered = filter_dataframe_by_percentile(
        df_2014_converted, income_vars, threshold_bin_index_2014, fraction_of_bin_2014, year_label="2014"
    )
    
    df_2019_filtered = filter_dataframe_by_percentile(
        df_2019_converted, income_vars, threshold_bin_index_2019, fraction_of_bin_2019, year_label="2019"
    )

    # Step 5: Apply external migration to filtered data
    print("\n" + "="*60)
    print("APPLYING MIGRATION NULL MODEL TO FILTERED DATA")
    print("="*60)
    
    df_2014_external = apply_external_migration(df_2014_filtered, df_2019_filtered)
    
    # Step 6: Apply internal migration
    df_2019_simulated = apply_internal_migration_swap(df_2014_external, df_2019_filtered)

    # Save dataframes to CSV files
    print("\n" + "="*60)
    print("Saving DataFrames to CSV")
    print("="*60)
    
    df_2014_filtered.to_csv('null_model/simulation_results/swapping_percentile/Census-Data-2014_80.csv', index=False)
    print(f"✓ Saved df_2014_filtered to 'null_model/simulation_results/swapping_percentile/Census-Data-2014_80.csv'")
    
    df_2019_simulated.to_csv('null_model/simulation_results/swapping_percentile/Simulated-Data-2019_80.csv', index=False)
    print(f"✓ Saved df_2019_simulated to 'null_model/simulation_results/swapping_percentile/Simulated-Data-2019_80.csv'")

    # Return final results
    print("\n" + "="*60)
    print("✓ Swapping percentile null model complete!")
    print("="*60)
    print(f"2014 DataFrame shape: {df_2014_filtered.shape}")
    print(f"2019 Simulated DataFrame shape: {df_2019_simulated.shape}")
    print(f"2014 Total population: {df_2014_filtered['B19001_001E'].sum():.0f}")
    print(f"2019 Simulated population: {df_2019_simulated['B19001_001E'].sum():.0f}")
    
    return df_2014_filtered, df_2019_simulated

if __name__ == "__main__": 
    df_2014, df_2019 = main()
