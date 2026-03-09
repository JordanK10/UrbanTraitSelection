import requests
import pandas as pd
import numpy as np

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


def fetch_block_group_data(year):
    """
    Fetch block group data for a single year from the Census API
    
    Args:
        year: Year to fetch (14 for 2014, 19 for 2019)
    
    Returns:
        DataFrame: Raw block group level data (no processing)
    """
    print(f"\n{'='*60}")
    print(f"Fetching Block Group Data for Year 20{year}")
    print(f"{'='*60}")
    
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
    
    return df_combined


def identify_and_remove_problematic_block_groups(df, year_label):
    """
    Identify and remove block groups with sentinel values in average household size
    
    Checks for sentinel values (-666666666.0) in the B25010_001E (Average Household Size) column,
    which indicates suppressed/missing data from the Census Bureau.
    
    Args:
        df: DataFrame with block group data
        year_label: String label for the year (e.g., "2014" or "2019")
    
    Returns:
        tuple: (cleaned_df, problematic_geoids)
            - cleaned_df: DataFrame with problematic block groups removed
            - problematic_geoids: Set of GEOIDs that were removed
    """
    print(f"\n{'='*60}")
    print(f"Identifying Problematic Block Groups - {year_label}")
    print(f"{'='*60}")
    
    # Census sentinel value for suppressed/missing data
    sentinel_value = -666666666.0
    
    initial_count = len(df)
    print(f"Initial block groups: {initial_count}")
    
    # Check for sentinel values in Average Household Size
    print(f"\nChecking Average Household Size (B25010_001E) for sentinel values...")
    mask_sentinel = (df['B25010_001E'] == sentinel_value)
    problematic_geoids = set(df.loc[mask_sentinel, 'GEOID'].values)
    count_sentinel = len(problematic_geoids)
    
    print(f"  - Sentinel values found: {count_sentinel}")
    if count_sentinel > 0:
        print(f"  - Problematic GEOIDs: {sorted(problematic_geoids)}")
    
    # Remove problematic block groups
    df_cleaned = df[~mask_sentinel].copy()
    final_count = len(df_cleaned)
    
    print(f"\n{'='*60}")
    print(f"Summary - {year_label}")
    print(f"{'='*60}")
    print(f"Initial block groups: {initial_count}")
    print(f"Problematic block groups removed: {count_sentinel}")
    print(f"Remaining block groups: {final_count}")
    print(f"Percentage removed: {count_sentinel / initial_count * 100:.2f}%")
    
    return df_cleaned, problematic_geoids


def check_and_align_dataframes(df_2014, df_2019):
    """
    Check which block groups exist in one dataframe but not in the other,
    and remove mismatched block groups to ensure both dataframes have the same GEOIDs
    IN THE SAME ORDER.
    
    Args:
        df_2014: DataFrame with 2014 block group data
        df_2019: DataFrame with 2019 block group data
    
    Returns:
        tuple: (df_2014_aligned, df_2019_aligned, alignment_info)
            - df_2014_aligned: 2014 DataFrame with only matching GEOIDs, sorted by GEOID
            - df_2019_aligned: 2019 DataFrame with only matching GEOIDs, sorted by GEOID
            - alignment_info: Dict with alignment statistics
    """
    print(f"\n{'='*60}")
    print(f"Checking Dataframe Alignment")
    print(f"{'='*60}")
    
    # Get sets of GEOIDs from each dataframe
    geoids_2014 = set(df_2014['GEOID'].values)
    geoids_2019 = set(df_2019['GEOID'].values)
    
    # Find differences
    only_in_2014 = geoids_2014 - geoids_2019
    only_in_2019 = geoids_2019 - geoids_2014
    in_both = geoids_2014 & geoids_2019
    
    print(f"\nBlock groups in 2014: {len(geoids_2014)}")
    print(f"Block groups in 2019: {len(geoids_2019)}")
    print(f"Block groups in BOTH years: {len(in_both)}")
    
    print(f"\n{'='*60}")
    print(f"Mismatches Found")
    print(f"{'='*60}")
    
    print(f"\nBlock groups ONLY in 2014 (not in 2019): {len(only_in_2014)}")
    if len(only_in_2014) > 0:
        print(f"  GEOIDs: {sorted(only_in_2014)}")
    
    print(f"\nBlock groups ONLY in 2019 (not in 2014): {len(only_in_2019)}")
    if len(only_in_2019) > 0:
        print(f"  GEOIDs: {sorted(only_in_2019)}")
    
    print(f"\n{'='*60}")
    print(f"Alignment Summary - Before Removal")
    print(f"{'='*60}")
    total_unique = len(geoids_2014 | geoids_2019)
    print(f"Total unique block groups across both years: {total_unique}")
    print(f"Percentage aligned (in both): {len(in_both) / total_unique * 100:.2f}%")
    print(f"Percentage mismatched: {(len(only_in_2014) + len(only_in_2019)) / total_unique * 100:.2f}%")
    
    # Remove mismatched block groups from both dataframes
    print(f"\n{'='*60}")
    print(f"Removing Mismatched Block Groups and Sorting by GEOID")
    print(f"{'='*60}")
    
    # Keep only block groups that are in both years
    df_2014_aligned = df_2014[df_2014['GEOID'].isin(in_both)].copy()
    df_2019_aligned = df_2019[df_2019['GEOID'].isin(in_both)].copy()
    
    # CRITICAL FIX: Sort both dataframes by GEOID to ensure row-by-row alignment
    df_2014_aligned = df_2014_aligned.sort_values('GEOID').reset_index(drop=True)
    df_2019_aligned = df_2019_aligned.sort_values('GEOID').reset_index(drop=True)
    
    print(f"Removed from 2014: {len(only_in_2014)} block groups")
    print(f"Removed from 2019: {len(only_in_2019)} block groups")
    print(f"✓ Both dataframes sorted by GEOID")
    print(f"\nAligned 2014 DataFrame shape: {df_2014_aligned.shape}")
    print(f"Aligned 2019 DataFrame shape: {df_2019_aligned.shape}")
    
    # Verify alignment (both GEOIDs and order)
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
    
    alignment_info = {
        'only_in_2014': only_in_2014,
        'only_in_2019': only_in_2019,
        'in_both': in_both,
        'removed_from_2014': len(only_in_2014),
        'removed_from_2019': len(only_in_2019),
        'final_aligned_count': len(in_both)
    }
    
    return df_2014_aligned, df_2019_aligned, alignment_info


def convert_households_to_population(df, year_label):
    """
    Convert income bins from household counts to population counts
    
    For each income bin, multiply the household count by the average household size
    to get the population count. Then sum all income bin populations to get the
    total block group population and overwrite B19001_001E (Total Households becomes Total Population).
    
    Args:
        df: DataFrame with block group data
        year_label: String label for the year (e.g., "2014" or "2019")
    
    Returns:
        DataFrame: Modified DataFrame with population counts instead of household counts
    """
    print(f"\n{'='*60}")
    print(f"Converting Households to Population - {year_label}")
    print(f"{'='*60}")
    
    df_converted = df.copy()
    
    # Get average household size for each block group
    avg_household_size = df_converted['B25010_001E']
    
    print(f"Converting {len(income_vars)} income bins from household counts to population counts...")
    
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
    
    # Store old population for comparison
    old_population = df_converted['B19001_001E'].copy()
    
    # Overwrite B19001_001E with the new population calculation
    df_converted['B19001_001E'] = total_population
    
    print(f"✓ Total population calculated and B19001_001E overwritten")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Conversion Summary - {year_label}")
    print(f"{'='*60}")
    print(f"Total block groups: {len(df_converted)}")
    print(f"Old B19001_001E (households) - Mean: {old_population.mean():.2f}, Sum: {old_population.sum():.0f}")
    print(f"New B19001_001E (population) - Mean: {total_population.mean():.2f}, Sum: {total_population.sum():.0f}")
    print(f"Average household size - Mean: {avg_household_size.mean():.2f}")
    
    # Drop B01003_001E (Total Population) and B25010_001E (Average Household Size)
    # These are no longer needed after conversion
    print(f"\nDropping B01003_001E and B25010_001E columns...")
    df_converted = df_converted.drop(columns=['B01003_001E', 'B25010_001E'])
    print(f"✓ Columns dropped. Final shape: {df_converted.shape}")
    
    return df_converted


def apply_external_migration(df_2014, df_2019):
    """
    Apply external migration to df_2014 to simulate people leaving/entering Chicago
    
    For each income bin:
    1. Calculate net change: sum(2019_bin) - sum(2014_bin)
    2. If negative (departures): randomly remove people from that bin in df_2014
    3. If positive (arrivals): randomly add people to that bin in df_2014
    
    Args:
        df_2014: DataFrame with 2014 block group data
        df_2019: DataFrame with 2019 block group data
    
    Returns:
        DataFrame: Modified df_2014 with external migration applied
    """
    print(f"\n{'='*60}")
    print(f"Applying External Migration to 2014 Data")
    print(f"{'='*60}")
    
    df_migrated = df_2014.copy()
    
    print("\nCalculating net change per income bin...")
    print(f"{'Income Bin':<15} {'2014 Total':>12} {'2019 Total':>12} {'Net Change':>12} {'Type':>12}")
    print("-" * 65)
    
    # Store net changes for each income bin
    net_changes = {}
    
    for income_var in income_vars:
        total_2014 = df_2014[income_var].sum()
        total_2019 = df_2019[income_var].sum()
        net_change = total_2019 - total_2014
        net_changes[income_var] = int(net_change)
        
        change_type = "Arrivals" if net_change > 0 else "Departures" if net_change < 0 else "No change"
        print(f"{income_var:<15} {total_2014:>12.0f} {total_2019:>12.0f} {net_change:>+12.0f} {change_type:>12}")
    
    # Apply migration for each income bin
    print(f"\n{'='*60}")
    print(f"Applying Random Migration")
    print(f"{'='*60}")
    
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
                eligible_mask = df_migrated[income_var] >= 1
                eligible_indices = df_migrated[eligible_mask].index.tolist()
                
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
    print(f"\n{'='*60}")
    print(f"External Migration Summary")
    print(f"{'='*60}")
    print(f"Original 2014 total population: {df_2014['B19001_001E'].sum():.0f}")
    print(f"Modified 2014 total population: {df_migrated['B19001_001E'].sum():.0f}")
    print(f"Target 2019 total population: {df_2019['B19001_001E'].sum():.0f}")
    print(f"Difference: {df_migrated['B19001_001E'].sum() - df_2019['B19001_001E'].sum():.0f}")
    
    return df_migrated


def apply_internal_migration(df_2014, df_2019):
    """
    Apply internal migration within the city based on population churn
    
    Steps:
    1. Calculate population difference per block group (2019 - 2014)
    2. Calculate total churn: sum of absolute differences
    3. Number of people to move = total churn / 2
    4. Randomly move people from one block group to another (same income bin)
    
    Args:
        df_2014: DataFrame with 2014 block group data (after external migration)
        df_2019: DataFrame with 2019 block group data
    
    Returns:
        DataFrame: Modified df_2014 with internal migration applied
    """
    print(f"\n{'='*60}")
    print(f"Applying Internal Migration")
    print(f"{'='*60}")
    
    df_internal = df_2014.copy()
    
    # Calculate population difference per block group
    pop_2014 = df_2014['B19001_001E']
    pop_2019 = df_2019['B19001_001E']
    difference = pop_2019 - pop_2014
    
    # Calculate total churn
    total_churn = difference.abs().sum()
    num_people_to_move = int(total_churn / 2)
    
    print(f"\nPopulation Analysis:")
    print(f"  2014 total population: {pop_2014.sum():.0f}")
    print(f"  2019 total population: {pop_2019.sum():.0f}")
    print(f"  Total churn (sum of |differences|): {total_churn:.0f}")
    print(f"  Number of people to move internally: {num_people_to_move}")
    
    print(f"\n{'='*60}")
    print(f"Randomly Moving {num_people_to_move} People")
    print(f"{'='*60}")
    
    # Move people one at a time
    for i in range(num_people_to_move):
        # Step 1: Randomly select a source block group and income bin
        # Find block groups with at least 1 person in at least one income bin
        eligible_for_departure = False
        source_idx = None
        source_income_bin = None
        
        # Try to find a valid source
        max_attempts = 100
        for attempt in range(max_attempts):
            # Randomly select a source block group
            source_idx = np.random.choice(df_internal.index)
            
            # Randomly select an income bin
            source_income_bin = np.random.choice(income_vars)
            
            # Check if this block group has at least 1 person in this income bin
            if df_internal.loc[source_idx, source_income_bin] >= 1:
                eligible_for_departure = True
                break
        
        if not eligible_for_departure:
            print(f"  WARNING: Could not find eligible source at iteration {i+1}/{num_people_to_move}")
            break
        
        # Step 2: Randomly select a destination block group (can be any block group)
        dest_idx = np.random.choice(df_internal.index)
        
        # Step 3: Move 1 person from source to destination (same income bin)
        df_internal.loc[source_idx, source_income_bin] -= 1
        df_internal.loc[source_idx, 'B19001_001E'] -= 1
        
        df_internal.loc[dest_idx, source_income_bin] += 1
        df_internal.loc[dest_idx, 'B19001_001E'] += 1
        
        # Progress update
        if (i + 1) % 10000 == 0 or (i + 1) == num_people_to_move:
            print(f"  Progress: {i+1}/{num_people_to_move} people moved")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Internal Migration Summary")
    print(f"{'='*60}")
    print(f"Original 2014 total population: {df_2014['B19001_001E'].sum():.0f}")
    print(f"Modified 2014 total population: {df_internal['B19001_001E'].sum():.0f}")
    print(f"Target 2019 total population: {df_2019['B19001_001E'].sum():.0f}")
    print(f"People moved: {num_people_to_move}")
    
    # Verify total population unchanged
    if abs(df_internal['B19001_001E'].sum() - df_2014['B19001_001E'].sum()) < 0.1:
        print(f"✓ Total population preserved (internal migration only)")
    else:
        print(f"✗ WARNING: Total population changed!")
    
    return df_internal


def main():
    """
    Main function to retrieve 2014 and 2019 block group data and remove problematic entries
    """
    print("="*60)
    print("Null Model V2 - Block Group Data Retrieval")
    print("Years: 2014, 2019")
    print("="*60)
    
    # Fetch data for both years
    df_2014_raw = fetch_block_group_data(14)
    df_2019_raw = fetch_block_group_data(19)
    
    # Check if data was fetched successfully
    if df_2014_raw is None or df_2019_raw is None:
        print("\nERROR: Failed to fetch data. Terminating.")
        return None, None, None, None
    
    print("\n" + "="*60)
    print("Data Retrieval Complete")
    print("="*60)
    print(f"2014 Raw DataFrame shape: {df_2014_raw.shape}")
    print(f"2019 Raw DataFrame shape: {df_2019_raw.shape}")
    
    # Identify and remove problematic block groups
    df_2014_cleaned, problematic_2014 = identify_and_remove_problematic_block_groups(df_2014_raw, "2014")
    df_2019_cleaned, problematic_2019 = identify_and_remove_problematic_block_groups(df_2019_raw, "2019")
    
    print("\n" + "="*60)
    print("Cleaned Data Summary")
    print("="*60)
    print(f"2014 Cleaned DataFrame shape: {df_2014_cleaned.shape}")
    print(f"2019 Cleaned DataFrame shape: {df_2019_cleaned.shape}")
    
    # Check alignment and remove mismatched block groups
    df_2014_aligned, df_2019_aligned, alignment = check_and_align_dataframes(df_2014_cleaned, df_2019_cleaned)
    
    print("\n" + "="*60)
    print("Aligned Data Summary")
    print("="*60)
    print(f"2014 Aligned DataFrame shape: {df_2014_aligned.shape}")
    print(f"2019 Aligned DataFrame shape: {df_2019_aligned.shape}")
    
    # Convert household counts to population counts
    df_2014 = convert_households_to_population(df_2014_aligned, "2014")
    df_2019 = convert_households_to_population(df_2019_aligned, "2019")
    
    print("\n" + "="*60)
    print("Data Conversion Complete")
    print("="*60)
    print(f"2014 DataFrame shape: {df_2014.shape}")
    print(f"2019 DataFrame shape: {df_2019.shape}")
    print(f"✓ Income bins now contain POPULATION counts (not household counts)")
    print(f"✓ B19001_001E now contains TOTAL POPULATION (sum of income bins)") 
    
    # Save both 2014 and 2019 data before applying migrations
    print("\n" + "="*60)
    print("Saving Data (Before Migration) - For Null Model")
    print("="*60)
    
    df_2019.to_csv('null_model/simulation_results/migration_all_block_groups/Census-Data-2019.csv', index=False)
    print(f"✓ Saved df_2019 to 'null_model/simulation_results/migration_all_block_groups/Census-Data-2019.csv'")
    print(f"  This file contains 2019 data with population counts")
    
    print(f"\nThese files can be used as input for loadNullModelData.py")
    
    # Apply external migration to 2014 data
    df_2014_external = apply_external_migration(df_2014, df_2019)
    
    # Apply internal migration to 2014 data
    df_2019_simulated = apply_internal_migration(df_2014_external, df_2019)
    
    print("\n" + "="*60)
    print("Final Data - Null Model Complete")
    print("="*60)
    print(f"Original 2014 DataFrame shape: {df_2014.shape}")
    print(f"After External Migration shape: {df_2014_external.shape}")
    print(f"After Internal Migration shape: {df_2019_simulated.shape}")
    print(f"2019 DataFrame shape: {df_2019.shape}")
    print(f"\n✓ External migration applied (income bin changes)")
    print(f"✓ Internal migration applied (random movement within city)")
    
    # Save dataframes to CSV files
    print("\n" + "="*60)
    print("Saving DataFrames to CSV")
    print("="*60)
    
    df_2014.to_csv('null_model/simulation_results/migration_all_block_groups/Census-Data-2014.csv', index=False)
    print(f"✓ Saved df_2014 to 'null_model/simulation_results/migration_all_block_groups/Census-Data-2014.csv'")
    
    df_2019_simulated.to_csv('null_model/simulation_results/migration_all_block_groups/Simulated-Data-2019.csv', index=False)
    print(f"✓ Saved df_2019_simulated to 'null_model/simulation_results/migration_all_block_groups/Simulated-Data-2019.csv'")
    
    return df_2014, df_2014_external, df_2019_simulated, df_2019, problematic_2014, problematic_2019, alignment


if __name__ == "__main__":
    df_2014, df_2014_external, df_2019_simulated, df_2019, problems_2014, problems_2019, alignment = main()

