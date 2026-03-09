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

def apply_internal_migration(df_2014, df_2019):
    print(f"{'-'*60}")
    print(f"Applying Internal Migration")
    print(f"{'-'*60}")
    
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
        for attempt in range(max_attempts): 
            dest_idx = np.random.choice(df_internal.index)
            if dest_idx != source_idx:
                break 
        
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
    # Step 1: Perform an API call to retrieve census data for the years 2014 and 2019 as dataframes
    # Step 2: Clean up 2014 dataframe and 2019 dataframe. This means performing the following actions
    # Step 2a: Making sure all the GEOIDs presents in the 2014 dataframe are present in the 2019 dataframe 
    # and visa versa 
    # Step 2b: Removing all GEOIDs where there are no people living (Avg Household size = -666.66666)
    # Step 2c: Check that the 2014 and 2019 dataframes are indexed by GEOID and are in 
    #          the same GEOID 
    # Step 3: Convert the dataframe from having a household count per income bin to having a population count
    # Step 4: Apply external (out-of-city) migration 
    # Step 5: Apply internal (between block groups) migration
    print("*" * 60)
    print("Running 'Migration' null model on all Chicago block groups.")
    print("*" * 60)

    df_2014_raw = fetch_data(14)
    df_2019_raw = fetch_data(19)

    if df_2014_raw is None or df_2019_raw is None:
        print("\nEither 2014 or 2019 data was unretrieved. Terminating program...")
        return None
    
    df_2014_clean, df_2019_clean = clean_dataframe(df_2014_raw, df_2019_raw)
    df_2014_converted = convert_households_to_population(df_2014_clean)
    df_2019_converted = convert_households_to_population(df_2019_clean)

    df_2014_external = apply_external_migration(df_2014_converted, df_2019_converted)
    df_2019_simulated = apply_internal_migration(df_2014_external, df_2019_converted)

    # Save dataframes to CSV files
    print("\n" + "="*60)
    print("Saving DataFrames to CSV")
    print("="*60)
    
    df_2014_converted.to_csv('null_model/simulation_results/migration_all_block_groups/Census-Data-2014.csv', index=False)
    print(f"✓ Saved df_2014 to 'null_model/simulation_results/migration_all_block_groups/Census-Data-2014.csv'")
    
    df_2019_simulated.to_csv('null_model/simulation_results/migration_all_block_groups/Simulated-Data-2019.csv', index=False)
    print(f"✓ Saved df_2019_simulated to 'null_model/simulation_results/migration_all_block_groups/Simulated-Data-2019.csv'")

if __name__ == "__main__": 
    main()
