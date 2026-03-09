"""
Null Model with Uniform Initial Population Distribution

Calculates migration parameters from census data and creates a null model
starting from a uniform population distribution.
"""

import pandas as pd
import numpy as np

# Income bins
income_vars = [
    "B19001_002E", "B19001_003E", "B19001_004E", "B19001_005E",
    "B19001_006E", "B19001_007E", "B19001_008E", "B19001_009E",
    "B19001_010E", "B19001_011E", "B19001_012E", "B19001_013E",
    "B19001_014E", "B19001_015E", "B19001_016E", "B19001_017E",
]

def clean_dataframe(df_2014, df_2019):
    print("-"*60)
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

    print("-"*60)
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


def calculate_migration_parameters(df_2014, df_2019):
    """
    Calculate migration parameters from census data.
    
    Returns:
        external_changes: dict mapping income vars to net external change
        total_internal_migration: total number of internal moves
    """
    print("\nCalculating migration parameters from census data...")
    
    # Ensure both dataframes have the same block groups
    # Match by GEOID
    df_2014_sorted = df_2014.sort_values('GEOID').reset_index(drop=True)
    df_2019_sorted = df_2019.sort_values('GEOID').reset_index(drop=True)
    
    # Verify we have matching block groups
    if not df_2014_sorted['GEOID'].equals(df_2019_sorted['GEOID']):
        print("  WARNING: Block groups don't match perfectly. Using intersection.")
        common_geoids = set(df_2014['GEOID']) & set(df_2019['GEOID'])
        df_2014_sorted = df_2014[df_2014['GEOID'].isin(common_geoids)].sort_values('GEOID').reset_index(drop=True)
        df_2019_sorted = df_2019[df_2019['GEOID'].isin(common_geoids)].sort_values('GEOID').reset_index(drop=True)
    
    print(f"  Matched {len(df_2014_sorted)} block groups")
    
    # Calculate external migration (net change per income bin across entire region)
    external_changes = {}
    change_per_income_bin = []
    
    for var in income_vars:
        total_2014 = df_2014_sorted[var].sum()
        total_2019 = df_2019_sorted[var].sum()
        net_change = total_2019 - total_2014
        external_changes[var] = net_change
        change_per_income_bin.append(net_change)
        print(f"  {var}: {net_change:+.0f} (external migration)")
    

    # Calculate population difference per block group
    pop_2014 = df_2014_sorted['B19001_001E']
    pop_2019 = df_2019_sorted['B19001_001E']
    difference = pop_2019 - pop_2014
    
    # Calculate total churn
    total_churn = difference.abs().sum()
    num_people_to_move = int(total_churn / 2)
    
    print(f"\n  Total external migration (net): {sum(change_per_income_bin):+.0f}")
    print(f"  Total internal migration (moves): {num_people_to_move:.0f}")
    
    return external_changes, num_people_to_move, change_per_income_bin


def create_uniform_distribution(df_template):
    """Create uniform population distribution using template for block groups"""
    print("\nCreating uniform 2014 distribution...")
    
    total_pop = df_template['B19001_001E'].sum()
    n_bg = len(df_template)
    pop_per_bg = int(total_pop / n_bg)
    pop_per_bin = int(pop_per_bg / len(income_vars))
    
    print(f"  Total population: {total_pop:.0f}")
    print(f"  Block groups: {n_bg}")
    print(f"  Pop per block group: {pop_per_bg}")
    print(f"  Pop per income bin: {pop_per_bin}")
    
    df = df_template.copy()
    
    # Set uniform population in each income bin
    for var in income_vars:
        df[var] = pop_per_bin
    
    # Recalculate total
    df['B19001_001E'] = df[income_vars].sum(axis=1)
    
    print(f"  Actual total: {df['B19001_001E'].sum():.0f}")
    return df


def apply_external_migration(df, external_changes):
    """Apply external migration: add/remove people per income bin randomly"""
    print("\nApplying external migration...")
    
    df_result = df.copy()
    
    for income_var, net_change in external_changes.items():
        if net_change == 0:
            continue
        
        change_type = "removing" if net_change < 0 else "adding"
        print(f"  {income_var}: {change_type} {abs(net_change)}")
        
        if net_change < 0:
            # Departures: remove people
            for _ in range(int(abs(net_change))):
                eligible = df_result[df_result[income_var] >= 1].index.tolist()
                if not eligible:
                    print(f"    WARNING: No eligible block groups left for {income_var}")
                    break
                idx = np.random.choice(eligible)
                df_result.loc[idx, income_var] -= 1
                df_result.loc[idx, 'B19001_001E'] -= 1
        else:
            # Arrivals: add people
            for _ in range(int(net_change)):
                idx = np.random.choice(df_result.index)
                df_result.loc[idx, income_var] += 1
                df_result.loc[idx, 'B19001_001E'] += 1
    
    print(f"  Total population after external migration: {df_result['B19001_001E'].sum():.0f}")
    return df_result


def apply_internal_migration(df, num_moves):
    """Randomly move people between block groups (same income bin)"""
    print(f"\nApplying internal migration ({num_moves} moves)...")
    
    df_result = df.copy()
    
    for i in range(num_moves):
        # Find source with at least 1 person in some income bin
        for _ in range(100):
            source_idx = np.random.choice(df_result.index)
            income_bin = np.random.choice(income_vars)
            if df_result.loc[source_idx, income_bin] >= 1:
                break
        else:
            continue
        
        # Random destination
        for _ in range(100): 
            dest_idx = np.random.choice(df_result.index)
            if dest_idx != source_idx:
                break 
        
        # Move 1 person (same income bin)
        df_result.loc[source_idx, income_bin] -= 1
        df_result.loc[source_idx, 'B19001_001E'] -= 1
        df_result.loc[dest_idx, income_bin] += 1
        df_result.loc[dest_idx, 'B19001_001E'] += 1
        
        if (i + 1) % 100000 == 0:
            print(f"  Progress: {i+1}/{num_moves}")
    
    print(f"  Done. Total population: {df_result['B19001_001E'].sum():.0f}")
    return df_result


def main():
    
    # Load census data
    print("\nLoading census data...")
    df_2014 = pd.read_csv('null_model/Census-Data-2014.csv')
    df_2019 = pd.read_csv('null_model/Census-Data-2019.csv')
    print(f"  Loaded 2014 data: {len(df_2014)} block groups, {df_2014['B19001_001E'].sum():.0f} total population")
    print(f"  Loaded 2019 data: {len(df_2019)} block groups, {df_2019['B19001_001E'].sum():.0f} total population")
    

    df_2014_clean, df_2019_clean = clean_dataframe(df_2014, df_2019)
    df_2014_converted = convert_households_to_population(df_2014_clean)
    df_2019_converted = convert_households_to_population(df_2019_clean)

    # Calculate migration parameters from the data
    external_changes, total_internal_migration, change_per_income_bin = calculate_migration_parameters(df_2014_converted, df_2019_converted)
    
    # Step 1: Create uniform 2014 distribution (use 2014 as template)
    df_2014_uniform = create_uniform_distribution(df_2014)
    
    # Step 2: Apply external migration
    df_after_external = apply_external_migration(df_2014_uniform, external_changes)
    
    # Step 3: Apply internal migration
    df_2019_null = apply_internal_migration(df_after_external, total_internal_migration)
    
    # Save results
    print("\n" + "="*60)
    print("Saving results")
    print("="*60)
    
    df_2014_uniform.to_csv('null_model/simulation_results/uniform/Uniform-Model-2014.csv', index=False)
    print("✓ Saved 'null_model/simulation_results/uniform/Uniform-Model-2014.csv'")
    
    df_2019_null.to_csv('null_model/simulation_results/uniform/Uniform-Model-2019.csv', index=False)
    print("✓ Saved 'null_model/simulation_results/uniform/Uniform-Model-2019.csv'")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    
    return df_2014_uniform, df_2019_null


if __name__ == "__main__":
    df_2014, df_2019 = main()