from operator import index
from typing import Any
import pandas as pd
import numpy as np
import random
from collections import Counter
from itertools import product  


# Income bins
income_vars = [
    "B19001_002E", "B19001_003E", "B19001_004E", "B19001_005E",
    "B19001_006E", "B19001_007E", "B19001_008E", "B19001_009E",
    "B19001_010E", "B19001_011E", "B19001_012E", "B19001_013E",
    "B19001_014E", "B19001_015E", "B19001_016E", "B19001_017E",
]

# The subset of income bins included in this simulation. 
# Most income bins are not included to simplify the migration null model.
included_income_vars = [
    "B19001_002E",  # <$10k        → $5,000
    "B19001_005E",  # $20k-$25k    → $22,500
    "B19001_007E",  # $30k-$35k    → $32,500
    "B19001_009E",  # $40k-$45k    → $42,500
    "B19001_011E",  # $50k-$60k    → $55,000
    "B19001_012E",  # $60k-$75k    → $67,500
    "B19001_014E",  # $100k-$125k  → $112,500
    "B19001_017E",  # $200k+       → $400,000
]

# Internal migration is done in bulk in order to speed up 
# the null model. 
number_of_internal_moves_per_iteration = 100

def calculate_external_migration_parameters(df_2014, df_2019):
    """
    Calculate migration parameters from census data.
    Returns:
        external_changes: dict mapping income vars to net external change
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
    
    for var in included_income_vars:
        total_2014 = df_2014_sorted[var].sum()
        total_2019 = df_2019_sorted[var].sum()
        net_change = total_2019 - total_2014
        external_changes[var] = net_change
        change_per_income_bin.append(net_change)
        print(f"  {var}: {net_change:+.0f} (external migration)")
    
    print(f"\n  Total external migration (net): {sum(change_per_income_bin):+.0f}")
    return external_changes, change_per_income_bin


def calculate_internal_migration_parameters(df_2014, df_2019):
    # Calculate population difference per block group
    pop_2014 = df_2014['B19001_001E']
    pop_2019 = df_2019['B19001_001E']
    difference = pop_2019 - pop_2014

    # Calculate total churn
    total_churn = difference.abs().sum()
    num_people_to_move = int(total_churn / 2)
    
    print(f"  Total internal migration (moves): {num_people_to_move:.0f}")
    return num_people_to_move


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
            # How many people to remove? 
            total_count = int(abs(net_change))
            # How many persons can we remove from each income bin? 
            max_counts = dict(zip(range(0, len(df_result)), df_result[income_var].tolist()))            # Build a sampling pool of people
            pool = []
            for num, max_count in max_counts.items():
                pool.extend([num] * int(max_count))
            # Check if we can remove the amount of people we want to remove
            if total_count > len(pool):
                raise ValueError("Cannot generate that many numbers respecting the max counts")

            # Randomly choose a 'total_count' number of people from pool
            random_numbers = random.sample(pool, total_count) 
            random_numbers.sort()
            
            # Generate a dictionary of block group index : how much to remove from block group 
            cnt = Counter(random_numbers)
            sorted_cnt = {k: cnt[k] for k in sorted(cnt)}

            print("Number of people removed from each block group is:")
            print(sorted_cnt)

            # Put the external migration numbers in a list to subtract from dataframe
            migration_pattern = [0] * len(df_result)

            for idx, count in sorted_cnt.items():
                migration_pattern[idx] = count

            df_result[income_var] -= migration_pattern

        else:
            # Arrivals: add people
            for _ in range(int(net_change)):
                idx = np.random.choice(df_result.index)
                df_result.loc[idx, income_var] += 1
    
    df_result['B19001_001E'] = df_result[included_income_vars].sum(axis=1)
    print(f"  Total population after external migration: {df_result['B19001_001E'].sum():.0f}")
    return df_result


def apply_internal_migration(df, num_moves):
    """Randomly move people between block groups (same income bin)"""
    print(f"\nApplying internal migration ({num_moves} moves)...")
    
    # Extract out the portions of the dataframe that include the income bins ONLY
    df_result = df.copy()
    number_of_moves_performed = 0
    
    print(f"Population before internal migration {df_result[included_income_vars].values.sum()}")
    locations = list(product(list(range(0,len(df_result))),list(range(0,len(df_result[included_income_vars].columns)))))
    
    # Makes sure that the number_of_internal_momves_per_iteration is still doable for this round
    while num_moves - number_of_moves_performed > number_of_internal_moves_per_iteration:
        # Generate the pool from which we draw people to move 
        # Needs to be redone every time we perform a movements because the population distribution changes
        #index_list = list(range(0,len(df_result)))
        #incomes_list = list(range(0,len(df_result.columns)))
        pool = []
        for i, j in locations:
            pool.extend([(i,j)] * int(df_result.iloc[i,j]))


        # The end goal, sorted_net_migration_cnt tells us how much to remove or add to each income bin 
        # and income group in the dataframe. 
        departure_bg = random.sample(pool, number_of_internal_moves_per_iteration) 
        arrival_bg = random.choices(locations, k = number_of_internal_moves_per_iteration)
        
        departure_cnt = Counter(departure_bg)
        arrival_cnt = Counter(arrival_bg)
        net_migration_cnt = Counter(arrival_cnt)  # Copy arrivals
        net_migration_cnt.subtract(departure_cnt)  # Subtract in-place (keeps negatives!)

        # Perform the movement of people 
        for i, j in net_migration_cnt:
            df_result.iloc[i,j] += net_migration_cnt[(i,j)]        

        number_of_moves_performed += number_of_internal_moves_per_iteration

    final_number_of_moves = num_moves % number_of_moves_performed

    if final_number_of_moves > 0:
        pool = []
        for i, j in locations:
            pool.extend([(i,j)] * int(df_result.iloc[i,j]))
        # The end goal, sorted_net_migration_cnt tells us how much to remove or add to each income bin 
        # and income group in the dataframe. 
        departure_bg = random.sample(pool, final_number_of_moves) 
        arrival_bg = random.choices(locations, k = final_number_of_moves)
        
        departure_cnt = Counter(departure_bg)
        arrival_cnt = Counter(arrival_bg)
        net_migration_cnt = Counter(arrival_cnt)  # Copy arrivals
        net_migration_cnt.subtract(departure_cnt)  # Subtract in-place (keeps negatives!)

        # Perform the movement of people 
        for i, j in net_migration_cnt:
            df_result.iloc[i,j] += net_migration_cnt[(i,j)]        
 
    df_result['B19001_001E'] = df_result[included_income_vars].sum(axis=1)
    print(f"Population after internal migration {df_result[included_income_vars].values.sum()}")
    print(f"Population check: {df_result['B19001_001E'].sum()}")
    return df_result
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
    df_2014 = pd.read_csv('null_model/simulation_scripts/subset_scripts/2014-2-Communities.csv', dtype = {'state': str, 'county': str, 'tract': str, 'block group': str, 'GEOID': str})
    df_2019 = pd.read_csv('null_model/simulation_scripts/subset_scripts/2019-2-Communities.csv', dtype = {'state': str, 'county': str, 'tract': str, 'block group': str, 'GEOID': str})
    print("\nCensus data loaded... ✓")

    # Right after loading data
    df_2014 = df_2014.sort_values('GEOID').reset_index(drop=True)
    df_2019 = df_2019.sort_values('GEOID').reset_index(drop=True)

    print("\nOverwriting population values accroding to 'included_income_vars'...")
    df_2014['B19001_001E'] = df_2014[included_income_vars].sum(axis=1)
    df_2019['B19001_001E'] = df_2019[included_income_vars].sum(axis=1)
    print("\nPopulaton Overwritten... ✓")

    print("\nRestricting dataframe to include only 'included_income_vars'...")
    df_2014 = df_2014[included_income_vars + ['B19001_001E','state', 'county', 'tract', 'block group', 'GEOID']]
    df_2019 = df_2019[included_income_vars + ['B19001_001E','state', 'county', 'tract', 'block group', 'GEOID']]
    print("\nSubset of dataframe taken... ✓")

    print(f"\nLoaded 2014 data: {len(df_2014)} block groups, {df_2014['B19001_001E'].sum():.0f} total population")
    print(f"\nLoaded 2019 data: {len(df_2019)} block groups, {df_2019['B19001_001E'].sum():.0f} total population")

    external_changes, change_per_income_bin = calculate_external_migration_parameters(df_2014, df_2019)

    print("\nLook below to check if the dataframes are ordered by GEOID:\n")
    print(df_2014)
    print(df_2019)
    print('\nThe external migration parameters are:\n')
    print(external_changes)
    print(change_per_income_bin)

    # CONTINUE FILLING FROM HERE
    print('=' * 60)
    print('\n Now, we apply external_migration')
    df_2014_external = apply_external_migration(df_2014, external_changes)
    print('\n External migration completed')
    print('This is how the dataframes looks after external migration has been applied')
    print(df_2014)
    print(df_2014_external)
    print('=' * 60)




    num_people_to_move = calculate_internal_migration_parameters(df_2014_external, df_2019)




    print('=' * 60)
    print('\n Now, we apply internal migration')
    df_2019_simulated = apply_internal_migration(df_2014_external, num_people_to_move)
    print("This is how the dataframes looks after internal migration has been applied")
    print(df_2014)
    print(df_2019_simulated)

    df_2014.to_csv('null_model/simulation_results/migration_check/Census-Data-2014.csv', index=False)
    print(f"✓ Saved df_2014 to 'null_model/simulation_results/migration_check/Census-Data-2014.csv'")
    
    df_2019_simulated.to_csv('null_model/simulation_results/migration_check/Simulated-Data-2019.csv', index=False)
    print(f"✓ Saved df_2019_simulated to 'null_model/simulation_results/migration_check/Simulated-Data-2019.csv'")

    return df_2014, df_2019_simulated

if __name__ == "__main__":
    df_2014, df_2019 = main()