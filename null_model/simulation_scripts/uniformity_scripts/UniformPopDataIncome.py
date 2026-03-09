"""
Null Model: Uniform Population Distribution + Real Income Distribution

- Makes all block groups have the same total population
- But retains the original income distribution within each block group
- Then applies external and internal migration
"""

import pandas as pd
import numpy as np

# Pre-calculated migration parameters
total_internal_migration = 2055394
change_per_income_bin = [-77685, -63432, -71422, -46052, -59074, -47822, -44580, -36290, -42196, -51999, -71695, -22788, 75263, 89660, 169225, 291253]

# Income bins (in order matching change_per_income_bin)
income_vars = [
    "B19001_002E", "B19001_003E", "B19001_004E", "B19001_005E",
    "B19001_006E", "B19001_007E", "B19001_008E", "B19001_009E",
    "B19001_010E", "B19001_011E", "B19001_012E", "B19001_013E",
    "B19001_014E", "B19001_015E", "B19001_016E", "B19001_017E",
]

# Map changes to income vars
external_changes = dict(zip(income_vars, change_per_income_bin))


def make_population_uniform(df):
    """
    Make all block groups have the same total population while retaining
    each block group's original income distribution.
    
    Steps:
    1. For each BG, calculate fraction of pop in each income bin
    2. Calculate uniform pop per BG = total_city_pop / num_block_groups
    3. Redistribute: new_count[bin] = fraction[bin] * uniform_pop
    """
    print("\nMaking population uniform while retaining income distribution...")
    
    df_result = df.copy()
    
    total_city_pop = df['B19001_001E'].sum()
    n_block_groups = len(df)
    uniform_pop_per_bg = int(total_city_pop / n_block_groups)
    
    print(f"  Total city population: {total_city_pop:.0f}")
    print(f"  Number of block groups: {n_block_groups}")
    print(f"  Uniform pop per block group: {uniform_pop_per_bg}")
    
    for idx in df_result.index:
        bg_total_pop = df_result.loc[idx, 'B19001_001E']
        
        if bg_total_pop > 0:
            # Calculate fraction in each income bin
            for var in income_vars:
                original_count = df_result.loc[idx, var]
                fraction = original_count / bg_total_pop
                # Redistribute according to uniform population
                df_result.loc[idx, var] = int(fraction * uniform_pop_per_bg)
        else:
            # If block group has zero population, distribute uniformly
            pop_per_bin = int(uniform_pop_per_bg / len(income_vars))
            for var in income_vars:
                df_result.loc[idx, var] = pop_per_bin
        
        # Recalculate total
        df_result.loc[idx, 'B19001_001E'] = df_result.loc[idx, income_vars].sum()
    
    print(f"  Total population after redistribution: {df_result['B19001_001E'].sum():.0f}")
    return df_result


def apply_external_migration(df):
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
            for _ in range(abs(net_change)):
                eligible = df_result[df_result[income_var] >= 1].index.tolist()
                if not eligible:
                    print(f"    WARNING: No eligible block groups left for {income_var}")
                    break
                idx = np.random.choice(eligible)
                df_result.loc[idx, income_var] -= 1
                df_result.loc[idx, 'B19001_001E'] -= 1
        else:
            # Arrivals: add people
            for _ in range(net_change):
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
        dest_idx = np.random.choice(df_result.index)
        
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
    print("="*60)
    print("Null Model: Uniform Pop Distribution + Real Income")
    print("="*60)
    
    # Load real 2014 data
    print("\nLoading Null-Model-2014.csv...")
    df_2014 = pd.read_csv('null_model/null_results/Null-Model-2014.csv')
    print(f"  Loaded {len(df_2014)} block groups")
    print(f"  Total population: {df_2014['B19001_001E'].sum():.0f}")
    
    # Step 1: Make population uniform while keeping income distribution
    df_2014_uniform_pop = make_population_uniform(df_2014)
    
    # Step 2: Apply external migration
    df_after_external = apply_external_migration(df_2014_uniform_pop)
    
    # Step 3: Apply internal migration
    df_2019_null = apply_internal_migration(df_after_external, total_internal_migration)
    
    # Save results
    print("\n" + "="*60)
    print("Saving results")
    print("="*60)
    
    df_2014_uniform_pop.to_csv('null_model/null_results/UniformPop-DataIncome-2014.csv', index=False)
    print("✓ Saved 'null_model/null_results/UniformPop-DataIncome-2014.csv'")
    
    df_2019_null.to_csv('null_model/null_results/UniformPop-DataIncome-2019.csv', index=False)
    print("✓ Saved 'null_model/null_results/UniformPop-DataIncome-2019.csv'")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    
    return df_2014_uniform_pop, df_2019_null


if __name__ == "__main__":
    df_2014, df_2019 = main()
