"""
Null Model V2 with Swap-Based Internal Migration

Same as migration-all-block-groups.py except internal migration uses swaps instead of random moves.
A swap exchanges people between two different block groups, preserving each block group's total population.
"""

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
    "B19001_002E", "B19001_003E", "B19001_004E", "B19001_005E",
    "B19001_006E", "B19001_007E", "B19001_008E", "B19001_009E",
    "B19001_010E", "B19001_011E", "B19001_012E", "B19001_013E",
    "B19001_014E", "B19001_015E", "B19001_016E", "B19001_017E",
]

# Population variables
pop_vars = ["B01003_001E", "B19001_001E", "B25010_001E"]

# All variables to fetch
all_vars = income_vars + pop_vars


def fetch_block_group_data(year):
    """Fetch block group data for a single year from the Census API"""
    print(f"\n{'='*60}")
    print(f"Fetching Block Group Data for Year 20{year}")
    print(f"{'='*60}")
    
    base_url = f"https://api.census.gov/data/20{year}/acs/acs5"
    all_dfs = []
    
    for state, county, cty in zip(states, counties, cty_name):
        print(f"Fetching {cty} County...", end=" ")
        bg_url = f"{base_url}?get={','.join(all_vars)}&for=block%20group:*&in=state:{state}&in=county:{county}&key={api_key}"
        
        try:
            response = requests.get(bg_url, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
                print(f"✓ {len(df)} block groups")
                all_dfs.append(df)
            else:
                print(f"✗ No data")
        except requests.exceptions.RequestException as e:
            print(f"✗ Error: {e}")
    
    if not all_dfs:
        return None
    
    df = pd.concat(all_dfs, ignore_index=True)
    df['GEOID'] = df['state'] + df['county'] + df['tract'] + df['block group']
    
    for col in all_vars:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"\n✓ Total block groups fetched: {len(df)}")
    return df


def clean_and_align_data(df_2014, df_2019):
    """Remove problematic block groups and align both dataframes"""
    print(f"\n{'='*60}")
    print("Cleaning and Aligning Data")
    print(f"{'='*60}")
    
    # Remove sentinel values
    sentinel = -666666666.0
    df_2014 = df_2014[df_2014['B25010_001E'] != sentinel].copy()
    df_2019 = df_2019[df_2019['B25010_001E'] != sentinel].copy()
    
    # Align block groups
    common_geoids = set(df_2014['GEOID']) & set(df_2019['GEOID'])
    df_2014 = df_2014[df_2014['GEOID'].isin(common_geoids)].copy()
    df_2019 = df_2019[df_2019['GEOID'].isin(common_geoids)].copy()
    
    # CRITICAL FIX: Sort both dataframes by GEOID to ensure row-by-row alignment
    df_2014 = df_2014.sort_values('GEOID').reset_index(drop=True)
    df_2019 = df_2019.sort_values('GEOID').reset_index(drop=True)

    final_geoids_2014 = df_2014['GEOID'].values
    final_geoids_2019 = df_2019['GEOID'].values
    
    if set(final_geoids_2014) == set(final_geoids_2019):
        print(f"\n✓ SUCCESS: Both dataframes have identical GEOIDs ({len(final_geoids_2014)} block groups)")
        
        # Check if order matches too
        if np.array_equal(final_geoids_2014, final_geoids_2019):
            print(f"✓ SUCCESS: GEOIDs are in the SAME ORDER (row-by-row alignment verified)")
        else:
            print(f"✗ ERROR: GEOIDs match but are in DIFFERENT ORDER!")
    else:
        print(f"\n✗ ERROR: Dataframes still misaligned!")

    print(f"  Common block groups: {len(common_geoids)}")
    return df_2014, df_2019


def convert_households_to_population(df, year_label):
    """Convert income bins from household counts to population counts"""
    print(f"\nConverting households to population - {year_label}...")
    
    df = df.copy()
    avg_hh_size = df['B25010_001E']
    
    for var in income_vars:
        df[var] = np.floor(df[var] * avg_hh_size)
    
    df['B19001_001E'] = df[income_vars].sum(axis=1)
    df = df.drop(columns=['B01003_001E', 'B25010_001E'])
    
    print(f"  Total population: {df['B19001_001E'].sum():.0f}")
    return df


def apply_external_migration(df_2014, df_2019):
    """Apply external migration to match income bin totals"""
    print(f"\n{'='*60}")
    print("Applying External Migration")
    print(f"{'='*60}")
    
    df = df_2014.copy()
    
    for var in income_vars:
        total_2014 = df_2014[var].sum()
        total_2019 = df_2019[var].sum()
        net_change = int(total_2019 - total_2014)
        
        if net_change == 0:
            continue
        
        if net_change < 0:
            # Remove people
            for _ in range(abs(net_change)):
                eligible = df[df[var] >= 1].index.tolist()
                if not eligible:
                    break
                idx = np.random.choice(eligible)
                df.loc[idx, var] -= 1
                df.loc[idx, 'B19001_001E'] -= 1
        else:
            # Add people
            for _ in range(net_change):
                idx = np.random.choice(df.index)
                df.loc[idx, var] += 1
                df.loc[idx, 'B19001_001E'] += 1
    
    print(f"  Population after external migration: {df['B19001_001E'].sum():.0f}")
    return df


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
    print("="*60)
    print("Null Model V2 - SWAP-BASED Internal Migration")
    print("="*60)
    
    # Fetch data
    df_2014_raw = fetch_block_group_data(14)
    df_2019_raw = fetch_block_group_data(19)
    
    if df_2014_raw is None or df_2019_raw is None:
        print("ERROR: Failed to fetch data")
        return None
    
    # Clean and align
    df_2014_clean, df_2019_clean = clean_and_align_data(df_2014_raw, df_2019_raw)
    
    # Convert to population
    df_2014 = convert_households_to_population(df_2014_clean, "2014")
    df_2019 = convert_households_to_population(df_2019_clean, "2019")
    
    # Apply external migration
    df_2014_external = apply_external_migration(df_2014, df_2019)
    
    # Apply swap-based internal migration
    df_2014_final = apply_internal_migration_swap(df_2014_external, df_2019)
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving Results")
    print(f"{'='*60}")
    
    df_2014.to_csv('null_model/simulation_results/swapping_all_block_groups/Census-Data-2014.csv', index=False)
    print("✓ Saved 'null_model/simulation_results/swapping_all_block_groups/Census-Data-2014.csv'")
    
    df_2014_final.to_csv('null_model/simulation_results/swapping_all_block_groups/Simulated-Data-2019.csv', index=False)
    print("✓ Saved 'null_model/simulation_results/swapping_all_block_groups/Simulated-Data-2019.csv'")
    
    return df_2014, df_2014_final, df_2019


if __name__ == "__main__":
    results = main()

