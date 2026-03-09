"""
Fetch ACS Block Group Data with Community Assignment

This script:
1. Retrieves block group data from ACS Census API
2. Adds a community column based on tract-to-community mapping
3. Provides a function to filter block groups by community membership
"""

import requests
import pandas as pd
import numpy as np
import sys

# Census API key
api_key = "35d314060d56f894db2f7621b0e5e5f7eca9af27"

# Counties in the Chicago metro area
counties = ["031", "043", "089", "093", "097", "111", "197"]
states = ["17", "17", "17", "17", "17", "17", "17"]
cty_name = ["Cook", "DuPg", "Kane", "Kndl", "Lke", "McHn", "Will"]
COUNTY_FIPS_TO_NAME_MAP = dict(zip(counties, cty_name))

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

def load_community_mapping():
    """
    Load community mapping data from CSV files.
    
    Returns:
        DataFrame with tract -> community mapping
    """
    print("Loading community mapping data...")
    
    communities = pd.concat([
        pd.read_csv('matched_chicago_data.csv'),
        pd.read_csv('matched_chicagoLand_data.csv')
    ], axis=0)

    communities['GEOID'] = communities['GEOID'].astype(str)
    communities = communities.rename(columns={'GEOID': 'tract_geoid'})
    communities = communities[~communities['community'].isna()]
    communities['community'] = communities['community'].str.upper()

    # Extract tract and county from GEOID (format: SSCCCTTTTTT)
    # State: first 2 chars, County: next 3, Tract: last 6
    communities['state'] = communities['tract_geoid'].str[:2]
    communities['county'] = communities['tract_geoid'].str[2:5]
    communities['tract'] = communities['tract_geoid'].str[5:]
    
    # Create county-specific community names
    communities['community'] = communities['community'] + '_' + communities['county']

    print(f"  Loaded {len(communities)} tract-to-community mappings")
    return communities


def fetch_block_group_data(year):
    """
    Fetch block group data for a single year from the Census API.
    
    Args:
        year: Year to fetch (e.g., 14 for 2014, 19 for 2019)
    
    Returns:
        DataFrame: Raw block group level data
    """
    print(f"\n{'='*60}")
    print(f"Fetching Block Group Data for Year 20{year}")
    print(f"{'='*60}")
    
    base_url = f"https://api.census.gov/data/20{year}/acs/acs5"
    all_dfs = []
    
    for state, county, cty in zip(states, counties, cty_name):
        print(f"Fetching {cty} County (State: {state}, County: {county})...", end=" ")
        
        bg_geography = f"block%20group:*&in=state:{state}&in=county:{county}"
        bg_url = f"{base_url}?get={','.join(all_vars)}&for={bg_geography}&key={api_key}"
        
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
        print("\nERROR: No data fetched for any county")
        return None
    
    # Combine all county dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Create GEOID for block groups (state + county + tract + block group)
    df['GEOID'] = df['state'] + df['county'] + df['tract'] + df['block group']
    
    # Convert numeric columns
    for col in all_vars:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"\n✓ Total block groups fetched: {len(df)}")

    return df


def add_community_column(df, community_mapping):
    """
    Add community column to block group dataframe based on tract-to-community mapping.
    
    Args:
        df: Block group DataFrame with 'state', 'county', 'tract' columns
        community_mapping: DataFrame with tract -> community mapping
    
    Returns:
        DataFrame with 'community' column added
    """
    print("\nAdding community column to block groups...")
    
    df = df.copy()
    
    # Ensure columns are strings and stripped
    for col in ['state', 'county', 'tract']:
        df[col] = df[col].astype(str).str.strip()
    
    # Prepare community mapping for merge
    comms = community_mapping[['state', 'county', 'tract', 'community']].copy()
    comms['state'] = comms['state'].astype(str).str.strip()
    comms['county'] = comms['county'].astype(str).str.strip()
    comms['tract'] = comms['tract'].astype(str).str.strip()

    comms = comms.drop_duplicates(subset=['state', 'county', 'tract'], keep='first')
    # Merge on state, county, tract
    df = pd.merge(
        df,
        comms[['state', 'county', 'tract', 'community']],
        on=['state', 'county', 'tract'],
        how='left'
    )

    # Fill missing communities with county name
    missing_mask = df['community'].isna()
    if missing_mask.any():
        print(f"  Found {missing_mask.sum()} block groups without community assignment.")
        print(f"  Filling with county name.")
        fill_values = df.loc[missing_mask, 'county'].map(COUNTY_FIPS_TO_NAME_MAP)
        df.loc[missing_mask, 'community'] = fill_values.str.upper() + '_' + df.loc[missing_mask, 'county']
    
    # Uppercase for consistency
    df['community'] = df['community'].str.upper()
    
    print(f"  ✓ Community column added. Unique communities: {df['community'].nunique()}")
    return df


def filter_by_communities(df, communities_to_include=None, communities_to_exclude=None):
    """
    Filter block groups based on community membership.
    
    Args:
        df: DataFrame with 'community' column
        communities_to_include: List of community names to include (if None, include all)
        communities_to_exclude: List of community names to exclude (if None, exclude none)
    
    Returns:
        DataFrame: Filtered block groups
    
    Example:
        # Include only specific communities
        df_filtered = filter_by_communities(df, communities_to_include=['LOOP_031', 'LINCOLN PARK_031'])
        
        # Exclude specific communities
        df_filtered = filter_by_communities(df, communities_to_exclude=['COOK', 'DUPAGE'])
        
        # Both include and exclude
        df_filtered = filter_by_communities(df, 
            communities_to_include=['LOOP_031', 'LINCOLN PARK_031', 'LAKE VIEW_031'],
            communities_to_exclude=['LAKE VIEW_031'])
    """
    print("\nFiltering block groups by community...")
    
    df_filtered = df.copy()
    original_count = len(df_filtered)
    
    # Apply inclusion filter
    if communities_to_include is not None:
        # Normalize input
        communities_to_include = [c.upper() for c in communities_to_include]
        df_filtered = df_filtered[df_filtered['community'].isin(communities_to_include)]
        print(f"  Applied inclusion filter: {len(communities_to_include)} communities")
    
    # Apply exclusion filter
    if communities_to_exclude is not None:
        # Normalize input
        communities_to_exclude = [c.upper() for c in communities_to_exclude]
        df_filtered = df_filtered[~df_filtered['community'].isin(communities_to_exclude)]
        print(f"  Applied exclusion filter: {len(communities_to_exclude)} communities")
    
    print(f"  Block groups: {original_count} → {len(df_filtered)}")
    return df_filtered


def get_unique_communities(df):
    """
    Get list of unique communities in the dataframe.
    
    Args:
        df: DataFrame with 'community' column
    
    Returns:
        List of unique community names sorted alphabetically
    """

    return sorted(df['community'].unique().tolist())

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

def main():
    print("="*60)
    print("Fetch ACS Block Group Data with Community Assignment")
    print("="*60)
    
    # Load community mapping
    community_mapping = load_community_mapping()
    
    # Fetch block group data for 2019
    df_2019 = fetch_block_group_data(19)
    
    if df_2019 is None:
        print("ERROR: Failed to fetch data")
        return None
    
    # Add community column
    df_2019 = add_community_column(df_2019, community_mapping)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total block groups: {len(df_2019)}")
    print(f"Unique communities: {df_2019['community'].nunique()}")
    print(f"\nSample of communities:")
    for comm in get_unique_communities(df_2019)[:10]:
        count = len(df_2019[df_2019['community'] == comm])
        print(f"  {comm}: {count} block groups")
    
    # Example: Filter to specific communities
    print(f"\n{'='*60}")
    print("Example: Filtering by Community")
    print(f"{'='*60}")
    
    # Get all Cook County communities (those ending with _031)
    communities_to_include = ['Burnside_031', 'Oakland_031']
    print(f"Found {len(communities_to_include)} communities to include.")
    
    # Filter to only Cook County block groups
    df_filtered = filter_by_communities(df_2019, communities_to_include)

    df_converted = convert_households_to_population(df_filtered)
    print(f"Number of block groups selected according to your criteria: {len(df_converted)}")
    df_converted.drop('community', axis=1, inplace=True)
    df_converted.to_csv('null_model/simulation_scripts/subset_scripts/2019-2-Communities.csv', index=False)
    print(f"✓ Saved 'df_converted' to 'null_model/simulation_scripts/subset_scripts/2019-2-Communities.csv'")

    return df_converted


if __name__ == "__main__":
    result = main()

