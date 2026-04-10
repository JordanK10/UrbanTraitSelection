import requests
import pandas as pd
import numpy as np

# Sentinel value and API key
sentinel_value = -666666666.0
api_key = "35d314060d56f894db2f7621b0e5e5f7eca9af27"

# Chicago metro area counties
counties = ["031", "043", "089", "093", "097", "111", "197"]
states = ["17"] * 7
cty_name = ["Cook", "DuPg", "Kane", "Kndl", "Lke", "McHn", "Will"]

# Income bracket variables (16 bins)
income_vars = [
    "B19001_002E", "B19001_003E", "B19001_004E", "B19001_005E",
    "B19001_006E", "B19001_007E", "B19001_008E", "B19001_009E",
    "B19001_010E", "B19001_011E", "B19001_012E", "B19001_013E",
    "B19001_014E", "B19001_015E", "B19001_016E", "B19001_017E",
]

# Population variables
pop_vars = ["B01003_001E", "B19001_001E", "B25010_001E"]
all_vars = income_vars + pop_vars


def fetch_data(year):
    """Fetch census data for a given year."""
    print(f"\nFetching 20{year} data...")
    base_url = f"https://api.census.gov/data/20{year}/acs/acs5"
    all_dfs = []
    
    for state, county, cty in zip(states, counties, cty_name):
        print(f"  {cty} County...", end=" ")
        bg_geography = f"block%20group:*&in=state:{state}&in=county:{county}"
        bg_url = f"{base_url}?get={','.join(all_vars)}&for={bg_geography}&key={api_key}"
        
        try:
            response = requests.get(bg_url, timeout=60)
            response.raise_for_status()
            data = response.json()
            if len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
                print(f"✓ {len(df)} BGs")
                all_dfs.append(df)
            else:
                print("✗ No data")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    df_combined['GEOID'] = (df_combined['state'] + df_combined['county'] + 
                            df_combined['tract'] + df_combined['block group'])
    
    for col in all_vars:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
    
    print(f"✓ Total: {len(df_combined)} block groups\n")
    return df_combined


def clean_and_align(df_2014, df_2019):
    """Remove sentinel values and align block groups."""
    print("Cleaning and aligning dataframes...")
    
    # Remove sentinel values
    df_2014 = df_2014[df_2014['B25010_001E'] >= 0]
    df_2019 = df_2019[df_2019['B25010_001E'] >= 0]
    
    # Keep only common block groups
    common_geoids = set(df_2014['GEOID']) & set(df_2019['GEOID'])
    df_2014 = df_2014[df_2014['GEOID'].isin(common_geoids)].sort_values('GEOID').reset_index(drop=True)
    df_2019 = df_2019[df_2019['GEOID'].isin(common_geoids)].sort_values('GEOID').reset_index(drop=True)
    
    print(f"✓ {len(df_2014)} aligned block groups\n")
    return df_2014, df_2019


def convert_to_population(df):
    """Convert household counts to population counts."""
    print("Converting households to population...")
    df = df.copy()
    
    avg_hh_size = df['B25010_001E']
    for var in income_vars:
        df[var] = np.floor(df[var] * avg_hh_size)
    
    df['B19001_001E'] = df[income_vars].sum(axis=1)
    df = df.drop(columns=['B01003_001E', 'B25010_001E'])
    
    print(f"✓ Population conversion complete\n")
    return df


def apply_internal_migration(df_2014, df_2019):
    """Apply internal migration using swap algorithm from simple_swap_test.py."""
    print("="*60)
    print("APPLYING INTERNAL MIGRATION (SWAP ALGORITHM)")
    print("="*60)
    
    # Visual alignment check
    print("\nAlignment verification (first 5 block groups):")
    print("  2014 GEOIDs:", df_2014['GEOID'].head(5).tolist())
    print("  2019 GEOIDs:", df_2019['GEOID'].head(5).tolist())
    
    print("\nAlignment verification (last 5 block groups):")
    print("  2014 GEOIDs:", df_2014['GEOID'].tail(5).tolist())
    print("  2019 GEOIDs:", df_2019['GEOID'].tail(5).tolist())
    
    # Verify all GEOIDs match in order
    if np.array_equal(df_2014['GEOID'].values, df_2019['GEOID'].values):
        print("\n✓ SUCCESS: All GEOIDs match in order (perfect alignment)")
    else:
        print("\n✗ ERROR: GEOIDs do not match in order!")
        return None
    
    # Calculate churn: sum(|differences|) - |total change|
    pop_2014 = df_2014[income_vars].values
    pop_2019 = df_2019[income_vars].values
    
    total_pop_2014 = pop_2014.sum()
    total_pop_2019 = pop_2019.sum()
    total_change = abs(total_pop_2019 - total_pop_2014)
    
    differences_per_cell = pop_2019 - pop_2014
    sum_abs_differences = np.abs(differences_per_cell).sum()
    
    churn = sum_abs_differences - total_change
    num_swaps = int(churn / 2)
    
    print(f"\n2014 total population: {total_pop_2014:.0f}")
    print(f"2019 total population: {total_pop_2019:.0f}")
    print(f"Total population change: {total_change:.0f}")
    print(f"Sum of |differences|: {sum_abs_differences:.0f}")
    print(f"Churn (internal moves): {churn:.0f}")
    print(f"Number of swaps to perform: {num_swaps}\n")
    
    # Start with 2014 data
    df_result = df_2014.copy()
    num_rows = len(df_result)
    num_cols = len(income_vars)
    
    # Perform swaps
    print("Performing swaps...")
    for i in range(num_swaps):
        # Get current population matrix (just income columns)
        pop_matrix = df_result[income_vars].values
        total_pop = pop_matrix.sum()
        
        if total_pop < 2:
            print(f"WARNING: Not enough population for swaps at iteration {i}")
            break
        
        # Flatten and calculate probabilities
        flat_pop = pop_matrix.flatten()
        probabilities = flat_pop / total_pop
        
        # Choose TWO different cells based on population-weighted probabilities
        flat_indices = np.random.choice(
            num_rows * num_cols,
            size=2,
            replace=False,
            p=probabilities
        )
        
        # Convert flat indices to (row, col) coordinates
        cell1_flat_idx = flat_indices[0]
        cell2_flat_idx = flat_indices[1]
        
        i_row = cell1_flat_idx // num_cols  # row of first cell
        j_col = cell1_flat_idx % num_cols   # column of first cell
        
        p_row = cell2_flat_idx // num_cols  # row of second cell
        q_col = cell2_flat_idx % num_cols   # column of second cell
        
        # Get the actual income variable names
        j_income_var = income_vars[j_col]
        q_income_var = income_vars[q_col]
        
        # Perform the swap:
        # Move person from (i_row, j_col) to (p_row, j_col)
        df_result.loc[i_row, j_income_var] -= 1
        df_result.loc[p_row, j_income_var] += 1
        
        # Move person from (p_row, q_col) to (i_row, q_col)
        df_result.loc[p_row, q_income_var] -= 1
        df_result.loc[i_row, q_income_var] += 1
        
        # Progress updates
        if (i + 1) % 50000 == 0 or (i + 1) == num_swaps:
            print(f"  Progress: {i+1}/{num_swaps} ({100*(i+1)/num_swaps:.1f}%)")
    
    # Update total population column
    df_result['B19001_001E'] = df_result[income_vars].sum(axis=1)
    
    print(f"\n✓ Migration complete")
    print(f"Final population: {df_result['B19001_001E'].sum():.0f}")
    print("="*60 + "\n")
    
    return df_result


def main():
    print("\n" + "="*60)
    print("DATA NULL SWAPPING MODEL")
    print("="*60 + "\n")
    
    # Fetch data
    df_2014_raw = fetch_data(14)
    df_2019_raw = fetch_data(19)
    
    # Clean and align
    df_2014_clean, df_2019_clean = clean_and_align(df_2014_raw, df_2019_raw)
    
    # Convert to population
    df_2014_pop = convert_to_population(df_2014_clean)
    df_2019_pop = convert_to_population(df_2019_clean)
    
    # Apply internal migration (no external migration)
    df_2019_simulated = apply_internal_migration(df_2014_pop, df_2019_pop)
    
    # Save results
    print("Saving results...")
    df_2014_pop.to_csv('null_model/simulation_results/data_null_swapping/Census-Data-2014.csv', index=False)
    df_2019_simulated.to_csv('null_model/simulation_results/data_null_swapping/Simulated-Data-2019.csv', index=False)
    print("✓ Saved to 'null_model/simulation_results/data_null_swapping/'\n")


if __name__ == "__main__":
    main()
