import pandas as pd
import numpy as np

# Income bracket variables (16 bins)
income_vars = [
    "B19001_002E", "B19001_003E", "B19001_004E", "B19001_005E",
    "B19001_006E", "B19001_007E", "B19001_008E", "B19001_009E",
    "B19001_010E", "B19001_011E", "B19001_012E", "B19001_013E",
    "B19001_014E", "B19001_015E", "B19001_016E", "B19001_017E",
]

# Income bin midpoints (for percentile calculation)
income_bin_midpoints = [
    5000, 12500, 17500, 22500, 27500, 32500, 37500, 42500,
    47500, 55000, 67500, 87500, 112500, 137500, 175000, 400000
]

# ==================== USER CONFIGURATION ====================
INCOME_PERCENTILE_THRESHOLD = 90  # Keep TOP X% (above this percentile)
INPUT_DIR = 'null_model/simulation_results/data_null_migration'
OUTPUT_DIR = f'null_model/simulation_results/data_null_migration_top{100 - INCOME_PERCENTILE_THRESHOLD}pct'
# ============================================================


def filter_top_income_percentile(df, percentile, year_label):
    """Keep only population above the specified income percentile (top earners)."""
    print(f"\nFiltering {year_label} to keep TOP {100 - percentile}% of population by income...")
    
    # Calculate total population in each income bin
    total_pop_per_bin = df[income_vars].sum(axis=0).values
    total_pop = total_pop_per_bin.sum()
    
    print(f"  Total population before filtering: {total_pop:.0f}")
    
    # Create weighted income distribution
    pop_income_pairs = []
    for i, (pop, midpoint) in enumerate(zip(total_pop_per_bin, income_bin_midpoints)):
        pop_income_pairs.extend([midpoint] * int(pop))
    
    # Calculate percentile threshold income
    threshold_income = np.percentile(pop_income_pairs, percentile)
    
    # Find which bin contains the threshold
    threshold_bin_idx = None
    for i, midpoint in enumerate(income_bin_midpoints):
        if midpoint <= threshold_income:
            threshold_bin_idx = i
        else:
            break
    
    print(f"  Threshold income: ${threshold_income:.0f}")
    print(f"  Keeping bins {threshold_bin_idx + 2}-{len(income_vars)}, partially keeping bin {threshold_bin_idx + 1}")
    
    # Calculate fraction of threshold bin to keep (the upper portion)
    cumulative_pop = np.cumsum(total_pop_per_bin)
    target_pop_to_remove = total_pop * (percentile / 100.0)
    
    if threshold_bin_idx >= 0 and threshold_bin_idx < len(income_vars) - 1:
        pop_up_to_threshold = cumulative_pop[threshold_bin_idx]
        pop_to_remove_from_next_bin = target_pop_to_remove - pop_up_to_threshold
        next_bin_total = total_pop_per_bin[threshold_bin_idx + 1]
        fraction_to_remove = pop_to_remove_from_next_bin / next_bin_total if next_bin_total > 0 else 0
        fraction_to_keep = 1.0 - max(0, min(1, fraction_to_remove))
    else:
        fraction_to_keep = 1.0
    
    # Apply filtering to dataframe
    df_filtered = df.copy()
    
    # Zero out all bins below threshold
    for i in range(0, threshold_bin_idx + 1):
        df_filtered[income_vars[i]] = 0
    
    # Partially filter the threshold bin (keep upper portion)
    if threshold_bin_idx + 1 < len(income_vars):
        df_filtered[income_vars[threshold_bin_idx + 1]] *= fraction_to_keep
        df_filtered[income_vars[threshold_bin_idx + 1]] = np.floor(df_filtered[income_vars[threshold_bin_idx + 1]])
    
    # Update total population
    df_filtered['B19001_001E'] = df_filtered[income_vars].sum(axis=1)
    
    print(f"  Population after filtering: {df_filtered['B19001_001E'].sum():.0f}")
    print(f"  Removed {100 * (total_pop - df_filtered['B19001_001E'].sum()) / total_pop:.1f}% of population")
    
    return df_filtered


def main():
    print("\n" + "="*60)
    print(f"FILTERING TO TOP {100 - INCOME_PERCENTILE_THRESHOLD}% OF INCOME EARNERS")
    print("="*60)
    
    # Load CSV files
    print(f"\nLoading data from {INPUT_DIR}...")
    df_2014 = pd.read_csv(f'{INPUT_DIR}/Census-Data-2014.csv', 
                          dtype={'state': str, 'county': str, 'tract': str, 'block group': str, 'GEOID': str})
    df_2019 = pd.read_csv(f'{INPUT_DIR}/Simulated-Data-2019.csv',
                          dtype={'state': str, 'county': str, 'tract': str, 'block group': str, 'GEOID': str})
    
    print(f"✓ Loaded 2014 data: {len(df_2014)} block groups")
    print(f"✓ Loaded 2019 data: {len(df_2019)} block groups")
    
    # Filter both years to keep top income earners
    df_2014_filtered = filter_top_income_percentile(df_2014, INCOME_PERCENTILE_THRESHOLD, "2014")
    df_2019_filtered = filter_top_income_percentile(df_2019, INCOME_PERCENTILE_THRESHOLD, "2019")
    
    # Save filtered results
    print(f"\nSaving filtered results to {OUTPUT_DIR}...")
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_2014_filtered.to_csv(f'{OUTPUT_DIR}/Census-Data-2014.csv', index=False)
    df_2019_filtered.to_csv(f'{OUTPUT_DIR}/Simulated-Data-2019.csv', index=False)
    
    print(f"✓ Saved filtered 2014 data")
    print(f"✓ Saved filtered 2019 data")
    print("\n" + "="*60)
    print("FILTERING COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
