import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
import sys

warnings.filterwarnings('ignore', category=FutureWarning)

# Check for 'null' argument to switch directories
if 'null' in sys.argv:
    INPUT_DIR = 'output_terms_null'
    BASE_OUTPUT_DIR = 'plots_null'
else:
    INPUT_DIR = 'output_terms'
    BASE_OUTPUT_DIR = 'plots'


# --- CONFIGURATION ---
STATE_FIPS = '17'  # Illinois
START_YEAR = 2014  # Represents 2010-2014 ACS 5-Year (matches your data start)
END_YEAR = 2019    # Represents 2015-2019 ACS 5-Year (matches your data end)

# CPI values for inflation adjustment (Annual average CPI-U)
# Source: U.S. Bureau of Labor Statistics
CPI_START = 236.736  # 2014 average
CPI_END = 255.657    # 2019 average
INFLATION_ADJUSTMENT_FACTOR = CPI_END / CPI_START

# Output directory for plots
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'inequality_scatter_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CENSUS API FUNCTIONS ---

def fetch_acs_data(year, level, state_fips):
    """
    Fetches Median Gross Rent data from the Census API for a given year, level, and state.
    """
    print(f"Fetching {year} ACS data for level: {level}...")
    
    if level == 'tract':
        geo_filter = f'for=tract:*&in=state:{state_fips}'
    elif level == 'block group':
        geo_filter = f'for=block%20group:*&in=state:{state_fips}&in=county:*'
    else:
        raise ValueError("Level must be 'tract' or 'block group'")

    # Table B25064: Gross Rent, Universe: Renter-occupied housing units
    # B25064_001E: Estimate!!Median gross rent ($)
    variable = 'B25064_001E'
    
    api_url = f'https://api.census.gov/data/{year}/acs/acs5?get=NAME,{variable}&{geo_filter}'
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching data from Census API: {e}")
        return None
    except ValueError:
        print(f"  Error decoding JSON from Census API response: {response.text}")
        return None

    if not data or len(data) <= 1:
        print("  No data returned from API.")
        return None

    df = pd.DataFrame(data[1:], columns=data[0])
    
    if level == 'tract':
        df['GEOID'] = df['state'] + df['county'] + df['tract']
    else: # block group
        df['GEOID'] = df['state'] + df['county'] + df['tract'] + df['block group']

    df = df[['GEOID', variable]]
    df = df.rename(columns={variable: f'MedianRent_{year}'})
    
    # Convert rent to numeric, coercing errors to NaN. Census uses negative values for missing data.
    df[f'MedianRent_{year}'] = pd.to_numeric(df[f'MedianRent_{year}'], errors='coerce')
    df.loc[df[f'MedianRent_{year}'] < 0, f'MedianRent_{year}'] = np.nan
    
    print(f"  Successfully fetched {len(df)} records.")
    return df

# --- GINI CALCULATION FUNCTIONS (Imported from gini_scatter.py) ---

def calculate_gini_coefficient(incomes, weights=None):
    """
    Calculate Gini coefficient for a distribution of incomes.
    """
    if weights is None:
        weights = np.ones_like(incomes)
    
    valid_mask = ~(np.isnan(incomes) | np.isnan(weights))
    incomes = incomes[valid_mask]
    weights = weights[valid_mask]
    
    if len(incomes) < 2:
        return np.nan
    
    sorted_indices = np.argsort(incomes)
    sorted_incomes = incomes[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    cum_weights = np.cumsum(sorted_weights)
    cum_income_weights = np.cumsum(sorted_incomes * sorted_weights)
    
    total_weight = cum_weights[-1]
    total_income_weight = cum_income_weights[-1]
    
    if total_weight == 0 or total_income_weight == 0:
        return np.nan
    
    cum_pop_share = cum_weights / total_weight
    cum_income_share = cum_income_weights / total_income_weight
    
    x = np.concatenate([[0], cum_pop_share])
    y = np.concatenate([[0], cum_income_share])
    
    area_under_lorenz = np.trapz(y, x)
    gini = 1 - 2 * area_under_lorenz
    return gini

# --- THEIL INDEX FUNCTIONS ---

def fetch_binned_income_data(year, state_fips):
    """
    Fetches binned household income data (B19001) for all tracts in a state.
    """
    print(f"Fetching {year} ACS binned income data for tracts...")
    geo_filter = f'for=tract:*&in=state:{state_fips}'
    
    # B19001: HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN INFLATION-ADJUSTED DOLLARS)
    # 17 columns: _001E is total, _002E to _017E are the bins
    variables = ','.join([f'B19001_{i:03d}E' for i in range(1, 18)])
    api_url = f'https://api.census.gov/data/{year}/acs/acs5?get=NAME,{variables}&{geo_filter}'

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  Error fetching binned income data: {e}")
        return None

    df = pd.DataFrame(data[1:], columns=data[0])
    df['GEOID'] = df['state'] + df['county'] + df['tract']
    
    # Convert all variable columns to numeric
    var_cols = [f'B19001_{i:03d}E' for i in range(1, 18)]
    df[var_cols] = df[var_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    print(f"  Successfully fetched binned income data for {len(df)} tracts.")
    return df

def calculate_theil_index(households, incomes):
    """
    Calculate Theil index for binned data.
    
    The correct Theil index formula for grouped data is:
    T = Σ (yi/Y) * ln((yi/Y)/(ni/N))
    where:
    yi = total income in group i (households_i * income_i)
    Y = total income across all groups
    ni = number of households in group i
    N = total number of households
    """
    total_households = np.sum(households)
    group_incomes = households * incomes
    total_income = np.sum(group_incomes)
    
    # Avoid division by zero
    if total_households == 0 or total_income == 0:
        return 0
        
    pop_shares = households / total_households
    income_shares = group_incomes / total_income
    
    # Only calculate for bins with both shares positive
    valid = (pop_shares > 0) & (income_shares > 0)
    
    if not np.any(valid):
        return 0
        
    # The correct formula implementation
    return np.sum(income_shares[valid] * np.log(income_shares[valid] / pop_shares[valid]))

def calculate_theil_decomposition(df_income_bins, year):
    # ... [earlier code remains the same until tract-level calculation] ...

    # Calculate tract-level Theil
    tract_theil = calculate_theil_index(households_in_bins, income_midpoints_adj)
    
    tract_stats.append({
        'GEOID': tract['GEOID'],
        'total_households': total_households,
        'total_income': total_income,
        'mean_income': mean_income,
        'theil_T': tract_theil
    })


def calculate_theil_from_individual_data(incomes):
    """Calculate Theil index from individual income data."""
    incomes = incomes[incomes > 0]  # Remove zeros
    if len(incomes) < 2:
        return 0
    
    mean_income = np.mean(incomes)
    income_ratios = incomes / mean_income
    return np.mean(income_ratios * np.log(income_ratios))


def calculate_gini_changes(income_data, group_by_col):
    """
    Calculate Gini coefficient change for each group.
    """
    print(f"Calculating Gini coefficients by {group_by_col}...")
    results = []
    
    for group_name, group in income_data.groupby(group_by_col):
        if len(group) < 2:
            continue
            
        initial_gini = calculate_gini_coefficient(
            group['LogAvgIncInitial_bg'].values, 
            group['PopInitial_bg'].values
        )
        
        final_gini = calculate_gini_coefficient(
            group['LogAvgIncFinal_bg'].values, 
            group['PopFinal_bg'].values
        )
        
        gini_change = final_gini - initial_gini if pd.notna(final_gini) and pd.notna(initial_gini) else np.nan
        
        results.append({
            group_by_col: group_name,
            'gini_change': gini_change
        })
    
    result_df = pd.DataFrame(results)
    print(f"  Calculated Gini changes for {len(result_df)} groups.")
    return result_df

# --- PLOTTING FUNCTION ---

def create_scatter_plot(data, x_col, y_col, size_col, color_col, title, xlabel, ylabel, filename):
    """
    Creates and saves a styled scatter plot with regression analysis.
    """
    # Drop rows with missing values for the core columns
    clean_data = data[[x_col, y_col, size_col, color_col]].dropna()
    
    if len(clean_data) < 20:
        print(f"Skipping plot '{title}': Not enough valid data points ({len(clean_data)}).")
        return

    # --- Prepare point sizes ---
    pop = clean_data[size_col]
    s_min, s_max = 15, 200
    sizes = s_min + ((pop - pop.min()) / (pop.max() - pop.min()) * (s_max - s_min)) if pop.min() != pop.max() else s_min

    # --- Prepare point colors ---
    # Bifurcate the color based on whether the initial income is above or below the median
    median_color_val = clean_data[color_col].median()
    colors = np.where(clean_data[color_col] > median_color_val, '#E77429', '#633673')

    # --- Create plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.scatter(clean_data[x_col], clean_data[y_col], s=sizes, c=colors, alpha=0.6, edgecolors='white', linewidth=0.5)

    # --- Regression Analysis ---
    slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[x_col], clean_data[y_col])
    r_squared = r_value**2
    
    x_range = np.linspace(clean_data[x_col].min(), clean_data[x_col].max(), 100)
    ax.plot(x_range, intercept + slope * x_range, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

    # --- Aesthetics ---
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Add stats box
    stats_text = f'$R^2 = {r_squared:.3f}$\np-value = {p_value:.3f}\nSlope = {slope:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    ax.axhline(0, color='grey', linestyle=':', linewidth=1)
    ax.axvline(0, color='grey', linestyle=':', linewidth=1)
    
    # Save plot
    full_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to {full_path}")

income_midpoints = np.array([
    5000, 12500, 17500, 22500, 27500, 32500, 37500, 42500, 47500, 
    55000, 67500, 87500, 112500, 137500, 175000, 350000 
])

def test_theil_calculation():
    """
    Test the Theil index calculation with known examples and verify properties.
    """
    print("Testing Theil Index Implementation...")
    
    # Test Case 1: Perfect equality
    # If everyone has the same income, Theil should be 0
    test_df = pd.DataFrame({
        'GEOID': ['test1'],
        'B19001_002E': [100],  # 100 households all in the same income bin
    })
    for i in range(3, 18):
        test_df[f'B19001_{i:03d}E'] = 0
    
    result1 = calculate_theil_decomposition(test_df, 2019)
    print("\nTest 1 - Perfect Equality:")
    print(f"Expected Theil: 0")
    print(f"Calculated Theil: {result1['overall_theil']:.6f}")
    
    # Test Case 2: Known inequality
    # Create a simple two-bin distribution where we can calculate by hand
    test_df2 = pd.DataFrame({
        'GEOID': ['test2'],
        'B19001_002E': [50],   # 50 households at $5000
        'B19001_003E': [50],   # 50 households at $12500
    })
    for i in range(4, 18):
        test_df2[f'B19001_{i:03d}E'] = 0
    
    result2 = calculate_theil_decomposition(test_df2, 2019)
    # Hand calculation for verification
    # μ = (50*5000 + 50*12500)/100 = 8750
    # T = 0.5*(5000/8750)*ln(5000/8750) + 0.5*(12500/8750)*ln(12500/8750)
    expected2 = 0.0823  # This value needs to be verified by hand
    print("\nTest 2 - Simple Two-Bin Case:")
    print(f"Expected Theil: {expected2:.4f}")
    print(f"Calculated Theil: {result2['overall_theil']:.4f}")
    
    # Test Case 3: Scale invariance
    # Multiply all incomes by 2, Theil should remain the same
    test_df3 = test_df2.copy()
    result3 = calculate_theil_decomposition(test_df3, 2019)
    print("\nTest 3 - Scale Invariance:")
    print(f"Original Theil: {result2['overall_theil']:.4f}")
    print(f"Scaled Theil: {result3['overall_theil']:.4f}")
    
    # Test Case 4: Decomposition Property
    # Total Theil should equal Between + Within
    test_df4 = pd.DataFrame({
        'GEOID': ['tract1', 'tract2'],
        'B19001_002E': [100, 0],    # First tract: all low income
        'B19001_003E': [0, 100],    # Second tract: all high income
    })
    for i in range(4, 18):
        test_df4[f'B19001_{i:03d}E'] = 0
    
    result4 = calculate_theil_decomposition(test_df4, 2019)
    decomp_sum = result4['between_group_theil'] + result4['within_group_theil']
    print("\nTest 4 - Decomposition Property:")
    print(f"Total Theil: {result4['overall_theil']:.4f}")
    print(f"Between + Within: {decomp_sum:.4f}")
    
    # Test Case 5: Population Weighting
    # A small rich tract and large poor tract should be weighted by population
    test_df5 = pd.DataFrame({
        'GEOID': ['tract1', 'tract2'],
        'B19001_002E': [1000, 0],    # 1000 households in poor tract
        'B19001_017E': [0, 100],     # 100 households in rich tract
    })
    for i in range(3, 17):
        test_df5[f'B19001_{i:03d}E'] = 0
    
    result5 = calculate_theil_decomposition(test_df5, 2019)
    print("\nTest 5 - Population Weighting:")
    print(f"Theil Index: {result5['overall_theil']:.4f}")
    print(f"Between-Group Share: {result5['between_group_share_pct']:.1f}%")

# Run the tests

# --- MAIN EXECUTION ---

def main():
    """
    Main function to fetch data, merge, and generate plots.
    """
    # --- THEIL INDEX ANALYSIS ---
    print(f"\n{'='*60}\nPerforming Statewide Theil Decomposition Analysis (Tracts)\n{'='*60}")
    
    # 1. Fetch binned income data for start and end years
    income_bins_start = fetch_binned_income_data(START_YEAR, STATE_FIPS)
    income_bins_end = fetch_binned_income_data(END_YEAR, STATE_FIPS)
    
    if income_bins_start is None or income_bins_end is None:
        print("Could not fetch binned income data. Skipping Theil analysis.")
    else:
        # 2. Calculate decomposition for both periods
        theil_start_results = calculate_theil_decomposition(income_bins_start, START_YEAR)
        theil_end_results = calculate_theil_decomposition(income_bins_end, END_YEAR)
        
        # 3. Print a summary report
        print("\n--- Illinois Statewide Inequality Decomposition Summary ---")
        print(f"Comparing {START_YEAR} ACS 5-Year Estimates to {END_YEAR} ACS 5-Year Estimates")
        print("-" * 55)
        print(f"{'Metric':<30} | {START_YEAR:<10} | {END_YEAR:<10} | {'Change':<10}")
        print("-" * 55)
        
        # --- TRACT-LEVEL THEIL CHANGE ANALYSIS ---
        if income_bins_start is not None and income_bins_end is not None:
            print(f"\nCalculating tract-level Theil changes...")
            
            # Get tract-level Theil data for both periods
            tract_theil_start = theil_start_results['tract_data'][['GEOID', 'theil_T']].rename(columns={'theil_T': 'theil_start'})
            tract_theil_end = theil_end_results['tract_data'][['GEOID', 'theil_T']].rename(columns={'theil_T': 'theil_end'})
            
            # Merge and calculate change
            tract_theil_change = pd.merge(tract_theil_start, tract_theil_end, on='GEOID', how='inner')
            tract_theil_change['theil_change'] = tract_theil_change['theil_end'] - tract_theil_change['theil_start']
            
            print(f"  Calculated Theil changes for {len(tract_theil_change)} tracts.")
            
            # Store this for later use in the scatter plot section
            theil_change_data = tract_theil_change
        else:
            theil_change_data = None

        start_between_share = theil_start_results['between_group_share_pct']
        end_between_share = theil_end_results['between_group_share_pct']
        change_share = end_between_share - start_between_share
        
        print(f"{'Overall Theil Index (T)':<30} | {theil_start_results['overall_theil']:.4f}     | {theil_end_results['overall_theil']:.4f}     | {theil_end_results['overall_theil'] - theil_start_results['overall_theil']:+.4f}")
        print(f"{'  - Between-Tract Share (%)':<30} | {start_between_share:.2f}%       | {end_between_share:.2f}%       | {change_share:+.2f} pp")
        print(f"{'  - Within-Tract Share (%)':<30} | {100-start_between_share:.2f}%       | {100-end_between_share:.2f}%       | {-change_share:+.2f} pp")
        print("-" * 55)
        print("Note: 'pp' stands for percentage points.")
        print("\nInterpretation:")
        if change_share > 0:
            print("The share of inequality BETWEEN tracts has INCREASED. This suggests that income segregation")
            print("across neighborhoods in Illinois has become more pronounced between the two periods.")
        elif change_share < 0:
            print("The share of inequality BETWEEN tracts has DECREASED. This suggests that while inequality")
            print("may still exist, it is becoming less defined by geography and more of a phenomenon within neighborhoods.")
        else:
            print("The share of inequality between tracts has not changed significantly.")

    # --- SCATTER PLOT ANALYSIS (TRACT ONLY) ---
    # Loop over both tract and block group levels. Commented out block group for now.
    # for level, level_short in [('tract', 'tr'), ('block group', 'bg')]:
    for level, level_short in [('tract', 'tr')]: # Only run tract-level analysis
        print(f"\n{'='*60}\nProcessing {level.upper()} level data\n{'='*60}")
        
        # 1. Fetch and process ACS rent data
        df_start = fetch_acs_data(START_YEAR, level, STATE_FIPS)
        df_end = fetch_acs_data(END_YEAR, level, STATE_FIPS)
        
        if df_start is None or df_end is None:
            print(f"Could not fetch ACS data for {level}. Skipping.")
            continue
            
        df_start[f'MedianRent_{START_YEAR}_adj'] = df_start[f'MedianRent_{START_YEAR}'] * INFLATION_ADJUSTMENT_FACTOR
        df_rent = pd.merge(df_start[['GEOID', f'MedianRent_{START_YEAR}_adj']], df_end, on='GEOID')
        df_rent['ChangeInRent'] = df_rent[f'MedianRent_{END_YEAR}'] - df_rent[f'MedianRent_{START_YEAR}_adj']
        
        # 2. Load local decomposition and Gini data
        print(f"Loading local data for {level}...")
        local_data_path = os.path.join(INPUT_DIR, f'bg_{level_short}_exported_terms.csv')
        if not os.path.exists(local_data_path):
            print(f"  Could not find local data file: {local_data_path}. Skipping.")
            continue
        df_local = pd.read_csv(local_data_path, dtype={'UnitName': str, 'ParentCommunity': str, 'ParentTract': str, 'ParentCounty': str})
        
        # --- GINI CALCULATION STEP ---
        gini_col = 'gini_change' # This will be the name of our newly computed column
        if level == 'tract':
            print("Loading block group data to calculate Gini changes for tracts...")
            bg_data_path = os.path.join(INPUT_DIR, 'bg_bg_exported_terms.csv')
            if not os.path.exists(bg_data_path):
                print(f"  Could not find block group data file: {bg_data_path}. Skipping Gini plot.")
                gini_col = None # Disable Gini plot if BG data is missing
            else:
                df_bg = pd.read_csv(bg_data_path, dtype={'ParentTract': str})
                df_gini_change = calculate_gini_changes(df_bg, 'ParentTract')
                # Merge Gini change data into the local tract data
                df_local = pd.merge(df_local, df_gini_change, left_on='UnitName', right_on='ParentTract', how='left')

        # The GEOID from census has a different format than the UnitName FIPS code, need to match them.
        # Census GEOID for tract: 17(state)031(county)081400(tract) -> 11 digits
        # Our UnitName for tract: 17031081400 -> 11 digits (MATCHES)
        # Census GEOID for BG: 17(state)031(county)081400(tract)1(bg) -> 12 digits
        # Our UnitName for BG: 170310814001 -> 12 digits (MATCHES)
        df_local['GEOID'] = df_local['UnitName']
        
        # 3. Merge ACS data with local data
        df_merged = pd.merge(df_local, df_rent[['GEOID', 'ChangeInRent']], on='GEOID', how='inner')
        print(f"  Merged {len(df_merged)} records.")
        
        # 4. Identify columns for plotting
        # For selection, we use the state-normalized PNC of the level's own selection effect.
        if level == 'tract':
            pop_pnc_col = 'Sel_tr_from_bg_pop_PNC_st'
        else: # block group - This part is currently disabled
            pop_pnc_col = 'Sel_bg_pop_PNC_st'
        
        size_col = f'PopInitial_{level_short}'
        color_col = f'LogAvgIncInitial_{level_short}'
        
        # Check for required columns for the rent plot
        required_rent_cols = [pop_pnc_col, size_col, color_col]
        if not all(col in df_merged.columns for col in required_rent_cols):
            print(f"  Missing one of the required columns for rent plot: {required_rent_cols}. Skipping.")
        else:
            # Plot 1: Population PNC vs. Change in Rent
            create_scatter_plot(
                data=df_merged,
                x_col=pop_pnc_col,
                y_col='ChangeInRent',
                size_col=size_col,
                color_col=color_col,
                title=f'{level.title()} Level: Population Selection vs. Change in Median Rent',
                xlabel='Population Selection PNC (State-Normalized)',
                ylabel='Inflation-Adjusted Change in Median Rent ($)',
                filename=f'{level_short}_pop_pnc_vs_rent_change.pdf'
            )

        # Check for required columns for the Gini plot
        if gini_col and gini_col in df_merged.columns:
             # Plot 2: Population PNC vs. Change in Gini
            create_scatter_plot(
                data=df_merged,
                x_col=pop_pnc_col,
                y_col=gini_col,
                size_col=size_col,
                color_col=color_col,
                title=f'{level.title()} Level: Population Selection vs. Change in Gini',
                xlabel='Population Selection PNC (State-Normalized)',
                ylabel='Change in Gini Coefficient',
                filename=f'{level_short}_pop_pnc_vs_gini_change.pdf'
            )
            
            # --- Sanity Check Plot ---
            # Plot 3: Income PNC vs. Change in Gini
            inc_pnc_col = 'Sel_tr_from_bg_inc_PNC_st'
            if inc_pnc_col in df_merged.columns:
                create_scatter_plot(
                    data=df_merged,
                    x_col=inc_pnc_col,
                    y_col=gini_col,
                    size_col=size_col,
                    color_col=color_col,
                    title=f'{level.title()} Level: Income Selection vs. Change in Gini (Sanity Check)',
                    xlabel='Income Selection PNC (State-Normalized)',
                    ylabel='Change in Gini Coefficient',
                    filename=f'{level_short}_inc_pnc_vs_gini_change_sanity_check.pdf'
                )
            else:
                print(f"  Skipping sanity check plot: Missing column '{inc_pnc_col}'")
        else:
            print("  Skipping Gini plot due to missing data or calculation failure.")

        # Plot 4: Population PNC vs. Change in Theil Index (if available)
        if theil_change_data is not None:
            # Merge Theil change data with the main tract data
            df_merged_with_theil = pd.merge(df_merged, theil_change_data[['GEOID', 'theil_change']], on='GEOID', how='inner')
            
            if 'theil_change' in df_merged_with_theil.columns and len(df_merged_with_theil) > 20:
                create_scatter_plot(
                    data=df_merged_with_theil,
                    x_col=pop_pnc_col,
                    y_col='theil_change',
                    size_col=size_col,
                    color_col=color_col,
                    title=f'{level.title()} Level: Population Selection vs. Change in Theil Index',
                    xlabel='Population Selection PNC (State-Normalized)',
                    ylabel='Change in Theil Index (2016-2021)',
                    filename=f'{level_short}_pop_pnc_vs_theil_change.pdf'
                )
            else:
                print("  Skipping Theil change plot due to insufficient data.")

    test_theil_calculation()

if __name__ == "__main__":
    main() 