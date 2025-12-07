import os
import sys
import csv
import pickle
import pandas as pd

# --- Configuration ---
# Check for 'null' argument to switch directories
if 'null' in sys.argv:
    INPUT_DIR = 'output_terms_null'
else:
    INPUT_DIR = 'output_terms'

DEMOGRAPHICS_DIR = os.path.join(INPUT_DIR, 'demographics')
HIERARCHY_LEVELS = ['bg', 'tr', 'cm', 'ct', 'st']
ANALYSIS_PERIOD_YEARS = 5
CENSUS_DATA_PATH = "calculation_scripts/data/census_data1.pkl"

def read_demographics_data():
    """Reads birth and death data, returning a dictionary mapping county code to total endogenous change."""
    births_file = os.path.join(DEMOGRAPHICS_DIR, 'Births.csv')
    deaths_file = os.path.join(DEMOGRAPHICS_DIR, 'Deaths.csv')
    
    county_demographics = {}

    # Read Births
    try:
        with open(births_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip the "Total" summary rows in the CDC data
                if row.get('Notes') == 'Total':
                    continue
                county_code = row.get('County Code')
                births = row.get('Births')
                if county_code and births and births.isdigit():
                    county_demographics.setdefault(county_code, {})['births'] = int(births)
    except FileNotFoundError:
        print(f"Warning: Births file not found at {births_file}")
    except Exception as e:
        print(f"Error reading births file: {e}")

    # Read Deaths
    try:
        with open(deaths_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                county_code = row.get('County Code')
                deaths = row.get('Deaths')
                if county_code and deaths and deaths.isdigit():
                    county_demographics.setdefault(county_code, {})['deaths'] = int(deaths)
    except FileNotFoundError:
        print(f"Warning: Deaths file not found at {deaths_file}")
    except Exception as e:
        print(f"Error reading deaths file: {e}")
        
    # Calculate total endogenous change for 5 years
    county_endogenous_change = {}
    for code, data in county_demographics.items():
        annual_change = data.get('births', 0) + data.get('deaths', 0)
        county_endogenous_change[code] = annual_change * ANALYSIS_PERIOD_YEARS
        
    return county_endogenous_change

def calculate_net_turnover_from_csv():
    """
    Calculates the 5-year NET population turnover for each county individually by
    summing the absolute population changes of all its constituent block groups.
    This is the |Pop_Final - Pop_Initial| calculation.
    """
    county_turnover = {}
    filepath = os.path.join(INPUT_DIR, 'bg_bg_exported_terms.csv') # Read the block-group level file

    if not os.path.exists(filepath):
        print(f"Error: Block group data file not found at {filepath}. Cannot calculate net turnover.")
        return {}
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                state_code = row.get('ParentState')
                county_code_part = row.get('ParentCounty')
                
                if state_code and county_code_part:
                    full_county_code = f"{state_code}{county_code_part.zfill(3)}"
                else:
                    continue

                pop_initial = row.get('PopInitial_bg')
                pop_final = row.get('PopFinal_bg')
                
                if pop_initial and pop_final:
                    try:
                        turnover = abs(float(pop_final) - float(pop_initial))
                        county_turnover[full_county_code] = county_turnover.get(full_county_code, 0) + turnover
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error processing block group turnover file: {e}")

    return county_turnover

def calculate_gross_annual_turnover_from_pkl():
    """
    Loads the census_data1.pkl file and calculates the GROSS year-over-year
    population turnover for each county by summing the annual absolute changes
    of all its constituent block groups.
    """
    print("--- Loading and processing census_data1.pkl for gross annual turnover ---")
    
    try:
        with open(CENSUS_DATA_PATH, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Census data pickle file not found at {CENSUS_DATA_PATH}")
        return {}
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return {}

    all_years_dfs = []
    # Years 2014 through 2019
    for year_int in range(14, 20):
        if year_int in data and 'bg' in data[year_int]:
            df = data[year_int]['bg']
            df['year'] = 2000 + year_int
            # Ensure UnitName is standard for merging
            if 'GEOID' in df.columns:
                 df['UnitName'] = df['GEOID']
            elif 'temp_bg_id' in df.columns:
                 df['UnitName'] = df['temp_bg_id']
            else: # Fallback to creating from state/county/tract/bg
                df['UnitName'] = df['state'].astype(str) + df['county'].astype(str).str.zfill(3) + df['tract'].astype(str).str.zfill(6) + df['block group'].astype(str)

            all_years_dfs.append(df[['UnitName', 'year', 'B19001_001E', 'state', 'county']])
    
    if len(all_years_dfs) < 2:
        print("Error: Not enough annual block group data found in the pickle file to calculate change.")
        return {}

    combined_df = pd.concat(all_years_dfs, ignore_index=True)
    combined_df.rename(columns={'B19001_001E': 'population'}, inplace=True)
    
    # Pivot to get years as columns for each block group
    pivot_df = combined_df.pivot(index='UnitName', columns='year', values='population').sort_index()

    # Get county mapping from the last available year
    county_map = combined_df.drop_duplicates('UnitName', keep='last').set_index('UnitName')
    county_map['full_county_code'] = county_map['state'].astype(str) + county_map['county'].astype(str).str.zfill(3)
    pivot_df = pivot_df.join(county_map['full_county_code'])
    pivot_df.dropna(subset=['full_county_code'], inplace=True)

    # Calculate year-over-year absolute changes
    years_to_calc = sorted([col for col in pivot_df.columns if isinstance(col, int)])
    total_gross_turnover = 0
    for i in range(len(years_to_calc) - 1):
        year1 = years_to_calc[i]
        year2 = years_to_calc[i+1]
        pivot_df[f'change_{year1}_{year2}'] = (pivot_df[year2] - pivot_df[year1]).abs()

    change_cols = [col for col in pivot_df.columns if 'change_' in str(col)]
    pivot_df['total_gross_turnover'] = pivot_df[change_cols].sum(axis=1)

    # Aggregate by county
    county_gross_turnover = pivot_df.groupby('full_county_code')['total_gross_turnover'].sum().to_dict()
    
    print("--- Finished calculating gross annual turnover ---")
    return county_gross_turnover


def main():
    """Main function to orchestrate the analysis and display results."""
    # --- Part 1: Original Grand Total Analysis ---
    print("--- Running Original Total Turnover Analysis ---")
    calculate_population_turnover_no_pandas()
    print("\n" + "="*80 + "\n")
    
    # --- Part 2: New County-by-County Analysis ---
    print("--- Running County-by-County Turnover vs. Endogenous Change Analysis ---")
    
    endogenous_change_data = read_demographics_data()
    net_turnover_data = calculate_net_turnover_from_csv()
    gross_annual_turnover_data = calculate_gross_annual_turnover_from_pkl()

    if not net_turnover_data:
        print("Could not perform county-level analysis due to missing turnover data.")
        return

    # Combine all datasets for comparison
    all_county_codes = set(net_turnover_data.keys()) | set(endogenous_change_data.keys()) | set(gross_annual_turnover_data.keys())
    
    comparison_data = []
    for code in all_county_codes:
        net_turnover = net_turnover_data.get(code, 0)
        gross_turnover = gross_annual_turnover_data.get(code, 0)
        endogenous_change = endogenous_change_data.get(code, 0)
        
        ratio_net = (net_turnover / endogenous_change) if endogenous_change > 0 else float('inf')
        ratio_gross = (gross_turnover / endogenous_change) if endogenous_change > 0 else float('inf')

        comparison_data.append({
            "County Code": code,
            "Net 5-Year Turnover": net_turnover,
            "Gross Annual Turnover (Sum)": gross_turnover,
            "5-Year Endogenous (Births+Deaths)": endogenous_change,
            "Ratio (Net)": ratio_net,
            "Ratio (Gross)": ratio_gross
        })

    if not comparison_data:
        print("No matching counties found between data sources.")
        return

    # --- Display County Comparison Table ---
    print("\n" + "="*25 + " County-Level Comparison " + "="*25)
    header = f"{'County Code':<12} {'Net 5-Yr Turnover':>20} {'Gross Ann. Turnover':>22} {'5-Yr Births+Deaths':>22} {'Ratio (Net)':>12} {'Ratio (Gross)':>15}"
    print(header)
    print("-" * len(header))

    comparison_data.sort(key=lambda x: x["Net 5-Year Turnover"], reverse=True)

    for row in comparison_data:
        print(f"{row['County Code']:<12} {row['Net 5-Year Turnover']:>20,.0f} {row['Gross Annual Turnover (Sum)']:>22,.0f} {row['5-Year Endogenous (Births+Deaths)']:>22,.0f} {row['Ratio (Net)']:>12.1f}x {row['Ratio (Gross)']:>15.1f}x")
    
    print("-" * len(header))
    print("Net Turnover: |Pop_2019 - Pop_2014|, summed over all BGs in the county.")
    print("Gross Turnover: Sum of |Pop_Yr+1 - Pop_Yr| for all BGs, summed over the 5-year period.")


def calculate_population_turnover_no_pandas():
    """
    Reads the exported CSVs for each hierarchical level using Python's
    built-in csv module and calculates the total absolute population change.
    This avoids using pandas to bypass a suspected environment issue.
    """
    turnover_results = {}

    for level in HIERARCHY_LEVELS:
        csv_filename = f"bg_{level}_exported_terms.csv"
        filepath = os.path.join(INPUT_DIR, csv_filename)

        if not os.path.exists(filepath):
            print(f"Warning: File not found for level '{level}': {filepath}. Skipping.")
            continue
        try:
            total_turnover = 0.0
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                pop_initial_col = f'PopInitial_{level}'
                pop_final_col = f'PopFinal_{level}'
                try:
                    idx_initial = header.index(pop_initial_col)
                    idx_final = header.index(pop_final_col)
                except ValueError:
                    continue
                for i, row in enumerate(reader):
                    try:
                        initial_pop = float(row[idx_initial])
                        final_pop = float(row[idx_final])
                        total_turnover += abs(final_pop - initial_pop)
                    except (ValueError, IndexError):
                        continue
            turnover_results[level] = total_turnover
        except Exception as e:
            print(f"An error occurred processing {filepath}: {e}")

    if not turnover_results:
        print("No results were calculated. Please check file paths and column names.")
        return
    print(f"{'Level':<10} {'Total Absolute Change (5 Years)':<35} {'Avg. Annual Absolute Change':<30}")
    print("-" * 75)
    for level, total_change in turnover_results.items():
        annual_change = total_change / ANALYSIS_PERIOD_YEARS
        print(f"{level:<10} {total_change:,.0f}{' ':<{35-len(f'{total_change:,.0f}')}} {annual_change:,.0f}")
    print("-" * 75)
    print("\nComparison value: Approx. 40,000 total individuals per year from births (20k) + deaths (20k).")

if __name__ == '__main__':
    main()
