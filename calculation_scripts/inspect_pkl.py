import pandas as pd
import pickle
import numpy as np # For isnan if needed later
import sys # Import sys to access command-line arguments

# Set display options for pandas to show more content
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1200)
pd.set_option('display.max_colwidth', 100)

def inspect_dataframe(df, filename):
    """Inspects a pandas DataFrame object."""
    print(f"--- Inspecting DataFrame from: {filename} ---")
    if df is None or not isinstance(df, pd.DataFrame):
        print("  Error: Object is not a valid pandas DataFrame.")
        return

    # --- ADDED: Calculate and print overall non-NaN percentage --- 
    total_elements = df.size
    if total_elements == 0:
        print("\\nOverall Non-NaN: DataFrame is empty (0 elements).")
    else:
        non_nan_count = df.count().sum() # Sum of non-NaN counts across all columns
        non_nan_percentage = (non_nan_count / total_elements) * 100
        print(f"\\nOverall Non-NaN: {non_nan_percentage:.2f}% ({non_nan_count}/{total_elements} cells)")
    # -----------------------------------------------------------

    print(f"\\nDataFrame Shape: {df.shape}")
    print("\\nDataFrame Columns:")
    print(df.columns.tolist())
    print("\\nDataFrame Info:")
    df.info()
    print("\\nNaN Counts per Column:")
    print(df.isnull().sum())

    # Check for specific columns expected in analysis_results.pkl
    if 'Agg' in df.columns:
        print("\\nUnique 'Agg' values:")
        try:
            print(df['Agg'].unique())
            print("\\nValue Counts for 'Agg':")
            print(df['Agg'].value_counts())
        except Exception as e:
            print(f"  Could not analyze 'Agg' column: {e}")

    if 'EndYear' in df.columns:
         print("\\nUnique 'EndYear' values:")
         try:
            # Sort unique years before printing
            unique_years = df['EndYear'].unique()
            # Handle potential NaNs before sorting
            unique_years_no_nan = unique_years[~np.isnan(unique_years)]
            print(np.sort(unique_years_no_nan))
         except Exception as e:
             print(f"  Could not analyze 'EndYear' column: {e}")
             
    if 'FilterType' in df.columns:
        print("\\nUnique 'FilterType' values:")
        try:
            print(df['FilterType'].unique())
            print("\\nValue Counts for 'FilterType':")
            print(df['FilterType'].value_counts())
        except Exception as e:
             print(f"  Could not analyze 'FilterType' column: {e}")
             
    print("\\nFirst 5 rows (Head):")
    print((df[df['Year']==14])[df['Agg']=='tr'])

    input(df[df['ChildTract']=='17031520100'])

    print(f"--- Finished Inspecting DataFrame: {filename} ---")


def inspect_dict_data(data_dict, filename):
    """Inspects a dictionary containing year and aggregation level keys."""
    print(f"--- Inspecting Dictionary from: {filename} ---")
    if data_dict is None or not isinstance(data_dict, dict):
        print("  Error: Object is not a valid dictionary.")
        return

    available_years = sorted(data_dict.keys())
    print(f"Available Years (Keys): {available_years}")

    for year in available_years:
        print(f"\\n--- Year: {year} ---")
        if not isinstance(data_dict.get(year), dict):
            print(f"  Value for year {year} is not a dictionary. Skipping.")
            continue

        agg_levels = sorted(data_dict[year].keys())
        print(f"  Available Aggregation Levels: {agg_levels}")

        for agg in agg_levels:
            df = data_dict[year].get(agg)
            if df is None:
                print(f"    {agg}: DataFrame is None.")
            elif not isinstance(df, pd.DataFrame):
                print(f"    {agg}: Object is not a DataFrame (Type: {type(df)}).")
            elif df.empty:
                print(f"    {agg}: DataFrame is empty.")
            else:
                # Calculate non-NaN percentage
                total_elements = df.size
                if total_elements == 0:
                    non_nan_percentage = 100.0 # Or 0.0, depending on convention for empty df
                    print(f"    {agg}: DataFrame is empty (0 elements).")
                else:
                    non_nan_count = df.count().sum()
                    non_nan_percentage = (non_nan_count / total_elements) * 100
                    print(f"    {agg}: {non_nan_percentage:.2f}% non-NaN values ({non_nan_count}/{total_elements} cells)")
                    
                # --- ADDED: Print head of the DataFrame --- 
                print("    Head:")
                print(df)
                # -------------------------------------------

    print(f"--- Finished Inspecting Dictionary: {filename} ---")

def inspect_analysis_price2(file_path):
    """Loads and inspects analysis_Price2.pkl, focusing on tract decomposition."""
    try:
        df = pd.read_pickle(file_path)
        print(f"Successfully loaded {file_path}")
        print(f"DataFrame Shape: {df.shape}")

        if df.empty:
            print("DataFrame is empty.")
            return

        # --- Basic Info ---
        print("\n--- DataFrame Info ---")
        df.info()
        print("\n--- AvgLogInc Description ---")
        print(df['UnitName'])
        print("\n--- AvgLogInc Histogram ---")
        print(df['AvgLogInc'].hist(bins=100))
        # --- Iterate through relevant Aggregation Levels ---
        agg_levels_to_inspect = ['tr', 'bg'] # Add other agg levels if needed
        max_n_to_check = 5 # Maximum N value to check for (e.g., CovIncTerm_5)

        for agg_level in agg_levels_to_inspect:
            df_agg = df[df['Agg'] == agg_level].copy()
            if df_agg.empty:
                print(f"\n--- No data found for Aggregation Level: {agg_level.upper()} ---")
                continue

            print(f"\n--- Inspecting {agg_level.upper()} Data (Total Rows: {len(df_agg)}) ---")

            # Define columns of interest dynamically
            cols_to_show_dynamic = ['UnitName', 'Year', 'FirstYear']
            for n in range(1, max_n_to_check + 1):
                # Population-related decomposition terms
                pop_terms = []
                if f'NumMembers_{n}' in df_agg.columns: pop_terms.append(f'NumMembers_{n}')
                if f'DeltaZbar_{n}' in df_agg.columns: pop_terms.append(f'DeltaZbar_{n}')
                if f'CovTerm_{n}' in df_agg.columns: pop_terms.append(f'CovTerm_{n}') # Population Cov
                if f'ExpPopTerm_{n}' in df_agg.columns: pop_terms.append(f'ExpPopTerm_{n}')
                if f'SanityCheckPop_{n}' in df_agg.columns: pop_terms.append(f'SanityCheckPop_{n}')
                cols_to_show_dynamic.extend(pop_terms)

                # Income-related decomposition terms (YOUR FOCUS)
                inc_terms = []
                if f'CovIncTerm_{n}' in df_agg.columns: inc_terms.append(f'CovIncTerm_{n}') # Income Cov
                if f'ExpIncTerm_{n}' in df_agg.columns: inc_terms.append(f'ExpIncTerm_{n}')
                if f'SanityCheckInc_{n}' in df_agg.columns: inc_terms.append(f'SanityCheckInc_{n}')
                cols_to_show_dynamic.extend(inc_terms)
            
            # Filter to columns that actually exist in df_agg to avoid KeyErrors
            cols_to_show_dynamic = [col for col in cols_to_show_dynamic if col in df_agg.columns]
            # Remove duplicates if any term was added twice by mistake (though logic prevents it here)
            cols_to_show_dynamic = sorted(list(set(cols_to_show_dynamic)), key = lambda x: (x.split('_')[-1] if x.split('_')[-1].isdigit() else '0', x) ) # Sort by N then name

            if not cols_to_show_dynamic or len(cols_to_show_dynamic) <= 3: # <=3 because UnitName, Year, FirstYear are base
                print(f"  None of the specified dynamic columns of interest found for {agg_level.upper()}.")
            else:
                print(f"\n  Showing relevant columns for {agg_level.upper()} (sample of first 15 rows):")
                # Sort by UnitName and Year to see lookbacks for the same unit together
                df_agg_sorted = df_agg.sort_values(by=['UnitName', 'Year'])
                print(df_agg_sorted[cols_to_show_dynamic].head(40).to_string())

            # --- NaN Analysis for CovIncTerm_N and CovTerm_N --- 
            print(f"\n  NaN Analysis for Covariance Terms ({agg_level.upper()}):")
            for n_val in range(1, max_n_to_check + 1):
                for cov_type in ['CovTerm', 'CovIncTerm']:
                    cov_col_name = f'{cov_type}_{n_val}'
                    if cov_col_name in df_agg.columns:
                        nan_count = df_agg[cov_col_name].isnull().sum()
                        total_count = len(df_agg[cov_col_name])
                        if total_count > 0:
                            non_nan_percent = (total_count - nan_count) / total_count * 100
                            print(f"    {cov_col_name}: {non_nan_percent:.2f}% non-NaN ({total_count - nan_count}/{total_count})")
                        else:
                            print(f"    {cov_col_name}: Column present but no data rows.")
                    # else:
                    #     print(f"    {cov_col_name}: Column not found.")

            # --- Check NumMembers_N --- 
            print(f"\n  Value Counts for NumMembers_N ({agg_level.upper()}):")
            for n_val in range(1, max_n_to_check + 1):
                num_members_col = f'NumMembers_{n_val}'
                if num_members_col in df_agg.columns:
                    print(f"    {num_members_col}:")
                    print(df_agg[num_members_col].value_counts().sort_index().to_string())
                # else:
                #     print(f"    {num_members_col}: Column not found.")

            # --- Check SanityCheck_N (both Pop and Inc) ---
            print(f"\n  Descriptive Stats for SanityCheck_N ({agg_level.upper()}):")
            for n_val in range(1, max_n_to_check + 1):
                for sanity_type in ['SanityCheckPop', 'SanityCheckInc']:
                    sanity_col_name = f'{sanity_type}_{n_val}'
                    if sanity_col_name in df_agg.columns:
                        print(f"    {sanity_col_name}:")
                        print(df_agg[sanity_col_name].describe().to_string())
                        # Check how many are very close to zero vs. NaN
                        close_to_zero = df_agg[np.isclose(df_agg[sanity_col_name].fillna(np.nan), 0, atol=1e-9)].shape[0]
                        num_nan = df_agg[sanity_col_name].isnull().sum()
                        print(f"      Count close to zero (abs < 1e-9): {close_to_zero}")
                        print(f"      Count NaN: {num_nan}")
                    # else:
                    #     print(f"    {sanity_col_name}: Column not found.")

        print(f"\nInspection of {file_path} focused on tr/bg complete.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

def test_population_hypothesis(file_path):
    """
    Loads analysis_Price2.pkl and tests the hypothesis that the total tract-level
    population is different from the total community-level population.
    """
    print(f"--- Testing Population Hypothesis on: {file_path} ---")
    try:
        df = pd.read_pickle(file_path)
        print("Successfully loaded DataFrame.")

        # Define the specific period for analysis, matching the pie charts
        TARGET_FIRST_YEAR = 14
        TARGET_N = 5
        TARGET_YEAR = TARGET_FIRST_YEAR + TARGET_N

        # Filter for the specific time period
        df_period = df[(df['FirstYear'] == TARGET_FIRST_YEAR) & (df['Year'] == TARGET_YEAR)].copy()

        # 1. Analyze Tract Level
        df_tr = df_period[df_period['Agg'] == 'tr'].copy()
        if not df_tr.empty:
            pop_initial_tr = df_tr['PopInitialN'].sum()
            print(f"Total Initial Population (Tract Level, N=5, FirstYear=14): {pop_initial_tr:,.0f}")
        else:
            pop_initial_tr = 0
            print("No Tract (tr) level data found for the specified period.")

        # 2. Analyze Community Level
        df_cm = df_period[df_period['Agg'] == 'cm'].copy()
        if not df_cm.empty:
            pop_initial_cm = df_cm['PopInitialN'].sum()
            print(f"Total Initial Population (Community Level, N=5, FirstYear=14): {pop_initial_cm:,.0f}")
        else:
            pop_initial_cm = 0
            print("No Community (cm) level data found for the specified period.")
            
        # 3. Compare and Conclude
        if pop_initial_tr > 0:
            difference = pop_initial_tr - pop_initial_cm
            percentage_diff = (difference / pop_initial_tr) * 100
            print(f"\\nDifference (Tract - Community): {difference:,.0f}")
            print(f"Percentage of Tract Population Missing from Community Aggregation: {percentage_diff:.2f}%")
            
            if difference > 0:
                print("\\nHypothesis Confirmed: The population discrepancy originates in this file.")
            else:
                print("\\nHypothesis Not Confirmed: Population totals are consistent or community is larger.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def inspect_csv_columns(file_path):
    """Loads a CSV and prints its column headers."""
    print(f"--- Inspecting Columns for: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
        print("Successfully loaded CSV file.")
        print("\\nColumn Headers:")
        print(df.columns.tolist())
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_pkl.py <path_to_pickle_file>")
        sys.exit(1)

    pickle_file_path = sys.argv[1]

    try:
        with open(pickle_file_path, 'rb') as f:
            loaded_object = pickle.load(f)
        print(f"Successfully loaded data from {pickle_file_path}")

        # Check the type of the loaded object and call the appropriate function
        if isinstance(loaded_object, pd.DataFrame):
            inspect_dataframe(loaded_object, pickle_file_path)
        elif isinstance(loaded_object, dict):
            inspect_dict_data(loaded_object, pickle_file_path)
        else:
            print(f"Error: Loaded object is neither a pandas DataFrame nor a dictionary (Type: {type(loaded_object)}). Cannot inspect.")

    except FileNotFoundError:
        print(f"Error: File not found at {pickle_file_path}")
    except Exception as e:
        print(f"Error loading or inspecting data from {pickle_file_path}: {e}")

if __name__ == "__main__":
    # Default file to inspect
    pkl_file_path = "results/analysis_Price2.pkl"
    
    # Allow overriding with a command-line argument
    if len(sys.argv) > 1:
        pkl_file_path = sys.argv[1]
        
    # inspect_analysis_price2(pkl_file_path)
    # test_population_hypothesis(pkl_file_path)
    inspect_csv_columns('output_terms/bg_cm_exported_terms.csv')

    # --- Remove or comment out the hardcoded inspection blocks below --- 
    # # --- Add inspection for outputs.pkl --- 
    # outputs_filepath = "results/outputs.pkl"
    # print(f"\\n\\n--- Inspecting {outputs_filepath} --- ")
    # try:
    #     with open(outputs_filepath, 'rb') as f:
    #         outputs_data = pickle.load(f)
    #     print("File loaded successfully.")
    # except FileNotFoundError:
    #     print(f"Error: File not found at {outputs_filepath}")
    #     outputs_data = None
    # except Exception as e:
    #     print(f"Error loading pickle file: {e}")
    #     outputs_data = None
    # 
    # if outputs_data is not None:
    #     # ... (rest of outputs.pkl inspection logic)
    #     pass # Placeholder
    # # --- End inspection for outputs.pkl --- 
    # 
    # # --- Add inspection for census_data.pkl --- 
    # census_data_filepath = "data/census_data.pkl" # Correct path
    # print(f"\\n\\n--- Inspecting {census_data_filepath} --- ")
    # try:
    #     with open(census_data_filepath, 'rb') as f:
    #         census_data = pickle.load(f)
    #     print("File loaded successfully.")
    # except FileNotFoundError:
    #     print(f"Error: File not found at {census_data_filepath}")
    #     census_data = None
    # except Exception as e:
    #     print(f"Error loading pickle file: {e}")
    #     census_data = None
    # 
    # if census_data is not None:
    #     # ... (rest of census_data.pkl inspection logic)
    #     pass # Placeholder
    # # --- End inspection for census_data.pkl --- 