import pandas as pd
import numpy as np
import pickle
import os
import sys

# Import functions from saveCensusDataV2
from saveCensusDataV2 import genAggregatedDFs, save_data, income_vars, pop_vars, COUNTY_FIPS_TO_NAME_MAP

# Model configurations
MODEL_CONFIGS = {
    'migration': {
        'files': {
            14: "null_model/simulation_results/migration_all_block_groups/Census-Data-2014.csv",
            19: "null_model/simulation_results/migration_all_block_groups/Simulated-Data-2019.csv"
        },
        'output': 'calculation_scripts/data/null_model/migration_all_BG.pkl',
        'name': 'Random Migration Null Model'
    },
    'swapping': {
        'files': {
            14: "null_model/simulation_results/swapping_all_block_groups/Census-Data-2014.csv",
            19: "null_model/simulation_results/swapping_all_block_groups/Simulated-Data-2019.csv"
        },
        'output': 'calculation_scripts/data/null_model/swapping_all_BG.pkl',
        'name': 'Random Swapping Null Model'
    },
    'uniform': {
        'files': {
            14: "null_model/simulation_results/uniform/Uniform-Model-2014.csv",
            19: "null_model/simulation_results/uniform/Uniform-Model-2019.csv"
        },
        'output': 'calculation_scripts/data/null_model/uniform_all_BG.pkl',
        'name': 'Uniform Migration Null Model'
    },
    'migration_percentile_90': {
        'files': {
            14: "null_model/simulation_results/migration_percentile/Census-Data-2014_90.csv",
            19: "null_model/simulation_results/migration_percentile/Simulated-Data-2019_90.csv"
        },
        'output': 'calculation_scripts/data/null_model/migration_percentile_90.pkl',
        'name': 'Migration Percentile 90 Null Model'
    },
    'migration_percentile_80': {
        'files': {
            14: "null_model/simulation_results/migration_percentile/Census-Data-2014_80.csv",
            19: "null_model/simulation_results/migration_percentile/Simulated-Data-2019_80.csv"
        },
        'output': 'calculation_scripts/data/null_model/migration_percentile_80.pkl',
        'name': 'Migration Percentile 80 Null Model'
    },
    'swapping_percentile_90': {
        'files': {
            14: "null_model/simulation_results/swapping_percentile/Census-Data-2014_90.csv",
            19: "null_model/simulation_results/swapping_percentile/Simulated-Data-2019_90.csv"
        },
        'output': 'calculation_scripts/data/null_model/swapping_percentile_90.pkl',
        'name': 'Swapping Percentile 90 Null Model'
    },
    'swapping_percentile_80': {
        'files': {
            14: "null_model/simulation_results/swapping_percentile/Census-Data-2014_80.csv",
            19: "null_model/simulation_results/swapping_percentile/Simulated-Data-2019_80.csv"
        },
        'output': 'calculation_scripts/data/null_model/swapping_percentile_80.pkl',
        'name': 'Swapping Percentile 80 Null Model'
    },
    'migration_check': {
        'files': {
            14: "null_model/simulation_results/migration_check/Census-Data-2014.csv",
            19: "null_model/simulation_results/migration_check/Simulated-Data-2019.csv"
        },
        'output': 'calculation_scripts/data/null_model/migraion_check.pkl',
        'name': 'Migration Check Null Model'
    },
    'migration_meeting': {
        'files': {
            14: "null_model/simulation_results/data_null_migration/Census-Data-2014.csv",
            19: "null_model/simulation_results/data_null_migration/Simulated-Data-2019.csv"
        },
        'output': 'calculation_scripts/data/null_model/migraion_meeting.pkl',
        'name': 'Migration Meeting'
    },
    'migration_gravity': {
        'files': {
            14: "null_model/simulation_results/data_null_gravity/Census-Data-2014.csv",
            19: "null_model/simulation_results/data_null_gravity/Simulated-Data-2019.csv"
        },
        'output': 'calculation_scripts/data/null_model/migration_gravity.pkl',
        'name': 'Gravity Migration Null Model (complete graph)'
    }
}

# Check command-line argument for model type
if 'swapping' in sys.argv:
    MODEL_TYPE = 'swapping'
elif 'uniform' in sys.argv:
    MODEL_TYPE = 'uniform'
elif 'migration' in sys.argv:
    MODEL_TYPE = 'migration'
elif 'migration_percentile_90' in sys.argv:
    MODEL_TYPE = 'migration_percentile_90'
elif 'migration_percentile_80' in sys.argv:
    MODEL_TYPE = 'migration_percentile_80'
elif 'swapping_percentile_90' in sys.argv:
    MODEL_TYPE = 'swapping_percentile_90'
elif 'swapping_percentile_80' in sys.argv:
    MODEL_TYPE = 'swapping_percentile_80'
elif 'migration_check' in sys.argv:
    MODEL_TYPE = 'migration_check'
elif 'migration_meeting' in sys.argv:
    MODEL_TYPE = 'migration_meeting'
elif 'migration_gravity' in sys.argv:
    MODEL_TYPE = 'migration_gravity'
else:
    MODEL_TYPE = 'migration'

# Get configuration for selected model
model_config = MODEL_CONFIGS[MODEL_TYPE]
null_model_files = model_config['files']
OUTPUT_FILE = model_config['output']
MODEL_NAME = model_config['name']

def loadNullModelData():
    """
    Load null model data from CSV files and process through the same aggregation pipeline
    as saveCensusDataV2.py, but reading from CSV files instead of Census API.
    """
    
    # --- Load Community Mapping Data --- 
    try:
        communities = pd.concat([
            pd.read_csv('matched_chicago_data.csv'), 
            pd.read_csv('matched_chicagoLand_data.csv')
        ], axis=0)
        communities['GEOID'] = communities['GEOID'].astype(str)
        communities = communities.rename(columns={'GEOID': 'tract'}).set_index('tract')
        communities = communities[~communities['community'].isna()]
        communities['community'] = communities['community'].str.upper()
        
        # Create county-specific community names
        communities['county'] = communities.index.str[-9:-6]
        communities['community'] = communities['community'] + '_' + communities['county']
        
        print("Successfully loaded and processed community mapping data.")
    except FileNotFoundError:
        print("ERROR: Community mapping CSV file(s) not found. Cannot proceed with community aggregation.")
        communities = None
    except Exception as e:
        print(f"ERROR: Failed to load or process community mapping data: {e}")
        communities = None
    # -----------------------------------

    datas = {}

    for year, csv_file in null_model_files.items():
        print(f"\n{'='*60}")
        print(f"Processing {MODEL_NAME} data for year 20{year}")
        print(f"{'='*60}\n")
        
        csv_path = csv_file
        
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"ERROR: File not found: {csv_path}")
            print(f"Skipping year {year}")
            continue
        
        try:
            # Read the CSV file
            print(f"Reading CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"  SUCCESS: Loaded {len(df)} rows from {csv_path}")
            
            # Display column names for verification
            print(f"  Columns in CSV: {list(df.columns)}")
            
            # Ensure required columns exist
            required_geo_cols = ['state', 'county', 'tract', 'block group']
            missing_geo_cols = [col for col in required_geo_cols if col not in df.columns]
            if missing_geo_cols:
                print(f"ERROR: Missing required geographic columns: {missing_geo_cols}")
                print(f"Skipping year {year}")
                continue
            
            # Check for income variables
            available_income_vars = [var for var in income_vars if var in df.columns]
            if not available_income_vars:
                print(f"ERROR: No income variables found in CSV")
                print(f"Skipping year {year}")
                continue
            
            print(f"  Found {len(available_income_vars)} income variables")
            
            # Ensure B19001_001E exists (population/households total)
            if 'B19001_001E' not in df.columns:
                print(f"ERROR: B19001_001E (population total) not found in CSV")
                print(f"Skipping year {year}")
                continue
            
            # Convert geographic identifiers to strings and ensure proper formatting
            for col in required_geo_cols:
                df[col] = df[col].astype(str).str.strip()
            
            # Pad tract to 6 digits if needed
            df['tract'] = df['tract'].str.zfill(6)
            
            # Pad block group to 1 digit if needed
            df['block group'] = df['block group'].str.zfill(1)
            
            # Pad county to 3 digits if needed
            df['county'] = df['county'].str.zfill(3)
            
            # Pad state to 2 digits if needed
            df['state'] = df['state'].str.zfill(2)
            
            print(f"  Formatted geographic identifiers")
            
            # Convert numeric columns to numeric type
            variables_to_process = available_income_vars + ['B19001_001E']
            for var in variables_to_process:
                if var in df.columns:
                    df[var] = pd.to_numeric(df[var], errors='coerce')
            
            # Filter out rows where all income variables are zero or NaN
            row_sums = df[available_income_vars].sum(axis=1)
            df = df[row_sums > 0]
            print(f"  After filtering zero-sum rows: {len(df)} rows remain")
            
            if df.empty:
                print(f"WARNING: No valid data rows after filtering for year {year}")
                continue
            
            # NOTE: Unlike saveCensusDataV2.py, we assume B19001_001E in the CSV
            # is already the population value (not households that need to be multiplied)
            # If this assumption is incorrect, the multiplication logic can be added here
            
            print(f"  Data loaded and prepared for aggregation")
            print(f"  Sample B19001_001E values: {df['B19001_001E'].head().tolist()}")
            
            # --- Prepare data for aggregation ---
            # The input level is 'bg' (block group) since we have block group data
            input_level = 'bg'
            
            # Put the dataframe in a list as genAggregatedDFs expects a list of dataframes
            data_for_aggregation = [df]
            
            print(f"\n  Calling genAggregatedDFs with input_level='{input_level}'")
            
            # Call the aggregation function
            aggs = ['bg', 'tr', 'cm', 'ct', 'st']
            try:
                aggregated_results, aggs_returned = genAggregatedDFs(
                    aggs=aggs,
                    data_to_aggregate=data_for_aggregation,
                    input_level=input_level,
                    variables=variables_to_process,
                    year=year,
                    community_mapping_df=communities
                )
                
                # Store results
                datas[year] = dict(zip(aggs_returned, aggregated_results))
                
                print(f"\n  SUCCESS: Aggregation completed for year {year}")
                print(f"  Aggregation levels created: {aggs_returned}")
                
                # Print summary statistics for each level
                for agg_level, agg_df in datas[year].items():
                    if not agg_df.empty:
                        print(f"    {agg_level}: {len(agg_df)} records")
                    else:
                        print(f"    {agg_level}: EMPTY")
                
            except Exception as e:
                print(f"ERROR: Aggregation failed for year {year}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        except Exception as e:
            print(f"ERROR: Failed to process {csv_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return datas


def main():
    """
    Main function to load null model data and save to pickle file
    """
    print("="*60)
    print(f"MODEL DATA LOADER: {MODEL_NAME}")
    print("="*60)
    print(f"Model type: {MODEL_TYPE}")
    print(f"Input files: {list(null_model_files.values())}")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*60 + "\n")
    
    # Load and process the data
    null_model_data = loadNullModelData()
    
    if not null_model_data:
        print("\nERROR: No data was successfully loaded")
        return
    
    # Save the data
    print(f"\n{'='*60}")
    print(f"Saving processed data to: {OUTPUT_FILE}")
    print(f"{'='*60}\n")
    
    save_data(null_model_data, OUTPUT_FILE)
    
    print("\n" + "="*60)
    print(f"{MODEL_NAME.upper()} DATA PROCESSING COMPLETE")
    print("="*60)
    print(f"Years processed: {list(null_model_data.keys())}")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*60)


if __name__ == "__main__":
    main()

