import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import scipy as sp

from scipy.optimize import curve_fit
from scipy.stats import lognorm

# Set the API key (replace 'YOUR_API_KEY' with your actual key)
api_key = "35d314060d56f894db2f7621b0e5e5f7eca9af27"
# years   = np.linspace(10,23,14,dtype='int')
# years   = np.linspace(10,23,14,dtype='int')
years   = np.linspace(9,23,15,dtype='int')

aggs    = ['bg','tr','cm','ct','st']
# Define the desired variables for wage distribution
# Here we use B19001 for household income brackets; you can adjust based on what you need.
income_vars = [
    # "B19001_001E",  # Total households
    "B19001_002E",  # Households with income less than $10,000
    "B19001_003E",  # Households with income between $10,000 and $14,999
    "B19001_004E",  # Households with income between $15,000 and $19,999
    "B19001_005E",  # Households with income between $20,000 and $24,999
    "B19001_006E",  # Households with income between $25,000 and $29,999
    "B19001_007E",  # Households with income between $30,000 and $34,999
    "B19001_008E",  # Households with income between $35,000 and $39,999
    "B19001_009E",  # Households with income between $40,000 and $44,999
    "B19001_010E",  # Households with income between $45,000 and $49,999
    "B19001_011E",  # Households with income between $50,000 and $59,999
    "B19001_012E",  # Households with income between $60,000 and $74,999
    "B19001_013E",  # Households with income between $75,000 and $99,999
    "B19001_014E",  # Households with income between $100,000 and $124,999
    "B19001_015E",  # Households with income between $125,000 and $149,999
    "B19001_016E",  # Households with income between $150,000 and $199,999
    "B19001_017E",  # Households with income $200,000 or more
]

pop_vars = [
    "B01003_001E",  # Population
    "B19001_001E",  # Total Households
    "B25010_001E"   # Average Household Size
]

geographies = {
    "state": {"for": "state:*"},
    "county": {"for": "county:*", "in": "state:17"},  # Example for Illinois (state code 17)
    "place": {"for": "place:*", "in": "state:17"},   # Example for Illinois places
    "tract": {"for": "tract:*", "in": "state:17&in=county:031"},  # Example for Cook County, IL
    "block group": {"for": "block group:*", "in": "state:17&in=county:031"},  # Example for Cook County, IL
}
variable=["NAME","B19083_001E","B19013_001E"]



# Define the variables you need (total population and population change, if available)
# population_vars =   # Total population variable

# Convert the list of variables to a comma-separated string
variable_str = ",".join(income_vars)

# Define the geography: Get data for block groups in the Chicago metro area
# List of counties in the Chicago metro area
counties = ["031",  "043",  "089",  "093",  "097", "111",  "197"]#,  "089", "127",  "059"]  
states   = ["17" ,  "17" ,  "17" ,  "17" ,  "17" , "17" ,  "17"]#,  "18" , "18" ,  "55" ]  # State codes
cty_name = ["Cook", "DuPg", "Kane", "Kndl", "Lke", "McHn", "Will"]#, "Lke", "Prtr", "Knsh"]

# Define the geography: Get data for block groups in the Chicago metro area
# List of counties in the Chicago metro area
# counties = ["031", "089"]
# states   = ["17", "17"]
# cty_name = ["Cook", "Lke"]

#GENERATES DATAFRAMES OF INCOME DISTRIBUTION FOR EACH LEVEL OF AGGREGATION

def matchTracts(df_tract, comms,identifiers=False):

    # Ensure input df_tract has necessary columns
    if 'tract' not in df_tract.columns or 'county' not in df_tract.columns:
        print("Error in matchTracts: Input DataFrame missing 'tract' or 'county' column.")
        # Decide how to handle: return unmodified, raise error, etc.
        # For now, return unmodified to avoid breaking flow, but flag it.
        return df_tract, False

    # Ensure comms has necessary columns
    if 'tract' not in comms.columns or 'county' not in comms.columns:
         print("Error in matchTracts: `comms` DataFrame missing 'tract' or 'county' column.")
         return df_tract, False

    # 1. Find '(tract, county)' pairs in comms
    # Correctly create a set of tuples
    tract_county_pairs_in_comms = set(tuple(x) for x in comms[['tract', 'county']].to_numpy())

    # Create (tract, county) pairs in the input df_tract for checking
    # Use list comprehension for direct boolean mask creation
    df_tract_pairs = list(zip(df_tract['tract'].astype(str), df_tract['county'].astype(str)))
    
    # Check which pairs from df_tract are NOT in the comms set
    leftover_mask = [pair not in tract_county_pairs_in_comms for pair in df_tract_pairs]
    
    # --- The rest of matchTracts remains the same, operating on leftovers ---
    # It will still potentially modify the 'tract' column of the leftovers based on 4-digit prefix

    # 2. Filter DataFrame A to get only the leftover rows
    df_tract_leftover = df_tract[leftover_mask].copy() # Use .copy()

    # --- If no leftovers, we can stop ---
    if df_tract_leftover.empty:
        return df_tract,False
        # 3. Create the grouping key (first 4 digits of 'tract') for the leftover rows
    
    df_tract_leftover['tract_prefix'] = df_tract_leftover['tract'].astype(str).str[:4]
    
    if not identifiers:
        df_tract_leftover['temp_tract_id'] = df_tract_leftover['temp_tract_id'].astype(str).str[:-2]


    # 4. Define aggregation rules: 'sum' for numeric, 'first' for others
    # Exclude the original 'tract' column and the new 'tract_prefix' from aggregation targets
    cols_to_aggregate = [col for col in df_tract_leftover.columns if col not in ['tract', 'tract_prefix']]
    agg_dict = {}
    for col in cols_to_aggregate:
        if pd.api.types.is_numeric_dtype(df_tract_leftover[col]):
            agg_dict[col] = 'sum'
        else:
            # Since identical within group, 'first' preserves the value
            agg_dict[col] = 'first'
    
    # 5. Group by prefix and apply the defined aggregations
    grouped_aggregated = df_tract_leftover.groupby('tract_prefix').agg(agg_dict)

    # print(grouped_aggregated)
    # input("problem children")

    # 6. Prepare the aggregated data for concatenation
    grouped_aggregated_reset = grouped_aggregated.reset_index() # Makes 'tract_prefix' a column
    # Rename 'tract_prefix' to 'tract'
    grouped_aggregated_prepared = grouped_aggregated_reset.rename(columns={'tract_prefix': 'tract'})
    # Append '00' to make tracts 6 digits
    grouped_aggregated_prepared['tract'] = grouped_aggregated_prepared['tract'].astype(str) + '00'
    if not identifiers:
        grouped_aggregated_prepared['temp_tract_id'] = grouped_aggregated_prepared['temp_tract_id'].astype(str) + '00'



    # Ensure column order matches original A (Important: Add 'tract' back to the list)
    original_cols_ordered = df_tract.columns.tolist() # Get original order
    grouped_aggregated_prepared = grouped_aggregated_prepared[original_cols_ordered]


    # 7. Get the rows from A that were *not* leftovers
    # Convert list mask to Pandas Series before negation
    leftover_mask_series = pd.Series(leftover_mask, index=df_tract.index)
    df_tract_kept = df_tract[~leftover_mask_series]

    # 8. Concatenate the kept original rows and the new aggregated rows
    df_tract_final = pd.concat([df_tract_kept, grouped_aggregated_prepared], ignore_index=True)
    if not identifiers:
        return df_tract_final.set_index('temp_tract_id',drop=True),True
    else:
        
        return df_tract_final.set_index(df_tract_final['state'] + df_tract_final['county'] + df_tract_final['tract'],drop=True),True

def genAggregatedDFs(aggs,data_to_aggregate, input_level, variables, year, community_mapping_df):
    """
    Aggregates census data from a starting geographic level (BG or TR) to higher ones.

    Args:
        data_to_aggregate (list): List of pandas DataFrames containing census data,
                                  all at the same geographic level (either 'bg' or 'tr').
        input_level (str): The geographic level of the data in data_to_aggregate ('bg' or 'tr').
        variables (list): List of census variable names (columns) to aggregate.
        year (int): The census year being processed.
        community_mapping_df (pd.DataFrame): DataFrame mapping tract IDs (index) to community, state, county.
    """
    print(f"  genAggregatedDFs ({year}): Starting aggregation from input level: '{input_level}' with {len(data_to_aggregate)} dataframe(s).")

    # If no data provided, return empty structures
    if not data_to_aggregate or input_level == 'none':
        print(f"  genAggregatedDFs ({year}): No data provided. Returning empty aggregation results.")
        # Define columns based on potential outputs and required identifiers
        cols_base = variables + ['state', 'county']
        cols_cm = cols_base + ['community']
        cols_tr = cols_cm + ['tract']
        cols_bg = cols_tr + ['block group']

        empty_df_bg = pd.DataFrame(columns=cols_bg)
        empty_df_tr = pd.DataFrame(columns=cols_tr)
        empty_df_cm = pd.DataFrame(columns=cols_cm)
        empty_df_ct = pd.DataFrame(columns=cols_base)
        empty_df_st = pd.DataFrame(columns=['state'] + variables)

        # Attempt to set expected indices, handling potential KeyErrors if columns are missing
        try:
            empty_df_bg = empty_df_bg.set_index(['state', 'county', 'tract', 'block group'])
        except KeyError: pass
        try:
            empty_df_tr = empty_df_tr.set_index(['state', 'county', 'tract'])
        except KeyError: pass
        try:
            empty_df_cm = empty_df_cm.set_index('community')
        except KeyError: pass
        try:
             empty_df_ct = empty_df_ct.set_index(['state', 'county'])
        except KeyError: pass
        try:
             empty_df_st = empty_df_st.set_index('state')
        except KeyError: pass

        # Return in the expected order: bg, tr, cm, ct, st
        return empty_df_bg, empty_df_tr, empty_df_cm, empty_df_ct, empty_df_st

    # --- Concatenate and Initial Cleaning --- 
    df_processed = pd.concat(data_to_aggregate, ignore_index=True)

    # Convert the relevant columns to numeric (income distributions are integers)
    # Ensure 'variables' only contains columns expected to be numeric
    numeric_vars = [v for v in variables if v in df_processed.columns] # Filter variables present
    df_processed[numeric_vars] = df_processed[numeric_vars].apply(pd.to_numeric, errors='coerce')

    # Filter out zero sum rows based on the numeric variables provided
    row_sums = df_processed[numeric_vars].sum(axis=1)
    df_processed = df_processed[row_sums != 0]

    if df_processed.empty:
         print(f"  genAggregatedDFs ({year}): Data became empty after cleaning. Returning empty results.")
         # Call self with empty list to get standard empty structures
         return genAggregatedDFs([], 'none', variables, year, None)

    # --- Prepare Identifiers & Community Mapping --- 
    # These steps rely on 'state', 'county', 'tract' existing in the input
    required_id_cols = ['state', 'county', 'tract']
    if not all(col in df_processed.columns for col in required_id_cols):
        print(f"ERROR in genAggregatedDFs ({year}): Input data missing required columns: {required_id_cols}. Cannot proceed.")
        return genAggregatedDFs([], 'none', variables, year, None) # Return empty

    # Use a temporary tract ID for linking with community data
    df_processed['temp_tract_id'] = df_processed['state'] + df_processed['county'] + df_processed['tract']

    # Create df_identifier from the unique tracts in the processed data
    df_identifier_unique = df_processed[required_id_cols].drop_duplicates()
    df_identifier = df_identifier_unique.set_index(df_identifier_unique['state'] + df_identifier_unique['county'] + df_identifier_unique['tract'])

    if community_mapping_df is None:
         print(f"ERROR in genAggregatedDFs ({year}): Community mapping DataFrame is not provided.")
         return genAggregatedDFs([], 'none', variables, year, None)

    # --- Add county column to community_mapping_df EARLY ---
    # Ensure index is string before slicing
    community_mapping_df.index = community_mapping_df.index.astype(str)
    community_mapping_df['tract'] = community_mapping_df.index.str[-6:]
    community_mapping_df['county'] = community_mapping_df.index.str[-9:-6] # Extract county
    community_mapping_df['state'] = community_mapping_df.index.str[:-9] # Extract county
    # Keep original index for potential later use if needed
    community_mapping_df['original_geoid_index'] = community_mapping_df.index 
    # Note: index is still the original GEOID here. Reset later if needed.
    # -------------------------------------------------------

    
    # --- Alignment using matchTracts (now expects county column) ---
    # df_identifier has 'county'. community_mapping_df has 'county'.
    df_identifier,_ = matchTracts(df_identifier, community_mapping_df,True)
    
    # This seems redundant/problematic - matchTracts is designed for census data -> comms, not vice-versa
    # community_mapping_df['state']=community_mapping_df['community'];community_mapping_df['county']=community_mapping_df['community'] #<<< This overwrites the county FIPS!
    # community_mapping_df,_ = matchTracts(community_mapping_df, df_identifier,True) #<<< Applying matchTracts to the map itself seems wrong
    # community_mapping_df = community_mapping_df.drop(columns=['state','county']).set_index(community_mapping_df['index']).drop(columns=['index']) #<<< This drops the county FIPS again
    
    # --- Revised comms creation (simpler, keeping county) ---
    # Let's use the *original* community mapping df (with added county) as the base
    # and merge necessary info from df_identifier if needed, but prioritize the map.
    # Reset index to use columns for merging/lookup
    # Select necessary columns from the map
    comms = community_mapping_df[['tract', 'county', 'community','state']].copy()
    comms.index.name = 'temp_tract_id'
    # Ensure tract/county are strings for consistent matching
    comms['tract'] = comms['tract'].astype(str).str.strip()
    comms['county'] = comms['county'].astype(str).str.strip()
    comms['state'] = comms['state'].astype(str).str.strip()
    # Drop duplicates based on tract and county to ensure unique mapping per tract/county pair
    comms = comms.drop_duplicates(subset=['tract', 'county','state'], keep='first')
    # --------------------------------------------------------
    # print(community_mapping_df[community_mapping_df['community']=='WINFIELD'])


    # Pre-calculate community identifiers for later merge
    # Use the cleaned 'comms' which already has unique community per tract/county
    commu_agg = comms.drop(['tract'],axis=1, errors='ignore').groupby(['community']).first()
    # --- Level-Specific Processing and Aggregation --- 
    df_blockgroup = None
    df_tract = None

    if input_level == 'bg':
        print(f"  genAggregatedDFs ({year}): Processing as Block Group input.")
        # Check for required columns first
        required_bg_cols = required_id_cols + ['block group']
        if not all(col in df_processed.columns for col in required_bg_cols):
             print(f"ERROR in genAggregatedDFs ({year}): Input level is 'bg' but required columns ({required_bg_cols}) are missing.")
             return genAggregatedDFs([], 'none', variables, year, None)
        
        # Strip whitespace from identifier columns BEFORE creating ID
        for col in required_bg_cols:
             if df_processed[col].dtype == 'object':
                 df_processed[col] = df_processed[col].astype(str).str.strip()

        # Set BG-level Index
        df_processed['ID'] = df_processed['state'] + df_processed['county'] + df_processed['tract'] + df_processed['block group']
        df_processed.set_index('ID', inplace=True, drop=True)
        df_processed = df_processed[~df_processed.index.duplicated(keep='first')]

        # Keep the processed BG data
        df_blockgroup = df_processed.copy()

        # Strip whitespace from temp_tract_id before potential use in matchTracts or aggregation
        if 'temp_tract_id' in df_blockgroup.columns:
            df_blockgroup['temp_tract_id'] = df_blockgroup['temp_tract_id'].astype(str).str.strip()

        # Ensure tract column is stripped before calling matchTracts (if it uses it)
        if 'tract' in df_blockgroup.columns:
            df_blockgroup['tract'] = df_blockgroup['tract'].astype(str).str.strip()

        # print(df_blockgroup)
        # input("bg - before matchTracts")


        df_blockgroup,success = matchTracts(df_blockgroup, comms) # matchTracts expects 'tract' and 'county' in comms
        if success:
             # This assignment might be wrong if matchTracts modified things based on 4-digit prefix
             # It assumes the state/county/tract *after* matchTracts should form the ID.
            df_blockgroup['temp_tract_id'] = df_blockgroup['state'] + df_blockgroup['county'] + df_blockgroup['tract']


        # --- Modify Merge to use Tract and County ---
        # Ensure df_blockgroup has 'tract' and 'county' columns before merge
        if 'tract' not in df_blockgroup.columns or 'county' not in df_blockgroup.columns:
             print("ERROR: df_blockgroup missing 'tract' or 'county' before final merge.")
             # Handle error... maybe return empty?
        else:
             # Ensure types match for merge keys
             df_blockgroup['tract'] = df_blockgroup['tract'].astype(str).str.strip()
             df_blockgroup['county'] = df_blockgroup['county'].astype(str).str.strip()
             
             # Merge on both tract and county
             # We only need the 'community' column from comms
             df_blockgroup = pd.merge(
                 df_blockgroup, 
                 comms[['tract', 'county', 'community']], # Select join keys + community
                 on=['tract', 'county'], # Use both columns for merging
                 how='left'
             )
        # --------------------------------------------
             
        df_blockgroup = df_blockgroup.dropna(subset=['community'])

        # Aggregate BG -> Tract
        # Ensure temp_tract_id (now index of df_blockgroup) is clean - should be from matchTracts if index is set there
        # If matchTracts doesn't set index, we need to strip before groupby
        # Assuming matchTracts returns with temp_tract_id as index
        cols_to_drop_for_tract_agg = ['tract','block group', 'state', 'county','community']
        
        df_tract = df_blockgroup.drop(columns=cols_to_drop_for_tract_agg, errors='ignore').groupby(['temp_tract_id']).sum()

        df_tract = pd.merge(df_tract, comms, left_index=True, right_index=True, how='left') # Keep all aggregated tracts

        df_blockgroup.set_index(df_blockgroup['temp_tract_id']+df_blockgroup['block group'],inplace=True,drop=True)
    


    elif input_level == 'tr':
        print(f"  genAggregatedDFs ({year}): Processing as Tract input.")
        # Check for required columns first
        if not all(col in df_processed.columns for col in required_id_cols):
             print(f"ERROR in genAggregatedDFs ({year}): Input level is 'tr' but required columns ({required_id_cols}) are missing.")
             return genAggregatedDFs([], 'none', variables, year, None)
        
        # Strip whitespace from identifier columns BEFORE creating ID
        for col in required_id_cols:
             if df_processed[col].dtype == 'object':
                 df_processed[col] = df_processed[col].astype(str).str.strip()
                 
        # Strip temp_tract_id as well before setting index
        if 'temp_tract_id' in df_processed.columns:
            df_processed['temp_tract_id'] = df_processed['temp_tract_id'].astype(str).str.strip()

        # Set Tract-level Index using temp_tract_id
        # df_processed['ID'] = df_processed['state'] + df_processed['county'] + df_processed['tract']
        df_processed.set_index('temp_tract_id', inplace=True, drop=True)
        df_processed = df_processed[~df_processed.index.duplicated(keep='first')]

        # The input data IS the tract data
        df_tract = df_processed.copy()


        # Merge tract data with community info
        df_tract = pd.merge(df_tract, comms.drop(columns=['state', 'county', 'tract'], errors='ignore'), left_index=True, right_index=True, how='left') # Avoid duplicating state/county/tract cols
        # df_tract now contains original tract numerics and state, county, tract, community identifiers

    else:
        print(f"ERROR in genAggregatedDFs ({year}): Invalid input_level '{input_level}'.")
        return genAggregatedDFs([], 'none', variables, year, None)

    # --- Aggregate Upwards from Tract --- 
    if df_tract is None or df_tract.empty:
        print(f"  genAggregatedDFs ({year}): Tract data is empty after initial processing. Returning empty results.")
        return genAggregatedDFs([], 'none', variables, year, None)

    # Check required columns for further aggregation
    # Also strip community column before grouping
    if 'community' in df_tract.columns:
        df_tract['community'] = df_tract['community'].astype(str).str.strip()
    else:
        print(f"ERROR in genAggregatedDFs ({year}): df_tract is missing community column needed for aggregation.")
        return df_blockgroup, df_tract, pd.DataFrame(), pd.DataFrame(), pd.DataFrame() 
        
    if not all(col in df_tract.columns for col in ['state', 'county']):
         print(f"ERROR in genAggregatedDFs ({year}): df_tract is missing state or county columns needed for aggregation.")
         return df_blockgroup, df_tract, pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Return what we have + empty

    # Aggregate Tract -> Community
    # Group by stripped community column
    cols_to_drop_for_cm_agg = list(set(df_tract.columns) - set(numeric_vars) - set(['community']))
    df_community = df_tract.drop(columns=cols_to_drop_for_cm_agg, errors='ignore').groupby(['community']).sum()

    # Ensure commu_agg index is stripped before merge
    commu_agg.index = commu_agg.index.astype(str).str.strip()
    df_community = pd.merge(df_community, commu_agg, left_index=True, right_index=True, how='left') # Add back state/county for this community
    
    # --- NEW: Create 5-digit county FIPS code for consistency ---
    if 'state' in df_community.columns and 'county' in df_community.columns:
        df_community['county'] = df_community['state'].astype(str).str.strip() + df_community['county'].astype(str).str.strip()
        print(f"    Created 5-digit FIPS for 'county' column in community DataFrame for year {year}.")
    # --- END NEW ---
    
    # Strip index after merge and apply upper case
    df_community.index = df_community.index.astype(str).str.strip().str.upper()

    # --- DEBUG PRINT for df_tract before county aggregation ---
    if str(year) == '14':
        print(f"  DEBUG genAggregatedDFs ({year}): df_tract before county aggregation:")
        if 'county' in df_tract.columns and 'state' in df_tract.columns:
            df_tract_cook_filter = (df_tract['county'].astype(str).str.strip() == '031') & (df_tract['state'].astype(str).str.strip() == '17')
            if df_tract_cook_filter.any():
                df_tract_cook = df_tract[df_tract_cook_filter]
                print(f"    df_tract_cook (Cook County tracts) dtypes:\n{df_tract_cook.dtypes}")
                if 'B19001_001E' in df_tract_cook.columns:
                    print(f"    df_tract_cook B19001_001E head:\n{df_tract_cook[['B19001_001E']].head()}")
                    print(f"    df_tract_cook B19001_001E sum: {df_tract_cook['B19001_001E'].sum()}")
                    print(f"    df_tract_cook B19001_001E min: {df_tract_cook['B19001_001E'].min()}")
                    print(f"    df_tract_cook B19001_001E max: {df_tract_cook['B19001_001E'].max()}")
                    print(f"    Is B19001_001E in df_tract_cook NaN? {df_tract_cook['B19001_001E'].isnull().any()}")
                    print(f"    Is B19001_001E in df_tract_cook Inf? {np.isinf(df_tract_cook['B19001_001E']).any()}")
                    print(f"    Sample B19001_001E values from df_tract_cook (first 10 non-NaN):\n{df_tract_cook['B19001_001E'].dropna().head(10)}")

                    # --- Add debug for problematic tracts ---
                    problematic_tracts_filter = df_tract_cook['B19001_001E'] < -1000000
                    if problematic_tracts_filter.any():
                        problematic_tract_ids = df_tract_cook[problematic_tracts_filter].index.tolist()
                        print(f"      Found problematic Cook County tract IDs in df_tract_cook with B19001_001E < -1,000,000: {problematic_tract_ids}")
                        print(f"      Problematic tract details from df_tract_cook:\n{df_tract_cook[problematic_tracts_filter][['B19001_001E', 'state', 'county', 'tract']]}")

                        # Now check the contributing block groups if df_blockgroup is available (i.e., input_level was 'bg')
                        if df_blockgroup is not None and not df_blockgroup.empty:
                            print(f"      Inspecting contributing block groups for these problematic tracts:")
                            # Ensure temp_tract_id is the index or a column in df_blockgroup to filter
                            # df_blockgroup index is temp_tract_id + block group id ; temp_tract_id is also a column
                            if 'temp_tract_id' in df_blockgroup.columns: # df_blockgroup.index is temp_tract_id+block group
                                for pt_id in problematic_tract_ids:
                                    contributing_bgs = df_blockgroup[df_blockgroup['temp_tract_id'] == pt_id]
                                    if not contributing_bgs.empty:
                                        print(f"        Block groups contributing to problematic tract {pt_id} (from df_blockgroup):")
                                        print(contributing_bgs[['B19001_001E', 'state', 'county', 'tract', 'block group', 'temp_tract_id']].head())
                                        print(f"        Sum of B19001_001E for BGs in tract {pt_id}: {contributing_bgs['B19001_001E'].sum()}")
                                        print(f"        Min/Max of B19001_001E for BGs in tract {pt_id}: {contributing_bgs['B19001_001E'].min()} / {contributing_bgs['B19001_001E'].max()}")
                                    else:
                                        print(f"        No block groups found in df_blockgroup for problematic tract ID: {pt_id}")
                            else:
                                print("        Cannot inspect block groups: 'temp_tract_id' not in df_blockgroup.columns or df_blockgroup index not set as expected.")
                        else:
                            print("        df_blockgroup is None or empty; cannot inspect contributing block groups. (Likely TR input level)")
                    else:
                        print("      No tracts in df_tract_cook found with B19001_001E < -1,000,000.")
                    # --- End debug for problematic tracts ---
                else:
                    print("      Warning: B19001_001E column not found in df_tract_cook.")
            else:
                print("      No Cook County (031 in state 17) tracts found in df_tract for year 14.")
        else:
            print("      Warning: 'county' or 'state' column not in df_tract for debug print.")
    # --- END DEBUG ---

    # Aggregate Tract -> County
    # Need state and county columns in df_tract, strip them before use
    if 'state' in df_tract.columns: df_tract['state'] = df_tract['state'].astype(str).str.strip()
    if 'county' in df_tract.columns: df_tract['county'] = df_tract['county'].astype(str).str.strip()
    
    county_identifiers = df_tract[['state', 'county']].drop_duplicates().set_index('county') # Index is already stripped county
    cols_to_drop_for_ct_agg = list(set(df_tract.columns) - set(numeric_vars) - set(['county']))
    df_county = df_tract.drop(columns=cols_to_drop_for_ct_agg, errors='ignore').groupby(['county']).sum()
    df_county = pd.merge(df_county, county_identifiers, left_index=True, right_index=True, how='left')
    # Set composite State+County index using stripped columns
    df_county.set_index(df_county['state'] + df_county.index, inplace=True) # state and index are already stripped
    
    # --- DEBUG PRINT for df_county before state aggregation ---
    if str(year) == '14':
        print(f"  DEBUG genAggregatedDFs ({year}): df_county before state aggregation:")
        print(f"    df_county dtypes:\n{df_county.dtypes}")
        print(f"    df_county B19001_001E head:\n{df_county[['B19001_001E']].head()}")
        print(f"    Is B19001_001E in df_county NaN? {df_county['B19001_001E'].isnull().any()}")
        print(f"    Is B19001_001E in df_county Inf? {np.isinf(df_county['B19001_001E']).any()}")
    # --- END DEBUG ---

    # Aggregate County -> State
    # Ensure state column is stripped before grouping
    # State is already part of the index, and was stripped before setting index
    cols_to_drop_for_st_agg = list(set(df_county.columns) - set(numeric_vars) - set(['state'])) # 'state' might be column or index level
    if 'state' in df_county.columns: # If state is still a column
         df_county['state'] = df_county['state'].astype(str).str.strip()
         df_state = df_county.drop(columns=cols_to_drop_for_st_agg, errors='ignore').groupby(['state']).sum()
    else: # If state is part of the index
        # We need to group by the state part of the index. Assuming index is 'state'+'county'
        df_state = df_county.drop(columns=[col for col in cols_to_drop_for_st_agg if col != 'state'], errors='ignore').groupby(level=0).sum()

    # --- DEBUG PRINT for df_state ---
    if str(year) == '14' and 'B19001_001E' in df_state.columns: # Ensure year is compared as string if it might be int
        print(f"  DEBUG genAggregatedDFs ({year}): df_state head for B19001_001E:")
        print(df_state[['B19001_001E']].head())
    # --- END DEBUG ---

    # Return the results in the standard order
    # Note: df_blockgroup will be None if input_level was 'tr'
    if input_level == 'bg':
        return (df_blockgroup, df_tract, df_community, df_county, df_state),aggs
    elif input_level == 'tr':
        return (df_tract, df_community, df_county, df_state),aggs[1:]


def fetchIncomeData(variable,y=False): # Remove unused 'y' parameter later if confirmed unused
    
    # --- Load Community Mapping Data --- 
    # Moved from main() - Load this once before looping through years
    try:
        communities = pd.concat([pd.read_csv('matched_chicago_data.csv'), pd.read_csv('matched_chicagoLand_data.csv')], axis=0)
        communities['GEOID'] = communities['GEOID'].astype(str)
        communities = communities.rename(columns={'GEOID': 'tract'}).set_index('tract')
        communities = communities[~communities['community'].isna()]
        communities['community'] = communities['community'].str.upper()
        print("Successfully loaded and processed community mapping data.")
    except FileNotFoundError:
        print("ERROR: Community mapping CSV file(s) not found. Cannot proceed with community aggregation.")
        communities = None # Set to None if loading fails
    except Exception as e:
        print(f"ERROR: Failed to load or process community mapping data: {e}")
        communities = None
    # -----------------------------------

    datas = {}

    for year in years: 
    
        # Define the base URL for the Census API

        base_url = "https://api.census.gov/data/20"+str(year)+"/acs/acs5"

        if year == 9:
            base_url = "https://api.census.gov/data/2009/acs/acs5"

        # Store fetch results temporarily
        fetch_results_bg = {} # key: (state, county), value: dataframe
        fetch_results_tr = {} # key: (state, county), value: dataframe
        any_bg_success = False
        any_tr_success = False
        
        for state, county in zip(states, counties):
            county_key = (state, county)
            data_fetched_for_county_bg = False # Flag for BG success this county

            # --- Attempt to Fetch Block Group Data ---
            bg_geography = f"block%20group:*&in=state:{state}&in=county:{county}"
            bg_url = f"{base_url}?get={','.join(variable)}&for={bg_geography}&key={api_key}"
            print(f"Attempting BG data fetch: State {state}, County {county}, Year {year}")
            try:
                bg_response = requests.get(bg_url, timeout=60)
                bg_response.raise_for_status()
                bg_data = bg_response.json()

                if len(bg_data) > 1:
                    print(f"  SUCCESS: BG Data fetched for {state}-{county}, Year {year}.")
                    bg_columns = bg_data[0]
                    bg_data_rows = bg_data[1:]
                    df = pd.DataFrame(bg_data_rows, columns=bg_columns)

                    # --- START ADDED DETAILED DEBUG FOR PROBLEMATIC BGs ---
                    if str(year) == '14' and state == '17' and county == '031':
                        problematic_bgs_specs = [
                            {'tract': '351100', 'block group': '2'},
                            {'tract': '836100', 'block group': '2'}
                        ]
                        # Find column indices from the header
                        header_map = {name: idx for idx, name in enumerate(bg_columns)}
                        b19001_idx = header_map.get('B19001_001E')
                        b25010_idx = header_map.get('B25010_001E')
                        tract_idx = header_map.get('tract')
                        bg_idx = header_map.get('block group')

                        if all(idx is not None for idx in [b19001_idx, b25010_idx, tract_idx, bg_idx]):
                            print(f"    DETAILED DEBUG ({year}-{state}-{county}, BG): Checking raw values for specific BGs:")
                            for row_data in bg_data_rows: # Iterate over raw rows
                                current_tract_raw = row_data[tract_idx]
                                current_bg_raw = row_data[bg_idx]
                                for spec in problematic_bgs_specs:
                                    if current_tract_raw == spec['tract'] and current_bg_raw == spec['block group']:
                                        raw_b19001 = row_data[b19001_idx]
                                        raw_b25010 = row_data[b25010_idx]
                                        print(f"      Problematic BG Spec: Tract {spec['tract']}, BG {spec['block group']}")
                                        print(f"        Raw API values: B19001_001E (HH): '{raw_b19001}', B25010_001E (AvgSize): '{raw_b25010}'")

                        # Now check values in the DataFrame after initial pd.DataFrame creation (before numeric conversion in the main logic)
                        df_filtered_problematic = df[
                            (df['tract'].astype(str) == '351100') & (df['block group'].astype(str) == '2') & (df['state'].astype(str) == '17') & (df['county'].astype(str) == '031') |
                            (df['tract'].astype(str) == '836100') & (df['block group'].astype(str) == '2') & (df['state'].astype(str) == '17') & (df['county'].astype(str) == '031')
                        ]
                        if not df_filtered_problematic.empty:
                            print(f"    DETAILED DEBUG ({year}-{state}-{county}, BG): DataFrame values for specific BGs (before explicit to_numeric in multiplication step):")
                            for _, prow in df_filtered_problematic.iterrows():
                                print(f"      BG: State {prow['state']}, County {prow['county']}, Tract {prow['tract']}, BG {prow['block group']}")
                                print(f"        DF values: B19001_001E (HH): '{prow.get('B19001_001E')}', B25010_001E (AvgSize): '{prow.get('B25010_001E')}'")
                    # --- END ADDED DETAILED DEBUG FOR PROBLEMATIC BGs ---

                    # --- Multiply B19001_001E by B25010_001E ---
                    if 'B19001_001E' in df.columns and 'B25010_001E' in df.columns and 'B01003_001E' in df.columns:
                        if str(year) == '14' and state == '17': # Ensure year is compared as string
                            # --- START MODIFIED DEBUG PRINT after to_numeric for problematic BGs ---
                            df_problematic_for_numeric_check = df[
                                (df['tract'].astype(str) == '351100') & (df['block group'].astype(str) == '2') & (df['state'].astype(str) == '17') & (df['county'].astype(str) == '031') |
                                (df['tract'].astype(str) == '836100') & (df['block group'].astype(str) == '2') & (df['state'].astype(str) == '17') & (df['county'].astype(str) == '031')
                            ]
                            if not df_problematic_for_numeric_check.empty:
                                print(f"    DETAILED DEBUG ({year}-{state}-{county}, BG): Values for specific BGs AFTER to_numeric but BEFORE multiplication:")
                                # We need to show b19001_col and b25010_col for these specific BGs.
                                # This requires aligning the Series with the filtered DataFrame's index.
                                # Temporarily convert for debug view
                                _b19001_col_debug = pd.to_numeric(df['B19001_001E'], errors='coerce')
                                _b25010_col_debug = pd.to_numeric(df['B25010_001E'], errors='coerce')
                                for idx_problem, row_problem in df_problematic_for_numeric_check.iterrows():
                                    try:
                                        val_b19001_numeric = _b19001_col_debug.loc[idx_problem]
                                        val_b25010_numeric = _b25010_col_debug.loc[idx_problem]
                                        print(f"      BG: Tract {row_problem['tract']}, BG {row_problem['block group']}")
                                        print(f"        Post-to_numeric: b19001_col (HH): {val_b19001_numeric} (type {type(val_b19001_numeric)}), b25010_col (AvgSize): {val_b25010_numeric} (type {type(val_b25010_numeric)})")
                                    except KeyError:
                                        print(f"      BG: Tract {row_problem['tract']}, BG {row_problem['block group']} - Indexing error for numeric values in debug.")
                            # --- END MODIFIED DEBUG PRINT ---
                            print(f"    DEBUG fetchIncomeData ({year}-{state}-{county}, BG): B19001_001E (Households), B25010_001E (AvgSize), and B01003_001E (TotalPop) before processing (sample for county {county}):")
                            print(df[['B19001_001E', 'B25010_001E', 'B01003_001E']].head())
                        # --- END DEBUG ---
                        b19001_hh_col = pd.to_numeric(df['B19001_001E'], errors='coerce') # Original Households
                        b25010_avg_size_col = pd.to_numeric(df['B25010_001E'], errors='coerce')
                        b01003_total_pop_col = pd.to_numeric(df['B01003_001E'], errors='coerce') # Total Population from API

                        sentinel_value = -666666666.0
                        mask_avg_size_is_sentinel = (b25010_avg_size_col == sentinel_value)
                        
                        # Initialize B19001_001E with calculated population (households * avg_size)
                        # This column will store the final population figure.
                        df['B19001_001E'] = b19001_hh_col * b25010_avg_size_col
                        
                        affected_indices_bg_sentinel = None # Store indices affected by sentinel replacement
                        if mask_avg_size_is_sentinel.any():
                            print(f"    INFO fetchIncomeData ({year}-{state}-{county}, BG): Found sentinel value ({sentinel_value}) in B25010_001E (Avg HH Size) for {mask_avg_size_is_sentinel.sum()} records.")
                            print(f"      Replacing calculated population with B01003_001E (Total Population) for these records.")
                            affected_indices_bg_sentinel = df.index[mask_avg_size_is_sentinel]

                            for idx in affected_indices_bg_sentinel:
                                original_hh_api_str = df.loc[idx, 'B19001_001E'] # This is before it's overwritten by calculated pop
                                original_avg_size_api_str = df.loc[idx, 'B25010_001E']
                                original_total_pop_api_str = df.loc[idx, 'B01003_001E']
                                
                                numeric_hh_val = b19001_hh_col.loc[idx]
                                numeric_total_pop_val = b01003_total_pop_col.loc[idx]
                                
                                id_cols = ['state', 'county', 'tract', 'block group']
                                id_str = "-".join([str(df.loc[idx, col_name]) for col_name in id_cols if col_name in df.columns])

                                print(f"      For BG ID: {id_str}")
                                print(f"        Original API strings: Households='{original_hh_api_str}', AvgHouseholdSize='{original_avg_size_api_str}', TotalPopulation='{original_total_pop_api_str}'")
                                print(f"        Numeric values: Households={numeric_hh_val}, AvgHouseholdSize (is sentinel), TotalPopulation={numeric_total_pop_val}")
                                print(f"        Action: Using TotalPopulation ({numeric_total_pop_val}) as the population figure for B19001_001E.")
                            
                            # Perform the replacement using B01003_001E for sentinel cases
                            df.loc[mask_avg_size_is_sentinel, 'B19001_001E'] = b01003_total_pop_col[mask_avg_size_is_sentinel]
                        
                        # --- User Requested: Set population to 0 if *original* households were 0 --- 
                        # This applies after the sentinel logic, based on the initial household count (b19001_hh_col).
                        zero_hh_mask_bg = (b19001_hh_col == 0)
                        if zero_hh_mask_bg.any():
                            # Identify records where population needs to be set to 0
                            records_to_update_to_zero_bg = zero_hh_mask_bg & (df['B19001_001E'] != 0)
                            if records_to_update_to_zero_bg.any():
                                df.loc[records_to_update_to_zero_bg, 'B19001_001E'] = 0.0
                                print(f"    INFO fetchIncomeData ({year}-{state}-{county}, BG): Set population (in B19001_001E) to 0.0 for {records_to_update_to_zero_bg.sum()} records where initial household count was 0.")
                        # --- END User Requested --- 

                        if affected_indices_bg_sentinel is not None and not affected_indices_bg_sentinel.empty:
                            print(f"    INFO fetchIncomeData ({year}-{state}-{county}, BG): Values in B19001_001E AFTER sentinel replacement and/or zero HH override for affected BGs:")
                            for idx in affected_indices_bg_sentinel:
                                final_pop_val = df.loc[idx, 'B19001_001E']
                                id_cols = ['state', 'county', 'tract', 'block group']
                                id_str = "-".join([str(df.loc[idx, col_name]) for col_name in id_cols if col_name in df.columns])
                                print(f"      For BG ID: {id_str}, Final Population (B19001_001E): {final_pop_val}")
                        # --- END ENHANCED HANDLING FOR SENTINEL VALUES ---

                        # --- DEBUG PRINT after multiplication (BG) ---
                        if str(year) == '14' and state == '17': # Ensure year is compared as string
                            print(f"    DEBUG fetchIncomeData ({year}-{state}-{county}, BG): B19001_001E (now population) after processing (sample for county {county}):")
                            print(df[['B19001_001E']].head())
                        # --- END DEBUG ---
                        print(f"    SUCCESS: Processed population and stored in B19001_001E for {state}-{county}, Year {year} (BG).")
                        df.drop(columns=['B25010_001E', 'B01003_001E'], inplace=True, errors='ignore') # Drop B01003_001E as well
                        print(f"    Dropped B25010_001E and B01003_001E columns after calculation for {state}-{county}, Year {year} (BG).")
                    else:
                        missing_cols_calc = []
                        if 'B19001_001E' not in df.columns: missing_cols_calc.append('B19001_001E (Households)')
                        if 'B25010_001E' not in df.columns: missing_cols_calc.append('B25010_001E (Avg HH Size)')
                        if 'B01003_001E' not in df.columns: missing_cols_calc.append('B01003_001E (Total Population)')
                        print(f"    WARNING: Could not calculate population. Missing columns: {missing_cols_calc} for {state}-{county}, Year {year} (BG).")
                    # --- End Multiplication ---

                    fetch_results_bg[county_key] = df
                    data_fetched_for_county_bg = True
                    any_bg_success = True
                else:
                    print(f"  WARNING: BG Data received for {state}-{county}, Year {year} but contains no data rows. Will attempt TR fallback.")

            except requests.exceptions.RequestException as e:
                print(f"  FAILED: BG data fetch for {state}-{county}, Year {year}. Error: {e}. Will attempt TR fallback.")
            # ------------------------------------

            # --- Fallback to Fetch Tract Data (if BG failed or had no data) ---
            if not data_fetched_for_county_bg:
                tr_geography = f"tract:*&in=state:{state}&in=county:{county}"
                tr_url = f"{base_url}?get={','.join(variable)}&for={tr_geography}&key={api_key}"
                print(f"Attempting TR data fetch (fallback): State {state}, County {county}, Year {year}")
                try:
                    tr_response = requests.get(tr_url, timeout=60)
                    tr_response.raise_for_status()
                    tr_data = tr_response.json()

                    if len(tr_data) > 1:
                        print(f"  SUCCESS: TR Data fetched for {state}-{county}, Year {year}.")
                        tr_columns = tr_data[0]
                        tr_data_rows = tr_data[1:]
                        df = pd.DataFrame(tr_data_rows, columns=tr_columns)

                        # --- Multiply B19001_001E by B25010_001E (for TR fallback) ---
                        if 'B19001_001E' in df.columns and 'B25010_001E' in df.columns and 'B01003_001E' in df.columns:
                            # --- DEBUG PRINT before multiplication (TR Fallback) ---
                            if str(year) == '14' and state == '17': # Ensure year is compared as string
                                print(f"    DEBUG fetchIncomeData ({year}-{state}-{county}, TR Fallback): B19001_001E (Households), B25010_001E (AvgSize), B01003_001E (TotalPop) before processing (sample for county {county}):")
                                print(df[['B19001_001E', 'B25010_001E', 'B01003_001E']].head())
                            # --- END DEBUG ---
                            b19001_hh_col_tr = pd.to_numeric(df['B19001_001E'], errors='coerce') # Original Households
                            b25010_avg_size_col_tr = pd.to_numeric(df['B25010_001E'], errors='coerce')
                            b01003_total_pop_col_tr = pd.to_numeric(df['B01003_001E'], errors='coerce') # Total Population from API

                            sentinel_value_tr = -666666666.0
                            mask_avg_size_is_sentinel_tr = (b25010_avg_size_col_tr == sentinel_value_tr)
                            
                            # Initialize B19001_001E with calculated population (households * avg_size)
                            df['B19001_001E'] = b19001_hh_col_tr * b25010_avg_size_col_tr

                            affected_indices_tr_sentinel = None # Store indices affected by sentinel replacement
                            if mask_avg_size_is_sentinel_tr.any():
                                print(f"    INFO fetchIncomeData ({year}-{state}-{county}, TR Fallback): Found sentinel value ({sentinel_value_tr}) in B25010_001E (Avg HH Size) for {mask_avg_size_is_sentinel_tr.sum()} records.")
                                print(f"      Replacing calculated population with B01003_001E (Total Population) for these records.")
                                affected_indices_tr_sentinel = df.index[mask_avg_size_is_sentinel_tr]

                                for idx in affected_indices_tr_sentinel:
                                    original_hh_api_str_tr = df.loc[idx, 'B19001_001E'] # Before overwrite
                                    original_avg_size_api_str_tr = df.loc[idx, 'B25010_001E']
                                    original_total_pop_api_str_tr = df.loc[idx, 'B01003_001E']

                                    numeric_hh_val_tr = b19001_hh_col_tr.loc[idx]
                                    numeric_total_pop_val_tr = b01003_total_pop_col_tr.loc[idx]

                                    id_cols_tr = ['state', 'county', 'tract']
                                    id_str_tr = "-".join([str(df.loc[idx, col_name]) for col_name in id_cols_tr if col_name in df.columns])
                                    
                                    print(f"      For TR ID: {id_str_tr}")
                                    print(f"        Original API strings: Households='{original_hh_api_str_tr}', AvgHouseholdSize='{original_avg_size_api_str_tr}', TotalPopulation='{original_total_pop_api_str_tr}'")
                                    print(f"        Numeric values: Households={numeric_hh_val_tr}, AvgHouseholdSize (is sentinel), TotalPopulation={numeric_total_pop_val_tr}")
                                    print(f"        Action: Using TotalPopulation ({numeric_total_pop_val_tr}) as the population figure for B19001_001E.")
                                
                                # Perform the replacement
                                df.loc[mask_avg_size_is_sentinel_tr, 'B19001_001E'] = b01003_total_pop_col_tr[mask_avg_size_is_sentinel_tr]
                            
                            # --- START User Requested: Set population to 0 if *original* households were 0 (TR Fallback) --- 
                            zero_hh_mask_tr = (b19001_hh_col_tr == 0)
                            if zero_hh_mask_tr.any():
                                records_to_update_to_zero_tr = zero_hh_mask_tr & (df['B19001_001E'] != 0)
                                if records_to_update_to_zero_tr.any():
                                    df.loc[records_to_update_to_zero_tr, 'B19001_001E'] = 0.0
                                    print(f"    INFO fetchIncomeData ({year}-{state}-{county}, TR Fallback): Set population (in B19001_001E) to 0.0 for {records_to_update_to_zero_tr.sum()} records where initial household count was 0.")
                            # --- END User Requested --- 

                            if affected_indices_tr_sentinel is not None and not affected_indices_tr_sentinel.empty:
                                print(f"    INFO fetchIncomeData ({year}-{state}-{county}, TR Fallback): Values in B19001_001E AFTER sentinel replacement and/or zero HH override for affected TRs:")
                                for idx in affected_indices_tr_sentinel:
                                    final_pop_val_tr = df.loc[idx, 'B19001_001E']
                                    id_cols_tr = ['state', 'county', 'tract']
                                    id_str_tr = "-".join([str(df.loc[idx, col_name]) for col_name in id_cols_tr if col_name in df.columns])
                                    print(f"      For TR ID: {id_str_tr}, Final Population (B19001_001E): {final_pop_val_tr}")
                            # --- END ENHANCED HANDLING FOR SENTINEL VALUES (TR Fallback) ---

                            # --- DEBUG PRINT after multiplication (TR Fallback) ---
                            if str(year) == '14' and state == '17': # Ensure year is compared as string
                                print(f"    DEBUG fetchIncomeData ({year}-{state}-{county}, TR Fallback): B19001_001E (now population) after processing (sample for county {county}):")
                                print(df[['B19001_001E']].head()) # B25010_001E is dropped
                            # --- END DEBUG ---
                            print(f"    SUCCESS: Processed population and stored in B19001_001E for {state}-{county}, Year {year} (TR Fallback).")
                            df.drop(columns=['B25010_001E', 'B01003_001E'], inplace=True, errors='ignore') # Drop B01003_001E as well
                            print(f"    Dropped B25010_001E and B01003_001E columns after calculation for {state}-{county}, Year {year} (TR Fallback).")
                        else:
                            missing_cols_calc_tr = []
                            if 'B19001_001E' not in df.columns: missing_cols_calc_tr.append('B19001_001E (Households)')
                            if 'B25010_001E' not in df.columns: missing_cols_calc_tr.append('B25010_001E (Avg HH Size)')
                            if 'B01003_001E' not in df.columns: missing_cols_calc_tr.append('B01003_001E (Total Population)')
                            print(f"    WARNING: Could not calculate population. Missing columns: {missing_cols_calc_tr} for {state}-{county}, Year {year} (TR Fallback).")
                        # --- End Multiplication (for TR fallback) ---

                        fetch_results_tr[county_key] = df
                        any_tr_success = True
                        print(f"    (Note: Using Tract data for {state}-{county} as BG data was unavailable/empty.)")
                    else:
                         print(f"  WARNING: TR Data received for {state}-{county}, Year {year} but contains no data rows.")

                except requests.exceptions.RequestException as e:
                    print(f"  FAILED: TR data fetch for {state}-{county}, Year {year}. Error: {e}")
            # --- End Fallback Logic ---

            # Report if neither was successful for this specific county
            if not data_fetched_for_county_bg and county_key not in fetch_results_tr:
                 print(f"  FAILURE: Could not fetch usable BG or TR data for State {state}, County {county}, Year {year}.")
            # --- End Fetch Logic for County ---

        # --- Determine Level and Prepare Data for Aggregation --- 
        level_for_year = 'none'
        data_for_year = []

        if any_bg_success:
            level_for_year = 'bg'
            data_for_year = list(fetch_results_bg.values())
            print(f"\nYear {year}: Determined aggregation level: 'bg'. Passing {len(data_for_year)} BG dataframe(s).")
        elif any_tr_success:
            level_for_year = 'tr'
            data_for_year = list(fetch_results_tr.values())
            print(f"\nYear {year}: Determined aggregation level: 'tr' (No BG data succeeded). Passing {len(data_for_year)} TR dataframe(s).")
        else:
            print(f"\nYear {year}: No usable BG or TR data fetched for any county.")
            # level_for_year remains 'none', data_for_year remains []
        # ------------------------------------------------------

        # --- Call Aggregation Function for the Year ---
        # Call genAggregatedDFs with the determined data and level, passing the loaded communities df
        aggs = ['bg','tr','cm','ct','st']
        aggregated_results,aggs = genAggregatedDFs(aggs,data_for_year, level_for_year, variable, year, communities) # Pass communities
        datas[year] = dict(zip(aggs, aggregated_results))
        # --------------------------------------------

    return datas



def save_data(data, filename='census_data.pkl'):
    """
    Save processed data to a pickle file
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)


    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

# communities = None # Removed global variable definition

def main():
    # Your main code here
    # global communities # Removed global keyword
    # --- Community loading logic moved to fetchIncomeData ---
    # communities = pd.concat([pd.read_csv('matched_chicago_data.csv'),pd.read_csv('matched_chicagoLand_data.csv')],axis=0)
    # communities['GEOID'] = communities['GEOID'].astype(str)
    # communities = communities.rename(columns={'GEOID': 'tract'}).set_index('tract')
    # communities = communities[~communities['community'].isna()]
    # communities['community'] = communities['community'].str.upper()
    # ------------------------------------------------------
    # Matched community names and census tracts
    # GENERATES POPULATION DATA FOR EACH YEAR
    
    incomes = fetchIncomeData(income_vars+pop_vars)

    save_data(incomes, 'data/census_data1.pkl')

    pass

if __name__ == "__main__":
    main()
