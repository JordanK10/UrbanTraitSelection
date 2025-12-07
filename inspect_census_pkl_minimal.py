import pickle
import sys
import pandas as pd

def inspect_census_data():
    file_path = "calculation_scripts/data/census_data1.pkl"
    print(f"Attempting to load {file_path}...")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Successfully loaded data. Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            # Inspect first key
            first_key = list(data.keys())[0]
            print(f"Type of value for key '{first_key}': {type(data[first_key])}")
            
            if isinstance(data[first_key], dict):
                print(f"  Keys for '{first_key}': {list(data[first_key].keys())}")
                first_inner_key = list(data[first_key].keys())[0]
                inner_val = data[first_key][first_inner_key]
                print(f"  Type of value for '{first_key} -> {first_inner_key}': {type(inner_val)}")
                
                if isinstance(inner_val, pd.DataFrame):
                    print(f"  Columns: {inner_val.columns.tolist()}")
                    print(f"  Head:\n{inner_val.head()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_census_data()


