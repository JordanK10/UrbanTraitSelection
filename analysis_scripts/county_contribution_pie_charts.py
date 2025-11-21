import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import sys

# Check for 'null' argument to switch directories
if 'null' in sys.argv:
    INPUT_DIR = 'output_terms_null'
    BASE_OUTPUT_DIR = 'plots_null'
else:
    INPUT_DIR = 'output_terms'
    BASE_OUTPUT_DIR = 'plots'

# --- CONFIGURATION ---
LEVELS_TO_ANALYZE = {
    'Tract': {
        'data_path': os.path.join(INPUT_DIR, 'bg_tr_exported_terms.csv'),
        'pop_col': 'PopInitial_tr',
        'sel_inc_col': 'Sel_tr_from_bg_inc',
        'sel_pop_col': 'Sel_tr_from_bg_pop',
        'county_col': 'ParentCounty'
    },
    'Community': {
        'data_path': os.path.join(INPUT_DIR, 'bg_cm_exported_terms.csv'),
        'pop_col': 'PopInitial_cm',
        'sel_inc_col': 'Sel_cm_from_tr_inc',
        'sel_pop_col': 'Sel_cm_from_tr_pop',
        'county_col': 'ParentCounty'
    }
}

OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'county_contribution_pies')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom Colors for Plots
CUSTOM_PURPLE = '#633673'
CUSTOM_ORANGE = '#E77429'

# Custom Colormaps
PURPLE_CMAP = LinearSegmentedColormap.from_list("custom_purple", ['#ffffff', CUSTOM_PURPLE])
ORANGE_CMAP = LinearSegmentedColormap.from_list("custom_orange", ['#ffffff', CUSTOM_ORANGE])

# A mapping from FIPS code to County Name for prettier labels
COUNTY_FIPS_MAP = {
    '031': 'Cook',
    '043': 'DuPage',
    '089': 'Kane',
    '093': 'Kendall',
    '097': 'Lake',
    '111': 'McHenry',
    '197': 'Will'
}

# --- PLOTTING FUNCTION ---

def create_pie_chart(ax, data, title, normalization_total=None, custom_cmap=None):
    """
    Creates a styled pie chart on a given matplotlib axis.
    If normalization_total is provided, percentages are calculated against it.
    """
    # Use the provided normalization_total if available, otherwise sum the data.
    total = normalization_total if normalization_total is not None else data.sum()
    if total == 0: # Avoid division by zero
        return

    # Group small slices into 'Other'
    threshold = 0.01  # Group anything less than 0.01%
    small_slices = data[data / total * 100 < threshold]
    other_sum = small_slices.sum()
    
    main_slices = data[data / total * 100 >= threshold]
    if other_sum > 0:
        main_slices['Other'] = other_sum
        
    # Sort slices for consistent color mapping
    main_slices = main_slices.sort_values(ascending=False)
        
    labels = main_slices.index
    # Use the definitive total for calculating percentages for the pie chart itself
    sizes_for_pct = main_slices.values
    
    # Define colors
    if custom_cmap:
        cmap = custom_cmap
    else:
        cmap = plt.get_cmap("inferno")
    # Apply colors from dark to light
    colors = cmap(np.flip(np.linspace(.9, 0, len(labels))))
    
    wedges, texts, autotexts = ax.pie(
        sizes_for_pct, 
        # labels=labels, 
        labels=[' ' for _ in labels], 
        # autopct=lambda pct: f'{(pct/100) * main_slices.sum() / total * 100:.1f}%', # Custom autopct
        autopct=lambda pct: f' ', # Custom autopct
        startangle=140,
        colors=colors,
        pctdistance=0.85,
        wedgeprops=dict(width=0.8, edgecolor='w')
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.setp(autotexts, size=16, weight="bold", color="white")
    plt.setp(texts, size=16)


# --- MAIN EXECUTION ---

def main():
    """
    Main function to generate the 2x3 grid of pie charts.
    """
    print(f"\n{'='*60}\nGenerating County Contribution Pie Charts\n{'='*60}")

    plt.style.use('seaborn-v0_8-whitegrid')
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    for row_idx, (level_name, config) in enumerate(LEVELS_TO_ANALYZE.items()):
        print(f"\nProcessing {level_name} level...")
        
        # Load data
        if not os.path.exists(config['data_path']):
            print(f"  Data file not found: {config['data_path']}. Skipping.")
            # Turn off unused axes
            for col_idx in range(3):
                axes[row_idx, col_idx].axis('off')
            continue
        df = pd.read_csv(config['data_path'])
        
        # --- Correctly handle County FIPS ---
        county_col = config['county_col']
        if level_name == 'Tract':
            # Tract data uses 3-digit codes
            df['CountyName'] = df[county_col].astype(str).str.strip().str.zfill(3).map(COUNTY_FIPS_MAP).fillna('Unknown')
        else: # Community level
            # Community data uses 5-digit codes, extract last 3 digits
            df['CountyName'] = df[county_col].astype(str).str.strip().str[-3:].map(COUNTY_FIPS_MAP).fillna('Unknown')
        
        analysis_types = ['Positive', 'Negative', 'Absolute']
        for col_idx, analysis_type in enumerate(analysis_types):
            ax = axes[row_idx, col_idx]
            sel_col = config['sel_pop_col'] # Using population selection as requested

            if sel_col not in df.columns:
                print(f"  Selection column '{sel_col}' not found. Skipping.")
                ax.axis('off')
                continue
                
            # Filter data for pos/neg/abs
            if analysis_type == 'Positive':
                subset_df = df[df[sel_col] > 0]
                contribution = subset_df.groupby('CountyName')[sel_col].sum()
            elif analysis_type == 'Negative':
                subset_df = df[df[sel_col] < 0]
                contribution = subset_df.groupby('CountyName')[sel_col].apply(lambda x: x.abs().sum())
            else: # Absolute
                contribution = df.groupby('CountyName')[sel_col].apply(lambda x: x.abs().sum())

            contribution = contribution[contribution > 0]
            
            if contribution.empty:
                print(f"  No {analysis_type} selection data for {level_name}. Skipping pie chart.")
                ax.axis('off')
            else:
                title = f'{level_name} Level\n{analysis_type} Population Selection'
                create_pie_chart(ax, contribution, title, custom_cmap=PURPLE_CMAP)
            
    # Final adjustments and save
    fig.suptitle('Share of Absolute Selection Effect by County', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = 'county_selection_contribution_summary.pdf'
    full_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(full_path, dpi=300)
    plt.close()
    
    print(f"\nPie chart summary saved to {full_path}")

    # --- FIGURE 2: SELECTION DENSITY (PER CAPITA) ---
    print(f"\n{'='*60}\nGenerating County Selection Density Pie Charts\n{'='*60}")
    
    fig2, axes2 = plt.subplots(2, 3, figsize=(24, 16))

    for row_idx, (level_name, config) in enumerate(LEVELS_TO_ANALYZE.items()):
        print(f"\nProcessing {level_name} level for density charts...")
        
        # We can reuse the loaded df from the first part if it's stored
        # For simplicity, we'll just reload it here
        if not os.path.exists(config['data_path']):
            for col_idx in range(3):
                axes2[row_idx, col_idx].axis('off')
            continue
        df = pd.read_csv(config['data_path'])
        
        county_col = config['county_col']
        if level_name == 'Tract':
            df['CountyName'] = df[county_col].astype(str).str.strip().str.zfill(3).map(COUNTY_FIPS_MAP).fillna('Unknown')
        else:
            df['CountyName'] = df[county_col].astype(str).str.strip().str[-3:].map(COUNTY_FIPS_MAP).fillna('Unknown')
        
        # Calculate total initial population per county
        county_pops = df.groupby('CountyName')[config['pop_col']].sum()

        analysis_types = ['Positive', 'Negative', 'Absolute']
        for col_idx, analysis_type in enumerate(analysis_types):
            ax = axes2[row_idx, col_idx]
            sel_col = config['sel_pop_col']

            if sel_col not in df.columns:
                ax.axis('off')
                continue

            if analysis_type == 'Positive':
                subset_df = df[df[sel_col] > 0]
                county_sel = subset_df.groupby('CountyName')[sel_col].sum()
            elif analysis_type == 'Negative':
                subset_df = df[df[sel_col] < 0]
                county_sel = subset_df.groupby('CountyName')[sel_col].apply(lambda x: x.abs().sum())
            else: # Absolute
                county_sel = df.groupby('CountyName')[sel_col].apply(lambda x: x.abs().sum())
            
            # Calculate per-capita selection density
            selection_density = county_sel / county_pops
            selection_density = selection_density[selection_density > 0].dropna()

            if selection_density.empty:
                print(f"  No selection density data for {level_name} - {analysis_type}. Skipping.")
                ax.axis('off')
            else:
                title = f'{level_name} Level\n{analysis_type} Population Selection Density'
                create_pie_chart(ax, selection_density, title, custom_cmap=ORANGE_CMAP)
    
    fig2.suptitle('Share of Per-Capita Selection Effect (Density) by County', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename2 = 'county_selection_density_summary.pdf'
    full_path2 = os.path.join(OUTPUT_DIR, filename2)
    plt.savefig(full_path2, dpi=300)
    plt.close()
    
    print(f"\nSelection density pie chart summary saved to {full_path2}")

    # --- FIGURE 3: POPULATION SHARE ---
    print(f"\n{'='*60}\nGenerating County Population Share Pie Charts\n{'='*60}")
    
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 8))

    # --- MODIFICATION START: Enforce consistent normalization ---
    
    # 1. First, determine the definitive total population from the Tract level data.
    tract_config = LEVELS_TO_ANALYZE.get('Tract')
    definitive_county_pops = pd.Series(dtype=float)
    if tract_config and os.path.exists(tract_config['data_path']):
        df_tr = pd.read_csv(tract_config['data_path'])
        df_tr['CountyName'] = df_tr[tract_config['county_col']].astype(str).str.strip().str.zfill(3).map(COUNTY_FIPS_MAP).fillna('Unknown')
        definitive_county_pops = df_tr.groupby('CountyName')[tract_config['pop_col']].sum()
        definitive_total_pop = definitive_county_pops.sum()
        print(f"  Established definitive total population from Tract level: {definitive_total_pop:,.0f}")
    else:
        definitive_total_pop = 0
        print("  Warning: Could not load Tract level data to establish definitive population total.")

    # Calculate community level pops for comparison table
    community_config = LEVELS_TO_ANALYZE.get('Community')
    community_county_pops = pd.Series(dtype=float)
    if community_config and os.path.exists(community_config['data_path']):
        df_cm = pd.read_csv(community_config['data_path'])
        df_cm['CountyName'] = df_cm[community_config['county_col']].astype(str).str.strip().str[-3:].map(COUNTY_FIPS_MAP).fillna('Unknown')
        community_county_pops = df_cm.groupby('CountyName')[community_config['pop_col']].sum()

    # Create and print the comparison DataFrame
    comparison_df = pd.DataFrame({
        'Population_Tract_Level': definitive_county_pops,
        'Population_Community_Level': community_county_pops
    }).fillna(0)
    comparison_df['Difference'] = comparison_df['Population_Tract_Level'] - comparison_df['Population_Community_Level']
    
    print("\n--- County Population Comparison ---")
    print(comparison_df.to_string(formatters={'Population_Tract_Level': '{:,.0f}'.format,
                                            'Population_Community_Level': '{:,.0f}'.format,
                                            'Difference': '{:,.0f}'.format}))
    print("------------------------------------\n")


    # 2. Iterate and plot, using the definitive total for normalization.
    for ax_idx, (level_name, config) in enumerate(LEVELS_TO_ANALYZE.items()):
        print(f"\nProcessing {level_name} level for population share...")
        ax = axes3[ax_idx]
        
        if not os.path.exists(config['data_path']):
            print(f"  Data file not found: {config['data_path']}. Skipping.")
            ax.axis('off')
            continue
            
        df = pd.read_csv(config['data_path'])
        
        county_col = config['county_col']
        if level_name == 'Tract':
            df['CountyName'] = df[county_col].astype(str).str.strip().str.zfill(3).map(COUNTY_FIPS_MAP).fillna('Unknown')
        else: # Community level (and any others)
            df['CountyName'] = df[county_col].astype(str).str.strip().str[-3:].map(COUNTY_FIPS_MAP).fillna('Unknown')
            
        # Calculate total initial population per county for the current level
        county_pops = df.groupby('CountyName')[config['pop_col']].sum()
        county_pops = county_pops[county_pops > 0]

        if county_pops.empty:
            print(f"  No population data for {level_name}. Skipping pie chart.")
            ax.axis('off')
        else:
            # For the community level, check for discrepancies and add an "Unaccounted" slice
            if level_name == 'Community' and definitive_total_pop > 0:
                current_total_pop = county_pops.sum()
                unaccounted_pop = definitive_total_pop - current_total_pop
                
                # --- DEBUGGING STATEMENTS START ---
                print(f"    DEBUG: Definitive Total (from Tracts): {definitive_total_pop:,.2f}")
                print(f"    DEBUG: Current Total (from Communities): {current_total_pop:,.2f}")
                print(f"    DEBUG: Calculated Unaccounted Population: {unaccounted_pop:,.2f}")
                # --- DEBUGGING STATEMENTS END ---

                if unaccounted_pop > 0.01: # Use a small threshold to avoid floating point issues
                    print(f"  Adding 'Unaccounted' slice with population: {unaccounted_pop:,.0f}")
                    county_pops['Unaccounted'] = unaccounted_pop

            title = f'{level_name} Level\nPopulation Share by County'
            # Pass the definitive_total_pop for consistent normalization
            create_pie_chart(ax, county_pops, title, normalization_total=definitive_total_pop)
            
    fig3.suptitle('Share of Population by County', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename3 = 'county_population_share_summary.pdf'
    full_path3 = os.path.join(OUTPUT_DIR, filename3)
    plt.savefig(full_path3, dpi=300)
    plt.close()
    
    print(f"\nPopulation share pie chart summary saved to {full_path3}")


if __name__ == "__main__":
    main()
