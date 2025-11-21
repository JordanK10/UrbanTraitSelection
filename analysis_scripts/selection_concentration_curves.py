import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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
    'tr': {
        'data_path': os.path.join(INPUT_DIR, 'bg_tr_exported_terms.csv'),
        'pop_col': 'PopInitial_tr',
        'sel_inc_col': 'Sel_tr_from_bg_inc',
        'sel_pop_col': 'Sel_tr_from_bg_pop',
        'level_name': 'Tract',
        'county_col': 'ParentCounty'
    },
    'cm': {
        'data_path': os.path.join(INPUT_DIR, 'bg_cm_exported_terms.csv'),
        'pop_col': 'PopInitial_cm',
        'sel_inc_col': 'Sel_cm_from_tr_inc',
        'sel_pop_col': 'Sel_cm_from_tr_pop',
        'level_name': 'Community',
        'county_col': 'ParentCounty'
    }
}

OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'concentration_curves')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- ANALYSIS FUNCTIONS ---

def calculate_curve_data(df, selection_col, population_col, analysis_type, use_unit_count_xaxis=False, sort_by_magnitude=False):
    """
    Helper function to calculate the x and y coordinates for a concentration curve.
    
    analysis_type: 'Positive', 'Negative', or 'Absolute'
    use_unit_count_xaxis: If True, x-axis is cumulative share of units, not population.
    sort_by_magnitude: If True, sorts by selection magnitude instead of per-capita intensity.
    """
    if df.empty:
        return None, None

    # 1. Separate data
    if analysis_type == 'Positive':
        sub_df = df[df[selection_col] > 0].copy()
        sel_col_to_process = selection_col
    elif analysis_type == 'Negative':
        sub_df = df[df[selection_col] < 0].copy()
        sel_col_to_process = selection_col
    else:  # Absolute
        sub_df = df.copy()
        sub_df['abs_selection'] = sub_df[selection_col].abs()
        sel_col_to_process = 'abs_selection'
    
    if sub_df.empty:
        return None, None

    # 2. Get totals for normalization
    total_sel = sub_df[sel_col_to_process].sum()
    if total_sel == 0:
        return None, None

    # 3. Sort units based on the chosen metric
    sort_ascending = True if analysis_type == 'Negative' else False
    
    if sort_by_magnitude:
        # For smooth curves, sort directly by the magnitude of the selection effect
        sorted_df = sub_df.sort_values(by=sel_col_to_process, ascending=sort_ascending)
    else:
        # For intensity analysis, sort by per-capita selection
        epsilon = 1e-9
        sub_df['selection_per_capita'] = sub_df[sel_col_to_process] / (sub_df[population_col] + epsilon)
        sorted_df = sub_df.sort_values(by='selection_per_capita', ascending=sort_ascending)

    # 4. Calculate cumulative shares
    if use_unit_count_xaxis:
        total_units = len(sorted_df)
        sorted_df['cum_x_share'] = np.arange(1, total_units + 1) / total_units
    else:
        subset_population = sub_df[population_col].sum()
        if subset_population == 0:
            return None, None
        sorted_df['cum_x_share'] = sorted_df[population_col].cumsum() / subset_population

    sorted_df['cum_sel_share'] = sorted_df[sel_col_to_process].cumsum() / total_sel
    
    # 5. Prepare plot data
    x_vals = np.concatenate([[0], sorted_df['cum_x_share']])
    y_vals = np.concatenate([[0], sorted_df['cum_sel_share']])
    
    return x_vals, y_vals

def create_county_comparison_plot(df_statewide, config, selection_type):
    """
    Generates a 1x3 plot comparing concentration curves for all unique counties found in the data.
    """
    level_name = config['level_name']
    population_col = config['pop_col']
    selection_col = config['sel_inc_col'] if selection_type == 'Income' else config['sel_pop_col']
    county_col = config['county_col']

    print(f"  Generating multi-county comparison plot for {level_name} - {selection_type} Selection...")

    # Get unique counties from the data
    unique_counties = sorted([c.strip() for c in df_statewide[county_col].unique()])
    
    # Map FIPS to names for a cleaner legend
    fips_to_name = {
        '031': 'Cook', '043': 'DuPage', '089': 'Kane',
        '097': 'Lake', '111': 'McHenry', '197': 'Will', '093': 'Kendall'
    }

    # Create a color map for the counties
    colors = plt.cm.inferno(np.linspace(0, 1, len(unique_counties)))
    county_color_map = {county: color for county, color in zip(unique_counties, colors)}

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    
    analysis_types = ['Positive', 'Negative', 'Absolute']
    
    for i, analysis_type in enumerate(analysis_types):
        ax = axes[i]
        
        # Plot a curve for each county
        for fips in unique_counties:
            df_county = df_statewide[df_statewide[county_col].str.strip() == fips]
            if not df_county.empty:
                label = fips_to_name.get(fips[-3:], f'County {fips}') # Use name if available, else FIPS
                color = county_color_map[fips]
                
                x, y = calculate_curve_data(df_county, selection_col, population_col, analysis_type)
                if x is not None:
                    ax.plot(x, y, label=label, color=color, linewidth=2,alpha=.75)
        
        # Plot the statewide curve for comparison
        x_state, y_state = calculate_curve_data(df_statewide, selection_col, population_col, analysis_type)
        if x_state is not None:
            ax.plot(x_state, y_state, label='Statewide', color='black', linestyle='--', linewidth=2, zorder=3)

        # Aesthetics for subplot
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_title(f'{analysis_type} Selection', fontsize=14, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
            #set tick label fontsize to 10
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.grid(True)

    # Aesthetics for the whole figure
    # axes[0].set_ylabel(f'Cumulative % of {selection_type} Selection Effect', fontsize=12)
     
    # Create a single legend for community level
    if level_name == 'Community':
        county_handles = [plt.Line2D([0], [0], color=county_color_map[fips], lw=2) for fips in unique_counties]
        statewide_handle = plt.Line2D([0], [0], color='black', lw=2, linestyle='--')
        handles = [statewide_handle] + county_handles
        county_labels = [fips_to_name.get(fips[-3:], f'County {fips}') for fips in unique_counties]
        labels = ['Statewide'] + county_labels
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.875, 0.55), fontsize=10)


    # Save plot
    filename = f'multi_county_{level_name.lower()}_{selection_type.lower()}_concentration.pdf'
    full_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.93]) # Adjust layout for external legend
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Multi-county comparison plot saved to {full_path}")

def create_unified_concentration_plot(df, population_col, selection_col, scope_name, selection_type):
    """
    Recreates the original plot format where positive, negative, and absolute
    curves are on the same axis, normalized by the total population.
    """
    print(f"  Generating unified plot for {scope_name} - {selection_type} Selection...")

    # 1. Separate data
    df_pos = df[df[selection_col] > 0].copy()
    df_neg = df[df[selection_col] < 0].copy()
    df_abs = df.copy()
    df_abs['abs_selection'] = df_abs[selection_col].abs()

    # 2. Get totals for normalization
    total_population = df[population_col].sum()
    total_selection_pos = df_pos[selection_col].sum()
    total_selection_neg = df_neg[selection_col].sum()
    total_selection_abs = df_abs['abs_selection'].sum()

    # Create figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    datasets = {
        'Positive Selection': (df_pos, selection_col, total_selection_pos, '#E77429'),
        'Negative Selection': (df_neg, selection_col, total_selection_neg, '#633673'),
        'Absolute Selection': (df_abs, 'abs_selection', total_selection_abs, 'black')
    }

    for label, (sub_df, sel_col, total_sel, color) in datasets.items():
        if sub_df.empty or total_sel == 0:
            continue

        # Sort by per-capita effect to maintain consistency
        epsilon = 1e-9
        sub_df['selection_per_capita'] = sub_df[sel_col] / (sub_df[population_col] + epsilon)
        sort_ascending = True if label == 'Negative Selection' else False
        sorted_df = sub_df.sort_values(by='selection_per_capita', ascending=sort_ascending)

        # Calculate shares
        # For Positive/Negative, normalize x-axis by the subset population.
        # For Absolute, normalize x-axis by total population (full scope).
        subset_population = sub_df[population_col].sum()
        pop_norm = subset_population if label in ['Positive Selection', 'Negative Selection'] else total_population
        if pop_norm == 0:
            continue

        sorted_df['cum_pop_share'] = sorted_df[population_col].cumsum() / pop_norm
        sorted_df['cum_sel_share'] = sorted_df[sel_col].cumsum() / total_sel
        
        x_vals = np.concatenate([[0], sorted_df['cum_pop_share']])
        y_vals = np.concatenate([[0], sorted_df['cum_sel_share']])
        
        ax.plot(x_vals, y_vals, label=label, color=color, linewidth=2)

    # Aesthetics
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Line of Equality')
    ax.set_title(f'{scope_name}: {selection_type} Selection Concentration', fontsize=16, fontweight='bold')
    ax.set_xlabel('Cumulative % of Total Population', fontsize=12)
    ax.set_ylabel(f'Cumulative % of {selection_type} Selection Effect', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left')
    ax.grid(True)

    # Save plot
    filename = f'unified_{scope_name.lower().replace(" ", "_")}_{selection_type.lower()}_concentration.pdf'
    full_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Unified plot saved to {full_path}")


def create_level_comparison_plot(scope_name, df_tract, df_community, tract_config, community_config, selection_type):
    """
    Generates a plot comparing Tract and Community absolute selection concentration on the same axes.
    """
    print(f"  Generating level comparison plot for {scope_name} - {selection_type} Selection...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5), sharey=True)
    
    analysis_type = 'Absolute'
    
    # Define custom colors
    CUSTOM_PURPLE = '#633673'
    CUSTOM_ORANGE = '#E77429'

    # --- Tract Data ---
    tract_pop_col = tract_config['pop_col']
    tract_sel_col = tract_config['sel_inc_col'] if selection_type == 'Income' else tract_config['sel_pop_col']
    x_tract, y_tract = calculate_curve_data(df_tract, tract_sel_col, tract_pop_col, analysis_type)

    if x_tract is not None:
        ax.plot(x_tract, y_tract, label='Tract', color=CUSTOM_ORANGE, linestyle='-', linewidth=2, alpha=.95, zorder=1)
        try:
            # Diagnostic print
            print(f"The share of Tract selection at {x_tract[np.where(x_tract > 0.198)[0][0]]:.2f}% of the population for {analysis_type} is {y_tract[np.where(x_tract > 0.198)[0][0]]:.2f}")
        except IndexError:
            pass

    # --- Community Data ---
    comm_pop_col = community_config['pop_col']
    comm_sel_col = community_config['sel_inc_col'] if selection_type == 'Income' else community_config['sel_pop_col']
    x_comm, y_comm = calculate_curve_data(df_community, comm_sel_col, comm_pop_col, analysis_type)
    
    if x_comm is not None:
        ax.plot(x_comm, y_comm, label='Community', color=CUSTOM_PURPLE, linestyle='-', linewidth=2, alpha=.95, zorder=1)
        try:
            # Diagnostic print
            print(f"The share of Community selection at {x_comm[np.where(x_comm > 0.198)[0][0]]:.2f}% of the population for {analysis_type} is {y_comm[np.where(x_comm > 0.198)[0][0]]:.2f}")
        except IndexError:
            pass

    # Aesthetics for subplot
    ax.plot([0, 1], [0, 1], '--', color='#3D3D3D', alpha=0.5, linewidth=1.5, zorder=0)
    ax.set_xlim(-.005, 1.005)
    ax.set_ylim(-.005, 1.005)
    ax.grid(True, zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', which='minor', labelsize=16)
    
    ax.legend()

    # Save plot
    filename = f'level_comparison_{scope_name.lower().replace(" ", "_")}_{selection_type.lower()}_concentration.pdf'
    full_path = os.path.join(OUTPUT_DIR, filename)
    print(filename)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Level comparison plot saved to {full_path}")


def create_level_comparison_plot_by_units(scope_name, df_tract, df_community, tract_config, community_config, selection_type):
    """
    Generates a plot comparing Tract and Community concentration on the same axes,
    with the X-axis representing the cumulative share of units, not population.
    """
    print(f"  Generating level comparison plot by unit count for {scope_name} - {selection_type} Selection...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5), sharey=True)
    
    analysis_type = 'Absolute'
    
    # Define custom colors
    CUSTOM_PURPLE = '#633673'
    CUSTOM_ORANGE = '#E77429'

    # --- Tract Data ---
    tract_pop_col = tract_config['pop_col']
    tract_sel_col = tract_config['sel_inc_col'] if selection_type == 'Income' else tract_config['sel_pop_col']
    x_tract, y_tract = calculate_curve_data(df_tract, tract_sel_col, tract_pop_col, analysis_type, use_unit_count_xaxis=True, sort_by_magnitude=True)

    if x_tract is not None:
        ax.plot(x_tract, y_tract, label='Tract', color=CUSTOM_ORANGE, linestyle='-', linewidth=2, alpha=.95, zorder=1)
        try:
            # Diagnostic print
            print(f"The share of Tract selection at {x_tract[np.where(x_tract > 0.198)[0][0]]:.2f}% of the NUMBER OF UNITS for {analysis_type} is {y_tract[np.where(x_tract > 0.198)[0][0]]:.2f}")
        except IndexError:
            pass

    # --- Community Data ---
    comm_pop_col = community_config['pop_col']
    comm_sel_col = community_config['sel_inc_col'] if selection_type == 'Income' else community_config['sel_pop_col']
    x_comm, y_comm = calculate_curve_data(df_community, comm_sel_col, comm_pop_col, analysis_type, use_unit_count_xaxis=True, sort_by_magnitude=True)
    
    if x_comm is not None:
        ax.plot(x_comm, y_comm, label='Community', color=CUSTOM_PURPLE, linestyle='-', linewidth=2, alpha=.95, zorder=1)

        try:
            # Diagnostic print
            print(f"The share of Community selection at {x_comm[np.where(x_comm > 0.198)[0][0]]:.2f}% of the NUMBER OF UNITS for {analysis_type} is {y_comm[np.where(x_comm > 0.198)[0][0]]:.2f}")
        except IndexError:
            pass

    # Aesthetics for subplot
    ax.plot([0, 1], [0, 1], '--', color='#3D3D3D', alpha=0.5, linewidth=1.5, zorder=0)
    ax.set_xlim(-.005, 1.005)
    ax.set_ylim(-.005, 1.005)
    ax.grid(True, zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    
    # New X-axis label
    # ax.set_xlabel('Cumulative % of Units', fontsize=12)
    # ax.set_ylabel(f'Cumulative % of Absolute {selection_type} Selection', fontsize=10)

    ax.legend()
    # fig.suptitle(f'{scope_name}: Selection Concentration by Unit Count', fontsize=14, fontweight='bold')

    # Save plot
    filename = f'level_comparison_by_unit_{scope_name.lower().replace(" ", "_")}_{selection_type.lower()}_concentration.pdf'
    full_path = os.path.join(OUTPUT_DIR, filename)
    print(filename)
    plt.tight_layout(rect=[0, 0, 1, 0.92]) # Adjust for suptitle
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Level comparison plot by unit count saved to {full_path}")


# --- MAIN EXECUTION ---

def main():
    """
    Main function to generate all concentration curve plots.
    """
    print(f"\n{'='*60}\nGenerating Selection Concentration Curves\n{'='*60}")

    # --- Load all data first ---
    statewide_data = {}
    cook_county_data = {}
    for level_code, config in LEVELS_TO_ANALYZE.items():
        if not os.path.exists(config['data_path']):
            print(f"Could not find data file: {config['data_path']}. Skipping this level.")
            continue
        
        # Determine dtype for reading CSV to preserve leading zeros in FIPS codes
        county_col = config.get('county_col')
        dtype_spec = {county_col: str} if county_col else None
        
        df = pd.read_csv(config['data_path'], dtype=dtype_spec)
        
        # --- START FIX: Reconstruct ParentCounty for community level if it's missing ---
        if level_code == 'cm' and county_col and county_col not in df.columns:
            print(f"  '{county_col}' not found for Community level. Attempting to reconstruct from 'UnitName'.")
            if 'UnitName' in df.columns:
                # Assumes UnitName is 'COMMUNITYNAME_FIPS' and state is Illinois ('17')
                df[county_col] = '17' + df['UnitName'].str.split('_').str[-1]
                print(f"    Successfully created '{county_col}' column.")
            else:
                print(f"    Warning: 'UnitName' column not found. Cannot reconstruct county column.")
        # --- END FIX ---
                
        statewide_data[level_code] = df
        
        if county_col and county_col in df.columns:
            # Filter for Cook County. No need for .astype(str) as it's already read as string.
            df_cook = df[df[county_col].str.strip().str.endswith('031')].copy()
            if not df_cook.empty:
                cook_county_data[level_code] = df_cook

    # --- Generate existing comparative and unified plots ---
    print("\n--- Generating Statewide vs. Cook County Plots (per level) ---")
    for level_code, config in LEVELS_TO_ANALYZE.items():
        if level_code not in statewide_data:
            continue
        
        df_state = statewide_data[level_code]
        df_cook = cook_county_data.get(level_code, pd.DataFrame())
        level_name = config['level_name']

        print(f"\nProcessing {level_name} level...")
        
        # Call comparative plot function
        create_county_comparison_plot(df_state, config, 'Population')
        
        # Call unified plot functions
        create_unified_concentration_plot(df_state, config['pop_col'], config['sel_pop_col'], f"Statewide {level_name}", 'Population')
        if not df_cook.empty:
            create_unified_concentration_plot(df_cook, config['pop_col'], config['sel_pop_col'], f"Cook County {level_name}", 'Population')

    # --- Generate NEW Tract vs. Community comparison plots ---
    print("\n--- Generating Tract vs. Community Plots (per scope) ---")
    if 'tr' in statewide_data and 'cm' in statewide_data:
        create_level_comparison_plot(
            'Statewide', 
            statewide_data['tr'], 
            statewide_data['cm'], 
            LEVELS_TO_ANALYZE['tr'], 
            LEVELS_TO_ANALYZE['cm'], 
            'Population'
        )
    if 'tr' in cook_county_data and 'cm' in cook_county_data:
        create_level_comparison_plot(
            'Cook County', 
            cook_county_data['tr'], 
            cook_county_data['cm'], 
            LEVELS_TO_ANALYZE['tr'], 
            LEVELS_TO_ANALYZE['cm'], 
            'Population'
        )
    
    # --- Generate NEW Tract vs. Community comparison plots BY UNIT COUNT ---
    print("\n--- Generating Tract vs. Community Plots by Unit Count (per scope) ---")
    if 'tr' in statewide_data and 'cm' in statewide_data:
        create_level_comparison_plot_by_units(
            'Statewide', 
            statewide_data['tr'], 
            statewide_data['cm'], 
            LEVELS_TO_ANALYZE['tr'], 
            LEVELS_TO_ANALYZE['cm'], 
            'Population'
        )
    if 'tr' in cook_county_data and 'cm' in cook_county_data:
        create_level_comparison_plot_by_units(
            'Cook County', 
            cook_county_data['tr'], 
            cook_county_data['cm'], 
            LEVELS_TO_ANALYZE['tr'], 
            LEVELS_TO_ANALYZE['cm'], 
            'Population'
        )


if __name__ == "__main__":
    main()
