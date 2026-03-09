import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ==================== USER CONFIGURATION ====================
# Adjust these parameters as needed

# Dataframe dimensions
NUM_ROWS = 500
NUM_COLS = 5

# Random number range for initial population
MIN_POPULATION = 20
MAX_POPULATION = 500

# Swap parameters
NUM_SWAPS = 10000000  # Total number of swaps to perform
PRINT_INTERVAL = 1000000   # Print dataframe every N swaps

# Histogram parameters
NUM_HISTOGRAM_BINS = 100     # Number of bins for the histogram plots
NUM_BLOCK_GROUPS_TO_PLOT = 3  # Number of block groups to include in right plot

# ============================================================


def create_random_dataframe(rows, cols, min_val, max_val):
    """Creates a dataframe with random integer values."""
    data = np.random.randint(min_val, max_val + 1, size=(rows, cols))
    df = pd.DataFrame(data, columns=[f'Column_{i}' for i in range(cols)])
    return df


def perform_swap(df, num_swaps, print_interval):
    """
    Performs swap operations within the dataframe.
    For each swap:
    - Choose two cells (i,j) and (p,q) with probability proportional to their values
    - Subtract 1 from (i,j) and add 1 to (p,j) [person from location i moves to location p in column j]
    - Subtract 1 from (p,q) and add 1 to (i,q) [person from location p moves to location i in column q]
    This effectively swaps people between two locations while keeping them in their respective columns.
    """
    print("\n" + "="*60)
    print("INITIAL STATE")
    print("="*60)
    print(df)
    print(f"\nTotal population: {df.values.sum()}")
    
    num_rows, num_cols = df.shape
    
    for iteration in range(1, num_swaps + 1):
        # Calculate total population
        total_population = df.values.sum()
        
        if total_population < 2:
            print(f"\nWarning: Not enough population to swap. Stopping at iteration {iteration}.")
            break
        
        # Flatten the dataframe to 1D array and calculate probabilities
        flat_values = df.values.flatten()
        probabilities = flat_values / total_population
        
        # Choose TWO different cells based on population-weighted probabilities
        # Sample without replacement to ensure they're different
        flat_indices = np.random.choice(
            num_rows * num_cols, 
            size=2, 
            replace=False, 
            p=probabilities
        )
        
        # Convert flat indices back to (row, col) for both cells
        cell1_flat_idx = flat_indices[0]
        cell2_flat_idx = flat_indices[1]
        
        i = cell1_flat_idx // num_cols  # row of first cell
        j = cell1_flat_idx % num_cols   # column of first cell
        
        p = cell2_flat_idx // num_cols  # row of second cell
        q = cell2_flat_idx % num_cols   # column of second cell
        
        # Perform the swap:
        # Move person from (i,j) to (p,j)
        df.iloc[i, j] -= 1
        df.iloc[p, j] += 1
        
        # Move person from (p,q) to (i,q)
        df.iloc[p, q] -= 1
        df.iloc[i, q] += 1
        
        # Print at specified intervals
        if iteration % print_interval == 0:
            print("\n" + "="*60)
            print(f"AFTER {iteration} SWAPS")
            print("="*60)
            print(df)
            print(f"\nTotal population: {df.values.sum()}")
    
    # Print final state if not already printed
    if num_swaps % print_interval != 0:
        print("\n" + "="*60)
        print(f"FINAL STATE AFTER {num_swaps} SWAPS")
        print("="*60)
        print(df)
        print(f"\nTotal population: {df.values.sum()}")
    
    return df


def plot_column_histograms(df_initial, df_final, num_bins=20, num_block_groups=10):
    """
    Plots two histograms showing:
    1. Distribution of all individual cell values in the dataframe
    2. Distribution of population values within a subset of block groups
    
    Args:
        df_initial: Initial dataframe before swaps
        df_final: Final dataframe after swaps
        num_bins: Number of bins for the histogram plots (default: 20)
        num_block_groups: Number of block groups (rows) to include in right plot (default: 10)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ======== LEFT PLOT: All Cell Values Distribution ========
    # Flatten all values in the dataframe
    initial_all_values = df_initial.values.flatten()  # All individual cell values
    final_all_values = df_final.values.flatten()
    
    # Determine bin edges based on number of bins
    all_values_combined = np.concatenate([initial_all_values, final_all_values])
    min_val_cells = all_values_combined.min()
    max_val_cells = all_values_combined.max()
    bins_cells = np.linspace(min_val_cells - 5, max_val_cells + 5, num_bins)
    
    # Plot all cell values histograms
    axes[0].hist(initial_all_values, bins=bins_cells, alpha=0.6, label='Initial', 
                 color='blue', edgecolor='black')
    axes[0].hist(final_all_values, bins=bins_cells, alpha=0.6, label='Final', 
                 color='red', edgecolor='black')
    
    axes[0].set_xlabel('Population Count per Cell', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of All Cell Values in Dataframe', 
                      fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add statistics for all cell values
    stats_text_cells = f'Initial: mean={initial_all_values.mean():.1f}, std={initial_all_values.std():.1f}\n'
    stats_text_cells += f'Final: mean={final_all_values.mean():.1f}, std={final_all_values.std():.1f}'
    axes[0].text(0.02, 0.98, stats_text_cells, transform=axes[0].transAxes, 
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ======== RIGHT PLOT: Selected Block Groups Distribution ========
    # Select first N block groups (rows)
    num_rows_to_plot = min(num_block_groups, len(df_initial))
    
    # Flatten values from the selected rows only
    initial_subset_values = df_initial.iloc[:num_rows_to_plot].values.flatten()
    final_subset_values = df_final.iloc[:num_rows_to_plot].values.flatten()
    
    # Determine bin edges based on number of bins
    all_subset_values = np.concatenate([initial_subset_values, final_subset_values])
    min_val_subset = all_subset_values.min()
    max_val_subset = all_subset_values.max()
    bins_subset = np.linspace(min_val_subset - 5, max_val_subset + 5, num_bins)
    
    # Plot subset histograms
    axes[1].hist(initial_subset_values, bins=bins_subset, alpha=0.6, label='Initial', 
                 color='blue', edgecolor='black')
    axes[1].hist(final_subset_values, bins=bins_subset, alpha=0.6, label='Final', 
                 color='red', edgecolor='black')
    
    axes[1].set_xlabel('Population Count per Cell', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Distribution of Population in First {num_rows_to_plot} Block Groups', 
                      fontsize=13, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add statistics for subset
    stats_text_subset = f'Initial: mean={initial_subset_values.mean():.1f}, std={initial_subset_values.std():.1f}\n'
    stats_text_subset += f'Final: mean={final_subset_values.mean():.1f}, std={final_subset_values.std():.1f}'
    axes[1].text(0.02, 0.98, stats_text_subset, transform=axes[1].transAxes, 
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('swap_histogram_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('swap_histogram_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Histograms saved to 'swap_histogram_comparison.pdf' and '.png'")
    plt.show()


def main():
    """Main function to run the swap simulation."""
    print("\n" + "="*60)
    print("SIMPLE SWAP SIMULATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Dataframe size: {NUM_ROWS} rows x {NUM_COLS} columns")
    print(f"  Initial population range: {MIN_POPULATION} to {MAX_POPULATION}")
    print(f"  Number of swaps: {NUM_SWAPS}")
    print(f"  Print interval: every {PRINT_INTERVAL} swaps")
    
    # Set random seed for reproducibility (optional)
    # np.random.seed(42)
    # random.seed(42)
    
    # Create initial dataframe
    df_initial = create_random_dataframe(NUM_ROWS, NUM_COLS, MIN_POPULATION, MAX_POPULATION)
    
    # Make a copy for swapping (to preserve original for comparison)
    df_working = df_initial.copy()
    
    # Perform swaps
    df_final = perform_swap(df_working, NUM_SWAPS, PRINT_INTERVAL)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    
    # Plot histograms comparing initial and final state of first column
    print("\nGenerating histogram comparison...")
    plot_column_histograms(df_initial, df_final, num_bins=NUM_HISTOGRAM_BINS, 
                          num_block_groups=NUM_BLOCK_GROUPS_TO_PLOT)
    

if __name__ == "__main__":
    main()
