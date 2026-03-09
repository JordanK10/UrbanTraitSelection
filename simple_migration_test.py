import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ==================== USER CONFIGURATION ====================
# Adjust these parameters as needed

# Dataframe dimensions
NUM_ROWS = 500
NUM_COLS = 1

# Random number range for initial population
MIN_POPULATION = 20
MAX_POPULATION = 500

# Migration parameters
NUM_MIGRATIONS = 10000000  # Total number of people to move
PRINT_INTERVAL = 1000000    # Print dataframe every N migrations

# Histogram parameters
NUM_HISTOGRAM_BINS = 100     # Number of bins for the histogram plot

# ============================================================


def create_random_dataframe(rows, cols, min_val, max_val):
    """Creates a dataframe with random integer values."""
    data = np.random.randint(min_val, max_val + 1, size=(rows, cols))
    df = pd.DataFrame(data, columns=[f'Column_{i}' for i in range(cols)])
    return df


def perform_migration(df, num_migrations, print_interval):
    """
    Performs internal migration within the dataframe.
    For each migration:
    - Choose a cell (row, col) with probability proportional to its value
    - Subtract 1 from that cell
    - Choose a destination row uniformly in the SAME column, add 1
    """
    print("\n" + "="*60)
    print("INITIAL STATE")
    print("="*60)
    print(df)
    print(f"\nTotal population: {df.values.sum()}")
    
    num_rows, num_cols = df.shape
    
    for i in range(1, num_migrations + 1):
        # Calculate total population
        total_population = df.values.sum()
        
        if total_population == 0:
            print(f"\nWarning: No population left to migrate. Stopping at iteration {i}.")
            break
        
        # Flatten the dataframe to 1D array and calculate probabilities
        flat_values = df.values.flatten()
        probabilities = flat_values / total_population
        
        # Choose a source cell based on population-weighted probabilities
        # This gives us a flat index (0 to num_rows*num_cols - 1)
        flat_source_idx = np.random.choice(num_rows * num_cols, p=probabilities)
        
        # Convert flat index back to (row, col)
        source_row = flat_source_idx // num_cols
        source_col = flat_source_idx % num_cols
        
        # Choose a destination row uniformly within the same column
        dest_row = random.randint(0, num_rows - 1)
        
        # Perform the migration
        df.iloc[source_row, source_col] -= 1
        df.iloc[dest_row, source_col] += 1
        
        # Print at specified intervals
        if i % print_interval == 0:
            print("\n" + "="*60)
            print(f"AFTER {i} MIGRATIONS")
            print("="*60)
            print(df)
            print(f"\nTotal population: {df.values.sum()}")
    
    # Print final state if not already printed
    if num_migrations % print_interval != 0:
        print("\n" + "="*60)
        print(f"FINAL STATE AFTER {num_migrations} MIGRATIONS")
        print("="*60)
        print(df)
        print(f"\nTotal population: {df.values.sum()}")
    
    return df


def plot_column_histograms(df_initial, df_final, column_name='Column_0', num_bins=20):
    """
    Plots histograms of initial and final values for a specified column.
    
    Args:
        df_initial: Initial dataframe before migration
        df_final: Final dataframe after migration
        column_name: Name of the column to plot (default: 'Column_0')
        num_bins: Number of bins for the histogram (default: 20)
    """
    plt.figure(figsize=(10, 6))
    
    # Get the column data
    initial_values = df_initial[column_name].values
    final_values = df_final[column_name].values
    
    # Determine appropriate bin edges to use for both histograms
    all_values = np.concatenate([initial_values, final_values])
    min_val = all_values.min()
    max_val = all_values.max()
    bins = np.linspace(min_val - 5, max_val + 5, num_bins)
    
    # Plot both histograms
    plt.hist(initial_values, bins=bins, alpha=0.6, label='Initial', color='blue', edgecolor='black')
    plt.hist(final_values, bins=bins, alpha=0.6, label='Final', color='red', edgecolor='black')
    
    # Add labels and title
    plt.xlabel('Population Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of {column_name} Values: Initial vs Final', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Initial: mean={initial_values.mean():.1f}, std={initial_values.std():.1f}\n'
    stats_text += f'Final: mean={final_values.mean():.1f}, std={final_values.std():.1f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('migration_histogram_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('migration_histogram_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Histograms saved to 'migration_histogram_comparison.pdf' and '.png'")
    plt.show()


def main():
    """Main function to run the migration simulation."""
    print("\n" + "="*60)
    print("SIMPLE MIGRATION SIMULATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Dataframe size: {NUM_ROWS} rows x {NUM_COLS} columns")
    print(f"  Initial population range: {MIN_POPULATION} to {MAX_POPULATION}")
    print(f"  Number of migrations: {NUM_MIGRATIONS}")
    print(f"  Print interval: every {PRINT_INTERVAL} migrations")
    
    # Set random seed for reproducibility (optional)
    # np.random.seed(42)
    # random.seed(42)
    
    # Create initial dataframe
    df_initial = create_random_dataframe(NUM_ROWS, NUM_COLS, MIN_POPULATION, MAX_POPULATION)
    
    # Make a copy for migration (to preserve original for comparison)
    df_working = df_initial.copy()
    
    # Perform migrations
    df_final = perform_migration(df_working, NUM_MIGRATIONS, PRINT_INTERVAL)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    
    # Plot histograms comparing initial and final state of first column
    print("\nGenerating histogram comparison...")
    plot_column_histograms(df_initial, df_final, column_name='Column_0', num_bins=NUM_HISTOGRAM_BINS)
    

if __name__ == "__main__":
    main()
