#!/usr/bin/env python3
"""
Null Model Parameter Distribution Analysis
Runs null model simulations repeatedly and analyzes the distribution of 
selection parameters across multiple runs.
"""

import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import importlib.util

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Number of simulation runs to perform
NUM_RUNS = 20

# Path to the simulation script to run (relative to project root)
SIMULATION_SCRIPT = "null_model/simulation_scripts/swapping_scripts/swapping-percentile-filtering.py"

# Command-line argument for loadNullModelData.py (e.g., 'migration', 'swapping', 'uniform')
LOAD_NULL_MODEL_ARG = "swapping"

# Output directory for distribution plots
OUTPUT_DIR = "plots/parameter_distributions"

# ============================================================================
# PIPELINE SCRIPTS (paths relative to project root)
# ============================================================================

PIPELINE_SCRIPTS = {
    'load_null': 'calculation_scripts/loadNullModelData.py',
    'proc_price': 'calculation_scripts/procPrice.py',
    'aggregate': 'calculation_scripts/aggregatePriceV5.py',
    'extract_params': 'analysis_scripts/specialty_histogram_pnc_only.py'
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_script(script_path, args=None):
    """Run a Python script as a subprocess."""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Script failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(result.stdout)
    return True


def import_and_run_module(script_path):
    """Import a module and run its main() function, returning the result."""
    # Get absolute path
    abs_path = os.path.abspath(script_path)
    
    # Create module name from path
    module_name = os.path.splitext(os.path.basename(script_path))[0]
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Run main() and return result
    if hasattr(module, 'main'):
        return module.main()
    else:
        print(f"Warning: {script_path} has no main() function")
        return None


def run_single_iteration(iteration_num):
    """Run one complete iteration of the pipeline."""
    print(f"\n{'#'*60}")
    print(f"# ITERATION {iteration_num + 1}/{NUM_RUNS}")
    print(f"{'#'*60}")
    
    # Step 1: Run simulation script
    if not run_script(SIMULATION_SCRIPT):
        print(f"Iteration {iteration_num + 1} failed at simulation step")
        return None
    
    # Step 2: Load null model data
    if not run_script(PIPELINE_SCRIPTS['load_null'], [LOAD_NULL_MODEL_ARG]):
        print(f"Iteration {iteration_num + 1} failed at load_null step")
        return None
    
    # Step 3: Run procPrice
    if not run_script(PIPELINE_SCRIPTS['proc_price']):
        print(f"Iteration {iteration_num + 1} failed at proc_price step")
        return None
    
    # Step 4: Run aggregatePrice
    if not run_script(PIPELINE_SCRIPTS['aggregate']):
        print(f"Iteration {iteration_num + 1} failed at aggregate step")
        return None
    
    # Step 5: Extract parameters
    print(f"\n{'='*60}")
    print(f"Extracting parameters from specialty_histogram_pnc_only.py")
    print(f"{'='*60}")
    
    try:
        results = import_and_run_module(PIPELINE_SCRIPTS['extract_params'])
        if results is None or len(results) == 0:
            print(f"Iteration {iteration_num + 1} failed: no parameters returned")
            return None
        return results
    except Exception as e:
        print(f"Iteration {iteration_num + 1} failed at parameter extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


def collect_parameters():
    """Run all iterations and collect parameters."""
    # Structure: all_params[metric_name][level][parameter] = [values across runs]
    all_params = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    successful_runs = 0
    
    for i in range(NUM_RUNS):
        results = run_single_iteration(i)
        
        if results is None:
            print(f"⚠️  Iteration {i + 1} failed, skipping...")
            continue
        
        # Process results
        # results is a list like:
        # [
        #   {'metric': 'Income PNC_st', 'community': {...}, 'tract': {...}},
        #   {'metric': 'Population PNC_st', 'community': {...}, 'tract': {...}}
        # ]
        
        for metric_result in results:
            metric_name = metric_result['metric']
            
            for level in ['community', 'tract']:
                if metric_result[level] is not None:
                    level_data = metric_result[level]
                    for param in ['df', 'loc', 'scale', 'skew', 'mean']:
                        if param in level_data:
                            all_params[metric_name][level][param].append(level_data[param])
        
        successful_runs += 1
        print(f"✓ Iteration {i + 1} completed successfully ({successful_runs}/{i + 1})")
    
    print(f"\n{'='*60}")
    print(f"Collection complete: {successful_runs}/{NUM_RUNS} successful runs")
    print(f"{'='*60}")
    
    return all_params, successful_runs


def create_distribution_plots(all_params, successful_runs):
    """Create distribution plots for all parameters."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Creating distribution plots...")
    print(f"{'='*60}")
    
    plot_count = 0
    
    for metric_name in all_params.keys():
        for level in ['community', 'tract']:
            for param in ['df', 'loc', 'scale', 'skew', 'mean']:
                
                if param not in all_params[metric_name][level]:
                    continue
                
                values = all_params[metric_name][level][param]
                
                if len(values) == 0:
                    print(f"  Skipping {metric_name} - {level} - {param}: no data")
                    continue
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot histogram
                ax.hist(values, bins=min(30, len(values)//3), 
                       edgecolor='black', alpha=0.7, color='steelblue')
                
                # Add statistics
                mean_val = np.mean(values)
                median_val = np.median(values)
                std_val = np.std(values)
                
                stats_text = f'Mean: {mean_val:.4f}\nMedian: {median_val:.4f}\nStd: {std_val:.4f}\nN: {len(values)}'
                ax.text(0.02, 0.98, stats_text, 
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=10)
                
                # Add vertical lines for mean and median
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label='Median')
                
                # Labels and title
                ax.set_xlabel(f'{param.capitalize()} Value', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                
                # Create clean title
                selection_type = 'Income' if 'Income' in metric_name else 'Population'
                title = f'Distribution of {param.capitalize()} - {level.capitalize()} Level\n{selection_type} Selection ({successful_runs} runs)'
                ax.set_title(title, fontsize=14, fontweight='bold')
                
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Save figure
                filename = f"{selection_type.lower()}_{level}_{param}_distribution.pdf"
                filepath = os.path.join(OUTPUT_DIR, filename)
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                plt.close()
                
                plot_count += 1
                print(f"  ✓ Created: {filename}")
    
    print(f"\n{'='*60}")
    print(f"✓ Created {plot_count} distribution plots in {OUTPUT_DIR}")
    print(f"{'='*60}")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("NULL MODEL PARAMETER DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Number of runs: {NUM_RUNS}")
    print(f"  - Simulation script: {SIMULATION_SCRIPT}")
    print(f"  - Load null model arg: {LOAD_NULL_MODEL_ARG}")
    print(f"  - Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    # Collect parameters from all runs
    all_params, successful_runs = collect_parameters()
    
    if successful_runs == 0:
        print("\n❌ ERROR: No successful runs. Cannot create plots.")
        return
    
    # Create distribution plots
    create_distribution_plots(all_params, successful_runs)
    
    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
