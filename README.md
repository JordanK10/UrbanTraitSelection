# Hierarchical Price Decomposition for Socio-Economic Analysis

This project implements a hierarchical Price decomposition to analyze changes in average log income across various geographic levels. The goal is to understand how much of the change at a higher level (e.g., county or MSA) is due to selection effects at that level versus effects transmitted from changes occurring within its constituent lower-level units (e.g., communities, tracts, block groups).

## Project Overview

The core idea is to apply the Price equation recursively across a defined geographic hierarchy. The Price equation decomposes the change in an average trait of a population into two components:
1.  A term representing selection acting on the entities within that population.
2.  A term representing the transmission of average trait changes from those entities.

By applying this hierarchically, we can attribute changes in socio-economic indicators, specifically average log income, to processes occurring at different spatial scales.

## Workflow and Scripts

The analysis pipeline consists of several Python scripts, each with a specific role:

1.  **`saveCensusDataV2.py`**:
    *   **Purpose**: This script is the foundational data acquisition step. It fetches raw socio-economic data—primarily household income distributions (ACS table B19001) and population counts (ACS table B01003)—from the U.S. Census Bureau's American Community Survey (ACS) API for specified years and geographic areas. It then processes this data by aggregating it to various predefined geographic levels (Block Group, Tract, Community, County, State) using a provided community mapping file for custom aggregations like Chicago community areas.
    *   **Inputs**:
        *   Census API Key (hardcoded).
        *   Predefined list of years for data fetching.
        *   Target geographic areas (defined by lists of FIPS codes for states and counties, e.g., for the Chicago metropolitan area).
        *   Specific ACS variable codes for income brackets and total population.
        *   Community mapping CSV files (`matched_chicago_data.csv`, `matched_chicagoLand_data.csv`) to link Census Tracts to community areas.
    *   **Workflow**:
        *   Loads community mapping files.
        *   Iterates through years and specified counties, attempting to fetch data at the Block Group level.
        *   Implements a fallback to fetch Tract-level data if Block Group data is unavailable.
        *   Aggregates the fetched data hierarchically (BG -> Tract -> Community -> County -> State) using the `genAggregatedDFs` function, which handles summing numeric data and linking geographic identifiers.
    *   **Outputs**:
        *   A primary pickle file (`data/census_data1.pkl`) containing a nested dictionary. Outer keys are years, inner keys are aggregation levels ('bg', 'tr', 'cm', 'ct', 'st'), and values are pandas DataFrames with the aggregated census data.
        *   Detailed console logs of the fetching and aggregation process.

2.  **`procHelpers.py`**:
    *   **Purpose**: Contains shared utility functions and constant definitions (e.g., income bin midpoints, list of income variables, geographic hierarchy levels) used by other processing scripts.
    *   **Inputs**: Not a standalone script; functions are imported.
    *   **Outputs**: Provides data and functions to other scripts.

3.  **`procPrice2.py`**:
    *   **Purpose**: This script (hypothetical, based on project structure) would perform initial calculations for individual geographic units across various N-year periods using the data from `saveCensusDataV2.py`. It focuses on within-unit changes, calculating metrics like the total change in average log income (ΔZ̄), population change metrics, and potentially direct Price decomposition components (covariance and expectation terms) if the analysis requires looking at internal shifts within the lowest-level units.
    *   **Inputs**:
        *   `data/census_data1.pkl` (output from `saveCensusDataV2.py`).
        *   Configuration for N-year periods.
        *   Constants for income calculations (e.g., income bin midpoints from `procHelpers.py`).
    *   **Workflow (Speculative)**:
        *   Loads `census_data1.pkl`.
        *   Loops through N-year periods and aggregation levels.
        *   For each unit, calculates initial and final average log income, total change (ΔZ̄), population changes, and Price components like `CovTerm_N`, `ExpTerm_N`, `CovIncTerm_N`, `ExpIncTerm_N`, `CovDistShift_N`, and `CovFracPopChg_N`.
        *   Includes sanity checks for the decomposed terms.
    *   **Outputs**:
        *   A pickle file (e.g., `results/analysis_Price2.pkl`) containing a comprehensive DataFrame. Each row represents a geographic unit for a specific N-year period, with columns for identifiers, parent identifiers, population and income data, total change in average log income, and the calculated Price components. This file is a key input for `aggregatePrice2.py` and `plotPrice.py`.

## Key Concepts

*   **Price Equation**: A covariance equation that provides a formal mathematical description of evolution and selection. In this context, it's used to decompose the change in average log income.
*   **Hierarchical Decomposition**: The process of applying the Price equation iteratively across nested levels of a system. This allows for attributing change at a macro-level to processes at various micro-levels and the transmission of those changes.
*   **Selection Term**: Represents the change in the average trait due to the covariance between the trait values of sub-units and their fitness (e.g., population growth).
*   **Transmission Term**: Represents the change in the average trait due to the average change within the sub-units, weighted by their fitness.

## How to Run (General Steps)

1.  **Obtain Data**: Run `saveCensusDataV2.py` (ensure a valid Census API key is set in the script) to download the necessary ACS data. This creates `data/census_data1.pkl`.
2.  **Process Base Metrics**: Run `procPrice2.py` to calculate Price components and other metrics for individual units. This creates `results/analysis_Price2.pkl`.
3.  **Perform Hierarchical Aggregation**: Run `aggregatePrice2.py` to perform the hierarchical decomposition. This uses the outputs from the previous steps and produces the main decomposition results.
4.  **Visualize/Inspect**: Use `plotPrice.py`, `plotCov.py`, and `inspect_pkl.py` to analyze and visualize the generated data.

Ensure all Python dependencies (pandas, numpy, requests, matplotlib, seaborn, etc.) are installed. 