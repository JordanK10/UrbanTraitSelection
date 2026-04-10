import argparse
import os
import requests
import pandas as pd
import numpy as np

# Sentinel value and API key
sentinel_value = -666666666.0
api_key = "35d314060d56f894db2f7621b0e5e5f7eca9af27"

# Chicago metro area counties
counties = ["031", "043", "089", "093", "097", "111", "197"]
states = ["17"] * 7
cty_name = ["Cook", "DuPg", "Kane", "Kndl", "Lke", "McHn", "Will"]

# Income bracket variables (16 bins)
income_vars = [
    "B19001_002E", "B19001_003E", "B19001_004E", "B19001_005E",
    "B19001_006E", "B19001_007E", "B19001_008E", "B19001_009E",
    "B19001_010E", "B19001_011E", "B19001_012E", "B19001_013E",
    "B19001_014E", "B19001_015E", "B19001_016E", "B19001_017E",
]

# Population variables
pop_vars = ["B01003_001E", "B19001_001E", "B25010_001E"]
all_vars = income_vars + pop_vars


def fetch_data(year):
    """Fetch census data for a given year."""
    print(f"\nFetching 20{year} data...")
    base_url = f"https://api.census.gov/data/20{year}/acs/acs5"
    all_dfs = []

    for state, county, cty in zip(states, counties, cty_name):
        print(f"  {cty} County...", end=" ")
        bg_geography = f"block%20group:*&in=state:{state}&in=county:{county}"
        bg_url = f"{base_url}?get={','.join(all_vars)}&for={bg_geography}&key={api_key}"

        try:
            response = requests.get(bg_url, timeout=60)
            response.raise_for_status()
            data = response.json()
            if len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
                print(f"✓ {len(df)} BGs")
                all_dfs.append(df)
            else:
                print("✗ No data")
        except Exception as e:
            print(f"✗ Error: {e}")

    df_combined = pd.concat(all_dfs, ignore_index=True)
    df_combined["GEOID"] = (
        df_combined["state"]
        + df_combined["county"]
        + df_combined["tract"]
        + df_combined["block group"]
    )

    for col in all_vars:
        df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce")

    print(f"✓ Total: {len(df_combined)} block groups\n")
    return df_combined


def clean_and_align(df_2014, df_2019):
    """Remove sentinel values and align block groups."""
    print("Cleaning and aligning dataframes...")

    df_2014 = df_2014[df_2014["B25010_001E"] >= 0]
    df_2019 = df_2019[df_2019["B25010_001E"] >= 0]

    common_geoids = set(df_2014["GEOID"]) & set(df_2019["GEOID"])
    df_2014 = df_2014[df_2014["GEOID"].isin(common_geoids)].sort_values("GEOID").reset_index(drop=True)
    df_2019 = df_2019[df_2019["GEOID"].isin(common_geoids)].sort_values("GEOID").reset_index(drop=True)

    print(f"✓ {len(df_2014)} aligned block groups\n")
    return df_2014, df_2019


def build_complete_graph_dest_probs(mass, alpha):
    """
    Distance-agnostic gravity on a complete graph: P(j | i) ∝ M_j^alpha / const
    for all origins i. mass_j is fixed baseline (e.g. 2014 BG population).
    alpha=0 yields uniform destinations over block groups.
    """
    mass = np.asarray(mass, dtype=np.float64)
    mass = np.maximum(mass, 0.0)
    if alpha == 0:
        w = np.ones_like(mass)
    else:
        w = np.power(mass, alpha)
    s = w.sum()
    if s <= 0:
        w = np.ones_like(mass)
        s = w.sum()
    return (w / s).astype(np.float64)


def convert_to_population(df):
    """Convert household counts to population counts."""
    print("Converting households to population...")
    df = df.copy()

    avg_hh_size = df["B25010_001E"]
    for var in income_vars:
        df[var] = np.floor(df[var] * avg_hh_size)

    df["B19001_001E"] = df[income_vars].sum(axis=1)
    df = df.drop(columns=["B01003_001E", "B25010_001E"])

    print(f"✓ Population conversion complete\n")
    return df


def apply_internal_migration(df_2014, df_2019, mode="population", alpha=1.0):
    """
    Internal migration micro-simulation. Source: cell ∝ initial 2014 population.

    mode 'population': destination BG ∝ initial 2014 BG population.
    mode 'gravity_network': complete-graph gravity; P(dest=j) ∝ M_j^alpha with
        fixed M_j = 2014 baseline BG population (distance-agnostic).

    Both modes are fully vectorized: all num_moves sources and destinations are
    sampled in two numpy calls, then applied with a single bulk delta update.
    Approximation: sampling from initial rather than updated distributions.
    The error is O(num_moves / total_population), negligible for a null model.
    """
    print("=" * 60)
    print("APPLYING INTERNAL MIGRATION (vectorized)")
    if mode == "gravity_network":
        print(f"MODE: gravity_network (complete graph, alpha={alpha})")
    else:
        print("MODE: population (dest ∝ 2014 BG population)")
    print("=" * 60)

    print("\nAlignment verification (first 5 block groups):")
    print("  2014 GEOIDs:", df_2014["GEOID"].head(5).tolist())
    print("  2019 GEOIDs:", df_2019["GEOID"].head(5).tolist())

    print("\nAlignment verification (last 5 block groups):")
    print("  2014 GEOIDs:", df_2014["GEOID"].tail(5).tolist())
    print("  2019 GEOIDs:", df_2019["GEOID"].tail(5).tolist())

    if np.array_equal(df_2014["GEOID"].values, df_2019["GEOID"].values):
        print("\n✓ SUCCESS: All GEOIDs match in order (perfect alignment)")
    else:
        print("\n✗ ERROR: GEOIDs do not match in order!")
        return None

    pop_2014 = df_2014[income_vars].values
    pop_2019 = df_2019[income_vars].values

    total_pop_2014 = pop_2014.sum()
    total_pop_2019 = pop_2019.sum()
    total_change = abs(total_pop_2019 - total_pop_2014)

    differences_per_cell = pop_2019 - pop_2014
    sum_abs_differences = np.abs(differences_per_cell).sum()

    churn = sum_abs_differences - total_change
    num_moves = int(churn / 2)

    print(f"\n2014 total population: {total_pop_2014:.0f}")
    print(f"2019 total population: {total_pop_2019:.0f}")
    print(f"Total population change: {total_change:.0f}")
    print(f"Sum of |differences|: {sum_abs_differences:.0f}")
    print(f"Churn (internal moves): {churn:.0f}")
    print(f"Number of people to move: {num_moves}\n")

    df_result = df_2014.copy()
    num_rows = len(df_result)
    num_cols = len(income_vars)

    # --- Precompute destination probabilities ---
    if mode == "gravity_network":
        mass_2014 = df_2014["B19001_001E"].values.astype(np.float64)
        dest_probs = build_complete_graph_dest_probs(mass_2014, alpha)
    else:
        bg_pop_2014 = pop_2014.sum(axis=1).astype(np.float64)
        dest_probs = bg_pop_2014 / bg_pop_2014.sum()

    # --- Batch-sample all sources from initial 2014 distribution ---
    print("Sampling sources and destinations...")
    flat_pop = pop_2014.flatten().astype(np.float64)
    src_probs = flat_pop / flat_pop.sum()
    flat_src = np.random.choice(num_rows * num_cols, size=num_moves, p=src_probs)
    src_rows = flat_src // num_cols
    src_cols = flat_src % num_cols

    # --- Batch-sample all destinations ---
    dest_rows = np.random.choice(num_rows, size=num_moves, p=dest_probs)

    # --- Compute net delta (income bin preserved on each move) ---
    print("Applying moves...")
    pop_array = pop_2014.astype(np.float64).copy()
    delta = np.zeros_like(pop_array)
    np.subtract.at(delta, (src_rows, src_cols), 1)
    np.add.at(delta, (dest_rows, src_cols), 1)
    pop_array += delta

    df_result[income_vars] = pop_array
    df_result["B19001_001E"] = pop_array.sum(axis=1)

    print(f"\n✓ Migration complete")
    print(f"Final population: {df_result['B19001_001E'].sum():.0f}")
    print("=" * 60 + "\n")

    return df_result


def main():
    parser = argparse.ArgumentParser(description="Null internal migration micro-simulation.")
    parser.add_argument(
        "--mode",
        choices=["population", "gravity_network"],
        default="population",
        help="'population': dest ∝ current BG pop; "
        "'gravity_network': complete graph, P(dest) ∝ fixed 2014 mass^alpha.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Mass exponent for gravity_network (ignored for population mode).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("DATA NULL MIGRATION MODEL")
    print("=" * 60 + "\n")

    if args.mode == "gravity_network":
        out_dir = "null_model/simulation_results/data_null_gravity"
    else:
        out_dir = "null_model/simulation_results/data_null_migration"
    os.makedirs(out_dir, exist_ok=True)

    df_2014_raw = fetch_data(14)
    df_2019_raw = fetch_data(19)

    df_2014_clean, df_2019_clean = clean_and_align(df_2014_raw, df_2019_raw)

    df_2014_pop = convert_to_population(df_2014_clean)
    df_2019_pop = convert_to_population(df_2019_clean)

    df_2019_simulated = apply_internal_migration(
        df_2014_pop, df_2019_pop, mode=args.mode, alpha=args.alpha
    )

    out_2014 = os.path.join(out_dir, "Census-Data-2014.csv")
    out_2019 = os.path.join(out_dir, "Simulated-Data-2019.csv")
    print("Saving results...")
    df_2014_pop.to_csv(out_2014, index=False)
    df_2019_simulated.to_csv(out_2019, index=False)
    print(f"✓ Saved to {out_dir}/\n")


if __name__ == "__main__":
    main()
