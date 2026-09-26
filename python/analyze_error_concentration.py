from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT
    / "data"
    / "M40_M160_field_comparison.csv"
)

OUTPUT_FILE = (
    PROJECT
    / "data"
    / "M40_M160_error_concentration.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("========================================")
print(" M40-M160 DIFFERENCE CONCENTRATION")
print("========================================")

print("\nCells:", len(df))

error = df["dU_mag_star"].to_numpy()

# Squared velocity difference.
# This is useful because it gives more weight to cells
# with the largest discrepancies.
error2 = error**2

total_error2 = np.sum(error2)

if total_error2 <= 0.0:
    raise RuntimeError(
        "Total squared difference is zero."
    )

# ============================================================
# SORT CELLS FROM LARGEST TO SMALLEST DIFFERENCE
# ============================================================

order = np.argsort(error2)[::-1]

sorted_df = df.iloc[order].copy().reset_index(drop=True)

sorted_df["dU_mag_star_squared"] = (
    sorted_df["dU_mag_star"]**2
)

sorted_df["cumulative_error2_fraction"] = (
    sorted_df["dU_mag_star_squared"].cumsum()
    / total_error2
)

sorted_df["cumulative_cell_fraction"] = (
    (np.arange(len(sorted_df)) + 1)
    / len(sorted_df)
)

# ============================================================
# TARGET CONCENTRATION LEVELS
# ============================================================

targets = [
    0.50,
    0.80,
    0.90,
    0.95,
    0.99
]

print("\n========================================")
print(" CELLS REQUIRED TO CAPTURE")
print(" SUM(|dU|^2)")
print("========================================\n")

for target in targets:

    idx = np.searchsorted(
        sorted_df["cumulative_error2_fraction"].to_numpy(),
        target
    )

    n_cells = idx + 1

    fraction_cells = (
        n_cells / len(sorted_df)
    )

    threshold_error = (
        sorted_df.loc[idx, "dU_mag_star"]
    )

    print(
        f"{target*100:5.1f}% of sum(|dU|^2): "
        f"{n_cells:4d} cells "
        f"({fraction_cells*100:6.2f}% of domain), "
        f"minimum |dU|* = {threshold_error:.6e}"
    )

# ============================================================
# FIXED CELL BUDGETS
# ============================================================

budgets = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.20
]

print("\n========================================")
print(" DIFFERENCE CAPTURED BY TOP CELLS")
print("========================================\n")

for budget in budgets:

    n_cells = max(
        1,
        int(np.ceil(
            budget * len(sorted_df)
        ))
    )

    captured = (
        sorted_df.loc[
            n_cells - 1,
            "cumulative_error2_fraction"
        ]
    )

    print(
        f"Top {budget*100:5.1f}% "
        f"({n_cells:4d} cells) captures "
        f"{captured*100:6.2f}% of sum(|dU|^2)"
    )

# ============================================================
# BMU CONTRIBUTION TO TOTAL SQUARED DIFFERENCE
# ============================================================

bmu_summary = (
    df.assign(error2=error2)
    .groupby("bmu_id")
    .agg(
        cells=("error2", "size"),
        sum_error2=("error2", "sum"),
        mean_error2=("error2", "mean"),
        mean_dU=("dU_mag_star", "mean"),
        max_dU=("dU_mag_star", "max")
    )
)

bmu_summary["fraction_total_error2"] = (
    bmu_summary["sum_error2"]
    / total_error2
)

bmu_summary["fraction_cells"] = (
    bmu_summary["cells"]
    / len(df)
)

bmu_summary = bmu_summary.sort_values(
    "fraction_total_error2",
    ascending=False
)

print("\n========================================")
print(" CONTRIBUTION BY SOM NEURON")
print("========================================\n")

print(
    bmu_summary.to_string(
        float_format=lambda x: f"{x:.6e}"
    )
)

# ============================================================
# TOP ERROR CELLS: WHICH BMUs OWN THEM?
# ============================================================

for fraction in [0.01, 0.05, 0.10]:

    n_cells = max(
        1,
        int(np.ceil(
            fraction * len(sorted_df)
        ))
    )

    top = sorted_df.iloc[:n_cells]

    counts = (
        top["bmu_id"]
        .value_counts()
        .sort_index()
    )

    print("\n----------------------------------------")
    print(
        f"BMUs inside top "
        f"{fraction*100:.0f}% highest-difference cells"
    )
    print("----------------------------------------")

    print(counts.to_string())

# ============================================================
# SAVE SORTED DATA
# ============================================================

sorted_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSorted concentration data saved:")
print(OUTPUT_FILE)

print("\nAnalysis completed.")
