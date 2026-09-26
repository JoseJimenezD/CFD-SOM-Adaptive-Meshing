from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

DATASET_FILE = (
    PROJECT / "data" / "M40_som_dataset.csv"
)

COMPARISON_FILE = (
    PROJECT / "data" / "M40_M160_field_comparison.csv"
)

OUTPUT_FILE = (
    PROJECT / "data" / "M40_indicator_comparison.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

features = pd.read_csv(DATASET_FILE)
comparison = pd.read_csv(COMPARISON_FILE)

if len(features) != len(comparison):
    raise RuntimeError(
        "Feature and comparison datasets have different lengths."
    )

# Check cell ordering
if not np.allclose(
    features["x_star"].values,
    comparison["x_star"].values
):
    raise RuntimeError("x coordinates do not match.")

if not np.allclose(
    features["y_star"].values,
    comparison["y_star"].values
):
    raise RuntimeError("y coordinates do not match.")

df = comparison.copy()

df["gradU_mag_star"] = (
    features["gradU_mag_star"].values
)

# Squared M40-M160 velocity difference
df["error2"] = df["dU_mag_star"]**2

total_error2 = df["error2"].sum()

print("========================================")
print(" REFINEMENT INDICATOR COMPARISON")
print("========================================")

print("\nCells:", len(df))
print(
    "Total sum(|dU|^2):",
    f"{total_error2:.8e}"
)

# ============================================================
# BMU RISK SCORE
# ============================================================
#
# IMPORTANT:
# This score uses M160 information.
#
# Therefore it is ONLY a diagnostic/oracle score used here
# to determine whether SOM clusters separate regions with
# different observed inter-grid discrepancies.
#
# It must NOT be used later as a production refinement
# criterion without a pilot-only way to estimate importance.
# ============================================================

bmu_error = (
    df.groupby("bmu_id")["error2"]
    .mean()
    .sort_values(ascending=False)
)

df["bmu_oracle_score"] = (
    df["bmu_id"].map(bmu_error)
)

print("\nBMU diagnostic ranking:")
print(
    bmu_error.to_string(
        float_format=lambda x: f"{x:.8e}"
    )
)

# ============================================================
# INDICATORS
# ============================================================

indicators = {
    "Gradient |gradU|":
        df["gradU_mag_star"].to_numpy(),

    "SOM quantization error":
        df["quantization_error"].to_numpy(),

    "SOM BMU oracle":
        df["bmu_oracle_score"].to_numpy(),
}

# ============================================================
# CELL BUDGETS
# ============================================================

budgets = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.30
]

rows = []

print("\n========================================")
print(" CAPTURED sum(|dU|^2)")
print(" SAME CELL BUDGET FOR EACH INDICATOR")
print("========================================")

for budget in budgets:

    n_cells = max(
        1,
        int(np.ceil(
            budget * len(df)
        ))
    )

    print(
        f"\n--- Top {budget*100:.0f}% "
        f"= {n_cells} cells ---"
    )

    # Theoretical best possible selection:
    # directly rank by observed M40-M160 difference.
    oracle_order = np.argsort(
        df["error2"].to_numpy()
    )[::-1]

    oracle_selected = (
        oracle_order[:n_cells]
    )

    oracle_capture = (
        df["error2"]
        .to_numpy()[oracle_selected]
        .sum()
        / total_error2
    )

    print(
        f"{'Perfect error oracle':28s}: "
        f"{oracle_capture*100:7.3f}%"
    )

    rows.append(
        {
            "budget_fraction": budget,
            "n_cells": n_cells,
            "indicator": "Perfect error oracle",
            "captured_error2_fraction":
                oracle_capture
        }
    )

    for name, values in indicators.items():

        order = np.argsort(values)[::-1]

        selected = order[:n_cells]

        capture = (
            df["error2"]
            .to_numpy()[selected]
            .sum()
            / total_error2
        )

        print(
            f"{name:28s}: "
            f"{capture*100:7.3f}%"
        )

        rows.append(
            {
                "budget_fraction": budget,
                "n_cells": n_cells,
                "indicator": name,
                "captured_error2_fraction":
                    capture
            }
        )

# ============================================================
# OVERLAP WITH TRUE TOP-ERROR CELLS
# ============================================================

print("\n========================================")
print(" OVERLAP WITH HIGHEST-DIFFERENCE CELLS")
print("========================================")

error_order = np.argsort(
    df["error2"].to_numpy()
)[::-1]

for budget in budgets:

    n_cells = max(
        1,
        int(np.ceil(
            budget * len(df)
        ))
    )

    true_top = set(
        error_order[:n_cells]
    )

    print(
        f"\n--- Top {budget*100:.0f}% "
        f"= {n_cells} cells ---"
    )

    for name, values in indicators.items():

        order = np.argsort(values)[::-1]

        selected = set(
            order[:n_cells]
        )

        overlap = len(
            true_top.intersection(selected)
        )

        overlap_fraction = (
            overlap / n_cells
        )

        print(
            f"{name:28s}: "
            f"{overlap:4d}/{n_cells:4d} "
            f"({overlap_fraction*100:6.2f}%)"
        )

# ============================================================
# CORRELATIONS
# ============================================================

print("\n========================================")
print(" CORRELATION WITH OBSERVED DIFFERENCE")
print("========================================\n")

for name, values in indicators.items():

    corr = np.corrcoef(
        values,
        df["dU_mag_star"].to_numpy()
    )[0, 1]

    print(
        f"{name:28s}: r = {corr:.6f}"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(rows)

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nResults saved:")
print(OUTPUT_FILE)

print("\nAnalysis completed.")
