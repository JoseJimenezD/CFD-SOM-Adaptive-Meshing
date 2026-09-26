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
    PROJECT / "data" / "M40_indicator_complementarity.csv"
)

# ============================================================
# SETTINGS
# ============================================================

# Compare both indicators using exactly the same number of cells
BUDGET_FRACTION = 0.10

# ============================================================
# LOAD DATA
# ============================================================

features = pd.read_csv(DATASET_FILE)
comparison = pd.read_csv(COMPARISON_FILE)

if len(features) != len(comparison):
    raise RuntimeError(
        "Feature and comparison datasets have different lengths."
    )

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

df["error2"] = (
    df["dU_mag_star"]**2
)

total_error2 = df["error2"].sum()

n_total = len(df)

n_select = int(
    np.ceil(
        BUDGET_FRACTION * n_total
    )
)

print("========================================")
print(" INDICATOR COMPLEMENTARITY ANALYSIS")
print("========================================")

print("\nTotal cells:", n_total)

print(
    f"Budget: {BUDGET_FRACTION*100:.1f}% "
    f"= {n_select} cells per indicator"
)

# ============================================================
# SELECT TOP CELLS
# ============================================================

gradient_order = np.argsort(
    df["gradU_mag_star"].to_numpy()
)[::-1]

som_order = np.argsort(
    df["quantization_error"].to_numpy()
)[::-1]

gradient_set = set(
    gradient_order[:n_select]
)

som_set = set(
    som_order[:n_select]
)

both = gradient_set.intersection(
    som_set
)

som_only = som_set.difference(
    gradient_set
)

gradient_only = gradient_set.difference(
    som_set
)

neither = (
    set(range(n_total))
    .difference(
        gradient_set.union(som_set)
    )
)

# ============================================================
# HELPER FUNCTION
# ============================================================

def summarize_group(name, indices):

    indices = sorted(indices)

    if len(indices) == 0:

        print(
            f"{name:20s}: 0 cells"
        )

        return {
            "group": name,
            "cells": 0,
            "fraction_domain": 0.0,
            "sum_error2": 0.0,
            "fraction_total_error2": 0.0,
            "mean_dU": np.nan,
            "max_dU": np.nan
        }

    subset = df.iloc[indices]

    group_error2 = (
        subset["error2"].sum()
    )

    fraction_error2 = (
        group_error2 / total_error2
    )

    mean_dU = (
        subset["dU_mag_star"].mean()
    )

    max_dU = (
        subset["dU_mag_star"].max()
    )

    print(
        f"{name:20s}: "
        f"{len(indices):4d} cells | "
        f"{100*len(indices)/n_total:6.2f}% domain | "
        f"{100*fraction_error2:7.3f}% error2 | "
        f"mean |dU|*={mean_dU:.6e} | "
        f"max={max_dU:.6e}"
    )

    return {
        "group": name,
        "cells": len(indices),
        "fraction_domain":
            len(indices) / n_total,
        "sum_error2":
            group_error2,
        "fraction_total_error2":
            fraction_error2,
        "mean_dU":
            mean_dU,
        "max_dU":
            max_dU
    }

# ============================================================
# GROUP COMPARISON
# ============================================================

print("\n========================================")
print(" SELECTION OVERLAP")
print("========================================\n")

rows = []

rows.append(
    summarize_group(
        "Both",
        both
    )
)

rows.append(
    summarize_group(
        "SOM only",
        som_only
    )
)

rows.append(
    summarize_group(
        "Gradient only",
        gradient_only
    )
)

rows.append(
    summarize_group(
        "Neither",
        neither
    )
)

# ============================================================
# UNION
# ============================================================

union = gradient_set.union(
    som_set
)

union_error2 = (
    df.iloc[sorted(union)]["error2"].sum()
)

print("\n========================================")
print(" UNION")
print("========================================")

print(
    f"\nGradient OR SOM selects "
    f"{len(union)} cells "
    f"({100*len(union)/n_total:.2f}% of domain)"
)

print(
    "Captured sum(|dU|^2): "
    f"{100*union_error2/total_error2:.3f}%"
)

# ============================================================
# UNIQUE CONTRIBUTION
# ============================================================

som_only_error2 = (
    df.iloc[sorted(som_only)]["error2"].sum()
)

gradient_only_error2 = (
    df.iloc[sorted(gradient_only)]["error2"].sum()
)

print("\n========================================")
print(" UNIQUE CONTRIBUTION")
print("========================================")

print(
    "\nError2 contained in SOM-only cells     : "
    f"{100*som_only_error2/total_error2:.3f}%"
)

print(
    "Error2 contained in Gradient-only cells: "
    f"{100*gradient_only_error2/total_error2:.3f}%"
)

difference = (
    som_only_error2
    - gradient_only_error2
)

print(
    "SOM-only minus Gradient-only           : "
    f"{100*difference/total_error2:+.3f}%"
)

# ============================================================
# BMUs OF SOM-ONLY CELLS
# ============================================================

if len(som_only) > 0:

    som_only_df = df.iloc[
        sorted(som_only)
    ]

    print("\n========================================")
    print(" BMUs IN SOM-ONLY REGION")
    print("========================================\n")

    print(
        som_only_df["bmu_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

# ============================================================
# CREATE CELL-BY-CELL OUTPUT
# ============================================================

df["selected_gradient"] = False
df["selected_som"] = False

df.loc[
    list(gradient_set),
    "selected_gradient"
] = True

df.loc[
    list(som_set),
    "selected_som"
] = True

conditions = [
    df["selected_gradient"]
    & df["selected_som"],

    ~df["selected_gradient"]
    & df["selected_som"],

    df["selected_gradient"]
    & ~df["selected_som"]
]

choices = [
    "both",
    "som_only",
    "gradient_only"
]

df["selection_group"] = np.select(
    conditions,
    choices,
    default="neither"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nCell-by-cell results saved:")
print(OUTPUT_FILE)

print("\nAnalysis completed.")
