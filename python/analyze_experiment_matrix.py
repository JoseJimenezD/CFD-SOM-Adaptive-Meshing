from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT / "results"
RESULTS_FILE = RESULTS_DIR / "experiment_results.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------

df = pd.read_csv(RESULTS_FILE)

required = {
    "name",
    "method",
    "budget",
    "final_cells",
    "execution_time_s",
    "mean_v_error",
    "rms_v_error",
    "max_error",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}"
    )


# ------------------------------------------------------------
# Verify controlled 3x4 experiment matrix
# ------------------------------------------------------------

expected = {
    "GRAD_05",
    "GRAD_10",
    "GRAD_15",
    "GRAD_20",
    "SOM_QE_05",
    "SOM_QE_10",
    "SOM_QE_15",
    "SOM_QE_20",
    "HYBRID_05",
    "HYBRID_10",
    "HYBRID_15",
    "HYBRID_20",
}

available = set(df["name"])

missing_cases = expected - available

if missing_cases:
    raise RuntimeError(
        f"Missing experiments: {sorted(missing_cases)}"
    )

# Keep only controlled matrix.
df = df[df["name"].isin(expected)].copy()

# Guard against accidental duplicate rows.
duplicates = df[df["name"].duplicated(keep=False)]

if not duplicates.empty:
    raise RuntimeError(
        "Duplicate experiment rows detected:\n"
        + duplicates[["name", "method", "budget"]].to_string(index=False)
    )

df = df.sort_values(
    ["method", "budget"]
).reset_index(drop=True)


# ------------------------------------------------------------
# Extract methods
# ------------------------------------------------------------

grad = (
    df[df["method"] == "gradient"]
    .sort_values("budget")
    .reset_index(drop=True)
)

som = (
    df[df["method"] == "som_qe"]
    .sort_values("budget")
    .reset_index(drop=True)
)

hybrid = (
    df[df["method"] == "hybrid"]
    .sort_values("budget")
    .reset_index(drop=True)
)

for method_name, group in [
    ("gradient", grad),
    ("som_qe", som),
    ("hybrid", hybrid),
]:
    if len(group) != 4:
        raise RuntimeError(
            f"Expected 4 cases for {method_name}, found {len(group)}"
        )


# ------------------------------------------------------------
# Same-budget comparison
#
# Sign convention for Hybrid vs Gradient:
# positive -> Hybrid has lower error (improvement)
# negative -> Hybrid has higher error (worse)
# ------------------------------------------------------------

comparison_rows = []

budgets = sorted(df["budget"].unique())

for budget in budgets:

    g = grad[np.isclose(grad["budget"], budget)].iloc[0]
    s = som[np.isclose(som["budget"], budget)].iloc[0]
    h = hybrid[np.isclose(hybrid["budget"], budget)].iloc[0]

    if not (
        int(g["final_cells"])
        == int(s["final_cells"])
        == int(h["final_cells"])
    ):
        raise RuntimeError(
            f"Cell-count mismatch at budget {budget}"
        )

    grad_vs_som_rms = (
        (s["rms_v_error"] - g["rms_v_error"])
        / s["rms_v_error"]
        * 100.0
    )

    grad_vs_som_mean = (
        (s["mean_v_error"] - g["mean_v_error"])
        / s["mean_v_error"]
        * 100.0
    )

    hybrid_vs_grad_rms = (
        (g["rms_v_error"] - h["rms_v_error"])
        / g["rms_v_error"]
        * 100.0
    )

    hybrid_vs_grad_mean = (
        (g["mean_v_error"] - h["mean_v_error"])
        / g["mean_v_error"]
        * 100.0
    )

    comparison_rows.append(
        {
            "budget_percent": 100.0 * budget,
            "cells": int(g["final_cells"]),

            "grad_rms": g["rms_v_error"],
            "som_rms": s["rms_v_error"],
            "hybrid_rms": h["rms_v_error"],

            "grad_rms_lower_than_som_percent":
                grad_vs_som_rms,

            "hybrid_rms_improvement_vs_grad_percent":
                hybrid_vs_grad_rms,

            "grad_mean": g["mean_v_error"],
            "som_mean": s["mean_v_error"],
            "hybrid_mean": h["mean_v_error"],

            "grad_mean_lower_than_som_percent":
                grad_vs_som_mean,

            "hybrid_mean_improvement_vs_grad_percent":
                hybrid_vs_grad_mean,
        }
    )

comparison = pd.DataFrame(comparison_rows)

comparison_file = (
    RESULTS_DIR
    / "three_method_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False,
)


# ------------------------------------------------------------
# Marginal improvements with increasing budget
# ------------------------------------------------------------

marginal_rows = []

for method_name, group in df.groupby("method"):

    group = group.sort_values("budget").reset_index(drop=True)

    for i in range(1, len(group)):

        previous = group.iloc[i - 1]
        current = group.iloc[i]

        rms_change_percent = (
            (
                previous["rms_v_error"]
                - current["rms_v_error"]
            )
            / previous["rms_v_error"]
            * 100.0
        )

        mean_change_percent = (
            (
                previous["mean_v_error"]
                - current["mean_v_error"]
            )
            / previous["mean_v_error"]
            * 100.0
        )

        marginal_rows.append(
            {
                "method": method_name,
                "from_budget_percent":
                    previous["budget"] * 100.0,
                "to_budget_percent":
                    current["budget"] * 100.0,
                "cells_added":
                    int(
                        current["final_cells"]
                        - previous["final_cells"]
                    ),
                "rms_improvement_percent":
                    rms_change_percent,
                "mean_improvement_percent":
                    mean_change_percent,
            }
        )

marginal = pd.DataFrame(marginal_rows)

marginal_file = (
    RESULTS_DIR
    / "marginal_improvements_3methods.csv"
)

marginal.to_csv(
    marginal_file,
    index=False,
)


# ------------------------------------------------------------
# Monotonicity check
# ------------------------------------------------------------

monotonic_rows = []

for method_name, group in df.groupby("method"):

    group = group.sort_values("budget")

    rms_values = group["rms_v_error"].to_numpy()
    mean_values = group["mean_v_error"].to_numpy()

    monotonic_rows.append(
        {
            "method": method_name,
            "rms_strictly_decreases":
                bool(np.all(np.diff(rms_values) < 0.0)),
            "mean_strictly_decreases":
                bool(np.all(np.diff(mean_values) < 0.0)),
        }
    )

monotonicity = pd.DataFrame(monotonic_rows)

monotonicity_file = (
    RESULTS_DIR
    / "monotonicity_check.csv"
)

monotonicity.to_csv(
    monotonicity_file,
    index=False,
)


# ------------------------------------------------------------
# Plot labels
# ------------------------------------------------------------

labels = {
    "gradient": "Gradient",
    "som_qe": "SOM-QE",
    "hybrid": "Hybrid-v1",
}

method_order = [
    "gradient",
    "som_qe",
    "hybrid",
]


# ------------------------------------------------------------
# Plot 1: RMS vs cells
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

for method_name in method_order:

    group = (
        df[df["method"] == method_name]
        .sort_values("final_cells")
    )

    plt.plot(
        group["final_cells"],
        group["rms_v_error"],
        marker="o",
        label=labels[method_name],
    )

plt.xlabel("Number of cells")
plt.ylabel("Volume-weighted RMS velocity error")
plt.title("Adaptive-mesh accuracy: RMS error vs cells")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

rms_cells_file = (
    PLOTS_DIR
    / "RMS_vs_cells_3methods.png"
)

plt.savefig(
    rms_cells_file,
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# Plot 2: Mean error vs cells
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

for method_name in method_order:

    group = (
        df[df["method"] == method_name]
        .sort_values("final_cells")
    )

    plt.plot(
        group["final_cells"],
        group["mean_v_error"],
        marker="o",
        label=labels[method_name],
    )

plt.xlabel("Number of cells")
plt.ylabel("Volume-weighted mean velocity error")
plt.title("Adaptive-mesh accuracy: mean error vs cells")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

mean_cells_file = (
    PLOTS_DIR
    / "Mean_vs_cells_3methods.png"
)

plt.savefig(
    mean_cells_file,
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# Plot 3: RMS vs solver execution time
#
# Timing is retained as diagnostic information only.
# Single-run execution times should not be used to claim
# statistically significant performance differences.
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

for method_name in method_order:

    group = (
        df[df["method"] == method_name]
        .sort_values("execution_time_s")
    )

    plt.plot(
        group["execution_time_s"],
        group["rms_v_error"],
        marker="o",
        label=labels[method_name],
    )

plt.xlabel("icoFoam execution time [s]")
plt.ylabel("Volume-weighted RMS velocity error")
plt.title("Adaptive-mesh accuracy vs solver execution time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

rms_cpu_file = (
    PLOTS_DIR
    / "RMS_vs_CPU_3methods.png"
)

plt.savefig(
    rms_cpu_file,
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

print()
print("=" * 86)
print("CONTROLLED 3 x 4 EXPERIMENT MATRIX")
print("=" * 86)

print(
    df[
        [
            "name",
            "final_cells",
            "execution_time_s",
            "mean_v_error",
            "rms_v_error",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8g}",
    )
)

print()
print("=" * 86)
print("EQUAL-BUDGET COMPARISON")
print("=" * 86)
print(
    "Hybrid-vs-Gradient sign convention: "
    "positive = Hybrid improves; negative = Hybrid worsens."
)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print()
print("=" * 86)
print("MARGINAL IMPROVEMENTS")
print("=" * 86)

print(
    marginal.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print()
print("=" * 86)
print("MONOTONICITY")
print("=" * 86)

print(
    monotonicity.to_string(index=False)
)

print()
print("Saved:")
print(comparison_file)
print(marginal_file)
print(monotonicity_file)
print(rms_cells_file)
print(mean_cells_file)
print(rms_cpu_file)
