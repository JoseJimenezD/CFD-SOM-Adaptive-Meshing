from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# PATHS
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT / "data" / "M40_som_dataset.csv"
RESULTS_FILE = PROJECT / "data" / "M40_som_results.csv"
WEIGHTS_FILE = PROJECT / "data" / "M40_som_weights.npy"

# These MUST match the features used during SOM training
FEATURES = [
    "Ux_star",
    "Uy_star",
    "U_mag_star",
    "gradU_mag_star",
]

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)
results = pd.read_csv(RESULTS_FILE)
weights = np.load(WEIGHTS_FILE)

print("========================================")
print(" SOM NEURON PHYSICAL ANALYSIS")
print("========================================")

print(f"\nCells   : {len(df)}")
print(f"Weights : {weights.shape}")

# ============================================================
# RECONSTRUCT THE ORIGINAL STANDARDIZATION
# ============================================================

X_physical = df[FEATURES].values

scaler = StandardScaler()
scaler.fit(X_physical)

print("\nScaler reconstruction:")

for i, feature in enumerate(FEATURES):
    print(
        f"{feature:18s} "
        f"mean = {scaler.mean_[i]: .6f}   "
        f"scale = {scaler.scale_[i]: .6f}"
    )

# ============================================================
# TRANSFORM SOM WEIGHTS BACK TO PHYSICAL FEATURE SPACE
# ============================================================

som_rows, som_cols, n_features = weights.shape

weights_flat = weights.reshape(
    som_rows * som_cols,
    n_features
)

weights_physical = scaler.inverse_transform(weights_flat)

# ============================================================
# CREATE NEURON TABLE
# ============================================================

rows = []

for neuron_id in range(som_rows * som_cols):

    row = neuron_id // som_cols
    col = neuron_id % som_cols

    mask = results["bmu_id"] == neuron_id
    n_cells = int(mask.sum())

    values = weights_physical[neuron_id]

    rows.append(
        {
            "neuron": neuron_id,
            "row": row,
            "col": col,
            "cells": n_cells,
            "Ux_star": values[0],
            "Uy_star": values[1],
            "U_mag_star": values[2],
            "gradU_mag_star": values[3],
        }
    )

neuron_df = pd.DataFrame(rows)

print("\n========================================")
print(" PHYSICAL SOM NEURON PROTOTYPES")
print("========================================\n")

print(
    neuron_df.to_string(
        index=False,
        float_format=lambda x: f"{x: .5f}"
    )
)

# ============================================================
# HIGHEST QUANTIZATION-ERROR CELLS
# ============================================================

print("\n========================================")
print(" TOP 20 QUANTIZATION-ERROR CELLS")
print("========================================\n")

top_error = (
    results[
        [
            "x_star",
            "y_star",
            "Ux_star",
            "Uy_star",
            "U_mag_star",
            "gradU_mag_star",
            "bmu_id",
            "quantization_error",
        ]
    ]
    .sort_values(
        "quantization_error",
        ascending=False
    )
    .head(20)
)

print(
    top_error.to_string(
        index=False,
        float_format=lambda x: f"{x: .5f}"
    )
)

# ============================================================
# SAVE NEURON TABLE
# ============================================================

OUTPUT_FILE = (
    PROJECT
    / "data"
    / "M40_som_neuron_analysis.csv"
)

neuron_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nNeuron analysis saved:")
print(OUTPUT_FILE)

print("\nAnalysis completed.")
