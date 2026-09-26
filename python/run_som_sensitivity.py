from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT / "data" / "M40_som_dataset.csv"

OUTPUT_DIR = PROJECT / "results" / "som_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHITECTURES = [3, 4, 5, 6]
SEEDS = [1, 11, 21, 31, 42]

N_ITERATIONS = 20000

ALPHA_INITIAL = 0.5
ALPHA_FINAL = 0.05

SIGMA_INITIAL = 2.0
SIGMA_FINAL = 0.5

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

X_physical = df[FEATURES].values

scaler = StandardScaler()
X = scaler.fit_transform(X_physical)

n_cells = len(X)
n_features = X.shape[1]

# ============================================================
# STRUCTURED GRID NEIGHBOURS
# ============================================================

xvals = np.sort(df["x_star"].unique())
yvals = np.sort(df["y_star"].unique())

nx = len(xvals)
ny = len(yvals)

if nx * ny != n_cells:
    raise RuntimeError(
        f"Expected structured grid, got nx*ny={nx*ny}, cells={n_cells}"
    )

x_to_ix = {x: i for i, x in enumerate(xvals)}
y_to_iy = {y: i for i, y in enumerate(yvals)}

grid = np.empty((ny, nx), dtype=int)

for idx, row in df.iterrows():
    ix = x_to_ix[row["x_star"]]
    iy = y_to_iy[row["y_star"]]
    grid[iy, ix] = idx


def neighbours(i):
    row = df.iloc[i]

    ix = x_to_ix[row["x_star"]]
    iy = y_to_iy[row["y_star"]]

    out = []

    if ix > 0:
        out.append(grid[iy, ix - 1])

    if ix < nx - 1:
        out.append(grid[iy, ix + 1])

    if iy > 0:
        out.append(grid[iy - 1, ix])

    if iy < ny - 1:
        out.append(grid[iy + 1, ix])

    return out


NEIGHBOURS = [neighbours(i) for i in range(n_cells)]

# ============================================================
# TRAINING FUNCTION
# Exact algorithm used by original train_som.py
# ============================================================

def train_som(size, seed):

    rng = np.random.default_rng(seed)

    weights = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(size, size, n_features)
    )

    rr, cc = np.meshgrid(
        np.arange(size),
        np.arange(size),
        indexing="ij"
    )

    for iteration in range(N_ITERATIONS):

        sample_index = rng.integers(n_cells)
        sample = X[sample_index]

        distances = np.linalg.norm(
            weights - sample,
            axis=2
        )

        bmu = np.unravel_index(
            np.argmin(distances),
            distances.shape
        )

        fraction = iteration / max(N_ITERATIONS - 1, 1)

        alpha = ALPHA_INITIAL * (
            ALPHA_FINAL / ALPHA_INITIAL
        ) ** fraction

        sigma = SIGMA_INITIAL * (
            SIGMA_FINAL / SIGMA_INITIAL
        ) ** fraction

        lattice_distance_sq = (
            (rr - bmu[0]) ** 2
            + (cc - bmu[1]) ** 2
        )

        neighborhood = np.exp(
            -lattice_distance_sq /
            (2.0 * sigma ** 2)
        )

        weights += (
            alpha
            * neighborhood[:, :, None]
            * (sample - weights)
        )

    # --------------------------------------------------------
    # Final BMU assignment
    # --------------------------------------------------------

    bmu_rows = np.zeros(n_cells, dtype=int)
    bmu_cols = np.zeros(n_cells, dtype=int)
    bmu_id = np.zeros(n_cells, dtype=int)
    qe = np.zeros(n_cells)

    for i, sample in enumerate(X):

        distances = np.linalg.norm(
            weights - sample,
            axis=2
        )

        bmu = np.unravel_index(
            np.argmin(distances),
            distances.shape
        )

        bmu_rows[i] = bmu[0]
        bmu_cols[i] = bmu[1]

        bmu_id[i] = (
            bmu[0] * size
            + bmu[1]
        )

        qe[i] = distances[bmu]

    return weights, bmu_rows, bmu_cols, bmu_id, qe


# ============================================================
# INTERFACE MASK
# ============================================================

def interface_mask(bmu_id):

    mask = np.zeros(n_cells, dtype=bool)

    for i in range(n_cells):

        bi = bmu_id[i]

        for j in NEIGHBOURS[i]:

            if bmu_id[j] != bi:
                mask[i] = True
                break

    return mask


# ============================================================
# RUN EXPERIMENT MATRIX
# ============================================================

summary = []

print()
print("============================================================")
print(" SOM ARCHITECTURE / RANDOM-SEED SENSITIVITY STUDY")
print("============================================================")
print(f"Cells          : {n_cells}")
print(f"Iterations/run : {N_ITERATIONS}")
print(f"Architectures  : {ARCHITECTURES}")
print(f"Seeds          : {SEEDS}")
print(f"Total runs     : {len(ARCHITECTURES) * len(SEEDS)}")
print("============================================================")
print()

for size in ARCHITECTURES:

    for seed in SEEDS:

        label = f"SOM_{size}x{size}_seed{seed}"

        print(f"Running {label} ...", flush=True)

        t0 = time.perf_counter()

        weights, br, bc, bid, qe = train_som(
            size,
            seed
        )

        elapsed = time.perf_counter() - t0

        interface = interface_mask(bid)

        occupied = np.unique(bid).size

        result = df.copy()

        result["bmu_row"] = br
        result["bmu_col"] = bc
        result["bmu_id"] = bid
        result["quantization_error"] = qe
        result["interface_cell"] = interface.astype(int)

        result_file = OUTPUT_DIR / f"{label}.csv"
        weights_file = OUTPUT_DIR / f"{label}_weights.npy"

        result.to_csv(
            result_file,
            index=False
        )

        np.save(
            weights_file,
            weights
        )

        summary.append(
            {
                "architecture": f"{size}x{size}",
                "size": size,
                "seed": seed,
                "neurons": size * size,
                "occupied_neurons": occupied,
                "mean_qe": qe.mean(),
                "max_qe": qe.max(),
                "interface_cells": interface.sum(),
                "interface_fraction": interface.mean(),
                "training_seconds": elapsed,
            }
        )

        print(
            f"  occupied={occupied:2d}/{size*size:2d} | "
            f"mean QE={qe.mean():.6f} | "
            f"interfaces={interface.sum():4d} "
            f"({100*interface.mean():5.2f}%) | "
            f"time={elapsed:.2f}s"
        )

# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)

summary_file = OUTPUT_DIR / "som_sensitivity_summary.csv"

summary_df.to_csv(
    summary_file,
    index=False
)

print()
print("============================================================")
print(" ALL SOM SENSITIVITY RUNS COMPLETED")
print("============================================================")
print()
print(summary_df.to_string(index=False))
print()
print("Summary saved:")
print(summary_file)
print()
print("Individual results saved in:")
print(OUTPUT_DIR)
print()
