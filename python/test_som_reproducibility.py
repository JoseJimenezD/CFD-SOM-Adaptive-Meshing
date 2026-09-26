from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT / "data" / "M40_som_dataset.csv"
ORIGINAL_FILE = PROJECT / "data" / "M40_som_results.csv"
ORIGINAL_WEIGHTS_FILE = PROJECT / "data" / "M40_som_weights.npy"

SOM_ROWS = 4
SOM_COLS = 4
N_ITERATIONS = 20000

ALPHA_INITIAL = 0.5
ALPHA_FINAL = 0.05

SIGMA_INITIAL = 2.0
SIGMA_FINAL = 0.5

RANDOM_SEED = 42

FEATURES = [
    "Ux_star",
    "Uy_star",
    "U_mag_star",
    "gradU_mag_star",
]

# ------------------------------------------------------------
# Load exactly the same data
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)
X_physical = df[FEATURES].values

scaler = StandardScaler()
X = scaler.fit_transform(X_physical)

# ------------------------------------------------------------
# Initialize exactly as original
# ------------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

n_features = X.shape[1]

weights = rng.normal(
    loc=0.0,
    scale=1.0,
    size=(SOM_ROWS, SOM_COLS, n_features)
)

rr, cc = np.meshgrid(
    np.arange(SOM_ROWS),
    np.arange(SOM_COLS),
    indexing="ij"
)

# ------------------------------------------------------------
# Train exactly as original
# ------------------------------------------------------------

for iteration in range(N_ITERATIONS):

    sample_index = rng.integers(len(X))
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
        -lattice_distance_sq / (2.0 * sigma ** 2)
    )

    weights += (
        alpha
        * neighborhood[:, :, None]
        * (sample - weights)
    )

# ------------------------------------------------------------
# Assign BMUs and QE
# ------------------------------------------------------------

bmu_id = np.zeros(len(X), dtype=int)
qe = np.zeros(len(X))

for i, sample in enumerate(X):

    distances = np.linalg.norm(
        weights - sample,
        axis=2
    )

    bmu = np.unravel_index(
        np.argmin(distances),
        distances.shape
    )

    bmu_id[i] = bmu[0] * SOM_COLS + bmu[1]
    qe[i] = distances[bmu]

# ------------------------------------------------------------
# Compare against original saved results
# ------------------------------------------------------------

original = pd.read_csv(ORIGINAL_FILE)
original_weights = np.load(ORIGINAL_WEIGHTS_FILE)

same_bmu = np.array_equal(
    bmu_id,
    original["bmu_id"].to_numpy(dtype=int)
)

bmu_match_fraction = np.mean(
    bmu_id == original["bmu_id"].to_numpy(dtype=int)
)

max_qe_difference = np.max(
    np.abs(
        qe -
        original["quantization_error"].to_numpy()
    )
)

max_weight_difference = np.max(
    np.abs(weights - original_weights)
)

print()
print("============================================")
print(" SOM REPRODUCIBILITY TEST")
print("============================================")

print(f"Cells                  : {len(X)}")
print(f"Architecture           : {SOM_ROWS} x {SOM_COLS}")
print(f"Seed                   : {RANDOM_SEED}")
print(f"Iterations             : {N_ITERATIONS}")

print()
print(f"Mean QE                : {qe.mean():.9f}")
print(f"Maximum QE             : {qe.max():.9f}")

print()
print(f"Exact BMU equality     : {same_bmu}")
print(f"BMU match fraction     : {bmu_match_fraction:.9f}")
print(f"Max QE difference      : {max_qe_difference:.12e}")
print(f"Max weight difference  : {max_weight_difference:.12e}")

print()

if (
    same_bmu
    and max_qe_difference < 1e-10
    and max_weight_difference < 1e-10
):
    print("PASS: original SOM reproduced exactly.")
else:
    print("FAIL: reproduction differs from original SOM.")
    print("Do NOT start sensitivity study yet.")

print("============================================")
