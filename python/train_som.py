from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT / "data" / "M40_som_dataset.csv"
OUTPUT_DIR = PROJECT / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

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

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

X_physical = df[FEATURES].values

print("========================================")
print(" KOHONEN SOM - M40 CAVITY")
print("========================================")

print(f"Cells       : {len(df)}")
print(f"Features    : {len(FEATURES)}")
print(f"SOM         : {SOM_ROWS} x {SOM_COLS}")
print(f"Neurons     : {SOM_ROWS * SOM_COLS}")
print(f"Iterations  : {N_ITERATIONS}")

print("\nFeatures:")
for feature in FEATURES:
    print(" -", feature)

# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()
X = scaler.fit_transform(X_physical)

print("\nStandardized feature means:")
print(np.round(X.mean(axis=0), 6))

print("Standardized feature std:")
print(np.round(X.std(axis=0), 6))

# ============================================================
# INITIALIZE SOM
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

n_features = X.shape[1]

weights = rng.normal(
    loc=0.0,
    scale=1.0,
    size=(SOM_ROWS, SOM_COLS, n_features)
)

# Coordinates of neurons on the SOM lattice
rr, cc = np.meshgrid(
    np.arange(SOM_ROWS),
    np.arange(SOM_COLS),
    indexing="ij"
)

# ============================================================
# TRAINING
# ============================================================

print("\nTraining SOM...")

for iteration in range(N_ITERATIONS):

    # Random training sample
    sample_index = rng.integers(len(X))
    sample = X[sample_index]

    # --------------------------------------------------------
    # FIND BMU
    # --------------------------------------------------------

    distances = np.linalg.norm(
        weights - sample,
        axis=2
    )

    bmu = np.unravel_index(
        np.argmin(distances),
        distances.shape
    )

    # --------------------------------------------------------
    # DECAY LEARNING PARAMETERS
    # --------------------------------------------------------

    fraction = iteration / max(N_ITERATIONS - 1, 1)

    alpha = ALPHA_INITIAL * (
        ALPHA_FINAL / ALPHA_INITIAL
    ) ** fraction

    sigma = SIGMA_INITIAL * (
        SIGMA_FINAL / SIGMA_INITIAL
    ) ** fraction

    # --------------------------------------------------------
    # NEIGHBORHOOD FUNCTION
    # --------------------------------------------------------

    lattice_distance_sq = (
        (rr - bmu[0]) ** 2
        + (cc - bmu[1]) ** 2
    )

    neighborhood = np.exp(
        -lattice_distance_sq / (2.0 * sigma ** 2)
    )

    # --------------------------------------------------------
    # UPDATE WEIGHTS
    # --------------------------------------------------------

    weights += (
        alpha
        * neighborhood[:, :, None]
        * (sample - weights)
    )

    if (iteration + 1) % 2000 == 0:
        print(
            f"Iteration "
            f"{iteration + 1:6d}/{N_ITERATIONS}"
        )

# ============================================================
# ASSIGN EACH CELL TO ITS BMU
# ============================================================

bmu_rows = np.zeros(len(X), dtype=int)
bmu_cols = np.zeros(len(X), dtype=int)
bmu_id = np.zeros(len(X), dtype=int)
quantization_error = np.zeros(len(X))

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
        bmu[0] * SOM_COLS
        + bmu[1]
    )

    quantization_error[i] = distances[bmu]

# ============================================================
# SAVE RESULTS
# ============================================================

results = df.copy()

results["bmu_row"] = bmu_rows
results["bmu_col"] = bmu_cols
results["bmu_id"] = bmu_id
results["quantization_error"] = quantization_error

OUTPUT_FILE = OUTPUT_DIR / "M40_som_results.csv"

results.to_csv(
    OUTPUT_FILE,
    index=False
)

WEIGHTS_FILE = OUTPUT_DIR / "M40_som_weights.npy"

np.save(
    WEIGHTS_FILE,
    weights
)

# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print(" SOM TRAINING COMPLETED")
print("========================================")

print(
    "\nMean quantization error:",
    f"{quantization_error.mean():.6f}"
)

print(
    "Maximum quantization error:",
    f"{quantization_error.max():.6f}"
)

print("\nCells assigned to each neuron:")

unique, counts = np.unique(
    bmu_id,
    return_counts=True
)

for neuron, count in zip(unique, counts):

    row = neuron // SOM_COLS
    col = neuron % SOM_COLS

    print(
        f"Neuron {neuron:2d} "
        f"({row},{col}) : "
        f"{count:4d} cells"
    )

print("\nResults saved:")
print(OUTPUT_FILE)

print("\nWeights saved:")
print(WEIGHTS_FILE)
