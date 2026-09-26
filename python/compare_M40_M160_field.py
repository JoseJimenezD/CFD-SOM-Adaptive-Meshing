from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# ============================================================
# PATHS
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

CASE_M40 = PROJECT / "cases" / "Cavity_SOM_002"
CASE_M160 = PROJECT / "cases" / "Cavity_SOM_004"

TIME = "5"

SOM_RESULTS = PROJECT / "data" / "M40_som_results.csv"

OUTPUT_FILE = (
    PROJECT
    / "data"
    / "M40_M160_field_comparison.csv"
)

# Physical cavity dimensions
L = 0.1
U_LID = 1.0


# ============================================================
# OPENFOAM FIELD READERS
# ============================================================

def read_vector_field(filename):
    """
    Read the internalField of an OpenFOAM volVectorField.

    Returns
    -------
    array : shape (N, 3)
    """

    text = Path(filename).read_text()

    pattern = (
        r"internalField\s+nonuniform\s+List<vector>\s*"
        r"(\d+)\s*\(\s*(.*?)\s*\)\s*;"
    )

    match = re.search(
        pattern,
        text,
        re.DOTALL
    )

    if match is None:
        raise RuntimeError(
            f"Could not read vector internalField from:\n{filename}"
        )

    expected_n = int(match.group(1))
    block = match.group(2)

    vectors = re.findall(
        r"\(\s*"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)"
        r"\s*\)",
        block
    )

    array = np.asarray(
        vectors,
        dtype=float
    )

    if len(array) != expected_n:
        raise RuntimeError(
            f"{filename}: expected {expected_n} vectors, "
            f"read {len(array)}"
        )

    return array


# ============================================================
# LOAD OPENFOAM DATA
# ============================================================

print("========================================")
print(" M40 vs M160 FIELD COMPARISON")
print("========================================")

print("\nReading M40...")

C40 = read_vector_field(
    CASE_M40 / TIME / "C"
)

U40 = read_vector_field(
    CASE_M40 / TIME / "U"
)

print("M40 cells :", len(C40))

print("\nReading M160...")

C160 = read_vector_field(
    CASE_M160 / TIME / "C"
)

U160 = read_vector_field(
    CASE_M160 / TIME / "U"
)

print("M160 cells:", len(C160))


# ============================================================
# BASIC CONSISTENCY CHECKS
# ============================================================

if len(C40) != len(U40):
    raise RuntimeError(
        "M40 C and U have different numbers of cells."
    )

if len(C160) != len(U160):
    raise RuntimeError(
        "M160 C and U have different numbers of cells."
    )


# ============================================================
# BUILD REGULAR M160 GRID
# ============================================================

x160 = np.unique(
    np.round(C160[:, 0], 12)
)

y160 = np.unique(
    np.round(C160[:, 1], 12)
)

print("\nM160 grid:")
print("Nx =", len(x160))
print("Ny =", len(y160))

if len(x160) * len(y160) != len(C160):
    raise RuntimeError(
        "M160 does not appear to be a regular 2D Cartesian grid."
    )

# Create arrays indexed as [y, x]
Ux160_grid = np.full(
    (len(y160), len(x160)),
    np.nan
)

Uy160_grid = np.full(
    (len(y160), len(x160)),
    np.nan
)

x_lookup = {
    value: i
    for i, value in enumerate(x160)
}

y_lookup = {
    value: i
    for i, value in enumerate(y160)
}

for cell, velocity in zip(C160, U160):

    x = round(cell[0], 12)
    y = round(cell[1], 12)

    ix = x_lookup[x]
    iy = y_lookup[y]

    Ux160_grid[iy, ix] = velocity[0]
    Uy160_grid[iy, ix] = velocity[1]


if (
    np.isnan(Ux160_grid).any()
    or np.isnan(Uy160_grid).any()
):
    raise RuntimeError(
        "Missing values while constructing M160 grid."
    )


# ============================================================
# INTERPOLATE M160 ONTO M40 CELL CENTRES
# ============================================================

interp_Ux = RegularGridInterpolator(
    (y160, x160),
    Ux160_grid,
    method="linear",
    bounds_error=False,
    fill_value=None
)

interp_Uy = RegularGridInterpolator(
    (y160, x160),
    Uy160_grid,
    method="linear",
    bounds_error=False,
    fill_value=None
)

# Interpolator expects points as (y, x)
points40 = np.column_stack(
    (
        C40[:, 1],
        C40[:, 0]
    )
)

Ux160_at40 = interp_Ux(points40)
Uy160_at40 = interp_Uy(points40)


# ============================================================
# COMPUTE M40-M160 DIFFERENCES
# ============================================================

dUx = U40[:, 0] - Ux160_at40
dUy = U40[:, 1] - Uy160_at40

dU_mag = np.sqrt(
    dUx**2 + dUy**2
)

# Normalize using lid velocity
dUx_star = dUx / U_LID
dUy_star = dUy / U_LID
dU_mag_star = dU_mag / U_LID


# ============================================================
# LOAD SOM CLASSIFICATION
# ============================================================

som = pd.read_csv(SOM_RESULTS)

if len(som) != len(C40):
    raise RuntimeError(
        "SOM result length does not match M40."
    )

# Check that cell ordering is consistent
x40_star = C40[:, 0] / L
y40_star = C40[:, 1] / L

if not np.allclose(
    x40_star,
    som["x_star"].values
):
    raise RuntimeError(
        "M40 x coordinates do not match SOM dataset ordering."
    )

if not np.allclose(
    y40_star,
    som["y_star"].values
):
    raise RuntimeError(
        "M40 y coordinates do not match SOM dataset ordering."
    )


# ============================================================
# CREATE OUTPUT TABLE
# ============================================================

comparison = pd.DataFrame(
    {
        "x_star": x40_star,
        "y_star": y40_star,

        "Ux_M40": U40[:, 0],
        "Uy_M40": U40[:, 1],

        "Ux_M160_interp": Ux160_at40,
        "Uy_M160_interp": Uy160_at40,

        "dUx_star": dUx_star,
        "dUy_star": dUy_star,
        "dU_mag_star": dU_mag_star,

        "bmu_id": som["bmu_id"].values,

        "quantization_error":
            som["quantization_error"].values,
    }
)

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

print("\n========================================")
print(" GLOBAL DIFFERENCE STATISTICS")
print("========================================")

print(
    "Mean |dU|* :",
    f"{dU_mag_star.mean():.8e}"
)

print(
    "RMS  |dU|* :",
    f"{np.sqrt(np.mean(dU_mag_star**2)):.8e}"
)

print(
    "Max  |dU|* :",
    f"{dU_mag_star.max():.8e}"
)

imax = np.argmax(dU_mag_star)

print("\nMaximum difference location:")
print(
    f"x/L = {x40_star[imax]:.5f}"
)

print(
    f"y/L = {y40_star[imax]:.5f}"
)

print(
    f"BMU = {som['bmu_id'].iloc[imax]}"
)


# ============================================================
# DIFFERENCE BY SOM NEURON
# ============================================================

summary = (
    comparison
    .groupby("bmu_id")["dU_mag_star"]
    .agg(
        cells="count",
        mean="mean",
        rms=lambda x: np.sqrt(
            np.mean(x**2)
        ),
        maximum="max"
    )
)

print("\n========================================")
print(" DIFFERENCE BY SOM NEURON")
print("========================================\n")

print(
    summary.to_string(
        float_format=lambda x: f"{x:.8e}"
    )
)


# ============================================================
# TOP DIFFERENCE CELLS
# ============================================================

print("\n========================================")
print(" TOP 20 M40-M160 DIFFERENCE CELLS")
print("========================================\n")

top20 = (
    comparison
    .sort_values(
        "dU_mag_star",
        ascending=False
    )
    .head(20)
)

print(
    top20[
        [
            "x_star",
            "y_star",
            "dUx_star",
            "dUy_star",
            "dU_mag_star",
            "bmu_id",
            "quantization_error",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6e}"
    )
)

print("\nComparison saved:")
print(OUTPUT_FILE)

print("\nAnalysis completed.")
