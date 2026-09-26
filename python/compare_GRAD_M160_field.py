from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

PROJECT = Path(__file__).resolve().parents[1]

CASE_SOM = PROJECT / "cases" / "Cavity_SOM_006"
CASE_REF = PROJECT / "cases" / "Cavity_SOM_004"

TIME = "5"

OUTFILE = PROJECT / "data" / "GRAD2080_M160_field_comparison.csv"


def read_vectors(path):
    text = path.read_text()

    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+\d+\s*\((.*?)\)\s*;",
        text,
        re.S,
    )

    if match is None:
        raise RuntimeError(f"Could not read vector field: {path}")

    block = match.group(1)

    values = re.findall(
        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
        block,
    )

    return np.asarray(values, dtype=float)


def read_scalars(path):
    text = path.read_text()

    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)\s*;",
        text,
        re.S,
    )

    if match is None:
        raise RuntimeError(f"Could not read scalar field: {path}")

    block = match.group(1)

    values = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", block)

    return np.asarray(values, dtype=float)


# ------------------------------------------------------------
# Read SOM mesh
# ------------------------------------------------------------

C_som = read_vectors(CASE_SOM / TIME / "C")
U_som = read_vectors(CASE_SOM / TIME / "U")
V_som = read_scalars(CASE_SOM / TIME / "V")

if not (len(C_som) == len(U_som) == len(V_som)):
    raise RuntimeError("SOM C/U/V sizes do not match.")

# ------------------------------------------------------------
# Read M160 reference
# ------------------------------------------------------------

C_ref = read_vectors(CASE_REF / TIME / "C")
U_ref = read_vectors(CASE_REF / TIME / "U")

if len(C_ref) != len(U_ref):
    raise RuntimeError("M160 C/U sizes do not match.")

# ------------------------------------------------------------
# Reconstruct structured M160 grid
# ------------------------------------------------------------

x_ref = np.unique(C_ref[:, 0])
y_ref = np.unique(C_ref[:, 1])

nx = len(x_ref)
ny = len(y_ref)

if nx * ny != len(C_ref):
    raise RuntimeError(
        f"M160 is not recognized as structured: nx={nx}, ny={ny}, cells={len(C_ref)}"
    )

Ux_grid = np.empty((nx, ny))
Uy_grid = np.empty((nx, ny))

x_index = {round(x, 12): i for i, x in enumerate(x_ref)}
y_index = {round(y, 12): j for j, y in enumerate(y_ref)}

for c, u in zip(C_ref, U_ref):
    i = x_index[round(c[0], 12)]
    j = y_index[round(c[1], 12)]

    Ux_grid[i, j] = u[0]
    Uy_grid[i, j] = u[1]

interp_Ux = RegularGridInterpolator(
    (x_ref, y_ref),
    Ux_grid,
    bounds_error=False,
    fill_value=None,
)

interp_Uy = RegularGridInterpolator(
    (x_ref, y_ref),
    Uy_grid,
    bounds_error=False,
    fill_value=None,
)

points = C_som[:, :2]

Ux_ref_at_som = interp_Ux(points)
Uy_ref_at_som = interp_Uy(points)

# ------------------------------------------------------------
# Differences
# ------------------------------------------------------------

dUx = U_som[:, 0] - Ux_ref_at_som
dUy = U_som[:, 1] - Uy_ref_at_som

dU = np.sqrt(dUx**2 + dUy**2)

# U_lid = 1 m/s, so numerically dU_star = dU
dU_star = dU

# ------------------------------------------------------------
# Volume-weighted statistics
# ------------------------------------------------------------

Vtotal = np.sum(V_som)

mean_V = np.sum(V_som * dU_star) / Vtotal

rms_V = np.sqrt(
    np.sum(V_som * dU_star**2) / Vtotal
)

max_error = np.max(dU_star)
imax = np.argmax(dU_star)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df = pd.DataFrame(
    {
        "x": C_som[:, 0],
        "y": C_som[:, 1],
        "z": C_som[:, 2],
        "V": V_som,
        "Ux_SOM": U_som[:, 0],
        "Uy_SOM": U_som[:, 1],
        "Ux_M160_interp": Ux_ref_at_som,
        "Uy_M160_interp": Uy_ref_at_som,
        "dUx": dUx,
        "dUy": dUy,
        "dU_mag_star": dU_star,
    }
)

OUTFILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTFILE, index=False)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print()
print("========================================")
print(" GRAD2080 vs M160 FIELD COMPARISON")
print("========================================")
print()

print(f"SOM cells          : {len(C_som)}")
print(f"M160 cells         : {len(C_ref)}")
print(f"SOM total volume   : {Vtotal:.12e}")
print()

print("VOLUME-WEIGHTED DIFFERENCE STATISTICS")
print("----------------------------------------")
print(f"Mean_V |dU|*       : {mean_V:.8e}")
print(f"RMS_V  |dU|*       : {rms_V:.8e}")
print(f"Max    |dU|*       : {max_error:.8e}")
print()

print("Maximum difference location:")
print(f"x/L = {C_som[imax,0] / 0.1:.6f}")
print(f"y/L = {C_som[imax,1] / 0.1:.6f}")
print(f"cell volume = {V_som[imax]:.8e}")
print()

print("Cell-volume range:")
print(f"min V = {V_som.min():.8e}")
print(f"max V = {V_som.max():.8e}")
print()

print("Saved:")
print(OUTFILE)
