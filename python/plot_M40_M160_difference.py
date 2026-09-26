from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    / "plots"
    / "M40_M160_velocity_difference.png"
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Cells loaded:", len(df))

x = df["x_star"].values
y = df["y_star"].values
error = df["dU_mag_star"].values

# ============================================================
# RECONSTRUCT REGULAR M40 GRID
# ============================================================

x_unique = np.sort(np.unique(x))
y_unique = np.sort(np.unique(y))

Nx = len(x_unique)
Ny = len(y_unique)

print("Grid:", Nx, "x", Ny)

if Nx * Ny != len(df):
    raise RuntimeError(
        "Data do not form a regular Cartesian grid."
    )

error_grid = np.full(
    (Ny, Nx),
    np.nan
)

x_lookup = {
    value: i
    for i, value in enumerate(x_unique)
}

y_lookup = {
    value: i
    for i, value in enumerate(y_unique)
}

for xi, yi, ei in zip(x, y, error):

    ix = x_lookup[xi]
    iy = y_lookup[yi]

    error_grid[iy, ix] = ei

if np.isnan(error_grid).any():
    raise RuntimeError(
        "Missing cells while reconstructing M40 grid."
    )

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 6.2)
)

image = ax.imshow(
    error_grid,
    origin="lower",
    extent=[
        x_unique.min(),
        x_unique.max(),
        y_unique.min(),
        y_unique.max()
    ],
    aspect="equal",
    interpolation="nearest"
)

cbar = fig.colorbar(
    image,
    ax=ax
)

cbar.set_label(
    r"$|\mathbf{U}_{40}-\mathbf{U}_{160\rightarrow40}|/U_{lid}$"
)

ax.set_xlabel(r"$x/L$")
ax.set_ylabel(r"$y/L$")

ax.set_title(
    "M40-M160 velocity difference"
)

# Mark location of maximum difference
imax = np.argmax(error)

xmax = x[imax]
ymax = y[imax]
emax = error[imax]

ax.plot(
    xmax,
    ymax,
    marker="x",
    markersize=9,
    markeredgewidth=2
)

ax.text(
    0.50,
    -0.12,
    f"Max = {emax:.4f} at x/L={xmax:.4f}, y/L={ymax:.4f}",
    transform=ax.transAxes,
    ha="center"
)

fig.tight_layout()

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", OUTPUT_FILE)

plt.show()
