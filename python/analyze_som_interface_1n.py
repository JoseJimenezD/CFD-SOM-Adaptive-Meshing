import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "data" / "M40_som_results.csv"
OUT_DATA = PROJECT / "results" / "M40_som_interface_1n_10.csv"
OUT_FIG = PROJECT / "results" / "plots" / "M40_som_interface_1n_10.png"

BUDGET = 0.10

df = pd.read_csv(INPUT).copy()

# ------------------------------------------------------------
# 1. Reconstruct the structured 40 x 40 M40 grid
# ------------------------------------------------------------
xs = np.sort(df["x_star"].unique())
ys = np.sort(df["y_star"].unique())

nx = len(xs)
ny = len(ys)

assert nx * ny == len(df), "Dataset is not a complete structured grid."

x_to_ix = {x: i for i, x in enumerate(xs)}
y_to_iy = {y: j for j, y in enumerate(ys)}

grid = np.empty((ny, nx), dtype=int)

for idx, row in df.iterrows():
    ix = x_to_ix[row["x_star"]]
    iy = y_to_iy[row["y_star"]]
    grid[iy, ix] = idx

print(f"Grid: {nx} x {ny}")
print(f"Cells: {len(df)}")

# ------------------------------------------------------------
# 2. Direct-neighbour definition: N, S, E, W only
# ------------------------------------------------------------
def neighbours(idx):
    row = df.loc[idx]
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

# ------------------------------------------------------------
# 3. Interface physics
#
# For each cell, only neighbours belonging to a DIFFERENT BMU
# participate in the interface calculation.
#
# velocity_jump = maximum vector velocity jump across a SOM
# interface:
#
# sqrt[(Ux_i-Ux_j)^2 + (Uy_i-Uy_j)^2]
# ------------------------------------------------------------
transition = np.zeros(len(df))
velocity_jump = np.zeros(len(df))
is_interface = np.zeros(len(df), dtype=bool)

for i in range(len(df)):
    neigh = neighbours(i)

    different = [
        j for j in neigh
        if int(df.loc[j, "bmu_id"]) != int(df.loc[i, "bmu_id"])
    ]

    if not different:
        continue

    is_interface[i] = True
    transition[i] = len(different) / len(neigh)

    jumps = []

    for j in different:
        dux = df.loc[i, "Ux_star"] - df.loc[j, "Ux_star"]
        duy = df.loc[i, "Uy_star"] - df.loc[j, "Uy_star"]

        jumps.append(np.sqrt(dux**2 + duy**2))

    velocity_jump[i] = max(jumps)

df["som_interface"] = is_interface.astype(int)
df["som_transition"] = transition
df["velocity_jump"] = velocity_jump

# ------------------------------------------------------------
# 4. Physics-based interface score
#
# SOM determines WHERE regime boundaries exist.
# Velocity jump and local velocity gradient determine HOW
# dynamically important the interface is.
#
# Normalize both physical quantities by their M40 maxima.
# No M160 information is used.
#
# Multiplicative score:
#
# I = T * sqrt(G* * dU*)
#
# This requires all three ingredients:
#   - SOM regime transition
#   - local gradient
#   - velocity-vector change across the interface
# ------------------------------------------------------------
G = df["gradU_mag_star"].to_numpy()
dU = df["velocity_jump"].to_numpy()
T = df["som_transition"].to_numpy()

Gstar = G / G.max()

if dU.max() > 0:
    dUstar = dU / dU.max()
else:
    dUstar = np.zeros_like(dU)

score = T * np.sqrt(Gstar * dUstar)
score[~is_interface] = 0.0

df["interface_score"] = score

# ------------------------------------------------------------
# 5. Rank interface seeds
# ------------------------------------------------------------
seed_order = np.argsort(-score)

# ------------------------------------------------------------
# 6. Seed + exactly ONE neighbour layer
#
# Budget counts UNIQUE cells in final refinement set.
# Neighbours do NOT recursively generate more neighbours.
# ------------------------------------------------------------
target = int(round(BUDGET * len(df)))

selected = set()
seeds = []

for seed in seed_order:

    if score[seed] <= 0:
        break

    candidate = [seed] + neighbours(seed)

    # Only cells not already selected
    new_cells = [c for c in candidate if c not in selected]

    if not new_cells:
        continue

    # Do not exceed equal-budget target.
    remaining = target - len(selected)

    if remaining <= 0:
        break

    # Add seed first, then its immediate neighbours.
    for c in new_cells[:remaining]:
        selected.add(c)

    seeds.append(seed)

    if len(selected) >= target:
        break

selected = np.array(sorted(selected), dtype=int)
seeds = np.array(seeds, dtype=int)

df["selected_interface_1n"] = 0
df.loc[selected, "selected_interface_1n"] = 1

df["interface_seed"] = 0
df.loc[seeds, "interface_seed"] = 1

# ------------------------------------------------------------
# 7. Diagnostics
# ------------------------------------------------------------
print()
print("=== SOM-Interface-1N ===")
print(f"Budget: {BUDGET*100:.0f}%")
print(f"Target refinement cells: {target}")
print(f"Selected unique cells: {len(selected)}")
print(f"Number of seeds: {len(seeds)}")
print(f"Interface cells in M40: {is_interface.sum()}")
print(f"Mean transition of seeds: {T[seeds].mean():.6f}")
print(f"Mean gradient of seeds: {G[seeds].mean():.6f}")
print(f"Mean velocity jump of seeds: {dU[seeds].mean():.6f}")

# ------------------------------------------------------------
# 8. Save complete diagnostic table
# ------------------------------------------------------------
OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_DATA, index=False)

# ------------------------------------------------------------
# 9. Plot complete cell map
# ------------------------------------------------------------
category = np.zeros((ny, nx), dtype=int)

seed_set = set(seeds.tolist())

for iy in range(ny):
    for ix in range(nx):
        idx = grid[iy, ix]

        if idx in selected:
            category[iy, ix] = 1

        if idx in seed_set:
            category[iy, ix] = 2

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

cmap = ListedColormap([
    "#eeeeee",
    "#80b1d3",
    "#d95f02",
])

fig, ax = plt.subplots(figsize=(7.2, 6.2))

ax.imshow(
    category,
    origin="lower",
    extent=[0, 1, 0, 1],
    interpolation="nearest",
    cmap=cmap,
    vmin=0,
    vmax=2,
    aspect="equal",
)

ax.set_xlabel(r"$x/L$")
ax.set_ylabel(r"$y/L$")
ax.set_title(
    "SOM-Interface-1N — 10% total refinement budget"
)

legend = [
    Patch(facecolor="#eeeeee", label="Not refined"),
    Patch(facecolor="#80b1d3", label="1-neighbour refinement"),
    Patch(facecolor="#d95f02", label="Interface seed"),
]

ax.legend(
    handles=legend,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True,
)

fig.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")

print()
print(f"Saved data: {OUT_DATA}")
print(f"Saved figure: {OUT_FIG}")
