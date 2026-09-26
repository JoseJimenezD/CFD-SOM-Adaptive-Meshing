import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path
from collections import deque

PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "data" / "M40_som_results.csv"

OUT_CSV = PROJECT / "results" / "M40_som_interface_fixed_seeds.csv"
OUT_FIG = PROJECT / "results" / "plots" / "SOM_Interface_fixed_seeds_1N_vs_2N.png"

N_SEEDS = 40

df = pd.read_csv(INPUT).copy()

# ============================================================
# 1. Reconstruct 40 x 40 structured grid
# ============================================================

xs = np.sort(df["x_star"].unique())
ys = np.sort(df["y_star"].unique())

nx = len(xs)
ny = len(ys)

assert nx == 40 and ny == 40
assert nx * ny == len(df)

x_to_ix = {x: i for i, x in enumerate(xs)}
y_to_iy = {y: j for j, y in enumerate(ys)}

grid = np.empty((ny, nx), dtype=int)

for idx, row in df.iterrows():
    ix = x_to_ix[row["x_star"]]
    iy = y_to_iy[row["y_star"]]
    grid[iy, ix] = idx


def direct_neighbours(idx):

    row = df.loc[idx]

    ix = x_to_ix[row["x_star"]]
    iy = y_to_iy[row["y_star"]]

    out = []

    if ix > 0:
        out.append(grid[iy, ix-1])
    if ix < nx-1:
        out.append(grid[iy, ix+1])
    if iy > 0:
        out.append(grid[iy-1, ix])
    if iy < ny-1:
        out.append(grid[iy+1, ix])

    return out


# ============================================================
# 2. Detect SOM interfaces and velocity jump
# ============================================================

N = len(df)

is_interface = np.zeros(N, dtype=bool)
transition = np.zeros(N)
velocity_jump = np.zeros(N)

for i in range(N):

    neigh = direct_neighbours(i)

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


# ============================================================
# 3. Physics-based interface score
#
# SOM: identifies regime boundary
# T: strength of regime transition
# G: local velocity gradient
# dU: velocity-vector jump across SOM interface
#
# No M160 information.
# ============================================================

G = df["gradU_mag_star"].to_numpy()
T = transition
dU = velocity_jump

Gstar = G / G.max()
dUstar = dU / dU.max()

score = T * np.sqrt(Gstar * dUstar)
score[~is_interface] = 0.0

df["som_interface"] = is_interface.astype(int)
df["som_transition"] = T
df["velocity_jump"] = dU
df["interface_score"] = score


# ============================================================
# 4. ONE frozen seed set
# ============================================================

ranking = np.argsort(-score)

seeds = np.array(
    [i for i in ranking if score[i] > 0][:N_SEEDS],
    dtype=int
)

assert len(seeds) == N_SEEDS


# ============================================================
# 5. Neighbourhood to a fixed graph distance
#
# radius 1:
#       X
#     X S X
#       X
#
# radius 2:
#       X
#     X X X
#   X X S X X
#     X X X
#       X
#
# Face-sharing connectivity only.
# ============================================================

def neighbourhood(seed, radius):

    visited = {seed: 0}
    queue = deque([seed])

    while queue:

        current = queue.popleft()
        dist = visited[current]

        if dist >= radius:
            continue

        for nb in direct_neighbours(current):

            if nb not in visited:
                visited[nb] = dist + 1
                queue.append(nb)

    return set(visited.keys())


def expand_fixed_seeds(seeds, radius):

    selected = set()

    for seed in seeds:
        selected.update(neighbourhood(seed, radius))

    return np.array(sorted(selected), dtype=int)


selected_1n = expand_fixed_seeds(seeds, radius=1)
selected_2n = expand_fixed_seeds(seeds, radius=2)

# Fundamental check:
# every seed MUST exist in both selections.

assert set(seeds).issubset(set(selected_1n))
assert set(seeds).issubset(set(selected_2n))


# ============================================================
# 6. Save
# ============================================================

df["interface_seed"] = 0
df.loc[seeds, "interface_seed"] = 1

df["selected_1N"] = 0
df.loc[selected_1n, "selected_1N"] = 1

df["selected_2N"] = 0
df.loc[selected_2n, "selected_2N"] = 1

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)


# ============================================================
# 7. Diagnostics
# ============================================================

print()
print("======================================")
print("FIXED SOM INTERFACE SEED EXPERIMENT")
print("======================================")
print(f"Interface cells        : {is_interface.sum()}")
print(f"Frozen seeds           : {len(seeds)}")
print(f"1N selected cells      : {len(selected_1n)}")
print(f"2N selected cells      : {len(selected_2n)}")
print()
print(f"Mean seed gradient     : {G[seeds].mean():.6f}")
print(f"Mean seed transition   : {T[seeds].mean():.6f}")
print(f"Mean seed velocity jump: {dU[seeds].mean():.6f}")

set1 = set(selected_1n)
set2 = set(selected_2n)

print()
print(f"1N subset of 2N        : {set1.issubset(set2)}")
print(f"Additional 2N cells    : {len(set2-set1)}")


# ============================================================
# 8. Visualization
# ============================================================

def category(selected):

    selected = set(selected)
    seedset = set(seeds)

    cat = np.zeros((ny, nx), dtype=int)

    for iy in range(ny):
        for ix in range(nx):

            idx = grid[iy, ix]

            if idx in selected:
                cat[iy, ix] = 1

            if idx in seedset:
                cat[iy, ix] = 2

    return cat


cat1 = category(selected_1n)
cat2 = category(selected_2n)

cmap = ListedColormap([
    "#eeeeee",
    "#80b1d3",
    "#d95f02",
])

fig, axes = plt.subplots(
    1, 2,
    figsize=(12.5, 5.5),
    constrained_layout=True
)

for ax, cat, title in zip(
    axes,
    [cat1, cat2],
    [
        f"SOM-Interface-1N — {N_SEEDS} fixed seeds",
        f"SOM-Interface-2N — {N_SEEDS} fixed seeds",
    ]
):

    ax.imshow(
        cat,
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
    ax.set_title(title)


legend = [
    Patch(facecolor="#eeeeee", label="Not refined"),
    Patch(facecolor="#80b1d3", label="Neighbourhood cell"),
    Patch(facecolor="#d95f02", label="Same interface seed"),
]

fig.legend(
    handles=legend,
    loc="center left",
    bbox_to_anchor=(1.01, 0.5)
)

OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUT_FIG,
    dpi=300,
    bbox_inches="tight"
)

print()
print(f"Saved CSV   : {OUT_CSV}")
print(f"Saved figure: {OUT_FIG}")
