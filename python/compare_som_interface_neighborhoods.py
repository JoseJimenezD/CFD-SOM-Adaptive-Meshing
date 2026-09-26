import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path
from collections import deque

PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "data" / "M40_som_results.csv"

OUT_CSV = PROJECT / "results" / "M40_som_interface_neighborhoods_10.csv"
OUT_FIG = PROJECT / "results" / "plots" / "SOM_Interface_1N_vs_2N_10.png"

BUDGET = 0.10

df = pd.read_csv(INPUT).copy()

# ============================================================
# 1. Reconstruct structured M40 grid
# ============================================================

xs = np.sort(df["x_star"].unique())
ys = np.sort(df["y_star"].unique())

nx = len(xs)
ny = len(ys)

assert nx * ny == len(df)
assert nx == 40 and ny == 40

x_to_ix = {x: i for i, x in enumerate(xs)}
y_to_iy = {y: j for j, y in enumerate(ys)}

grid = np.empty((ny, nx), dtype=int)

for idx, row in df.iterrows():
    ix = x_to_ix[row["x_star"]]
    iy = y_to_iy[row["y_star"]]
    grid[iy, ix] = idx

print(f"Grid: {nx} x {ny}")
print(f"Cells: {len(df)}")

# ============================================================
# 2. Direct face-sharing neighbours
# ============================================================

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
# 3. SOM interface detection
# ============================================================

N = len(df)

transition = np.zeros(N)
velocity_jump = np.zeros(N)
is_interface = np.zeros(N, dtype=bool)

for i in range(N):

    neigh = direct_neighbours(i)

    different = [
        j for j in neigh
        if int(df.loc[j, "bmu_id"]) != int(df.loc[i, "bmu_id"])
    ]

    if len(different) == 0:
        continue

    is_interface[i] = True

    transition[i] = len(different) / len(neigh)

    jumps = []

    for j in different:

        dux = df.loc[i, "Ux_star"] - df.loc[j, "Ux_star"]
        duy = df.loc[i, "Uy_star"] - df.loc[j, "Uy_star"]

        jumps.append(
            np.sqrt(dux**2 + duy**2)
        )

    velocity_jump[i] = max(jumps)


# ============================================================
# 4. Interface physics score
#
# I = T * sqrt(G* dU*)
#
# IMPORTANT:
# M160 is NOT used here.
# ============================================================

G = df["gradU_mag_star"].to_numpy()
dU = velocity_jump
T = transition

Gstar = G / G.max()

if dU.max() > 0:
    dUstar = dU / dU.max()
else:
    dUstar = np.zeros_like(dU)

score = T * np.sqrt(Gstar * dUstar)

score[~is_interface] = 0.0

df["som_interface"] = is_interface.astype(int)
df["som_transition"] = T
df["velocity_jump"] = dU
df["interface_score"] = score

ranking = [
    i for i in np.argsort(-score)
    if score[i] > 0
]


# ============================================================
# 5. Fixed-radius neighbourhood
#
# radius = 1:
#
#       X
#     X S X
#       X
#
# maximum 5 cells in interior.
#
#
# radius = 2:
#
#       X
#     X X X
#   X X S X X
#     X X X
#       X
#
# maximum 13 cells in interior.
#
# Distance is graph/Manhattan distance using face neighbours.
# ============================================================

def neighbourhood(seed, radius):

    visited = {seed: 0}
    queue = deque([seed])

    while queue:

        current = queue.popleft()
        distance = visited[current]

        if distance >= radius:
            continue

        for nb in direct_neighbours(current):

            if nb not in visited:
                visited[nb] = distance + 1
                queue.append(nb)

    return set(visited.keys())


# ============================================================
# 6. Selection under fixed TOTAL budget
#
# We add complete neighbourhoods whenever possible.
#
# If the next complete neighbourhood would exceed the budget,
# we skip it and continue searching for another seed whose
# neighbourhood fits.
#
# Finally, if a few cells are still missing, they are filled
# deterministically from the highest-ranked interface cells
# and their nearest available neighbours.
#
# No recursive expansion beyond requested radius.
# ============================================================

def select_with_neighbourhood(radius, target):

    selected = set()
    seeds = []

    # First pass: complete neighbourhoods only
    for seed in ranking:

        region = neighbourhood(seed, radius)

        new_cells = region - selected

        if not new_cells:
            continue

        if len(selected) + len(new_cells) <= target:

            selected.update(new_cells)
            seeds.append(seed)

        if len(selected) == target:
            break

    # --------------------------------------------------------
    # Controlled completion if exact target was not reached.
    # --------------------------------------------------------

    if len(selected) < target:

        for seed in ranking:

            if len(selected) >= target:
                break

            region = neighbourhood(seed, radius)

            # Sort region:
            # seed first, then by Manhattan/grid distance,
            # then by index for deterministic behavior.
            sx = x_to_ix[df.loc[seed, "x_star"]]
            sy = y_to_iy[df.loc[seed, "y_star"]]

            ordered = sorted(
                region,
                key=lambda idx: (
                    abs(x_to_ix[df.loc[idx, "x_star"]] - sx)
                    + abs(y_to_iy[df.loc[idx, "y_star"]] - sy),
                    idx
                )
            )

            added_from_seed = False

            for c in ordered:

                if c in selected:
                    continue

                selected.add(c)
                added_from_seed = True

                if len(selected) >= target:
                    break

            if added_from_seed and seed not in seeds:
                seeds.append(seed)

    return (
        np.array(sorted(selected), dtype=int),
        np.array(seeds, dtype=int)
    )


# ============================================================
# 7. Generate 1N and 2N selections
# ============================================================

target = int(round(BUDGET * N))

sel_1n, seeds_1n = select_with_neighbourhood(
    radius=1,
    target=target
)

sel_2n, seeds_2n = select_with_neighbourhood(
    radius=2,
    target=target
)

assert len(sel_1n) == target
assert len(sel_2n) == target


# ============================================================
# 8. Store selections
# ============================================================

df["selected_1N"] = 0
df["seed_1N"] = 0

df.loc[sel_1n, "selected_1N"] = 1
df.loc[seeds_1n, "seed_1N"] = 1

df["selected_2N"] = 0
df["seed_2N"] = 0

df.loc[sel_2n, "selected_2N"] = 1
df.loc[seeds_2n, "seed_2N"] = 1

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)


# ============================================================
# 9. Diagnostics
# ============================================================

print()
print("========================================")
print("SOM INTERFACE NEIGHBOURHOOD EXPERIMENT")
print("========================================")

print(f"Budget              : {BUDGET*100:.0f}%")
print(f"Target cells        : {target}")
print(f"Interface cells     : {is_interface.sum()}")

for name, selected, seeds in [
    ("1N", sel_1n, seeds_1n),
    ("2N", sel_2n, seeds_2n),
]:

    print()
    print(f"--- {name} ---")
    print(f"Selected cells      : {len(selected)}")
    print(f"Seeds               : {len(seeds)}")

    print(
        f"Mean seed gradient  : "
        f"{G[seeds].mean():.6f}"
    )

    print(
        f"Mean seed transition: "
        f"{T[seeds].mean():.6f}"
    )

    print(
        f"Mean seed dU        : "
        f"{dU[seeds].mean():.6f}"
    )


# ============================================================
# 10. Overlap 1N vs 2N
# ============================================================

set1 = set(sel_1n.tolist())
set2 = set(sel_2n.tolist())

common = set1 & set2

print()
print("--- 1N vs 2N ---")
print(f"Common cells        : {len(common)}")
print(f"1N only             : {len(set1-common)}")
print(f"2N only             : {len(set2-common)}")
print(
    f"Overlap             : "
    f"{100*len(common)/target:.2f}%"
)


# ============================================================
# 11. Full-cell visualization
# ============================================================

def make_category(selected, seeds):

    selected = set(selected.tolist())
    seeds = set(seeds.tolist())

    cat = np.zeros((ny, nx), dtype=int)

    for iy in range(ny):
        for ix in range(nx):

            idx = grid[iy, ix]

            if idx in selected:
                cat[iy, ix] = 1

            if idx in seeds:
                cat[iy, ix] = 2

    return cat


cat1 = make_category(sel_1n, seeds_1n)
cat2 = make_category(sel_2n, seeds_2n)

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
        "SOM-Interface-1N — 10%",
        "SOM-Interface-2N — 10%",
    ],
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
    Patch(
        facecolor="#eeeeee",
        label="Not refined"
    ),
    Patch(
        facecolor="#80b1d3",
        label="Neighbourhood cell"
    ),
    Patch(
        facecolor="#d95f02",
        label="Interface seed"
    ),
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
