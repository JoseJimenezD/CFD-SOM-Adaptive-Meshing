from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DIR = PROJECT / "results" / "som_sensitivity"

SIZES = [3, 4, 5, 6]
SEEDS = [1, 11, 21, 31, 42]
N_TOP = 40

# ============================================================
# GRID
# ============================================================

base = pd.read_csv(DIR / "SOM_4x4_seed42.csv")

xvals = np.sort(base["x_star"].unique())
yvals = np.sort(base["y_star"].unique())

x_to_ix = {x: i for i, x in enumerate(xvals)}
y_to_iy = {y: i for i, y in enumerate(yvals)}

coords = np.array([
    [
        x_to_ix[row["x_star"]],
        y_to_iy[row["y_star"]]
    ]
    for _, row in base.iterrows()
])

# Direct face neighbours
nx = len(xvals)
ny = len(yvals)

grid = np.empty((ny, nx), dtype=int)

for idx, row in base.iterrows():
    grid[
        y_to_iy[row["y_star"]],
        x_to_ix[row["x_star"]]
    ] = idx

neigh = []

for i, (ix, iy) in enumerate(coords):

    q = []

    if ix > 0:
        q.append(grid[iy, ix-1])
    if ix < nx-1:
        q.append(grid[iy, ix+1])
    if iy > 0:
        q.append(grid[iy-1, ix])
    if iy < ny-1:
        q.append(grid[iy+1, ix])

    neigh.append(q)

# ============================================================
# RECONSTRUCT TOP-40 EXACTLY AS PREVIOUS ANALYSIS
# ============================================================

def get_top40(df):

    bmu = df["bmu_id"].to_numpy(dtype=int)
    ux = df["Ux_star"].to_numpy(float)
    uy = df["Uy_star"].to_numpy(float)
    grad = df["gradU_mag_star"].to_numpy(float)

    n = len(df)

    T = np.zeros(n)
    dU = np.zeros(n)
    interface = np.zeros(n, dtype=bool)

    for i in range(n):

        different = [
            j for j in neigh[i]
            if bmu[j] != bmu[i]
        ]

        if different:

            interface[i] = True

            T[i] = (
                len(different) /
                len(neigh[i])
            )

            dU[i] = max(
                np.hypot(
                    ux[i] - ux[j],
                    uy[i] - uy[j]
                )
                for j in different
            )

    Gstar = grad / grad.max()

    dUstar = (
        dU / dU.max()
        if dU.max() > 0
        else dU.copy()
    )

    score = np.zeros(n)

    score[interface] = (
        T[interface]
        * np.sqrt(
            Gstar[interface]
            * dUstar[interface]
        )
    )

    ranking = np.argsort(-score)
    ranking = ranking[score[ranking] > 0]

    return ranking[:N_TOP]


# ============================================================
# SPATIAL MATCH
# ============================================================

def directed_match(A, B, radius):
    """
    Fraction of seeds in A having at least one seed in B
    within Manhattan grid distance <= radius.
    """

    Bcoords = coords[B]

    matched = 0

    for i in A:

        d = np.abs(
            Bcoords - coords[i]
        ).sum(axis=1)

        if np.min(d) <= radius:
            matched += 1

    return matched / len(A)


def symmetric_match(A, B, radius):

    ab = directed_match(A, B, radius)
    ba = directed_match(B, A, radius)

    return 0.5 * (ab + ba)


def exact_jaccard(A, B):

    A = set(A.tolist())
    B = set(B.tolist())

    return len(A & B) / len(A | B)


# ============================================================
# LOAD TOP-40
# ============================================================

top = {}

for size in SIZES:

    for seed in SEEDS:

        df = pd.read_csv(
            DIR / f"SOM_{size}x{size}_seed{seed}.csv"
        )

        top[(size, seed)] = get_top40(df)

# ============================================================
# PAIRWISE ANALYSIS
# ============================================================

rows = []

print()
print("============================================================")
print(" TOP-40 SPATIAL STABILITY")
print("============================================================")

for size in SIZES:

    local = []

    for s1, s2 in combinations(SEEDS, 2):

        A = top[(size, s1)]
        B = top[(size, s2)]

        row = {
            "size": size,
            "architecture": f"{size}x{size}",
            "seed_a": s1,
            "seed_b": s2,
            "exact_jaccard": exact_jaccard(A, B),
            "match_r0": symmetric_match(A, B, 0),
            "match_r1": symmetric_match(A, B, 1),
            "match_r2": symmetric_match(A, B, 2),
        }

        rows.append(row)
        local.append(row)

    temp = pd.DataFrame(local)

    print()
    print(f"{size}x{size}")

    print(
        f"  Exact Jaccard : "
        f"{temp['exact_jaccard'].mean():.4f}"
    )

    print(
        f"  Exact match   : "
        f"{100*temp['match_r0'].mean():.1f}%"
    )

    print(
        f"  Match <= 1N   : "
        f"{100*temp['match_r1'].mean():.1f}%"
    )

    print(
        f"  Match <= 2N   : "
        f"{100*temp['match_r2'].mean():.1f}%"
    )

# ============================================================
# SAVE
# ============================================================

pairwise = pd.DataFrame(rows)

pairwise.to_csv(
    DIR / "som_seed_spatial_stability_pairwise.csv",
    index=False
)

summary = (
    pairwise
    .groupby(["size", "architecture"])
    .agg(
        exact_jaccard_mean=("exact_jaccard", "mean"),
        exact_jaccard_std=("exact_jaccard", "std"),
        match_r0_mean=("match_r0", "mean"),
        match_r1_mean=("match_r1", "mean"),
        match_r1_min=("match_r1", "min"),
        match_r2_mean=("match_r2", "mean"),
        match_r2_min=("match_r2", "min"),
    )
    .reset_index()
)

summary.to_csv(
    DIR / "som_seed_spatial_stability_summary.csv",
    index=False
)

print()
print("============================================================")
print(" SUMMARY")
print("============================================================")
print()
print(summary.to_string(index=False))

print()
print("Saved:")
print(DIR / "som_seed_spatial_stability_pairwise.csv")
print(DIR / "som_seed_spatial_stability_summary.csv")
print()
