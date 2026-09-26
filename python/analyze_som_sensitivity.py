from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DIR = PROJECT / "results" / "som_sensitivity"

SIZES = [3, 4, 5, 6]
SEEDS = [1, 11, 21, 31, 42]
N_SEEDS_INTERFACE = 40

# ------------------------------------------------------------
# Load one result and construct structured-grid neighbours
# ------------------------------------------------------------

base = pd.read_csv(DIR / "SOM_4x4_seed42.csv")
n = len(base)

xvals = np.sort(base["x_star"].unique())
yvals = np.sort(base["y_star"].unique())

nx = len(xvals)
ny = len(yvals)

x_to_ix = {x: i for i, x in enumerate(xvals)}
y_to_iy = {y: i for i, y in enumerate(yvals)}

grid = np.empty((ny, nx), dtype=int)

for idx, row in base.iterrows():
    grid[
        y_to_iy[row["y_star"]],
        x_to_ix[row["x_star"]]
    ] = idx

neigh = []

for i, row in base.iterrows():

    ix = x_to_ix[row["x_star"]]
    iy = y_to_iy[row["y_star"]]

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

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def interface_and_seeds(df):

    bmu = df["bmu_id"].to_numpy(dtype=int)
    ux = df["Ux_star"].to_numpy(float)
    uy = df["Uy_star"].to_numpy(float)
    grad = df["gradU_mag_star"].to_numpy(float)

    interface = np.zeros(n, dtype=bool)
    T = np.zeros(n)
    dU = np.zeros(n)

    for i in range(n):

        different = [
            j for j in neigh[i]
            if bmu[j] != bmu[i]
        ]

        if different:

            interface[i] = True
            T[i] = len(different) / len(neigh[i])

            jumps = [
                np.hypot(
                    ux[i] - ux[j],
                    uy[i] - uy[j]
                )
                for j in different
            ]

            dU[i] = max(jumps)

    Gstar = grad / grad.max()

    if dU.max() > 0:
        dUstar = dU / dU.max()
    else:
        dUstar = dU.copy()

    score = np.zeros(n)

    score[interface] = (
        T[interface]
        * np.sqrt(
            Gstar[interface]
            * dUstar[interface]
        )
    )

    ranking = np.argsort(-score)

    seeds = ranking[
        score[ranking] > 0
    ][:N_SEEDS_INTERFACE]

    seed_mask = np.zeros(n, dtype=bool)
    seed_mask[seeds] = True

    return interface, seed_mask


def jaccard(a, b):

    intersection = np.sum(a & b)
    union = np.sum(a | b)

    if union == 0:
        return 1.0

    return intersection / union


def coassignment_agreement(labels_a, labels_b):
    """
    Fraction of unordered cell pairs for which the two SOMs agree
    on whether the cells belong to the same BMU.

    Invariant to BMU label permutation.
    """

    same_a = labels_a[:, None] == labels_a[None, :]
    same_b = labels_b[:, None] == labels_b[None, :]

    iu = np.triu_indices(n, k=1)

    return np.mean(
        same_a[iu] == same_b[iu]
    )


# ------------------------------------------------------------
# Load all runs
# ------------------------------------------------------------

runs = {}

for size in SIZES:
    for seed in SEEDS:

        df = pd.read_csv(
            DIR / f"SOM_{size}x{size}_seed{seed}.csv"
        )

        interface, seed_mask = interface_and_seeds(df)

        runs[(size, seed)] = {
            "df": df,
            "bmu": df["bmu_id"].to_numpy(dtype=int),
            "interface": interface,
            "seeds": seed_mask,
        }

# ------------------------------------------------------------
# Pairwise stability WITHIN each architecture
# ------------------------------------------------------------

rows = []

print()
print("============================================================")
print(" WITHIN-ARCHITECTURE SOM STABILITY")
print("============================================================")

for size in SIZES:

    architecture_rows = []

    for s1, s2 in combinations(SEEDS, 2):

        A = runs[(size, s1)]
        B = runs[(size, s2)]

        row = {
            "size": size,
            "architecture": f"{size}x{size}",
            "seed_a": s1,
            "seed_b": s2,
            "coassignment_agreement":
                coassignment_agreement(A["bmu"], B["bmu"]),
            "interface_jaccard":
                jaccard(A["interface"], B["interface"]),
            "seed_jaccard":
                jaccard(A["seeds"], B["seeds"]),
            "common_seeds":
                np.sum(A["seeds"] & B["seeds"]),
        }

        rows.append(row)
        architecture_rows.append(row)

    temp = pd.DataFrame(architecture_rows)

    print()
    print(f"{size}x{size}")
    print(
        f"  Co-assignment agreement : "
        f"{temp['coassignment_agreement'].mean():.4f} "
        f"+/- {temp['coassignment_agreement'].std(ddof=1):.4f}"
    )
    print(
        f"  Interface Jaccard       : "
        f"{temp['interface_jaccard'].mean():.4f} "
        f"+/- {temp['interface_jaccard'].std(ddof=1):.4f}"
    )
    print(
        f"  40-seed Jaccard         : "
        f"{temp['seed_jaccard'].mean():.4f} "
        f"+/- {temp['seed_jaccard'].std(ddof=1):.4f}"
    )
    print(
        f"  Common seeds            : "
        f"{temp['common_seeds'].mean():.2f} / 40"
    )

pairwise = pd.DataFrame(rows)

pairwise.to_csv(
    DIR / "som_sensitivity_pairwise.csv",
    index=False
)

# ------------------------------------------------------------
# Architecture summary
# ------------------------------------------------------------

summary = (
    pairwise
    .groupby(["size", "architecture"])
    .agg(
        coassignment_mean=("coassignment_agreement", "mean"),
        coassignment_std=("coassignment_agreement", "std"),
        interface_jaccard_mean=("interface_jaccard", "mean"),
        interface_jaccard_std=("interface_jaccard", "std"),
        seed_jaccard_mean=("seed_jaccard", "mean"),
        seed_jaccard_std=("seed_jaccard", "std"),
        common_seeds_mean=("common_seeds", "mean"),
        common_seeds_min=("common_seeds", "min"),
        common_seeds_max=("common_seeds", "max"),
    )
    .reset_index()
)

summary.to_csv(
    DIR / "som_sensitivity_stability_summary.csv",
    index=False
)

print()
print("============================================================")
print(" STABILITY SUMMARY")
print("============================================================")
print()
print(summary.to_string(index=False))

# ------------------------------------------------------------
# Consensus frequency maps
# ------------------------------------------------------------

consensus_rows = []

for size in SIZES:

    interface_frequency = np.mean(
        np.vstack([
            runs[(size, seed)]["interface"]
            for seed in SEEDS
        ]),
        axis=0
    )

    seed_frequency = np.mean(
        np.vstack([
            runs[(size, seed)]["seeds"]
            for seed in SEEDS
        ]),
        axis=0
    )

    temp = base[
        ["x_star", "y_star"]
    ].copy()

    temp["architecture"] = f"{size}x{size}"
    temp["interface_frequency"] = interface_frequency
    temp["seed_frequency"] = seed_frequency

    consensus_rows.append(temp)

consensus = pd.concat(
    consensus_rows,
    ignore_index=True
)

consensus.to_csv(
    DIR / "som_sensitivity_consensus_maps.csv",
    index=False
)

print()
print("Saved:")
print(DIR / "som_sensitivity_pairwise.csv")
print(DIR / "som_sensitivity_stability_summary.csv")
print(DIR / "som_sensitivity_consensus_maps.csv")
print()
