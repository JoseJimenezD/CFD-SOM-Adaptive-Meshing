from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

PROJECT = Path(__file__).resolve().parents[1]
DIR = PROJECT / "results" / "som_sensitivity"

SIZES = [3, 4, 5, 6]
SEEDS = [1, 11, 21, 31, 42]


def same_cluster_jaccard(a, b):
    """
    Jaccard similarity between the sets of unordered cell pairs
    assigned to the same BMU.

    Invariant to BMU numbering.
    """

    n = len(a)
    iu = np.triu_indices(n, k=1)

    same_a = (a[:, None] == a[None, :])[iu]
    same_b = (b[:, None] == b[None, :])[iu]

    intersection = np.sum(same_a & same_b)
    union = np.sum(same_a | same_b)

    if union == 0:
        return 1.0

    return intersection / union


# ============================================================
# LOAD BMU PARTITIONS
# ============================================================

runs = {}

for size in SIZES:
    for seed in SEEDS:

        file = DIR / f"SOM_{size}x{size}_seed{seed}.csv"

        df = pd.read_csv(file)

        runs[(size, seed)] = (
            df["bmu_id"].to_numpy(dtype=int)
        )


# ============================================================
# WITHIN-ARCHITECTURE COMPARISONS
# ============================================================

rows = []

print()
print("============================================================")
print(" SOM PARTITION STABILITY — CORRECTED METRICS")
print("============================================================")

for size in SIZES:

    local = []

    for s1, s2 in combinations(SEEDS, 2):

        a = runs[(size, s1)]
        b = runs[(size, s2)]

        ari = adjusted_rand_score(a, b)

        pair_jaccard = same_cluster_jaccard(a, b)

        row = {
            "size": size,
            "architecture": f"{size}x{size}",
            "seed_a": s1,
            "seed_b": s2,
            "ari": ari,
            "same_cluster_jaccard": pair_jaccard,
        }

        rows.append(row)
        local.append(row)

    temp = pd.DataFrame(local)

    print()
    print(f"{size}x{size}")

    print(
        f"  ARI                    : "
        f"{temp['ari'].mean():.4f} "
        f"+/- {temp['ari'].std(ddof=1):.4f}"
    )

    print(
        f"  Same-cluster Jaccard   : "
        f"{temp['same_cluster_jaccard'].mean():.4f} "
        f"+/- {temp['same_cluster_jaccard'].std(ddof=1):.4f}"
    )

    print(
        f"  ARI range              : "
        f"{temp['ari'].min():.4f} "
        f"to {temp['ari'].max():.4f}"
    )

    print(
        f"  Pair-Jaccard range     : "
        f"{temp['same_cluster_jaccard'].min():.4f} "
        f"to {temp['same_cluster_jaccard'].max():.4f}"
    )


# ============================================================
# SAVE PAIRWISE RESULTS
# ============================================================

pairwise = pd.DataFrame(rows)

pairwise_file = (
    DIR / "som_partition_stability_pairwise.csv"
)

pairwise.to_csv(
    pairwise_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    pairwise
    .groupby(["size", "architecture"])
    .agg(
        ari_mean=("ari", "mean"),
        ari_std=("ari", "std"),
        ari_min=("ari", "min"),
        ari_max=("ari", "max"),
        same_cluster_jaccard_mean=(
            "same_cluster_jaccard", "mean"
        ),
        same_cluster_jaccard_std=(
            "same_cluster_jaccard", "std"
        ),
        same_cluster_jaccard_min=(
            "same_cluster_jaccard", "min"
        ),
        same_cluster_jaccard_max=(
            "same_cluster_jaccard", "max"
        ),
    )
    .reset_index()
)

summary_file = (
    DIR / "som_partition_stability_summary.csv"
)

summary.to_csv(
    summary_file,
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
print(pairwise_file)
print(summary_file)
print()
