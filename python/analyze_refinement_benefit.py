from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from framework.config import PROJECT, DATA_DIR, RESULTS_DIR, L, U_LID
from framework.validator import (
    read_vectors,
    build_reference_interpolators,
)


TIME = "5"

M40 = PROJECT / "cases" / "Cavity_SOM_002"
GRAD10 = PROJECT / "experiments" / "GRAD_10"
SOM10 = PROJECT / "experiments" / "SOM_QE_10"

PLOTS = RESULTS_DIR / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Read M40 cell centres and velocity
# ------------------------------------------------------------

C40 = read_vectors(M40 / TIME / "C")
U40 = read_vectors(M40 / TIME / "U")

if len(C40) != 1600:
    raise RuntimeError(
        f"Expected 1600 M40 cells, found {len(C40)}"
    )

points40 = C40[:, :2]


# ------------------------------------------------------------
# M160 reference evaluated at M40 centres
# ------------------------------------------------------------

interp_ref_x, interp_ref_y, _ = build_reference_interpolators()

Ux_ref = interp_ref_x(points40)
Uy_ref = interp_ref_y(points40)

Uref = np.column_stack([Ux_ref, Uy_ref])


# ------------------------------------------------------------
# Interpolate adaptive solutions onto M40 centres
# ------------------------------------------------------------

def interpolate_case(case_dir):

    C = read_vectors(case_dir / TIME / "C")
    U = read_vectors(case_dir / TIME / "U")

    points = C[:, :2]

    Ux = griddata(
        points,
        U[:, 0],
        points40,
        method="linear",
    )

    Uy = griddata(
        points,
        U[:, 1],
        points40,
        method="linear",
    )

    # M40 centres should be inside the adaptive point cloud,
    # but use nearest neighbour only as a safe fallback.
    missing = np.isnan(Ux) | np.isnan(Uy)

    if np.any(missing):
        print(
            f"WARNING: {missing.sum()} interpolation points "
            f"required nearest-neighbour fallback."
        )

        Ux[missing] = griddata(
            points,
            U[:, 0],
            points40[missing],
            method="nearest",
        )

        Uy[missing] = griddata(
            points,
            U[:, 1],
            points40[missing],
            method="nearest",
        )

    return np.column_stack([Ux, Uy])


UG = interpolate_case(GRAD10)
US = interpolate_case(SOM10)


# ------------------------------------------------------------
# Local errors relative to M160
# ------------------------------------------------------------

def velocity_error(U):
    d = U - Uref
    return np.sqrt(
        d[:, 0]**2 + d[:, 1]**2
    ) / U_LID


e40 = velocity_error(U40[:, :2])
eG = velocity_error(UG)
eS = velocity_error(US)

benefit_G = e40 - eG
benefit_S = e40 - eS


# ------------------------------------------------------------
# Recover 10% indicator selections
# ------------------------------------------------------------

dataset = pd.read_csv(
    DATA_DIR / "M40_som_dataset.csv"
)

som_results = pd.read_csv(
    DATA_DIR / "M40_som_results.csv"
)

n_select = 160

grad_idx = set(
    dataset.nlargest(
        n_select,
        "gradU_mag_star"
    ).index
)

som_idx = set(
    som_results.nlargest(
        n_select,
        "quantization_error"
    ).index
)

both = grad_idx & som_idx
grad_only = grad_idx - som_idx
som_only = som_idx - grad_idx
neither = (
    set(range(1600))
    - (grad_idx | som_idx)
)


category = np.full(
    1600,
    "Neither",
    dtype=object,
)

category[list(grad_only)] = "Gradient only"
category[list(som_only)] = "SOM-QE only"
category[list(both)] = "Both"


# ------------------------------------------------------------
# Save cell-by-cell results
# ------------------------------------------------------------

df = pd.DataFrame(
    {
        "x_over_L": C40[:, 0] / L,
        "y_over_L": C40[:, 1] / L,
        "category": category,
        "error_M40": e40,
        "error_GRAD10": eG,
        "error_SOM10": eS,
        "benefit_GRAD10": benefit_G,
        "benefit_SOM10": benefit_S,
    }
)

output_csv = (
    RESULTS_DIR
    / "refinement_benefit_10.csv"
)

df.to_csv(output_csv, index=False)


# ------------------------------------------------------------
# Category statistics
# ------------------------------------------------------------

order = [
    "Both",
    "Gradient only",
    "SOM-QE only",
    "Neither",
]

rows = []

for name in order:

    mask = category == name

    rows.append(
        {
            "category": name,
            "cells": int(mask.sum()),

            "initial_mean_error":
                float(e40[mask].mean()),

            "GRAD10_mean_error":
                float(eG[mask].mean()),

            "SOM10_mean_error":
                float(eS[mask].mean()),

            "GRAD10_mean_benefit":
                float(benefit_G[mask].mean()),

            "SOM10_mean_benefit":
                float(benefit_S[mask].mean()),

            "GRAD10_fraction_improved":
                float(
                    np.mean(benefit_G[mask] > 0)
                ),

            "SOM10_fraction_improved":
                float(
                    np.mean(benefit_S[mask] > 0)
                ),
        }
    )

stats = pd.DataFrame(rows)

stats_file = (
    RESULTS_DIR
    / "refinement_benefit_categories_10.csv"
)

stats.to_csv(stats_file, index=False)


# ------------------------------------------------------------
# Spatial benefit maps
# ------------------------------------------------------------

vmax = max(
    np.max(np.abs(benefit_G)),
    np.max(np.abs(benefit_S)),
)

for name, values, title in [
    (
        "GRAD10",
        benefit_G,
        "Local benefit of Gradient-10 refinement",
    ),
    (
        "SOM10",
        benefit_S,
        "Local benefit of SOM-QE-10 refinement",
    ),
]:

    plt.figure(figsize=(7, 6))

    sc = plt.scatter(
        C40[:, 0] / L,
        C40[:, 1] / L,
        c=values,
        s=28,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    plt.colorbar(
        sc,
        label="Benefit = M40 error - adaptive error",
    )

    plt.xlabel("x/L")
    plt.ylabel("y/L")
    plt.title(title)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.gca().set_aspect(
        "equal",
        adjustable="box",
    )

    plt.tight_layout()

    output = (
        PLOTS
        / f"refinement_benefit_{name}.png"
    )

    plt.savefig(output, dpi=220)
    plt.close()


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print()
print("=" * 90)
print("REFINEMENT BENEFIT ANALYSIS — 10% BUDGET")
print("=" * 90)

print()
print("Positive benefit = adaptive solution closer to M160")
print("Negative benefit = adaptive solution farther from M160")

print()
print(stats.to_string(
    index=False,
    float_format=lambda x: f"{x:.6e}",
))

print()
print("=" * 90)
print("GLOBAL STATISTICS ON COMMON M40 POINTS")
print("=" * 90)

print(f"M40 mean error    : {e40.mean():.8e}")
print(f"GRAD10 mean error : {eG.mean():.8e}")
print(f"SOM10 mean error  : {eS.mean():.8e}")

print()

print(
    "GRAD10 points improved : "
    f"{100*np.mean(benefit_G > 0):.2f}%"
)

print(
    "SOM10 points improved  : "
    f"{100*np.mean(benefit_S > 0):.2f}%"
)

print()
print(f"Saved: {output_csv}")
print(f"Saved: {stats_file}")
print(f"Saved plots in: {PLOTS}")
