from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from framework.indicators import (
    load_m40_dataset,
    select_gradient_cells,
    select_som_interface_cells,
)

RESULTS = Path("results/field_comparisons")


def key(x, y):
    return (round(float(x), 10), round(float(y), 10))


def keyset(df):
    return {
        key(r.x_star, r.y_star)
        for _, r in df.iterrows()
    }


def interpolate_error(csv_file, xq, yq):
    """
    Interpolate adaptive-mesh dU_mag_star onto original M40 centres.
    Linear interpolation is primary; nearest-neighbour is used only
    if a query point falls outside the linear interpolation hull.
    """
    df = pd.read_csv(csv_file)

    pts = df[["x", "y"]].to_numpy() / 0.1
    values = df["dU_mag_star"].to_numpy()

    linear = LinearNDInterpolator(pts, values)
    nearest = NearestNDInterpolator(pts, values)

    q = np.column_stack([xq, yq])

    out = np.asarray(linear(q), dtype=float)

    missing = np.isnan(out)

    if np.any(missing):
        out[missing] = nearest(q[missing])

    return out


base = load_m40_dataset().copy()

# M40 error against M160, evaluated at M40 centres.
m40_file = RESULTS / "Cavity_SOM_002_vs_M160.csv"

if not m40_file.exists():
    raise RuntimeError(
        f"Missing {m40_file}. "
        "We need the existing M40-vs-M160 comparison file."
    )

m40_cmp = pd.read_csv(m40_file)

# Match M40 comparison by physical coordinates.
m40_error_lookup = {
    key(r.x / 0.1, r.y / 0.1): float(r.dU_mag_star)
    for _, r in m40_cmp.iterrows()
}

base["key"] = [
    key(x, y)
    for x, y in zip(base.x_star, base.y_star)
]

missing = [
    k for k in base["key"]
    if k not in m40_error_lookup
]

if missing:
    raise RuntimeError(
        f"{len(missing)} M40 centres missing from Cavity_SOM_002_vs_M160.csv"
    )

base["error_M40"] = [
    m40_error_lookup[k]
    for k in base["key"]
]

xq = base["x_star"].to_numpy()
yq = base["y_star"].to_numpy()


experiments = [
    {
        "label": "1N",
        "radius": 1,
        "n_grad": 78,
        "grad_file": RESULTS / "Cavity_GRAD_MATCH_1N_vs_M160.csv",
        "som_file": RESULTS / "Cavity_SOM_INT_40S_1N_vs_M160.csv",
    },
    {
        "label": "2N",
        "radius": 2,
        "n_grad": 114,
        "grad_file": RESULTS / "Cavity_GRAD_MATCH_2N_vs_M160.csv",
        "som_file": RESULTS / "Cavity_SOM_INT_40S_2N_vs_M160.csv",
    },
]


for exp in experiments:

    grad_sel = select_gradient_cells(
        exp["n_grad"] / 1600
    )

    som_sel = select_som_interface_cells(
        radius=exp["radius"],
        n_seeds=40,
    )

    G = keyset(grad_sel)
    S = keyset(som_sel)

    both = G & S
    grad_only = G - S
    som_only = S - G
    neither = set(base["key"]) - (G | S)

    categories = {
        "Both": both,
        "Gradient-only": grad_only,
        "SOM-only": som_only,
        "Neither": neither,
    }

    grad_error = interpolate_error(
        exp["grad_file"],
        xq,
        yq,
    )

    som_error = interpolate_error(
        exp["som_file"],
        xq,
        yq,
    )

    work = base[
        ["x_star", "y_star", "key", "error_M40"]
    ].copy()

    work["error_Gradient"] = grad_error
    work["error_SOM"] = som_error

    work["benefit_Gradient"] = (
        work["error_M40"]
        - work["error_Gradient"]
    )

    work["benefit_SOM"] = (
        work["error_M40"]
        - work["error_SOM"]
    )

    work["SOM_minus_Gradient_benefit"] = (
        work["benefit_SOM"]
        - work["benefit_Gradient"]
    )

    print()
    print("=" * 78)
    print(f"{exp['label']} REFINEMENT-UTILITY ANALYSIS")
    print("=" * 78)

    rows = []

    for name, cells in categories.items():

        sub = work[
            work["key"].isin(cells)
        ]

        row = {
            "Region": name,
            "N": len(sub),
            "M40_error_mean": sub["error_M40"].mean(),
            "Gradient_error_mean": sub["error_Gradient"].mean(),
            "SOM_error_mean": sub["error_SOM"].mean(),
            "Gradient_benefit_mean": sub["benefit_Gradient"].mean(),
            "SOM_benefit_mean": sub["benefit_SOM"].mean(),
            "SOM_minus_Gradient_benefit":
                sub["SOM_minus_Gradient_benefit"].mean(),
            "SOM_better_fraction":
                np.mean(
                    sub["error_SOM"]
                    < sub["error_Gradient"]
                ),
        }

        rows.append(row)

    summary = pd.DataFrame(rows)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.8e}",
        )
    )

    outfile = (
        RESULTS
        / f"refinement_utility_{exp['label']}.csv"
    )

    summary.to_csv(outfile, index=False)

    print()
    print(f"Saved: {outfile}")

