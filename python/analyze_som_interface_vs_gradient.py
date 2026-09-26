from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from framework.indicators import (
    load_m40_dataset,
    select_gradient_cells,
    select_som_interface_cells,
)

OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)


def keyset(df):
    return {
        (
            round(float(r.x_star), 10),
            round(float(r.y_star), 10),
        )
        for _, r in df.iterrows()
    }


base = load_m40_dataset().copy()

cases = [
    {
        "name": "1N",
        "radius": 1,
        "n_gradient": 78,
    },
    {
        "name": "2N",
        "radius": 2,
        "n_gradient": 114,
    },
]

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5.5),
    constrained_layout=True,
)

for ax, cfg in zip(axes, cases):

    ngrad = cfg["n_gradient"]

    grad = select_gradient_cells(
        ngrad / 1600
    )

    som = select_som_interface_cells(
        radius=cfg["radius"],
        n_seeds=40,
    )

    G = keyset(grad)
    S = keyset(som)

    both = G & S
    grad_only = G - S
    som_only = S - G
    neither = (
        keyset(base) - (G | S)
    )

    print()
    print("=" * 60)
    print(
        f"{cfg['name']}: "
        f"Gradient {len(G)} vs SOM {len(S)}"
    )
    print("=" * 60)

    print(f"Both          : {len(both)}")
    print(f"Gradient only : {len(grad_only)}")
    print(f"SOM only      : {len(som_only)}")
    print(f"Neither       : {len(neither)}")

    overlap = (
        100.0 * len(both) / len(G)
    )

    print(f"Overlap       : {overlap:.2f}%")

    categories = {
        "Neither": neither,
        "Both": both,
        "Gradient only": grad_only,
        "SOM-interface only": som_only,
    }

    marker_map = {
        "Neither": "s",
        "Both": "s",
        "Gradient only": "s",
        "SOM-interface only": "s",
    }

    size_map = {
        "Neither": 18,
        "Both": 30,
        "Gradient only": 30,
        "SOM-interface only": 30,
    }

    for label, cells in categories.items():

        rows = []

        for x, y in cells:
            rows.append((x, y))

        if not rows:
            continue

        arr = np.asarray(rows)

        ax.scatter(
            arr[:, 0],
            arr[:, 1],
            marker=marker_map[label],
            s=size_map[label],
            label=label,
        )

    ax.set_title(
        f"{cfg['name']}: "
        f"{len(G)} selected cells"
    )

    ax.set_xlabel("x/L")
    ax.set_ylabel("y/L")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_aspect("equal")

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
)

fig.suptitle(
    "Cell-matched Gradient vs SOM-interface selection"
)

outfile = (
    OUT
    / "Gradient_vs_SOM_Interface_cell_matched.png"
)

fig.savefig(
    outfile,
    dpi=220,
    bbox_inches="tight",
)

print()
print(f"Saved: {outfile}")
