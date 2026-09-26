from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from framework.indicators import (
    select_gradient_cells,
    select_hybrid_cells,
)


PROJECT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT / "data" / "M40_som_dataset.csv"
PLOTS_DIR = PROJECT / "results" / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

budgets = [0.05, 0.10, 0.15, 0.20]


def make_keys(frame):
    return set(
        zip(
            frame["x_star"].round(10),
            frame["y_star"].round(10),
        )
    )


# ---------------------------------------------------------------------
# Recover the structured M40 grid from cell-centre coordinates.
# ---------------------------------------------------------------------

x_values = np.sort(df["x_star"].unique())
y_values = np.sort(df["y_star"].unique())

nx = len(x_values)
ny = len(y_values)

if nx * ny != len(df):
    raise RuntimeError(
        f"Expected a structured grid but obtained "
        f"nx={nx}, ny={ny}, cells={len(df)}"
    )

print(f"Detected structured grid: {nx} x {ny} = {nx*ny} cells")

x_index = {
    round(float(x), 10): i
    for i, x in enumerate(x_values)
}

y_index = {
    round(float(y), 10): j
    for j, y in enumerate(y_values)
}


# Category codes:
# 0 = Neither
# 1 = Gradient only
# 2 = Hybrid-v1 only
# 3 = Both

category_labels = [
    "Neither",
    "Gradient only",
    "Hybrid-v1 only",
    "Both",
]

# Use Matplotlib default categorical colors.
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

cmap = ListedColormap(
    [
        "0.90",            # Neither: light gray
        default_colors[1], # Gradient only
        default_colors[2], # Hybrid only
        default_colors[3], # Both
    ]
)


fig, axes = plt.subplots(
    2,
    2,
    figsize=(10.5, 9.0),
)

axes = axes.ravel()


for ax, budget in zip(axes, budgets):

    g = select_gradient_cells(budget)
    h = select_hybrid_cells(
        budget,
        alpha=0.5,
    )

    G = make_keys(g)
    H = make_keys(h)

    both = G & H
    grad_only = G - H
    hybrid_only = H - G

    grid = np.zeros(
        (ny, nx),
        dtype=int,
    )

    for _, row in df.iterrows():

        key = (
            round(float(row["x_star"]), 10),
            round(float(row["y_star"]), 10),
        )

        i = x_index[key[0]]
        j = y_index[key[1]]

        if key in both:
            code = 3
        elif key in grad_only:
            code = 1
        elif key in hybrid_only:
            code = 2
        else:
            code = 0

        grid[j, i] = code

    neither_count = int(np.sum(grid == 0))

    # Draw complete CFD cells rather than point markers.
    ax.imshow(
        grid,
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        interpolation="nearest",
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        aspect="equal",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel("x/L")
    ax.set_ylabel("y/L")

    ax.set_title(
        f"{int(100*budget)}% refinement budget"
    )

    # Cell-map visualization: no grid overlay.
    ax.grid(False)

    legend_handles = [
        Patch(
            facecolor=cmap(0),
            label=f"Neither ({neither_count})",
        ),
        Patch(
            facecolor=cmap(1),
            label=f"Gradient only ({len(grad_only)})",
        ),
        Patch(
            facecolor=cmap(2),
            label=f"Hybrid-v1 only ({len(hybrid_only)})",
        ),
        Patch(
            facecolor=cmap(3),
            label=f"Both ({len(both)})",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        borderaxespad=0.0,
    )


fig.suptitle(
    "Gradient vs Hybrid-v1 cell selection",
    fontsize=14,
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96],
    w_pad=6.0,
    h_pad=2.0,
)

output = (
    PLOTS_DIR
    / "Gradient_vs_Hybrid_selection_all_budgets.png"
)

plt.savefig(
    output,
    dpi=220,
    bbox_inches="tight",
)

plt.close()


print()
print("=" * 70)
print("GRADIENT vs HYBRID-v1 SELECTION MAP")
print("=" * 70)

for budget in budgets:

    g = select_gradient_cells(budget)
    h = select_hybrid_cells(
        budget,
        alpha=0.5,
    )

    G = make_keys(g)
    H = make_keys(h)

    print(
        f"{100*budget:4.0f}% : "
        f"Both={len(G & H):3d}  "
        f"Gradient-only={len(G-H):2d}  "
        f"Hybrid-only={len(H-G):2d}  "
        f"Overlap={100*len(G&H)/len(G):5.2f}%"
    )

print()
print("Saved:", output)
