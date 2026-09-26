from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]

INPUT = (
    PROJECT
    / "results"
    / "som_sensitivity"
    / "som_sensitivity_consensus_maps.csv"
)

OUTPUT = (
    PROJECT
    / "results"
    / "som_sensitivity"
    / "som_sensitivity_consensus.png"
)

df = pd.read_csv(INPUT)

architectures = ["3x3", "4x4", "5x5", "6x6"]

xvals = np.sort(df["x_star"].unique())
yvals = np.sort(df["y_star"].unique())

nx = len(xvals)
ny = len(yvals)

x_to_ix = {x: i for i, x in enumerate(xvals)}
y_to_iy = {y: i for i, y in enumerate(yvals)}

fig, axes = plt.subplots(
    2,
    4,
    figsize=(16, 8),
    constrained_layout=True
)

for col, architecture in enumerate(architectures):

    sub = df[
        df["architecture"] == architecture
    ]

    interface_map = np.zeros((ny, nx))
    seed_map = np.zeros((ny, nx))

    for _, row in sub.iterrows():

        ix = x_to_ix[row["x_star"]]
        iy = y_to_iy[row["y_star"]]

        interface_map[iy, ix] = (
            row["interface_frequency"]
        )

        seed_map[iy, ix] = (
            row["seed_frequency"]
        )

    # --------------------------------------------------------
    # Interface consensus
    # --------------------------------------------------------

    im1 = axes[0, col].imshow(
        interface_map,
        origin="lower",
        extent=[
            xvals.min(),
            xvals.max(),
            yvals.min(),
            yvals.max()
        ],
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest"
    )

    axes[0, col].set_title(
        f"{architecture} — Interface frequency"
    )

    axes[0, col].set_xlabel("x*")

    if col == 0:
        axes[0, col].set_ylabel("y*")

    # --------------------------------------------------------
    # Top-40 seed consensus
    # --------------------------------------------------------

    im2 = axes[1, col].imshow(
        seed_map,
        origin="lower",
        extent=[
            xvals.min(),
            xvals.max(),
            yvals.min(),
            yvals.max()
        ],
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest"
    )

    axes[1, col].set_title(
        f"{architecture} — Top-40 seed frequency"
    )

    axes[1, col].set_xlabel("x*")

    if col == 0:
        axes[1, col].set_ylabel("y*")


cbar1 = fig.colorbar(
    im1,
    ax=axes[0, :],
    shrink=0.85
)

cbar1.set_label(
    "Fraction of random seeds classified as interface"
)

cbar2 = fig.colorbar(
    im2,
    ax=axes[1, :],
    shrink=0.85
)

cbar2.set_label(
    "Fraction of random seeds selected in top-40"
)

fig.suptitle(
    "SOM sensitivity — spatial consensus across random initialization",
    fontsize=15
)

fig.savefig(
    OUTPUT,
    dpi=250,
    bbox_inches="tight"
)

plt.close(fig)

print()
print("Consensus figure saved:")
print(OUTPUT)
print()
