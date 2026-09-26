import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import ListedColormap

PROJECT = Path(__file__).resolve().parents[1]

som = pd.read_csv(PROJECT / "data" / "M40_som_results.csv")
sel = pd.read_csv(PROJECT / "results" / "M40_som_interface_fixed_seeds.csv")

OUT = PROJECT / "results" / "plots" / "SOM_regions_interfaces_audit.png"

xs = np.sort(som["x_star"].unique())
ys = np.sort(som["y_star"].unique())

nx = len(xs)
ny = len(ys)

xmap = {x:i for i,x in enumerate(xs)}
ymap = {y:j for j,y in enumerate(ys)}

grid = np.empty((ny,nx), dtype=int)

for idx,row in som.iterrows():
    grid[ymap[row["y_star"]], xmap[row["x_star"]]] = idx


def neighbours(idx):
    row = som.loc[idx]
    ix = xmap[row["x_star"]]
    iy = ymap[row["y_star"]]

    out = []

    if ix > 0:
        out.append(grid[iy,ix-1])
    if ix < nx-1:
        out.append(grid[iy,ix+1])
    if iy > 0:
        out.append(grid[iy-1,ix])
    if iy < ny-1:
        out.append(grid[iy+1,ix])

    return out


# ----------------------------------------------------------
# Original BMU map
# ----------------------------------------------------------

bmu_map = np.zeros((ny,nx), dtype=int)

for iy in range(ny):
    for ix in range(nx):
        idx = grid[iy,ix]
        bmu_map[iy,ix] = int(som.loc[idx,"bmu_id"])


# ----------------------------------------------------------
# Interface definition:
# a cell is interface if ANY face neighbour has another BMU
# ----------------------------------------------------------

interface = np.zeros(len(som), dtype=bool)

for i in range(len(som)):
    bi = int(som.loc[i,"bmu_id"])

    for j in neighbours(i):
        if int(som.loc[j,"bmu_id"]) != bi:
            interface[i] = True
            break


interface_map = np.zeros((ny,nx), dtype=int)

for iy in range(ny):
    for ix in range(nx):
        idx = grid[iy,ix]
        interface_map[iy,ix] = int(interface[idx])


# ----------------------------------------------------------
# Current frozen seeds
# ----------------------------------------------------------

seed = sel["interface_seed"].to_numpy(dtype=bool)

seed_map = np.zeros((ny,nx), dtype=int)

for iy in range(ny):
    for ix in range(nx):
        idx = grid[iy,ix]

        if interface[idx]:
            seed_map[iy,ix] = 1

        if seed[idx]:
            seed_map[iy,ix] = 2


print("===================================")
print("SOM INTERFACE AUDIT")
print("===================================")
print("Cells                 :", len(som))
print("Unique BMUs           :", som["bmu_id"].nunique())
print("Interface cells       :", interface.sum())
print("Interface fraction    :", f"{100*interface.mean():.2f}%")
print("Current seeds         :", seed.sum())
print(
    "Seeds on interface    :",
    np.sum(seed & interface),
    "/",
    seed.sum()
)


# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

fig,axes = plt.subplots(
    1,3,
    figsize=(16,5),
    constrained_layout=True
)

im = axes[0].imshow(
    bmu_map,
    origin="lower",
    extent=[0,1,0,1],
    interpolation="nearest",
    cmap="tab20",
    aspect="equal"
)

axes[0].set_title("Original SOM regions — BMU ID")
axes[0].set_xlabel(r"$x/L$")
axes[0].set_ylabel(r"$y/L$")

cbar = fig.colorbar(im, ax=axes[0], fraction=0.046)
cbar.set_label("BMU ID")


axes[1].imshow(
    interface_map,
    origin="lower",
    extent=[0,1,0,1],
    interpolation="nearest",
    cmap=ListedColormap(["#eeeeee","#222222"]),
    vmin=0,
    vmax=1,
    aspect="equal"
)

axes[1].set_title("All BMU interfaces")
axes[1].set_xlabel(r"$x/L$")
axes[1].set_ylabel(r"$y/L$")


axes[2].imshow(
    seed_map,
    origin="lower",
    extent=[0,1,0,1],
    interpolation="nearest",
    cmap=ListedColormap([
        "#eeeeee",
        "#9e9e9e",
        "#d95f02"
    ]),
    vmin=0,
    vmax=2,
    aspect="equal"
)

axes[2].set_title("Interfaces + current 40 seeds")
axes[2].set_xlabel(r"$x/L$")
axes[2].set_ylabel(r"$y/L$")

OUT.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUT,
    dpi=300,
    bbox_inches="tight"
)

print()
print("Saved:", OUT)
