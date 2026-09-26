from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT
    / "data"
    / "M40_som_results.csv"
)

OUTPUT_DIR = PROJECT / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

print("Cells loaded:", len(df))

# ============================================================
# SPATIAL BMU MAP
# ============================================================

plt.figure(figsize=(8, 7))

sc = plt.scatter(
    df["x_star"],
    df["y_star"],
    c=df["bmu_id"],
    s=55,
    marker="s",
    cmap="tab20",
    vmin=-0.5,
    vmax=15.5
)

cbar = plt.colorbar(
    sc,
    ticks=range(16)
)

cbar.set_label("SOM neuron / BMU")

plt.xlabel("x/L")
plt.ylabel("y/L")

plt.title(
    "Kohonen SOM classification - M40 cavity\n"
    "Physics-only features"
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.gca().set_aspect("equal")

plt.tight_layout()

output = OUTPUT_DIR / "M40_som_clusters.png"

plt.savefig(
    output,
    dpi=200
)

plt.close()

print("Saved:", output)


# ============================================================
# QUANTIZATION ERROR MAP
# ============================================================

plt.figure(figsize=(8, 7))

sc = plt.scatter(
    df["x_star"],
    df["y_star"],
    c=df["quantization_error"],
    s=55,
    marker="s"
)

plt.colorbar(
    sc,
    label="Quantization error"
)

plt.xlabel("x/L")
plt.ylabel("y/L")

plt.title(
    "SOM quantization error - M40 cavity"
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.gca().set_aspect("equal")

plt.tight_layout()

output = OUTPUT_DIR / "M40_som_quantization_error.png"

plt.savefig(
    output,
    dpi=200
)

plt.close()

print("Saved:", output)
