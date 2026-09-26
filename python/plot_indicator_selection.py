from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
PLOTS = PROJECT / "results" / "plots"

PLOTS.mkdir(parents=True, exist_ok=True)

dataset = pd.read_csv(DATA / "M40_som_dataset.csv")
som = pd.read_csv(DATA / "M40_som_results.csv")

if len(dataset) != 1600 or len(som) != 1600:
    raise RuntimeError("Expected exactly 1600 M40 cells.")

# Controlled 10% budget
n_select = 160

# Gradient selection
grad_idx = set(
    dataset.nlargest(n_select, "gradU_mag_star").index
)

# SOM-QE selection
som_idx = set(
    som.nlargest(n_select, "quantization_error").index
)

both = grad_idx & som_idx
grad_only = grad_idx - som_idx
som_only = som_idx - grad_idx
neither = set(range(1600)) - (grad_idx | som_idx)

print("=" * 70)
print("10% SELECTION OVERLAP")
print("=" * 70)
print(f"Gradient selected : {len(grad_idx)}")
print(f"SOM-QE selected   : {len(som_idx)}")
print(f"Both              : {len(both)}")
print(f"Gradient only     : {len(grad_only)}")
print(f"SOM-QE only       : {len(som_only)}")
print(f"Neither           : {len(neither)}")
print(f"Union             : {len(grad_idx | som_idx)}")

# Build plotting dataframe using the physical cell-centre coordinates.
plot_df = dataset[["x_star", "y_star"]].copy()
plot_df["category"] = "Neither"

plot_df.loc[list(grad_only), "category"] = "Gradient only"
plot_df.loc[list(som_only), "category"] = "SOM-QE only"
plot_df.loc[list(both), "category"] = "Both"

# Plot each category separately so the legend is clear.
plt.figure(figsize=(7, 7))

categories = [
    ("Neither", ".", 12),
    ("Gradient only", "s", 32),
    ("SOM-QE only", "^", 38),
    ("Both", "o", 42),
]

for category, marker, size in categories:
    part = plot_df[plot_df["category"] == category]

    plt.scatter(
        part["x_star"],
        part["y_star"],
        s=size,
        marker=marker,
        label=f"{category} ({len(part)})",
        alpha=0.8,
    )

plt.xlabel("x/L")
plt.ylabel("y/L")
plt.title("Gradient vs SOM-QE cell selection — 10% budget")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(True, alpha=0.2)
plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0.0,
)
plt.tight_layout(rect=[0, 0, 0.78, 1])
plt.tight_layout()

output = PLOTS / "Gradient_vs_SOM_selection_10.png"

plt.savefig(output, dpi=220)
plt.close()

print()
print(f"Saved: {output}")
