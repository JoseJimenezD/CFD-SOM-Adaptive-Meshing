from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT / "data" / "M40_som_dataset.csv"
OUTPUT_DIR = PROJECT / "plots"

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# READ DATASET
# ============================================================

data = np.genfromtxt(
    DATA_FILE,
    delimiter=",",
    names=True
)

x = data["x_star"]
y = data["y_star"]

U_mag = data["U_mag_star"]
gradU = data["gradU_mag_star"]
omega = data["omega_mag_star"]

print("Cells loaded:", len(x))

# ============================================================
# PLOTTING FUNCTION
# ============================================================

def plot_field(values, title, label, filename):

    plt.figure(figsize=(7, 6))

    sc = plt.scatter(
        x,
        y,
        c=values,
        s=55,
        marker="s"
    )

    plt.colorbar(sc, label=label)

    plt.xlabel("x/L")
    plt.ylabel("y/L")

    plt.title(title)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.gca().set_aspect("equal")

    plt.tight_layout()

    output = OUTPUT_DIR / filename

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print("Saved:", output)


# ============================================================
# CREATE FEATURE MAPS
# ============================================================

plot_field(
    U_mag,
    "Velocity magnitude",
    "|U| / U_lid",
    "M40_velocity_magnitude.png"
)

plot_field(
    gradU,
    "Velocity-gradient magnitude",
    "|grad(U)| L / U_lid",
    "M40_velocity_gradient.png"
)

plot_field(
    omega,
    "Vorticity magnitude",
    "|omega| L / U_lid",
    "M40_vorticity.png"
)

print()
print("Feature maps created successfully.")
