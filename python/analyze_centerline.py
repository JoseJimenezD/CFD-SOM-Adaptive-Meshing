import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Case parameters
# -----------------------------
L = 0.1          # cavity length [m]
U_lid = 1.0      # lid velocity [m/s]

# -----------------------------
# Read OpenFOAM sampled data
# Columns: y, Ux, Uy, Uz
# -----------------------------
file_path = "postProcessing/sampleDict/5/verticalCenterline_U.xy"

data = np.loadtxt(file_path)

y  = data[:, 0]
Ux = data[:, 1]
Uy = data[:, 2]

# -----------------------------
# Non-dimensional variables
# -----------------------------
y_star = y / L
Ux_star = Ux / U_lid

# -----------------------------
# Basic information
# -----------------------------
print("Number of sampled points:", len(y))
print("Minimum Ux =", Ux.min(), "m/s")
print("Maximum Ux =", Ux.max(), "m/s")

i_min = np.argmin(Ux)

print("Location of minimum Ux:")
print("y =", y[i_min], "m")
print("y/L =", y_star[i_min])

# -----------------------------
# Plot
# -----------------------------
plt.figure()

plt.plot(Ux_star, y_star, "o-", markersize=3)

plt.xlabel("Ux / U_lid")
plt.ylabel("y / L")
plt.title("Lid-Driven Cavity - Re = 100 - Mesh 20x20")

plt.grid(True)

plt.tight_layout()
plt.savefig("python/vertical_centerline_M20.png", dpi=300)

plt.show()
