from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Project configuration
# --------------------------------------------------
project = Path(__file__).resolve().parents[1]
cases_dir = project / "cases"

cases = {
    "M20":  cases_dir / "Cavity_SOM_001",
    "M40":  cases_dir / "Cavity_SOM_002",
    "M80":  cases_dir / "Cavity_SOM_003",
    "M160": cases_dir / "Cavity_SOM_004",
}

L = 0.1
U_lid = 1.0
time = "5"

results = {}

# --------------------------------------------------
# Read each mesh
# --------------------------------------------------
for name, case in cases.items():

    file_path = (
        case
        / "postProcessing"
        / "sampleDict"
        / time
        / "verticalCenterline_U.xy"
    )

    data = np.loadtxt(file_path)

    y = data[:, 0]
    Ux = data[:, 1]

    y_star = y / L
    Ux_star = Ux / U_lid

    i_min = np.argmin(Ux)

    results[name] = {
        "y": y_star,
        "Ux": Ux_star,
        "Ux_min": Ux_star[i_min],
        "y_min": y_star[i_min],
    }

# --------------------------------------------------
# Numerical comparison M20 vs M40
# --------------------------------------------------
diff_20_40   = results["M20"]["Ux"]  - results["M40"]["Ux"]
diff_40_80   = results["M40"]["Ux"]  - results["M80"]["Ux"]
diff_80_160  = results["M80"]["Ux"]  - results["M160"]["Ux"]

L2_20_40  = np.sqrt(np.mean(diff_20_40**2))
L2_40_80  = np.sqrt(np.mean(diff_40_80**2))
L2_80_160 = np.sqrt(np.mean(diff_80_160**2))

Linf_20_40  = np.max(np.abs(diff_20_40))
Linf_40_80  = np.max(np.abs(diff_40_80))
Linf_80_160 = np.max(np.abs(diff_80_160))

print()
print("===== MESH COMPARISON =====")
print()

for name, r in results.items():
    print(
        f"{name}: "
        f"Ux_min = {r['Ux_min']:.8f}, "
        f"y/L = {r['y_min']:.4f}"
    )

print()
print("M20 vs M40:")
print(f"L2 difference   = {L2_20_40:.8e}")
print(f"Linf difference = {Linf_20_40:.8e}")

print()
print("M40 vs M80:")
print(f"L2 difference   = {L2_40_80:.8e}")
print(f"Linf difference = {Linf_40_80:.8e}")

print()
print("M80 vs M160:")
print(f"L2 difference   = {L2_80_160:.8e}")
print(f"Linf difference = {Linf_80_160:.8e}")

r1 = L2_20_40 / L2_40_80
r2 = L2_40_80 / L2_80_160

p1 = np.log(r1) / np.log(2)
p2 = np.log(r2) / np.log(2)

print()
print("L2 convergence:")
print(f"E20-40 / E40-80   = {r1:.6f}")
print(f"E40-80 / E80-160  = {r2:.6f}")
print(f"Observed p (20-40-80)  = {p1:.6f}")
print(f"Observed p (40-80-160) = {p2:.6f}")

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure()

for name, data in results.items():
    plt.plot(
        data["Ux"],
        data["y"],
        label=name
    )

plt.xlabel("Ux / U_lid")
plt.ylabel("y / L")
plt.title("Lid-Driven Cavity - Re = 100")
plt.grid(True)
plt.legend()
plt.tight_layout()

output = project / "mesh_comparison.png"
plt.savefig(output, dpi=300)

print()
print("Figure saved to:")
print(output)

plt.show()
