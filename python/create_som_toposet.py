from pathlib import Path
import pandas as pd

# ============================================================
# Configuration
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

RESULTS_FILE = PROJECT / "data" / "M40_som_results.csv"

CASE = PROJECT / "cases" / "Cavity_SOM_005"
OUTPUT_FILE = CASE / "system" / "topoSetDict"

BUDGET_FRACTION = 0.10

# Physical dimensions used to nondimensionalize the cavity
L = 0.1
Z_CENTER = 0.005


# ============================================================
# Read SOM results
# ============================================================

df = pd.read_csv(RESULTS_FILE)

required_columns = [
    "x_star",
    "y_star",
    "quantization_error",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise RuntimeError(
        f"Missing required columns in {RESULTS_FILE}: {missing}"
    )

n_cells = len(df)

n_select = int(round(BUDGET_FRACTION * n_cells))

if n_select <= 0:
    raise RuntimeError("Number of selected cells is zero.")


# ============================================================
# Select cells using ONLY SOM quantization error
# ============================================================

selected = (
    df
    .sort_values("quantization_error", ascending=False)
    .head(n_select)
    .copy()
)

# Convert nondimensional coordinates back to physical coordinates
selected["x"] = selected["x_star"] * L
selected["y"] = selected["y_star"] * L
selected["z"] = Z_CENTER


# ============================================================
# Build OpenFOAM topoSetDict
# ============================================================

lines = []

lines.append(
"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2606                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/

FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}

actions
(
    {
        name    cellsToRefine;
        type    cellSet;
        action  new;
        source  nearestToCell;

        points
        (
"""
)

for _, row in selected.iterrows():

    x = row["x"]
    y = row["y"]
    z = row["z"]

    lines.append(
        f"            ({x:.10f} {y:.10f} {z:.10f})\n"
    )

lines.append(
"""        );
    }
);

// ************************************************************************* //
"""
)

OUTPUT_FILE.write_text("".join(lines))


# ============================================================
# Report
# ============================================================

print("========================================")
print(" SOM -> topoSetDict")
print("========================================")
print()
print(f"Total M40 cells       : {n_cells}")
print(f"Budget fraction       : {BUDGET_FRACTION:.1%}")
print(f"Requested SOM cells   : {n_select}")
print()
print("Quantization-error range selected:")
print(
    f"min QE = {selected['quantization_error'].min():.8e}"
)
print(
    f"max QE = {selected['quantization_error'].max():.8e}"
)
print()
print("Output:")
print(OUTPUT_FILE)
print()
print("IMPORTANT:")
print("Selection uses SOM quantization error only.")
print("M160 comparison data are NOT used.")
