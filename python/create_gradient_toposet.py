from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT / "data" / "M40_som_dataset.csv"
CASE = PROJECT / "cases" / "Cavity_SOM_006"
OUTPUT_FILE = CASE / "system" / "topoSetDict"

BUDGET_FRACTION = 0.10

L = 0.1
Z_CENTER = 0.005

df = pd.read_csv(DATA_FILE)

required = ["x_star", "y_star", "gradU_mag_star"]

for col in required:
    if col not in df.columns:
        raise RuntimeError(f"Missing column: {col}")

n_total = len(df)
n_select = round(BUDGET_FRACTION * n_total)

selected = (
    df.sort_values("gradU_mag_star", ascending=False)
      .head(n_select)
      .copy()
)

selected["x"] = selected["x_star"] * L
selected["y"] = selected["y_star"] * L

lines = []

lines.append("FoamFile")
lines.append("{")
lines.append("    version     2.0;")
lines.append("    format      ascii;")
lines.append("    class       dictionary;")
lines.append("    object      topoSetDict;")
lines.append("}")
lines.append("")
lines.append("actions")
lines.append("(")
lines.append("    {")
lines.append("        name    cellsToRefine;")
lines.append("        type    cellSet;")
lines.append("        action  new;")
lines.append("        source  nearestToCell;")
lines.append("        points")
lines.append("        (")

for _, row in selected.iterrows():
    lines.append(
        f"            ({row['x']:.10f} {row['y']:.10f} {Z_CENTER:.10f})"
    )

lines.append("        );")
lines.append("    }")
lines.append(");")
lines.append("")
lines.append("// ************************************************************************* //")

OUTPUT_FILE.write_text("\n".join(lines))

print()
print("========================================")
print(" GRADIENT -> topoSetDict")
print("========================================")
print()

print(f"Total M40 cells       : {n_total}")
print(f"Budget fraction       : {BUDGET_FRACTION*100:.1f}%")
print(f"Requested grad cells  : {n_select}")
print()

print("Velocity-gradient range selected:")
print(f"min |gradU|* = {selected['gradU_mag_star'].min():.8e}")
print(f"max |gradU|* = {selected['gradU_mag_star'].max():.8e}")
print()

print("Output:")
print(OUTPUT_FILE)
print()
print("Selection uses M40 velocity-gradient magnitude only.")
print("M160 comparison data are NOT used.")
