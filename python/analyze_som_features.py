from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT / "data" / "M40_som_dataset.csv"

# Read dataset
df = pd.read_csv(DATA_FILE)

print("\n========================================")
print(" SOM FEATURE ANALYSIS")
print("========================================")

print("\nDataset shape:")
print(df.shape)

print("\nFeatures:")
for column in df.columns:
    print(" -", column)

# ------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n========================================")
print(" DESCRIPTIVE STATISTICS")
print("========================================")

print(
    df.describe().loc[
        ["mean", "std", "min", "max"]
    ].T
)

# ------------------------------------------------------------
# CORRELATION MATRIX
# ------------------------------------------------------------

corr = df.corr()

print("\n========================================")
print(" CORRELATION MATRIX")
print("========================================")

print(corr.round(3))

# ------------------------------------------------------------
# STRONG CORRELATIONS
# ------------------------------------------------------------

print("\n========================================")
print(" STRONG CORRELATIONS |r| >= 0.80")
print("========================================")

columns = corr.columns

found = False

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        r = corr.iloc[i, j]

        if abs(r) >= 0.80:

            found = True

            print(
                f"{columns[i]:18s} <-> "
                f"{columns[j]:18s} "
                f"r = {r: .4f}"
            )

if not found:
    print("No correlations above threshold.")

print("\nAnalysis completed.")
