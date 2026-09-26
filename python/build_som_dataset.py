from pathlib import Path
import re
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

# Pilot CFD solution used to generate SOM features
CASE = PROJECT / "cases" / "Cavity_SOM_002"       # M40

TIME = "5"

L = 0.1
U_LID = 1.0

time_dir = CASE / TIME

print("========================================")
print(" BUILDING SOM DATASET")
print("========================================")
print(f"Case : {CASE.name}")
print(f"Time : {TIME}")
print()


# ============================================================
# GENERIC OPENFOAM INTERNAL FIELD READER
# ============================================================

def read_internal_field(file_path, n_components):

    text = file_path.read_text()

    pattern = (
        r"internalField\s+nonuniform\s+List<[^>]+>\s*"
        r"(\d+)\s*\(\s*(.*?)\s*\)\s*;"
    )

    match = re.search(pattern, text, re.DOTALL)

    if not match:
        raise RuntimeError(
            f"Could not read internalField from:\n{file_path}"
        )

    n_cells = int(match.group(1))
    field_text = match.group(2)

    if n_components == 1:

        values = np.fromstring(field_text, sep=" ")

        if len(values) != n_cells:
            raise RuntimeError(
                f"{file_path.name}: expected {n_cells} values, "
                f"found {len(values)}"
            )

        return values

    rows = re.findall(r"\(([^()]*)\)", field_text)

    data = np.array(
        [
            [float(v) for v in row.split()]
            for row in rows
        ]
    )

    if data.shape != (n_cells, n_components):
        raise RuntimeError(
            f"{file_path.name}: expected "
            f"({n_cells},{n_components}), found {data.shape}"
        )

    return data


# ============================================================
# READ OPENFOAM FIELDS
# ============================================================

print("Reading OpenFOAM fields...")

C = read_internal_field(
    time_dir / "C",
    3
)

U = read_internal_field(
    time_dir / "U",
    3
)

gradU = read_internal_field(
    time_dir / "grad(U)",
    9
)

vorticity = read_internal_field(
    time_dir / "vorticity",
    3
)


# ============================================================
# VERIFY CELL COUNTS
# ============================================================

n = len(C)

if not (
    len(U) == n
    and len(gradU) == n
    and len(vorticity) == n
):
    raise RuntimeError(
        "The OpenFOAM fields do not contain the same "
        "number of cells."
    )

print(f"Cells read: {n}")


# ============================================================
# BASIC VARIABLES
# ============================================================

x = C[:, 0]
y = C[:, 1]

Ux = U[:, 0]
Uy = U[:, 1]

U_mag = np.linalg.norm(U, axis=1)

# Frobenius norm of velocity-gradient tensor
gradU_mag = np.linalg.norm(gradU, axis=1)

# Magnitude of vorticity vector
omega_mag = np.linalg.norm(vorticity, axis=1)


# ============================================================
# PHYSICAL NON-DIMENSIONALIZATION
# ============================================================

x_star = x / L
y_star = y / L

Ux_star = Ux / U_LID
Uy_star = Uy / U_LID
U_mag_star = U_mag / U_LID

# grad(U) has units 1/s.
# Natural cavity scale = U_lid / L.
#
# Therefore:
#
# gradU* = gradU * L / U_lid
# omega* = omega * L / U_lid

gradU_star = gradU_mag * L / U_LID
omega_star = omega_mag * L / U_LID


# ============================================================
# BUILD PHYSICAL DATASET
# ============================================================

dataset = np.column_stack(
    [
        x_star,
        y_star,
        Ux_star,
        Uy_star,
        U_mag_star,
        gradU_star,
        omega_star,
    ]
)

headers = [
    "x_star",
    "y_star",
    "Ux_star",
    "Uy_star",
    "U_mag_star",
    "gradU_mag_star",
    "omega_mag_star",
]


# ============================================================
# CHECK DATA
# ============================================================

if not np.all(np.isfinite(dataset)):
    raise RuntimeError(
        "Dataset contains NaN or infinite values."
    )


# ============================================================
# SAVE
# ============================================================

output_dir = PROJECT / "data"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "M40_som_dataset.csv"

np.savetxt(
    output_file,
    dataset,
    delimiter=",",
    header=",".join(headers),
    comments="",
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("Dataset shape:")
print(dataset.shape)

print()
print("Feature ranges:")

for i, name in enumerate(headers):

    print(
        f"{name:18s} "
        f"min = {dataset[:, i].min(): .6e}   "
        f"max = {dataset[:, i].max(): .6e}"
    )

print()
print("Dataset saved to:")
print(output_file)

print()
print("========================================")
print(" DATASET CREATED SUCCESSFULLY")
print("========================================")
