from pathlib import Path
import shutil
import subprocess
import re

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

TEMPLATE_CASE = PROJECT / "cases" / "Cavity_SOM_003"
NEW_CASE = PROJECT / "cases" / "Cavity_SOM_004"

N = 160

L = 0.1
U_LID = 1.0
END_TIME = 5.0

# Keep approximately the same Courant number used previously
DT = 0.005 * 20 / N

# Write every 0.1 physical seconds
WRITE_INTERVAL = round(0.1 / DT)

print("========================================")
print(" AUTOMATIC CFD MESH STUDY")
print("========================================")
print(f"Mesh             : {N} x {N}")
print(f"Expected cells   : {N*N}")
print(f"deltaT           : {DT}")
print(f"writeInterval    : {WRITE_INTERVAL}")
print(f"endTime          : {END_TIME}")
print()


# ============================================================
# CREATE CLEAN CASE
# ============================================================

if NEW_CASE.exists():
    raise RuntimeError(
        f"{NEW_CASE} already exists.\n"
        "Delete it manually if you really want to recreate it."
    )

print("Creating new case...")

shutil.copytree(TEMPLATE_CASE, NEW_CASE)


# ============================================================
# REMOVE OLD RESULTS
# ============================================================

print("Cleaning inherited results...")

for item in NEW_CASE.iterdir():

    if item.is_dir():

        # Preserve initial condition directory 0
        if item.name == "0":
            continue

        # Remove numerical time directories such as 0.1, 1, 5
        try:
            float(item.name)
            shutil.rmtree(item)
        except ValueError:
            pass

# Remove inherited post-processing
post = NEW_CASE / "postProcessing"

if post.exists():
    shutil.rmtree(post)

# Remove inherited logs
for log in NEW_CASE.glob("log.*"):
    log.unlink()


# ============================================================
# MODIFY blockMeshDict
# ============================================================

block_file = NEW_CASE / "system" / "blockMeshDict"

text = block_file.read_text()

text, count = re.subn(
    r"\(\s*80\s+80\s+1\s*\)",
    f"({N} {N} 1)",
    text
)

if count != 1:
    raise RuntimeError(
        "Could not uniquely replace the mesh dimensions "
        "in blockMeshDict."
    )

block_file.write_text(text)


# ============================================================
# MODIFY controlDict
# ============================================================

control_file = NEW_CASE / "system" / "controlDict"

text = control_file.read_text()

text = re.sub(
    r"startFrom\s+\w+;",
    "startFrom       startTime;",
    text
)

text = re.sub(
    r"startTime\s+[^;]+;",
    "startTime       0;",
    text
)

text = re.sub(
    r"endTime\s+[^;]+;",
    f"endTime         {END_TIME};",
    text
)

text = re.sub(
    r"deltaT\s+[^;]+;",
    f"deltaT          {DT};",
    text
)

text = re.sub(
    r"writeInterval\s+[^;]+;",
    f"writeInterval   {WRITE_INTERVAL};",
    text
)

control_file.write_text(text)


# ============================================================
# RUN COMMAND HELPER
# ============================================================

def run(command, logfile=None):

    print()
    print("Running:", " ".join(command))

    if logfile is None:

        subprocess.run(
            command,
            cwd=NEW_CASE,
            check=True
        )

    else:

        log_path = NEW_CASE / logfile

        with log_path.open("w") as f:

            subprocess.run(
                command,
                cwd=NEW_CASE,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True
            )


# ============================================================
# GENERATE MESH
# ============================================================

run(["blockMesh"], "log.blockMesh")


# ============================================================
# CHECK MESH
# ============================================================

run(["checkMesh"], "log.checkMesh")

check_text = (NEW_CASE / "log.checkMesh").read_text()

if "Mesh OK." not in check_text:

    raise RuntimeError(
        "Mesh quality check FAILED. CFD will NOT be executed."
    )

print()
print("Mesh quality: OK")


# ============================================================
# RUN CFD
# ============================================================

run(["icoFoam"], "log.M160")


# ============================================================
# POST-PROCESS CENTERLINE
# ============================================================

run(
    ["postProcess", "-func", "sampleDict", "-latestTime"],
    "log.postProcess"
)


# ============================================================
# FINISHED
# ============================================================

print()
print("========================================")
print(" CFD CASE COMPLETED SUCCESSFULLY")
print("========================================")
print()
print("Case:")
print(NEW_CASE)
print()
print("Centerline:")
print(
    NEW_CASE
    / "postProcessing"
    / "sampleDict"
    / "5"
    / "verticalCenterline_U.xy"
)
