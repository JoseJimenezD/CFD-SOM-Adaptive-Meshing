from pathlib import Path
import shutil
import subprocess
import re

from framework.config import BASE_CASE
from framework.indicators import (
    select_gradient_cells,
    select_som_qe_cells,
    select_hybrid_cells,
    select_som_interface_cells,
    write_toposet_dict,
)


REFINE_DICT_SOURCE = (
    BASE_CASE.parent
    / "Cavity_SOM_005"
    / "system"
    / "refineMeshDict"
)

BASE_CELLS = 1600
CHILDREN_PER_PARENT = 4


def run_command(command, case_dir):
    result = subprocess.run(
        command,
        cwd=case_dir,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(
            f"Command failed: {' '.join(command)}"
        )

    return result.stdout


def read_cell_count(case_dir):
    output = run_command(["checkMesh"], case_dir)

    match = re.search(
        r"^\s*cells:\s+(\d+)",
        output,
        re.MULTILINE,
    )

    if not match:
        raise RuntimeError(
            "Could not determine cell count."
        )

    return int(match.group(1)), output


def select_cells(method, budget):

    if method == "gradient":
        return select_gradient_cells(budget)

    if method == "som_qe":
        return select_som_qe_cells(budget)

    if method == "hybrid":
        return select_hybrid_cells(
            budget,
            alpha=0.5,
        )

    if method == "som_interface_1n":
        return select_som_interface_cells(
            radius=1,
            n_seeds=40,
        )

    if method == "som_interface_2n":
        return select_som_interface_cells(
            radius=2,
            n_seeds=40,
        )

    raise RuntimeError(
        f"Unsupported refinement method: {method}"
    )


def mesh_experiment(case_dir, method, budget):

    case_dir = Path(case_dir)

    # ---------------------------------------------------------
    # SAFETY CHECK:
    # Case must still contain untouched M40 mesh.
    # ---------------------------------------------------------

    n_before, _ = read_cell_count(case_dir)

    if n_before != BASE_CELLS:
        raise RuntimeError(
            f"SAFETY STOP: expected {BASE_CELLS} cells "
            f"before refinement, but found {n_before}. "
            "The case may already have been refined."
        )

    # ---------------------------------------------------------
    # Select cells using ONLY M40 information.
    # ---------------------------------------------------------

    selected = select_cells(
        method,
        budget,
    )

    n_selected = len(selected)

    expected_cells = (
        BASE_CELLS
        - n_selected
        + CHILDREN_PER_PARENT * n_selected
    )

    print(f"Method             : {method}")
    print(f"Initial cells      : {n_before}")
    print(f"Selected parents   : {n_selected}")
    print(f"Expected cells     : {expected_cells}")

    # ---------------------------------------------------------
    # Generate topoSetDict.
    # ---------------------------------------------------------

    topo_dict = write_toposet_dict(
        selected,
        case_dir,
    )

    print(f"topoSetDict        : {topo_dict}")

    # ---------------------------------------------------------
    # Run topoSet.
    # ---------------------------------------------------------

    topo_output = run_command(
        ["topoSet"],
        case_dir,
    )

    match = re.search(
        r"cellSet cellsToRefine now size\s+(\d+)",
        topo_output,
    )

    if not match:
        raise RuntimeError(
            "Could not verify cellsToRefine size."
        )

    topo_count = int(match.group(1))

    if topo_count != n_selected:
        raise RuntimeError(
            f"SAFETY STOP: requested {n_selected} cells "
            f"but topoSet created {topo_count}."
        )

    print(f"topoSet cells      : {topo_count}")

    # ---------------------------------------------------------
    # Install validated refineMeshDict.
    # ---------------------------------------------------------

    destination = (
        case_dir
        / "system"
        / "refineMeshDict"
    )

    shutil.copy2(
        REFINE_DICT_SOURCE,
        destination,
    )

    # ---------------------------------------------------------
    # Refine exactly ONCE.
    # ---------------------------------------------------------

    run_command(
        ["refineMesh", "-overwrite"],
        case_dir,
    )

    # ---------------------------------------------------------
    # Validate resulting mesh.
    # ---------------------------------------------------------

    n_after, check_output = read_cell_count(
        case_dir
    )

    if n_after != expected_cells:
        raise RuntimeError(
            f"SAFETY STOP: expected {expected_cells} cells "
            f"after refinement, but found {n_after}."
        )

    if "Mesh OK." not in check_output:
        raise RuntimeError(
            "SAFETY STOP: checkMesh did not report Mesh OK."
        )

    print(f"Final cells        : {n_after}")
    print("Mesh status        : OK")

    return {
        "method": method,
        "initial_cells": n_before,
        "selected_cells": n_selected,
        "final_cells": n_after,
        "mesh_ok": True,
    }


# Backward-compatible function.
def mesh_gradient_experiment(case_dir, budget):
    return mesh_experiment(
        case_dir,
        "gradient",
        budget,
    )
