from pathlib import Path
import re
import subprocess

from framework.config import END_TIME


def run_check_mesh(case_dir):
    """
    Run checkMesh without modifying the case.
    Returns the complete checkMesh output.
    """
    case_dir = Path(case_dir)

    result = subprocess.run(
        ["checkMesh"],
        cwd=case_dir,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"checkMesh failed for: {case_dir}\n"
            f"{result.stdout}\n"
            f"{result.stderr}"
        )

    return result.stdout


def get_cell_count(case_dir):
    """
    Return the current number of cells in the case.
    """
    output = run_check_mesh(case_dir)

    match = re.search(
        r"^\s*cells:\s+(\d+)",
        output,
        re.MULTILINE,
    )

    if not match:
        raise RuntimeError(
            f"Could not determine cell count for: {case_dir}"
        )

    return int(match.group(1))


def mesh_is_ok(case_dir):
    """
    Check whether OpenFOAM reports Mesh OK.
    """
    output = run_check_mesh(case_dir)

    return "Mesh OK." in output


def final_time_directory(case_dir):
    """
    Return the expected final-time directory.
    Example: END_TIME=5.0 -> case/5
    """
    case_dir = Path(case_dir)

    return case_dir / f"{END_TIME:g}"


def solver_is_complete(case_dir):
    """
    Determine whether the CFD solution reached END_TIME.

    A final-time directory alone is not considered sufficient.
    If log.icoFoam exists, verify that the solver log reached
    the configured END_TIME.
    """
    case_dir = Path(case_dir)
    final_dir = final_time_directory(case_dir)

    if not final_dir.exists():
        return False

    log_file = case_dir / "log.icoFoam"

    if not log_file.exists():
        return False

    text = log_file.read_text(
        errors="ignore"
    )

    times = re.findall(
        r"^Time = ([0-9.eE+-]+)",
        text,
        re.MULTILINE,
    )

    if not times:
        return False

    final_logged_time = float(times[-1])

    return abs(final_logged_time - END_TIME) < 1.0e-12


def postprocessing_is_complete(case_dir):
    """
    Validation requires:
        final-time/C
        final-time/U
        final-time/V
    """
    final_dir = final_time_directory(case_dir)

    required = [
        final_dir / "C",
        final_dir / "U",
        final_dir / "V",
    ]

    return all(path.exists() for path in required)


def detect_case_state(case_dir):
    """
    Classify an experiment case.

    Possible states:
        NEW
        INVALID_MESH
        MESHED
        SOLVED
        POSTPROCESSED

    NEW means that the experiment directory does not exist.
    """
    case_dir = Path(case_dir)

    if not case_dir.exists():
        return {
            "state": "NEW",
            "exists": False,
            "cells": None,
            "mesh_ok": None,
            "solver_complete": False,
            "postprocessed": False,
        }

    cells = get_cell_count(case_dir)
    mesh_ok = mesh_is_ok(case_dir)

    if not mesh_ok:
        return {
            "state": "INVALID_MESH",
            "exists": True,
            "cells": cells,
            "mesh_ok": False,
            "solver_complete": False,
            "postprocessed": False,
        }

    solved = solver_is_complete(case_dir)
    postprocessed = postprocessing_is_complete(case_dir)

    if solved and postprocessed:
        state = "POSTPROCESSED"

    elif solved:
        state = "SOLVED"

    else:
        state = "MESHED"

    return {
        "state": state,
        "exists": True,
        "cells": cells,
        "mesh_ok": mesh_ok,
        "solver_complete": solved,
        "postprocessed": postprocessed,
    }
