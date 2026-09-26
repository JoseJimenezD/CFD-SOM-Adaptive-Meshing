from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from framework.config import (
    REFERENCE_CASE,
    RESULTS_DIR,
    L,
    U_LID,
    END_TIME,
)


def read_vectors(path):
    text = Path(path).read_text()

    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+\d+\s*\((.*?)\)\s*;",
        text,
        re.S,
    )

    if match is None:
        raise RuntimeError(
            f"Could not read vector field: {path}"
        )

    values = re.findall(
        r"\(\s*([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s*\)",
        match.group(1),
    )

    return np.asarray(values, dtype=float)


def read_scalars(path):
    text = Path(path).read_text()

    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)\s*;",
        text,
        re.S,
    )

    if match is None:
        raise RuntimeError(
            f"Could not read scalar field: {path}"
        )

    values = re.findall(
        r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?",
        match.group(1),
    )

    return np.asarray(values, dtype=float)


def build_reference_interpolators():
    """
    Read the M160 reference and construct velocity interpolators.

    IMPORTANT:
    M160 is used only for validation, never for cell selection.
    """

    time_name = f"{END_TIME:g}"
    ref_time = REFERENCE_CASE / time_name

    C_ref = read_vectors(ref_time / "C")
    U_ref = read_vectors(ref_time / "U")

    if len(C_ref) != len(U_ref):
        raise RuntimeError(
            "M160 C/U sizes do not match."
        )

    x_ref = np.unique(C_ref[:, 0])
    y_ref = np.unique(C_ref[:, 1])

    nx = len(x_ref)
    ny = len(y_ref)

    if nx * ny != len(C_ref):
        raise RuntimeError(
            "M160 reference is not recognized as structured: "
            f"nx={nx}, ny={ny}, cells={len(C_ref)}"
        )

    Ux_grid = np.empty((nx, ny))
    Uy_grid = np.empty((nx, ny))

    x_index = {
        round(x, 12): i
        for i, x in enumerate(x_ref)
    }

    y_index = {
        round(y, 12): j
        for j, y in enumerate(y_ref)
    }

    for c, u in zip(C_ref, U_ref):
        i = x_index[round(c[0], 12)]
        j = y_index[round(c[1], 12)]

        Ux_grid[i, j] = u[0]
        Uy_grid[i, j] = u[1]

    interp_Ux = RegularGridInterpolator(
        (x_ref, y_ref),
        Ux_grid,
        bounds_error=False,
        fill_value=None,
    )

    interp_Uy = RegularGridInterpolator(
        (x_ref, y_ref),
        Uy_grid,
        bounds_error=False,
        fill_value=None,
    )

    return interp_Ux, interp_Uy, len(C_ref)


def validate_against_reference(case_dir):
    """
    Compare an adaptive experiment against the M160 reference.

    Returns volume-weighted velocity-difference statistics.
    """

    case_dir = Path(case_dir)
    time_name = f"{END_TIME:g}"
    case_time = case_dir / time_name

    required = [
        case_time / "C",
        case_time / "U",
        case_time / "V",
    ]

    for path in required:
        if not path.exists():
            raise RuntimeError(
                f"Required validation field not found: {path}"
            )

    C = read_vectors(case_time / "C")
    U = read_vectors(case_time / "U")
    V = read_scalars(case_time / "V")

    if not (len(C) == len(U) == len(V)):
        raise RuntimeError(
            "Experiment C/U/V sizes do not match."
        )

    interp_Ux, interp_Uy, n_reference = (
        build_reference_interpolators()
    )

    points = C[:, :2]

    Ux_ref = interp_Ux(points)
    Uy_ref = interp_Uy(points)

    dUx = U[:, 0] - Ux_ref
    dUy = U[:, 1] - Uy_ref

    dU = np.sqrt(
        dUx**2 + dUy**2
    )

    dU_star = dU / U_LID

    Vtotal = np.sum(V)

    mean_v = (
        np.sum(V * dU_star)
        / Vtotal
    )

    rms_v = np.sqrt(
        np.sum(V * dU_star**2)
        / Vtotal
    )

    max_error = np.max(dU_star)
    imax = np.argmax(dU_star)

    experiment_name = case_dir.name

    output_dir = RESULTS_DIR / "field_comparisons"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / f"{experiment_name}_vs_M160.csv"
    )

    df = pd.DataFrame(
        {
            "x": C[:, 0],
            "y": C[:, 1],
            "z": C[:, 2],
            "V": V,
            "Ux": U[:, 0],
            "Uy": U[:, 1],
            "Ux_M160_interp": Ux_ref,
            "Uy_M160_interp": Uy_ref,
            "dUx": dUx,
            "dUy": dUy,
            "dU_mag_star": dU_star,
        }
    )

    df.to_csv(
        output_file,
        index=False,
    )

    result = {
        "cells": len(C),
        "reference_cells": n_reference,
        "total_volume": float(Vtotal),
        "mean_v_error": float(mean_v),
        "rms_v_error": float(rms_v),
        "max_error": float(max_error),
        "max_error_x_over_L": float(
            C[imax, 0] / L
        ),
        "max_error_y_over_L": float(
            C[imax, 1] / L
        ),
        "min_cell_volume": float(V.min()),
        "max_cell_volume": float(V.max()),
        "comparison_file": str(output_file),
    }

    return result
