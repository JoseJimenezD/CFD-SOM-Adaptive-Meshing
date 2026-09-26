from pathlib import Path
import pandas as pd

from framework.config import DATA_DIR, L


DATASET = DATA_DIR / "M40_som_dataset.csv"


def load_m40_dataset():
    df = pd.read_csv(DATASET)

    required = {
        "x_star",
        "y_star",
        "Ux_star",
        "Uy_star",
        "U_mag_star",
        "gradU_mag_star",
        "omega_mag_star",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns in M40 dataset: {sorted(missing)}"
        )

    if len(df) != 1600:
        raise RuntimeError(
            f"Expected 1600 M40 cells, found {len(df)}"
        )

    return df


def select_gradient_cells(budget):
    """
    Select M40 parent cells using ONLY gradU_mag_star.

    M160/reference information is never used here.
    """

    df = load_m40_dataset()

    n_select = round(len(df) * budget)

    selected = (
        df.nlargest(n_select, "gradU_mag_star")
        .copy()
        .reset_index(drop=True)
    )

    return selected


def write_toposet_dict(selected, case_dir):
    """
    Write OpenFOAM topoSetDict using the physical centres
    of the selected M40 cells.
    """

    case_dir = Path(case_dir)

    output = case_dir / "system" / "topoSetDict"

    lines = [
        "FoamFile",
        "{",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      topoSetDict;",
        "}",
        "",
        "actions",
        "(",
        "    {",
        "        name    cellsToRefine;",
        "        type    cellSet;",
        "        action  new;",
        "        source  nearestToCell;",
        "        points",
        "        (",
    ]

    z = 0.005

    for _, row in selected.iterrows():

        x = row["x_star"] * L
        y = row["y_star"] * L

        lines.append(
            f"            ({x:.12g} {y:.12g} {z:.12g})"
        )

    lines += [
        "        );",
        "    }",
        ");",
        "",
    ]

    output.write_text("\n".join(lines))

    return output



def load_som_results():
    som_file = DATA_DIR / "M40_som_results.csv"

    df = pd.read_csv(som_file)

    required = {
        "x_star",
        "y_star",
        "quantization_error",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns in SOM results: {sorted(missing)}"
        )

    if len(df) != 1600:
        raise RuntimeError(
            f"Expected 1600 M40 SOM cells, found {len(df)}"
        )

    return df


def select_som_qe_cells(budget):
    """
    Select M40 parent cells using ONLY SOM quantization error.

    Higher quantization_error = higher refinement priority.

    M160/reference information is never used here.
    """

    df = load_som_results()

    n_select = round(len(df) * budget)

    selected = (
        df.nlargest(n_select, "quantization_error")
        .copy()
        .reset_index(drop=True)
    )

    return selected


def compute_som_transition():
    """
    Compute the SOM regime-transition indicator T for the structured
    40x40 M40 grid.

    T_i = fraction of direct von-Neumann neighbours
          (left/right/up/down) having a different BMU.

    T ranges from 0 to 1.

    IMPORTANT:
    - Uses only M40 SOM classification.
    - Does NOT use M160/reference information.
    """

    import numpy as np

    df = load_som_results().copy()

    if "bmu_id" not in df.columns:
        raise RuntimeError(
            "M40_som_results.csv does not contain bmu_id"
        )

    # Robust coordinate ordering
    xvals = np.sort(df["x_star"].unique())
    yvals = np.sort(df["y_star"].unique())

    if len(xvals) != 40 or len(yvals) != 40:
        raise RuntimeError(
            f"Expected structured 40x40 M40 grid, "
            f"found {len(xvals)}x{len(yvals)}"
        )

    # Map spatial position -> dataframe index
    lookup = {}

    for idx, row in df.iterrows():
        key = (
            round(float(row["x_star"]), 10),
            round(float(row["y_star"]), 10),
        )
        lookup[key] = idx

    transition = np.zeros(len(df))

    for idx, row in df.iterrows():

        x = float(row["x_star"])
        y = float(row["y_star"])
        bmu = int(row["bmu_id"])

        ix = int(np.argmin(np.abs(xvals - x)))
        iy = int(np.argmin(np.abs(yvals - y)))

        neighbours = []

        if ix > 0:
            neighbours.append((xvals[ix - 1], y))
        if ix < len(xvals) - 1:
            neighbours.append((xvals[ix + 1], y))
        if iy > 0:
            neighbours.append((x, yvals[iy - 1]))
        if iy < len(yvals) - 1:
            neighbours.append((x, yvals[iy + 1]))

        different = 0

        for xn, yn in neighbours:

            key = (
                round(float(xn), 10),
                round(float(yn), 10),
            )

            j = lookup[key]

            if int(df.loc[j, "bmu_id"]) != bmu:
                different += 1

        transition[idx] = different / len(neighbours)

    df["som_transition"] = transition

    return df


def select_hybrid_cells(budget, alpha=0.5):
    """
    HYBRID-v1 refinement indicator.

        I_hybrid = G* (1 + alpha*T_SOM)

    where:

        G*    = gradU_mag_star / max(gradU_mag_star)
        T_SOM = local SOM regime-transition indicator
        alpha = 0.5 (fixed a priori for HYBRID-v1)

    Gradient remains the primary refinement signal.
    SOM transition only modulates its priority.

    IMPORTANT:
    M160/reference information is never used for selection.
    """

    grad = load_m40_dataset().copy()
    som = compute_som_transition().copy()

    # Robust coordinate keys
    for df in (grad, som):
        df["x_key"] = df["x_star"].round(10)
        df["y_key"] = df["y_star"].round(10)

    df = grad.merge(
        som[
            [
                "x_key",
                "y_key",
                "som_transition",
            ]
        ],
        on=["x_key", "y_key"],
        how="inner",
    )

    if len(df) != 1600:
        raise RuntimeError(
            f"Expected 1600 matched M40 cells, found {len(df)}"
        )

    max_grad = df["gradU_mag_star"].max()

    if max_grad <= 0:
        raise RuntimeError(
            "Maximum velocity-gradient magnitude must be positive"
        )

    df["G_normalized"] = (
        df["gradU_mag_star"] / max_grad
    )

    df["hybrid_indicator"] = (
        df["G_normalized"]
        * (1.0 + alpha * df["som_transition"])
    )

    n_select = round(len(df) * budget)

    selected = (
        df.nlargest(n_select, "hybrid_indicator")
        .copy()
        .reset_index(drop=True)
    )

    return selected


def select_som_interface_cells(radius, n_seeds=40):
    """
    SOM-Interface fixed-seed refinement.

    Step 1:
        SOM identifies flow-state regions through BMU classification.

    Step 2:
        A cell is an interface cell when at least one direct
        face-sharing neighbour has a different BMU.

    Step 3:
        Interface cells are ranked using

            I = T * sqrt(G* * dU*)

        where:
            T   = fraction of direct neighbours with different BMU
            G*  = normalized local velocity-gradient magnitude
            dU* = normalized maximum velocity-vector jump across
                  a SOM interface

    Step 4:
        The same n_seeds highest-ranked interface cells are frozen.

    Step 5:
        Refinement is expanded by exactly 'radius' graph layers
        using face-sharing neighbours.

        radius=1 -> seed + one neighbouring layer
        radius=2 -> seed + two neighbouring layers

    M160/reference information is NEVER used.
    """

    import numpy as np
    from collections import deque

    if radius not in (1, 2):
        raise ValueError(
            "SOM-Interface radius must be 1 or 2"
        )

    df = load_som_results().copy()

    required_som = {
        "x_star",
        "y_star",
        "Ux_star",
        "Uy_star",
        "bmu_id",
        "gradU_mag_star",
    }

    missing = required_som - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing SOM columns: {sorted(missing)}"
        )

    if len(df) != 1600:
        raise RuntimeError(
            f"Expected 1600 M40 cells, found {len(df)}"
        )

    # --------------------------------------------------------
    # Reconstruct structured 40x40 grid.
    # --------------------------------------------------------

    xvals = np.sort(df["x_star"].unique())
    yvals = np.sort(df["y_star"].unique())

    if len(xvals) != 40 or len(yvals) != 40:
        raise RuntimeError(
            "Expected structured 40x40 M40 grid"
        )

    x_to_ix = {x: i for i, x in enumerate(xvals)}
    y_to_iy = {y: j for j, y in enumerate(yvals)}

    grid = np.empty((40, 40), dtype=int)

    for idx, row in df.iterrows():
        ix = x_to_ix[row["x_star"]]
        iy = y_to_iy[row["y_star"]]
        grid[iy, ix] = idx

    def direct_neighbours(idx):

        row = df.loc[idx]

        ix = x_to_ix[row["x_star"]]
        iy = y_to_iy[row["y_star"]]

        out = []

        if ix > 0:
            out.append(grid[iy, ix - 1])

        if ix < 39:
            out.append(grid[iy, ix + 1])

        if iy > 0:
            out.append(grid[iy - 1, ix])

        if iy < 39:
            out.append(grid[iy + 1, ix])

        return out

    # --------------------------------------------------------
    # Interface quantities.
    # --------------------------------------------------------

    n = len(df)

    transition = np.zeros(n)
    velocity_jump = np.zeros(n)
    interface = np.zeros(n, dtype=bool)

    for i in range(n):

        neigh = direct_neighbours(i)

        different = [
            j for j in neigh
            if int(df.loc[j, "bmu_id"])
            != int(df.loc[i, "bmu_id"])
        ]

        if not different:
            continue

        interface[i] = True

        transition[i] = (
            len(different) / len(neigh)
        )

        jumps = []

        for j in different:

            dux = (
                df.loc[i, "Ux_star"]
                - df.loc[j, "Ux_star"]
            )

            duy = (
                df.loc[i, "Uy_star"]
                - df.loc[j, "Uy_star"]
            )

            jumps.append(
                np.sqrt(dux**2 + duy**2)
            )

        velocity_jump[i] = max(jumps)

    # --------------------------------------------------------
    # Frozen interface ranking.
    # --------------------------------------------------------

    G = df["gradU_mag_star"].to_numpy()

    Gstar = G / G.max()

    if velocity_jump.max() <= 0:
        raise RuntimeError(
            "No positive velocity jumps found."
        )

    dUstar = (
        velocity_jump
        / velocity_jump.max()
    )

    score = (
        transition
        * np.sqrt(Gstar * dUstar)
    )

    score[~interface] = 0.0

    ranking = np.argsort(-score)

    seeds = [
        i for i in ranking
        if score[i] > 0
    ][:n_seeds]

    if len(seeds) != n_seeds:
        raise RuntimeError(
            f"Expected {n_seeds} seeds, "
            f"found {len(seeds)}"
        )

    # --------------------------------------------------------
    # Expand EXACTLY radius layers.
    # No recursive propagation beyond radius.
    # --------------------------------------------------------

    def neighbourhood(seed):

        visited = {seed: 0}
        queue = deque([seed])

        while queue:

            current = queue.popleft()
            distance = visited[current]

            if distance >= radius:
                continue

            for nb in direct_neighbours(current):

                if nb not in visited:
                    visited[nb] = distance + 1
                    queue.append(nb)

        return set(visited)

    selected_indices = set()

    for seed in seeds:
        selected_indices.update(
            neighbourhood(seed)
        )

    selected_indices = sorted(
        selected_indices
    )

    selected = (
        df.loc[selected_indices]
        .copy()
        .reset_index(drop=True)
    )

    # Diagnostic columns are useful for later analysis.
    selected["som_interface_radius"] = radius
    selected["som_interface_n_seeds"] = n_seeds

    return selected
