from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

PROJECT = Path(__file__).resolve().parents[1]
L = 0.1


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*?$", "", text, flags=re.M)
    return text


def read_points(path):
    text = strip_comments(path.read_text())

    matches = re.findall(
        r"\(\s*"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)"
        r"\s*\)",
        text,
    )

    points = np.array(
        [[float(a), float(b), float(c)] for a, b, c in matches],
        dtype=float,
    )

    if len(points) == 0:
        raise RuntimeError(f"No points found in {path}")

    return points


def read_faces(path):
    text = strip_comments(path.read_text())

    matches = re.findall(
        r"(\d+)\s*\(\s*([0-9\s]+?)\s*\)",
        text,
    )

    faces = []

    for n_str, body in matches:
        n = int(n_str)
        ids = [int(v) for v in body.split()]

        if len(ids) == n:
            faces.append(ids)

    if len(faces) == 0:
        raise RuntimeError(f"No faces found in {path}")

    return faces


def extract_xy_edges(case_dir):
    mesh_dir = case_dir / "constant" / "polyMesh"

    points = read_points(mesh_dir / "points")
    faces = read_faces(mesh_dir / "faces")

    edges = set()

    for face in faces:
        n = len(face)

        for k in range(n):
            i = face[k]
            j = face[(k + 1) % n]

            p1 = points[i]
            p2 = points[j]

            x1 = round(float(p1[0] / L), 10)
            y1 = round(float(p1[1] / L), 10)
            x2 = round(float(p2[0] / L), 10)
            y2 = round(float(p2[1] / L), 10)

            if x1 == x2 and y1 == y2:
                continue

            a = (x1, y1)
            b = (x2, y2)

            edge = (a, b) if a <= b else (b, a)
            edges.add(edge)

    return [
        [edge[0], edge[1]]
        for edge in sorted(edges)
    ]

CASES = [
    (
        "Cavity_SOM_INT_40S_1N",
        "SOM-Interface: 40 fixed seeds + 1 layer",
        1834,
    ),
    (
        "Cavity_SOM_INT_40S_2N",
        "SOM-Interface: 40 fixed seeds + 2 layers",
        1942,
    ),
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 5.2),
)

for ax, (case_name, title, cells) in zip(axes, CASES):

    case_dir = PROJECT / "cases" / case_name

    if not case_dir.exists():
        raise RuntimeError(
            f"Case does not exist: {case_dir}"
        )

    segments = extract_xy_edges(case_dir)

    lines = LineCollection(
        segments,
        linewidths=0.28,
    )

    ax.add_collection(lines)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("x/L")
    ax.set_ylabel("y/L")

    ax.set_title(
        f"{title}\n{cells} cells"
    )

    ax.grid(False)

    print(
        f"{case_name}: "
        f"{len(segments)} unique projected mesh edges"
    )

fig.suptitle(
    "SOM-interface adaptive meshes with identical 40 seeds",
    fontsize=14,
)

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

output = (
    PROJECT
    / "results"
    / "plots"
    / "SOM_Interface_real_mesh_1N_vs_2N.png"
)

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print()
print("Saved:", output)
