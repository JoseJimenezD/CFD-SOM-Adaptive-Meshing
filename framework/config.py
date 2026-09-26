from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]
CASES_DIR = PROJECT / "cases"

FRAMEWORK_DIR = PROJECT / "framework"
EXPERIMENTS_DIR = PROJECT / "experiments"
RESULTS_DIR = PROJECT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"

DATA_DIR = PROJECT / "data"

# ============================================================
# PROTECTED REFERENCE CASES
# ============================================================
# These cases must NEVER be modified by the experiment manager.

BASE_CASE = CASES_DIR / "Cavity_SOM_002"      # M40
REFERENCE_CASE = CASES_DIR / "Cavity_SOM_004" # M160

BASE_MESH_NAME = "M40"
REFERENCE_MESH_NAME = "M160"

# ============================================================
# PHYSICAL / NUMERICAL SETTINGS
# ============================================================

L = 0.1
U_LID = 1.0
REYNOLDS = 100

END_TIME = 5.0

# Fine-cell size after one local refinement level
DX_FINE = 0.00125

# Same time step used for SOM-005 and GRAD-006
DELTA_T_ADAPTIVE = 0.00125
WRITE_INTERVAL = 80

# ============================================================
# SAFETY / VALIDATION LIMITS
# ============================================================

MAX_COURANT_ALLOWED = 1.05

REQUIRE_MESH_OK = True

# Reference solution is NEVER allowed for selecting cells.
ALLOW_REFERENCE_IN_SELECTION = False

# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42

# ============================================================
# EXPERIMENT MATRIX
# ============================================================
#
# IMPORTANT:
# "budget" is the fraction of M40 parent cells selected
# for one level of local x-y refinement.
#
# The reference M160 solution is used ONLY after a simulation
# has finished, for independent validation.
# ============================================================

EXPERIMENTS = [

    # --------------------------------------------------------
    # Conventional gradient baselines
    # --------------------------------------------------------

    {
        "name": "GRAD_05",
        "method": "gradient",
        "budget": 0.05,
    },

    {
        "name": "GRAD_10",
        "method": "gradient",
        "budget": 0.10,
    },

    {
        "name": "GRAD_15",
        "method": "gradient",
        "budget": 0.15,
    },

    {
        "name": "GRAD_20",
        "method": "gradient",
        "budget": 0.20,
    },

    # --------------------------------------------------------
    # SOM quantization-error experiments
    # --------------------------------------------------------

    {
        "name": "SOM_QE_05",
        "method": "som_qe",
        "budget": 0.05,
        "som_shape": (4, 4),
        "features": [
            "Ux_star",
            "Uy_star",
            "U_mag_star",
            "gradU_mag_star",
        ],
    },

    {
        "name": "SOM_QE_10",
        "method": "som_qe",
        "budget": 0.10,
        "som_shape": (4, 4),
        "features": [
            "Ux_star",
            "Uy_star",
            "U_mag_star",
            "gradU_mag_star",
        ],
    },

    {
        "name": "SOM_QE_15",
        "method": "som_qe",
        "budget": 0.15,
        "som_shape": (4, 4),
        "features": [
            "Ux_star",
            "Uy_star",
            "U_mag_star",
            "gradU_mag_star",
        ],
    },

    {
        "name": "SOM_QE_20",
        "method": "som_qe",
        "budget": 0.20,
        "som_shape": (4, 4),
        "features": [
            "Ux_star",
            "Uy_star",
            "U_mag_star",
            "gradU_mag_star",
        ],
    },

    # --------------------------------------------------------
    # HYBRID-v1: Gradient modulated by SOM regime transition
    #
    # I_hybrid = G* (1 + alpha * T_SOM)
    #
    # alpha = 0.5 fixed a priori before M160 validation.
    # Reference information is NOT used for cell selection.
    # --------------------------------------------------------

    {
        "name": "HYBRID_05",
        "method": "hybrid",
        "budget": 0.05,
        "alpha": 0.5,
    },

    {
        "name": "HYBRID_10",
        "method": "hybrid",
        "budget": 0.10,
        "alpha": 0.5,
    },

    {
        "name": "HYBRID_15",
        "method": "hybrid",
        "budget": 0.15,
        "alpha": 0.5,
    },

    {
        "name": "HYBRID_20",
        "method": "hybrid",
        "budget": 0.20,
        "alpha": 0.5,
    },

]
