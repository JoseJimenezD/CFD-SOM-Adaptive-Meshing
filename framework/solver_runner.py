from pathlib import Path
import re
import subprocess

from framework.config import (
    END_TIME,
    DELTA_T_ADAPTIVE,
    WRITE_INTERVAL,
    MAX_COURANT_ALLOWED,
)


def modify_control_dict(case_dir):
    """
    Configure an adaptive experiment with the validated
    transient settings used for the adaptive meshes.
    """

    case_dir = Path(case_dir)
    control_dict = case_dir / "system" / "controlDict"

    if not control_dict.exists():
        raise RuntimeError(
            f"controlDict not found: {control_dict}"
        )

    text = control_dict.read_text()

    replacements = {
        "startFrom": "startTime",
        "startTime": "0",
        "endTime": str(END_TIME),
        "deltaT": str(DELTA_T_ADAPTIVE),
        "writeInterval": str(WRITE_INTERVAL),
    }

    for key, value in replacements.items():

        pattern = rf"^(\s*{key}\s+)[^;]+;"

        new_text, count = re.subn(
            pattern,
            rf"\g<1>{value};",
            text,
            flags=re.MULTILINE,
        )

        if count != 1:
            raise RuntimeError(
                f"Expected exactly one '{key}' entry "
                f"in controlDict, found {count}."
            )

        text = new_text

    control_dict.write_text(text)

    return control_dict


def run_solver_command(command, case_dir):
    """
    Execute an OpenFOAM command and return its stdout.
    """

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


def parse_solver_log(log_text):
    """
    Extract basic numerical diagnostics from icoFoam output.
    """

    times = re.findall(
        r"^Time = ([0-9.eE+-]+)",
        log_text,
        flags=re.MULTILINE,
    )

    courant = re.findall(
        r"Courant Number mean:\s*([0-9.eE+-]+)"
        r"\s+max:\s*([0-9.eE+-]+)",
        log_text,
    )

    execution = re.findall(
        r"ExecutionTime =\s*([0-9.eE+-]+)\s*s",
        log_text,
    )

    if not times:
        raise RuntimeError(
            "Could not find simulation times in solver log."
        )

    if not courant:
        raise RuntimeError(
            "Could not find Courant numbers in solver log."
        )

    if not execution:
        raise RuntimeError(
            "Could not find ExecutionTime in solver log."
        )

    final_time = float(times[-1])
    courant_mean = float(courant[-1][0])
    courant_max = max(
        float(item[1])
        for item in courant
    )
    execution_time = float(execution[-1])

    return {
        "final_time": final_time,
        "courant_mean_final": courant_mean,
        "courant_max": courant_max,
        "execution_time": execution_time,
    }


def solve_experiment(case_dir):
    """
    Configure and run icoFoam for one adaptive experiment.
    """

    case_dir = Path(case_dir)

    # ---------------------------------------------------------
    # Safety check:
    # Do not overwrite an already solved experiment.
    # ---------------------------------------------------------

    final_directory = case_dir / f"{END_TIME:g}"

    if final_directory.exists():
        raise RuntimeError(
            f"SAFETY STOP: final time directory already exists: "
            f"{final_directory}"
        )

    # ---------------------------------------------------------
    # Configure controlDict.
    # ---------------------------------------------------------

    control_dict = modify_control_dict(case_dir)

    print(f"controlDict        : {control_dict}")
    print(f"endTime            : {END_TIME}")
    print(f"deltaT             : {DELTA_T_ADAPTIVE}")
    print(f"writeInterval      : {WRITE_INTERVAL}")

    # ---------------------------------------------------------
    # Run icoFoam.
    # ---------------------------------------------------------

    print()
    print("Running icoFoam...")

    log_text = run_solver_command(
        ["icoFoam"],
        case_dir,
    )

    log_file = case_dir / "log.icoFoam"
    log_file.write_text(log_text)

    # ---------------------------------------------------------
    # Parse and validate solver result.
    # ---------------------------------------------------------

    diagnostics = parse_solver_log(log_text)

    if abs(diagnostics["final_time"] - END_TIME) > 1e-9:
        raise RuntimeError(
            f"SAFETY STOP: solver ended at "
            f"{diagnostics['final_time']} instead of {END_TIME}."
        )

    if diagnostics["courant_max"] > MAX_COURANT_ALLOWED:
        raise RuntimeError(
            f"SAFETY STOP: maximum Courant number "
            f"{diagnostics['courant_max']:.6f} exceeds "
            f"allowed value {MAX_COURANT_ALLOWED:.6f}."
        )

    print()
    print("Solver completed successfully.")
    print(
        f"Final time         : "
        f"{diagnostics['final_time']}"
    )
    print(
        f"Final mean Co      : "
        f"{diagnostics['courant_mean_final']:.6f}"
    )
    print(
        f"Maximum Co         : "
        f"{diagnostics['courant_max']:.6f}"
    )
    print(
        f"Execution time     : "
        f"{diagnostics['execution_time']:.3f} s"
    )

    return diagnostics
