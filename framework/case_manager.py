from pathlib import Path
import shutil
import re

from framework.config import (
    BASE_CASE,
    REFERENCE_CASE,
    EXPERIMENTS_DIR,
)


def is_numeric_directory(path: Path) -> bool:
    """Return True if directory name represents an OpenFOAM time."""
    try:
        float(path.name)
        return True
    except ValueError:
        return False


def protect_reference_cases(destination: Path):
    """
    Safety check: never allow the experiment manager to overwrite
    the M40 pilot or M160 reference cases.
    """
    destination = destination.resolve()

    protected = {
        BASE_CASE.resolve(),
        REFERENCE_CASE.resolve(),
    }

    if destination in protected:
        raise RuntimeError(
            f"SAFETY ERROR: attempt to modify protected case {destination}"
        )


def clean_case(case_dir: Path):
    """
    Remove previous numerical results from a copied case.

    Keeps:
        0/
        constant/
        system/

    Removes:
        numerical time directories > 0
        postProcessing/
        processor*/
        log files
    """

    protect_reference_cases(case_dir)

    for item in case_dir.iterdir():

        if item.is_dir():

            if is_numeric_directory(item) and item.name != "0":
                shutil.rmtree(item)

            elif item.name == "postProcessing":
                shutil.rmtree(item)
            elif item.name == "python":
                shutil.rmtree(item)


            elif re.fullmatch(r"processor\d+", item.name):
                shutil.rmtree(item)

        elif item.is_file():

            if item.name.startswith("log."):
                item.unlink()


def create_experiment_case(experiment, overwrite=False):
    """
    Create a clean experiment case from the protected M40 pilot.
    """

    name = experiment["name"]

    destination = EXPERIMENTS_DIR / name

    protect_reference_cases(destination)

    if destination.exists():

        if not overwrite:
            raise FileExistsError(
                f"Experiment already exists: {destination}"
            )

        shutil.rmtree(destination)

    print(f"Creating experiment: {name}")
    print(f"Source             : {BASE_CASE}")
    print(f"Destination        : {destination}")

    shutil.copytree(BASE_CASE, destination)

    clean_case(destination)

    print("Case created and cleaned.")

    return destination
