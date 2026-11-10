"""Run one of Isaac Lab's own rsl_rl scripts (train.py / play.py) unchanged, with the HYPER tasks registered."""
import os
import runpy
import sys
from pathlib import Path

HYPER_ISAACLAB = Path(__file__).resolve().parents[1]


def isaaclab_dir() -> Path:
    if os.environ.get("ISAACLAB_PATH"):
        return Path(os.environ["ISAACLAB_PATH"])
    import isaaclab

    return Path(isaaclab.__file__).resolve().parents[3]  # <IsaacLab>/source/isaaclab/isaaclab/__init__.py


def run(name: str):
    rsl = isaaclab_dir() / "scripts" / "reinforcement_learning" / "rsl_rl"
    sys.path.insert(0, str(rsl))  # cli_args.py
    sys.path.insert(0, str(HYPER_ISAACLAB))
    import hyper_lab  # noqa: F401  (registers Isaac-Velocity-*-HYPER-*)

    # SimulationApp.close() can hang on some machines after the run is saved; exit right away instead.
    if os.environ.get("HYPER_FAST_EXIT", "1") == "1":
        from isaacsim.simulation_app import SimulationApp

        def _close(self, *args, **kwargs):
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

        SimulationApp.close = _close
    sys.argv[0] = str(rsl / name)
    runpy.run_path(str(rsl / name), run_name="__main__")
