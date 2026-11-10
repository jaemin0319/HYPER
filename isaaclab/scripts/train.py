"""Isaac Lab's scripts/reinforcement_learning/rsl_rl/train.py with the HYPER tasks. Same arguments, e.g.

    python isaaclab/scripts/train.py --task Isaac-Velocity-Flat-HYPER-v0 --headless
    python isaaclab/scripts/train.py --task Isaac-Velocity-Rough-HYPER-v0 --headless

Logs go to ./logs/rsl_rl/<experiment_name>/<time>.
"""
from _isaaclab_script import run

run("train.py")
