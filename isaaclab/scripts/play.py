"""Isaac Lab's scripts/reinforcement_learning/rsl_rl/play.py with the HYPER tasks. Same arguments, e.g.

    python isaaclab/scripts/play.py --task Isaac-Velocity-Flat-HYPER-Play-v0 --num_envs 16

It also exports the policy to <run>/exported/policy.pt and policy.onnx (Isaac Lab behaviour).
"""
from _isaaclab_script import run

run("play.py")
