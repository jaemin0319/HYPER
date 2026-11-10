"""Sanity check of a HYPER task: build the env, print what the policy sees, actuator parameters and the MDP terms,
then hold the default pose (zero action) for a few seconds.

    python isaaclab/scripts/check_env.py --task Isaac-Velocity-Flat-HYPER-v0 --num_envs 16
"""
import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-HYPER-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=4.0)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import hyper_lab  # noqa: E402,F401
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

code = 1
try:
    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    uw = env.unwrapped
    obs, _ = env.reset()
    robot = uw.scene["robot"]
    act = robot.actuators["base_legs"]
    om = uw.observation_manager
    info = {
        "task": args.task,
        "physics_dt": uw.physics_dt,
        "step_dt": uw.step_dt,
        "decimation": uw.cfg.decimation,
        "joint_names": robot.joint_names,
        "body_names": robot.body_names,
        "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
        "soft_joint_pos_limits": robot.data.soft_joint_pos_limits[0].tolist(),
        "actuator": {
            "class": type(act).__name__,
            "stiffness": act.stiffness[0].tolist(),
            "damping": act.damping[0].tolist(),
            "effort_limit": act.effort_limit[0].tolist(),
            "velocity_limit": act.velocity_limit[0].tolist(),
            "saturation_effort": float(act.cfg.saturation_effort),
            "sim_stiffness": robot.data.joint_stiffness[0].tolist(),
            "sim_damping": robot.data.joint_damping[0].tolist(),
        },
        "action_scale": float(uw.action_manager.get_term("joint_pos").cfg.scale),
        "obs": {g: {"terms": om.active_terms[g], "dims": [list(d) for d in om.group_obs_term_dim[g]],
                    "shape": list(obs[g].shape)} for g in om.active_terms},
        "rewards": {n: uw.reward_manager.get_term_cfg(n).weight for n in uw.reward_manager.active_terms},
        "events": {m: uw.event_manager.active_terms[m] for m in uw.event_manager.active_terms},
        "terminations": uw.termination_manager.active_terms,
        "commands": uw.command_manager.get_term("base_velocity").cfg.ranges.to_dict(),
        "total_mass_kg": float(robot.root_physx_view.get_masses()[0].sum()),
    }
    print(json.dumps(info, indent=1), flush=True)

    # hold the default pose: action 0 -> target = default joint pos
    a = torch.zeros(uw.num_envs, uw.action_manager.total_action_dim, device=uw.device)
    n_term = 0
    for _ in range(int(args.seconds / uw.step_dt)):
        obs, rew, term, trunc, _ = env.step(a)
        n_term += int(term.sum())
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
    z = robot.data.root_pos_w[:, 2] - uw.scene.env_origins[:, 2]
    g = robot.data.projected_gravity_b
    tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1)))
    stand = {"seconds": args.seconds, "terminations": n_term, "base_height_mean": float(z.mean()),
             "base_height_min": float(z.min()), "tilt_deg_max": float(tilt.max())}
    print("[stand]", json.dumps(stand), flush=True)
    info["stand"] = stand
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(info, indent=1))
    code = 0
except Exception:
    import traceback

    traceback.print_exc()
sys.stdout.flush()
os._exit(code)
