"""Drive one HYPER in the Isaac Sim GUI with the trained policy (keyboard, or a gamepad with --joystick).

    python isaaclab/scripts/teleop.py                                  # deploy/policies/flat/policy.pt
    python isaaclab/scripts/teleop.py --joystick /dev/input/js0        # gamepad, same mapping as the real robot

Keyboard (click the viewport first; hold keys, release = stop):
    UP / DOWN      forward / backward  (vx +-1.0 m/s)
    LEFT / RIGHT   step left / right   (vy +-0.5 m/s)
    Z / X          turn left / right   (wz +-1.0 rad/s)
    L              zero the command    R  reset the robot
Gamepad (deploy/config/hardware.example.json "joystick"): left stick vx/vy, right stick x yaw rate.

The policy sees only what the real robot has (command, joint pos/vel, last action). Green arrow above the robot =
command, blue = actual base velocity. Close the window to quit.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "deploy"))

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-HYPER-Play-v0")
parser.add_argument("--policy", default=str(REPO / "deploy" / "policies" / "flat" / "policy.pt"))
parser.add_argument("--joystick", default=None, help="Linux joystick device, e.g. /dev/input/js0")
parser.add_argument("--joystick_config", default=str(REPO / "deploy" / "config" / "hardware.example.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import hyper_lab  # noqa: E402,F401
import isaaclab_tasks  # noqa: E402,F401
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=1)
cb = env_cfg.commands.base_velocity  # the command comes from the keyboard / gamepad only
cb.heading_command = False
cb.resampling_time_range = (1.0e9, 1.0e9)
cb.rel_standing_envs = 0.0
cb.debug_vis = True
env_cfg.episode_length_s = 1.0e6  # no time-out resets; a fall still resets
env_cfg.viewer = ViewerCfg(eye=(1.4, -1.4, 0.8), lookat=(0.0, 0.0, 0.15), origin_type="asset_root", asset_name="robot")
env = gym.make(args.task, cfg=env_cfg)
uw = env.unwrapped
term = uw.command_manager.get_term("base_velocity")
policy = torch.jit.load(args.policy, map_location=uw.device).eval()
r = cb.ranges

keyboard = Se2Keyboard(Se2KeyboardCfg(v_x_sensitivity=r.lin_vel_x[1], v_y_sensitivity=0.5, omega_z_sensitivity=r.ang_vel_z[1]))
reset_requested = [False]
keyboard.add_callback("R", lambda: reset_requested.__setitem__(0, True))
joystick = None
if args.joystick:
    from hyper_deploy.commands import LinuxJoystick

    jcfg = dict(json.loads(Path(args.joystick_config).read_text())["joystick"], device=args.joystick)
    joystick = LinuxJoystick(jcfg, {"lin_vel_x": r.lin_vel_x, "lin_vel_y": r.lin_vel_y, "ang_vel_z": r.ang_vel_z})
print(__doc__, flush=True)

lo = torch.tensor([r.lin_vel_x[0], r.lin_vel_y[0], r.ang_vel_z[0]], device=uw.device)
hi = torch.tensor([r.lin_vel_x[1], r.lin_vel_y[1], r.ang_vel_z[1]], device=uw.device)
robot = uw.scene["robot"]
obs, _ = env.reset()
t_print = 0.0
while app.is_running():
    t0 = time.time()
    cmd = torch.as_tensor(keyboard.advance(), dtype=torch.float32, device=uw.device).flatten()[:3]
    if joystick is not None:
        cmd = cmd + torch.as_tensor(joystick.velocity(0.0), dtype=torch.float32, device=uw.device)
    term.vel_command_b[:] = torch.clamp(cmd, lo, hi)
    if reset_requested[0]:
        reset_requested[0] = False
        obs, _ = env.reset()
    obs = uw.observation_manager.compute()  # policy input with the new command
    with torch.inference_mode():
        action = policy(obs["policy"])
    obs, _, term_, trunc, _ = env.step(action)
    if time.time() - t_print > 0.5:
        t_print = time.time()
        v = robot.data.root_lin_vel_b[0]
        w = robot.data.root_ang_vel_b[0, 2]
        c = term.vel_command_b[0]
        print(f"\rcmd vx {c[0]:+.2f} vy {c[1]:+.2f} wz {c[2]:+.2f} | actual vx {v[0]:+.2f} vy {v[1]:+.2f} wz {w:+.2f}   ",
              end="", flush=True)
    rest = uw.step_dt - (time.time() - t0)
    if rest > 0:
        time.sleep(rest)
sys.stdout.flush()
os._exit(0)
