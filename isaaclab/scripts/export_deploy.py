"""Export a trained HYPER policy for deployment and check the deployment code against Isaac Lab.

    python isaaclab/scripts/export_deploy.py --task Isaac-Velocity-Flat-HYPER-Play-v0 \
        --checkpoint logs/rsl_rl/hyper_flat/<run>/model_299.pt --out deploy/policies/flat

Writes to --out:
  policy.pt / policy.onnx   Isaac Lab exporters (same as play.py)
  policy.npz                actor weights for deploy/hyper_deploy (numpy)
  deploy.json               joint order, default pose, action scale, obs layout, dt, PD gains, motor limits, ranges
  isaac_eval.json           (1) deploy code vs Isaac Lab on a live rollout: observation, action, joint target and
                            DC motor torque; (2) velocity tracking and falls for fixed joystick commands
"""
import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "deploy"))

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-HYPER-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--seconds", type=float, default=10.0, help="evaluation length per command")
parser.add_argument("--settle", type=float, default=3.0, help="seconds skipped before measuring tracking")
parser.add_argument("--envs_per_command", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import hyper_lab  # noqa: E402,F401
import isaaclab_tasks  # noqa: E402,F401
from hyper_deploy.policy_io import MLPPolicy, ObservationBuilder, PolicySpec, action_to_joint_target, dc_motor_torque  # noqa: E402
from isaaclab.utils.types import ArticulationActions  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

COMMANDS = [  # (vx, vy, wz) joystick commands to evaluate
    (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.5, 0.0),
    (0.0, -0.5, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.5, 0.0, 0.5), (0.7, 0.4, 0.0),
]

code = 1
try:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    num_envs = len(COMMANDS) * args.envs_per_command
    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=num_envs)
    cb = env_cfg.commands.base_velocity  # fixed commands: no heading control, no resampling, no standing envs
    cb.heading_command = False
    cb.resampling_time_range = (1.0e9, 1.0e9)
    cb.rel_standing_envs = 0.0
    cb.debug_vis = False
    env_cfg.episode_length_s = args.seconds + 5.0
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg)
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    uw = env.unwrapped
    runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(Path(args.checkpoint).resolve()))
    policy = runner.get_inference_policy(device=uw.device)
    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(out), filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=str(out), filename="policy.onnx")
    if normalizer is not None and not isinstance(normalizer, torch.nn.Identity):
        raise RuntimeError("actor observation normalization is on; fold it into policy.npz first")

    # actor MLP -> npz
    layers = [m for m in policy_nn.actor if isinstance(m, torch.nn.Linear)]
    acts = {type(m).__name__ for m in policy_nn.actor if not isinstance(m, torch.nn.Linear)}
    assert acts == {"ELU"} and all(m.alpha == 1.0 for m in policy_nn.actor if isinstance(m, torch.nn.ELU)), acts
    np.savez(out / "policy.npz", num_layers=len(layers), activation="elu",
             **{f"w{i}": l.weight.detach().cpu().numpy() for i, l in enumerate(layers)},
             **{f"b{i}": l.bias.detach().cpu().numpy() for i, l in enumerate(layers)})

    # deploy.json from the live env
    robot = uw.scene["robot"]
    act = robot.actuators["base_legs"]
    om = uw.observation_manager
    term = uw.command_manager.get_term("base_velocity")
    train_cfg = parse_env_cfg(args.task.replace("-Play", ""), device="cpu", num_envs=1)
    spec_d = {
        "task": args.task.replace("-Play", ""),
        "checkpoint": os.path.relpath(Path(args.checkpoint).resolve(), REPO),
        "joint_names": robot.joint_names,
        "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
        "joint_pos_limits": robot.data.joint_pos_limits[0].tolist(),
        "soft_joint_pos_limits": robot.data.soft_joint_pos_limits[0].tolist(),
        "action_scale": float(uw.action_manager.get_term("joint_pos").cfg.scale),
        "policy_obs": {"terms": om.active_terms["policy"], "dims": [int(np.prod(d)) for d in om.group_obs_term_dim["policy"]]},
        "step_dt": uw.step_dt,
        "physics_dt": uw.physics_dt,
        "decimation": uw.cfg.decimation,
        "actuator": {
            "model": type(act).__name__,
            "stiffness": act.stiffness[0].tolist(),
            "damping": act.damping[0].tolist(),
            "effort_limit": act.effort_limit[0].tolist(),
            "saturation_effort": [float(act.cfg.saturation_effort)] * robot.num_joints,
            "velocity_limit": act.velocity_limit[0].tolist(),
        },
        "command_ranges": train_cfg.commands.base_velocity.ranges.to_dict(),
        "motor": {"type": "GO_M8010_6", "gear_ratio": 6.33, "calf_belt_ratio": 1.0},
    }
    (out / "deploy.json").write_text(json.dumps(spec_d, indent=1))
    spec = PolicySpec.load(out / "deploy.json")
    builder = ObservationBuilder(spec)
    np_policy = MLPPolicy(out / "policy.npz")
    jit = torch.jit.load(str(out / "policy.pt"), map_location="cpu")

    # (1) deploy code vs Isaac Lab on a live rollout + (2) tracking evaluation
    cmd = torch.tensor(COMMANDS, device=uw.device, dtype=torch.float32).repeat_interleave(args.envs_per_command, 0)
    obs = venv.get_observations()
    term.vel_command_b[:] = cmd
    obs = venv.get_observations()  # recompute with the fixed commands
    om.compute()
    last_action = np.zeros((num_envs, spec.num_joints), np.float32)
    err = {"obs": 0.0, "action_numpy_vs_torch": 0.0, "action_jit_vs_torch": 0.0, "joint_target": 0.0}
    steps = int(args.seconds / uw.step_dt)
    settle = int(args.settle / uw.step_dt)
    v_hist, h_hist, falls = [], [], torch.zeros(num_envs, device=uw.device)
    for k in range(steps):
        term.vel_command_b[:] = cmd
        o_isaac = obs["policy"].cpu().numpy()
        q = robot.data.joint_pos.cpu().numpy()
        dq = robot.data.joint_vel.cpu().numpy()
        o_dep = np.stack([builder(cmd[i].cpu().numpy(), q[i], dq[i], last_action[i]) for i in range(num_envs)])
        err["obs"] = max(err["obs"], float(np.abs(o_dep - o_isaac).max()))
        with torch.inference_mode():
            a = policy(obs)
        a_np = a.cpu().numpy()
        err["action_numpy_vs_torch"] = max(err["action_numpy_vs_torch"], float(np.abs(np_policy(o_dep) - a_np).max()))
        err["action_jit_vs_torch"] = max(err["action_jit_vs_torch"], float((jit(torch.from_numpy(o_dep)) - a.cpu()).abs().max()))
        obs, _, dones, extras = venv.step(a)
        last_action = a_np.copy()
        last_action[dones.cpu().numpy().astype(bool)] = 0.0
        tgt = robot.data.joint_pos_target.cpu().numpy()
        alive = ~dones.cpu().numpy().astype(bool)
        err["joint_target"] = max(err["joint_target"], float(np.abs(np.stack([action_to_joint_target(spec, x) for x in a_np]) - tgt)[alive].max()))
        falls += (dones & ~extras["time_outs"].to(dones.device)).float()
        if k >= settle:
            v_hist.append(torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], 1).cpu().numpy())
            gz = robot.data.projected_gravity_b[:, 2].clamp(-1, 1)
            h_hist.append(torch.stack([robot.data.root_pos_w[:, 2] - uw.scene.env_origins[:, 2],
                                       torch.rad2deg(torch.acos(-gz))], 1).cpu().numpy())

    # DC motor torque: Isaac Lab's actuator object vs deploy dc_motor_torque on random states
    g = torch.Generator(device="cpu").manual_seed(0)
    n = 4096
    q_t = robot.data.default_joint_pos[:1].cpu().repeat(n, 1) + 0.6 * (torch.rand(n, 12, generator=g) - 0.5)
    q_m = robot.data.default_joint_pos[:1].cpu().repeat(n, 1) + 0.6 * (torch.rand(n, 12, generator=g) - 0.5)
    dq_m = 80.0 * (torch.rand(n, 12, generator=g) - 0.5)
    from isaaclab.actuators import DCMotor

    dc = DCMotor(act.cfg, joint_names=robot.joint_names, joint_ids=slice(None), num_envs=n, device="cpu",
                 stiffness=act.stiffness[:1].cpu().repeat(n, 1), damping=act.damping[:1].cpu().repeat(n, 1),
                 armature=torch.zeros(n, 12), friction=torch.zeros(n, 12),
                 effort_limit=act.effort_limit[:1].cpu().repeat(n, 1), velocity_limit=act.velocity_limit[:1].cpu().repeat(n, 1))
    res = dc.compute(ArticulationActions(joint_positions=q_t, joint_velocities=torch.zeros(n, 12), joint_efforts=torch.zeros(n, 12)),
                     joint_pos=q_m, joint_vel=dq_m)
    tau_dep = dc_motor_torque(spec, q_t.numpy(), q_m.numpy(), dq_m.numpy())
    err["dc_motor_torque"] = float(np.abs(tau_dep - res.joint_efforts.numpy()).max())
    err["dc_motor_samples_saturated"] = float((np.abs(dc.computed_effort.numpy() - res.joint_efforts.numpy()) > 1e-6).mean())

    v = np.stack(v_hist)  # (T, envs, 3)
    h = np.stack(h_hist)  # (T, envs, 2) height, tilt
    tracking = []
    for c_i, c in enumerate(COMMANDS):
        sl = slice(c_i * args.envs_per_command, (c_i + 1) * args.envs_per_command)
        vm = v[:, sl].reshape(-1, 3)
        tracking.append({"command": c, "mean_velocity": vm.mean(0).round(3).tolist(),
                         "mean_abs_error": np.abs(vm - np.array(c)).mean(0).round(3).tolist(),
                         "mean_height": round(float(h[:, sl, 0].mean()), 3), "max_tilt_deg": round(float(h[:, sl, 1].max()), 1),
                         "falls": int(falls[sl].sum().item())})
    result = {"deploy_vs_isaac_max_abs_error": err, "tracking": tracking,
              "eval": {"seconds": args.seconds, "settle": args.settle, "envs_per_command": args.envs_per_command,
                       "measured": "base lin vel x, y and yaw rate in the base frame (Isaac Lab ground truth)"}}
    (out / "isaac_eval.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)
    code = 0
except Exception:
    import traceback

    traceback.print_exc()
sys.stdout.flush()
os._exit(code)
