"""Policy input/output exactly as the Isaac Lab task computes it (isaaclab/hyper_lab/tasks/velocity/env_cfg.py).

deploy.json (written by isaaclab/scripts/export_deploy.py from the live Isaac Lab env) holds every number:
joint order, default joint positions, action scale, observation terms, control period, PD gains and motor limits.

Isaac Lab policy observation (no scaling, no clipping):
    [ velocity_commands (vx, vy, wz)          3   mdp.generated_commands
      joint_pos - default_joint_pos          12   mdp.joint_pos_rel
      joint_vel - default_joint_vel (= 0)    12   mdp.joint_vel_rel
      last action (raw policy output)        12   mdp.last_action ]
Action (mdp.JointPositionAction, use_default_offset): joint_pos_target = default_joint_pos + scale * action.
Actuator (isaaclab.actuators.DCMotor): tau = kp (target - q) - kd dq, clipped by the 4-quadrant torque-speed line.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PolicySpec:
    joint_names: list[str]
    default_joint_pos: np.ndarray
    action_scale: float
    obs_terms: list[str]
    obs_dims: list[int]
    step_dt: float  # policy period (decimation * physics dt)
    physics_dt: float
    stiffness: np.ndarray
    damping: np.ndarray
    effort_limit: np.ndarray
    saturation_effort: np.ndarray
    velocity_limit: np.ndarray
    joint_pos_limits: np.ndarray  # (12, 2) hard limits
    command_ranges: dict

    @classmethod
    def load(cls, path: str | Path) -> "PolicySpec":
        d = json.loads(Path(path).read_text())
        a = d["actuator"]
        return cls(
            joint_names=list(d["joint_names"]),
            default_joint_pos=np.asarray(d["default_joint_pos"], np.float64),
            action_scale=float(d["action_scale"]),
            obs_terms=list(d["policy_obs"]["terms"]),
            obs_dims=list(d["policy_obs"]["dims"]),
            step_dt=float(d["step_dt"]),
            physics_dt=float(d["physics_dt"]),
            stiffness=np.asarray(a["stiffness"], np.float64),
            damping=np.asarray(a["damping"], np.float64),
            effort_limit=np.asarray(a["effort_limit"], np.float64),
            saturation_effort=np.asarray(a["saturation_effort"], np.float64),
            velocity_limit=np.asarray(a["velocity_limit"], np.float64),
            joint_pos_limits=np.asarray(d["joint_pos_limits"], np.float64),
            command_ranges=d["command_ranges"],
        )

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


class ObservationBuilder:
    """Builds the actor observation from motor feedback (q, dq), the joystick command and the last action."""

    SUPPORTED = ("velocity_commands", "joint_pos", "joint_vel", "actions")

    def __init__(self, spec: PolicySpec):
        if tuple(spec.obs_terms) != self.SUPPORTED:
            raise ValueError(f"policy observation terms {spec.obs_terms} != {self.SUPPORTED}")
        self.spec = spec

    def __call__(self, command, joint_pos, joint_vel, last_action) -> np.ndarray:
        obs = np.concatenate([
            np.asarray(command, np.float64),
            np.asarray(joint_pos, np.float64) - self.spec.default_joint_pos,
            np.asarray(joint_vel, np.float64),
            np.asarray(last_action, np.float64),
        ])
        assert obs.shape == (sum(self.spec.obs_dims),)
        return obs.astype(np.float32)


def action_to_joint_target(spec: PolicySpec, action) -> np.ndarray:
    """mdp.JointPositionAction with use_default_offset=True."""
    return spec.default_joint_pos + spec.action_scale * np.asarray(action, np.float64)


def dc_motor_torque(spec: PolicySpec, q_target, q, dq, tau_ff=0.0, kp=None, kd=None) -> np.ndarray:
    """isaaclab.actuators.DCMotor.compute (PD + _clip_effort), joint space."""
    kp = spec.stiffness if kp is None else kp
    kd = spec.damping if kd is None else kd
    tau = kp * (np.asarray(q_target) - q) - kd * np.asarray(dq) + tau_ff
    v_lim = spec.velocity_limit * (1.0 + spec.effort_limit / spec.saturation_effort)
    v = np.clip(dq, -v_lim, v_lim)
    top = np.minimum(spec.saturation_effort * (1.0 - v / spec.velocity_limit), spec.effort_limit)
    bottom = np.maximum(spec.saturation_effort * (-1.0 - v / spec.velocity_limit), -spec.effort_limit)
    return np.clip(tau, bottom, top)


class MLPPolicy:
    """The rsl_rl actor (Linear + ELU ... Linear) evaluated with numpy, weights from policy.npz."""

    def __init__(self, path: str | Path):
        z = np.load(path)
        n = int(z["num_layers"])
        self.layers = [(z[f"w{i}"].astype(np.float32), z[f"b{i}"].astype(np.float32)) for i in range(n)]
        self.activation = str(z["activation"])
        if self.activation != "elu":
            raise ValueError(f"unsupported activation {self.activation}")
        self.obs_dim = self.layers[0][0].shape[1]
        self.act_dim = self.layers[-1][0].shape[0]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, np.float32)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU(alpha=1)
        return x
