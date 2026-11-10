"""Control loop shared by sim2sim and the real robot.

States:  passive (zero torque) --stand--> stand (ramp to the default pose, PD 25/0.5) --walk--> walk (policy, 50 Hz)
         any state --damp or a motor fault--> damp (kp 0, kd = damp_kd); damp is final.

Each tick (step_dt = 0.02 s, the Isaac Lab policy period):
  q, dq = motors.read()                       motor encoders only
  obs   = [cmd, q - q_default, dq, last_a]    ObservationBuilder (identical to the Isaac Lab policy group)
  a     = policy(obs)
  q*    = q_default + 0.25 a                  JointPositionAction, clipped to the hard joint limits
  motors.command(q*, kp=25, kd=0.5)           PD on the motor (DC motor torque-speed limit in sim)
"""
from __future__ import annotations

import numpy as np

from .policy_io import MLPPolicy, ObservationBuilder, PolicySpec, action_to_joint_target


class Controller:
    def __init__(self, spec: PolicySpec, policy: MLPPolicy, motors, commands, stand_s=2.0, damp_kd=1.0,
                 clip_to_limits=True, limit_margin=None, log=None):
        self.spec = spec
        self.policy = policy
        self.motors = motors
        self.commands = commands
        self.obs_builder = ObservationBuilder(spec)
        self.stand_s = stand_s
        self.damp_kd = damp_kd
        self.clip_to_limits = clip_to_limits
        self.limit_margin = limit_margin  # damp if a joint leaves its hard limits by more than this (rad)
        self.state = "passive"
        self.t = 0.0
        self.t_state = 0.0
        self.q_start = None
        self.last_action = np.zeros(spec.num_joints, np.float32)
        self.log = log
        self.reason = ""
        if policy.obs_dim != sum(spec.obs_dims) or policy.act_dim != spec.num_joints:
            raise ValueError("policy and deploy.json disagree on dimensions")

    def _enter(self, state, reason=""):
        if state == self.state or self.state == "damp":
            return
        if state == "walk" and self.state != "stand":
            return  # walk only from a finished stand
        self.state, self.t_state, self.reason = state, 0.0, reason
        if state == "walk":
            self.last_action[:] = 0.0  # Isaac Lab resets the action buffer to zero
        print(f"[controller] t={self.t:.2f}s -> {state} {reason}", flush=True)

    def tick(self):
        s = self.spec
        q, dq = self.motors.read()
        faults = self.motors.faults()
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            faults = faults + ["no motor state yet"] if self.state != "passive" else faults
        if self.limit_margin is not None and self.state in ("stand", "walk") and np.all(np.isfinite(q)):
            lim = s.joint_pos_limits
            out = (q < lim[:, 0] - self.limit_margin) | (q > lim[:, 1] + self.limit_margin)
            faults = faults + [f"{s.joint_names[i]} at {q[i]:+.3f} rad outside its limits" for i in np.flatnonzero(out)]
        if faults:
            self._enter("damp", "; ".join(faults))
        for ev in self.commands.events(self.t):
            if ev == "stand" and self.state == "passive":
                self.q_start = q.copy()
                self._enter("stand")
            elif ev == "walk" and self.state == "stand" and self.t_state >= self.stand_s - 1e-6:
                self._enter("walk")
            elif ev == "damp":
                self._enter("damp", "requested")

        cmd = np.zeros(3)
        action = None
        if self.state == "passive":
            self.motors.command(np.nan_to_num(q), 0.0, 0.0)
        elif self.state == "stand":
            alpha = min(1.0, self.t_state / self.stand_s)
            target = (1 - alpha) * self.q_start + alpha * s.default_joint_pos
            self.motors.command(target, s.stiffness, s.damping)
        elif self.state == "walk":
            cmd = self.commands.velocity(self.t)
            obs = self.obs_builder(cmd, q, dq, self.last_action)
            action = self.policy(obs)
            self.last_action = action
            target = action_to_joint_target(s, action)
            if self.clip_to_limits:
                target = np.clip(target, s.joint_pos_limits[:, 0], s.joint_pos_limits[:, 1])
            self.motors.command(target, s.stiffness, s.damping)
        else:  # damp
            self.motors.command(np.nan_to_num(q), 0.0, self.damp_kd)
        if self.log is not None:
            self.log(self.t, self.state, cmd, q, dq, action)
        self.motors.wait_tick(s.step_dt)
        self.t += s.step_dt
        self.t_state += s.step_dt
