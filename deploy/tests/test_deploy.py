"""Deployment code tests that need no hardware and no Isaac Sim (numpy; mujoco and torch tests skip if missing).

    python -m pytest deploy/tests -q        or        python deploy/tests/test_deploy.py
"""
import json
import os
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy"))

from hyper_deploy import fake_unitree_sdk as fake  # noqa: E402
from hyper_deploy.backends import GoM8010Backend  # noqa: E402
from hyper_deploy.commands import LinuxJoystick, ScriptedCommand  # noqa: E402
from hyper_deploy.policy_io import MLPPolicy, ObservationBuilder, PolicySpec, action_to_joint_target, dc_motor_torque  # noqa: E402

POLICY = REPO / "deploy" / "policies" / "flat"
SPEC = PolicySpec.load(POLICY / "deploy.json")
HW = json.loads((REPO / "deploy" / "config" / "hardware.example.json").read_text())


def test_spec_matches_isaac_task():
    assert SPEC.joint_names[:4] == ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
    assert SPEC.obs_terms == ["velocity_commands", "joint_pos", "joint_vel", "actions"] and SPEC.obs_dims == [3, 12, 12, 12]
    assert SPEC.action_scale == 0.25 and abs(SPEC.step_dt - 0.02) < 1e-9 and abs(SPEC.physics_dt - 0.005) < 1e-9
    assert np.allclose(SPEC.stiffness, 25) and np.allclose(SPEC.damping, 0.5)
    assert np.allclose(SPEC.effort_limit, 23.7, atol=1e-5) and np.allclose(SPEC.velocity_limit, 30.0)


def test_observation_layout():
    b = ObservationBuilder(SPEC)
    q = SPEC.default_joint_pos + np.arange(12) * 0.01
    dq = np.arange(12) * 0.1
    a = np.arange(12) * -0.2
    o = b([0.5, -0.2, 0.3], q, dq, a)
    assert o.dtype == np.float32 and o.shape == (39,)
    assert np.allclose(o[:3], [0.5, -0.2, 0.3]) and np.allclose(o[3:15], np.arange(12) * 0.01, atol=1e-6)
    assert np.allclose(o[15:27], dq) and np.allclose(o[27:], a)


def test_action_and_dc_motor():
    a = np.linspace(-1, 1, 12)
    assert np.allclose(action_to_joint_target(SPEC, a), SPEC.default_joint_pos + 0.25 * a)
    q = SPEC.default_joint_pos
    # stall: full torque, no-load speed: zero torque in the driving direction, braking allowed
    assert np.allclose(dc_motor_torque(SPEC, q + 2.0, q, np.zeros(12)), 23.7, atol=1e-5)
    assert np.allclose(dc_motor_torque(SPEC, q + 2.0, q, np.full(12, 30.0)), 0.0, atol=1e-4)
    assert np.allclose(dc_motor_torque(SPEC, q - 2.0, q, np.full(12, 15.0)), -23.7, atol=1e-5)
    assert np.allclose(dc_motor_torque(SPEC, q + 0.1, q, np.zeros(12)), 2.5)


def test_policy_numpy_matches_torchscript():
    try:
        import torch
    except ImportError:
        return
    pol = MLPPolicy(POLICY / "policy.npz")
    jit = torch.jit.load(str(POLICY / "policy.pt"), map_location="cpu")
    x = np.random.default_rng(0).normal(size=(256, 39)).astype(np.float32)
    assert np.abs(pol(x) - jit(torch.from_numpy(x)).detach().numpy()).max() < 1e-4


def _backend(direction=-1.0, offset=0.3):
    hw = json.loads(json.dumps(HW))
    for j in hw["joints"].values():
        j["direction"], j["offset"] = direction, offset
    return GoM8010Backend(SPEC, hw, sdk=fake)


def test_go_m8010_rotor_conversions():
    m = _backend()
    try:
        q_r, kp_r, kd_r, tau_r = m.joint_to_rotor_cmd(0, 0.5, 25.0, 0.5, 2.0)
        g = fake.GEAR
        assert np.isclose(q_r, -(0.5 - 0.3) * g) and np.isclose(kp_r, 25 / g**2) and np.isclose(kd_r, 0.5 / g**2)
        assert np.isclose(tau_r, -2.0 / g)
        q, dq = m.rotor_to_joint_state(0, q_r, 10.0)
        assert np.isclose(q, 0.5) and np.isclose(dq, -10.0 / g)
    finally:
        m.close(seconds=0.0)


def test_go_m8010_backend_reaches_target_and_stiffness():
    m = _backend()
    try:
        t0 = time.time()
        while not np.all(np.isfinite(m.read()[0])) and time.time() - t0 < 1:
            time.sleep(0.01)
        target = SPEC.default_joint_pos
        m.command(target, 25.0, 0.5)
        time.sleep(1.5)
        q, dq = m.read()
        assert np.abs(q - target).max() < 0.02, q - target
        # joint-space stiffness of the rotor-side command: motor at joint angle q0, target q0 + 0.1 rad
        g, s = fake.GEAR, m.direction[0]
        q0 = 0.7
        q_r_now = m.joint_to_rotor_cmd(0, q0, 0, 0, 0)[0]
        q_r_des, kp_r, _, _ = m.joint_to_rotor_cmd(0, q0 + 0.1, 25.0, 0.5, 0.0)
        tau_joint = s * g * kp_r * (q_r_des - q_r_now)  # rotor torque -> output torque -> joint torque
        assert np.isclose(tau_joint, 25.0 * 0.1), tau_joint
        assert not m.faults()
    finally:
        m.close(seconds=0.0)


def test_scripted_command_and_events():
    c = ScriptedCommand([(1.0, 0.5, 0, 0), (1.0, 0, 0, 2.0)], SPEC.command_ranges, stand_s=2.0)
    assert c.events(0.0) == ["stand"] and c.events(1.0) == [] and "walk" in c.events(2.0)
    assert np.allclose(c.velocity(2.5), [0.5, 0, 0]) and np.allclose(c.velocity(3.5), [0, 0, 1.0])  # yaw clipped to range


def test_linux_joystick_parsing():
    d = tempfile.mkdtemp()
    fifo = os.path.join(d, "js0")
    os.mkfifo(fifo)
    cfg = dict(HW["joystick"], device=fifo)
    holder = {}
    th = threading.Thread(target=lambda: holder.setdefault("js", LinuxJoystick(cfg, SPEC.command_ranges)))
    th.start()
    w = os.open(fifo, os.O_WRONLY)
    th.join()
    js = holder["js"]
    ev = lambda v, t, n: struct.pack("IhBB", 0, v, t, n)  # noqa: E731
    os.write(w, ev(-32767, 2, 1) + ev(16384, 2, 0) + ev(1, 1, 7) + ev(0, 1, 7))  # LY full up, LX half right, START
    time.sleep(0.2)
    v = js.velocity(0)
    assert np.isclose(v[0], 1.0, atol=1e-3)            # up = forward, full range
    assert -0.5 < v[1] < -0.3                           # right = -y
    assert js.events(0) == ["stand"]
    os.close(w)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__} {e}")
    sys.exit(1 if fails else 0)
