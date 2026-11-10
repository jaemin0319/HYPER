"""Motor backends. The controller only sees joint positions / velocities (motor encoders) in the policy's joint order.

    MujocoBackend   sim2sim on mujoco/hyper_scene.xml; each MuJoCo step applies the motor PD + DC motor limit
                    (like the GO-M8010-6 running its PD on board).
    GoM8010Backend  real Unitree GO-M8010-6 motors through unitree_actuator_sdk (one IO thread per RS-485 port).

Joint <-> motor (output shaft) mapping for the real robot, per joint in hardware.json:
    q_joint = direction * q_motor_output + offset           (calf: belt 1:1, motor on the thigh -> same form)
    q_motor_output = q_motor_rotor / gear_ratio             (GO-M8010-6: 6.33)
SDK commands are rotor side (unitree_actuator_sdk README): q*gear, dq*gear, kp/gear^2, kd/gear^2, tau/gear.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from .policy_io import PolicySpec, dc_motor_torque


class MotorBackend:
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def command(self, q_target, kp, kd, tau_ff=None):
        raise NotImplementedError

    def wait_tick(self, dt: float):
        raise NotImplementedError

    def faults(self) -> list[str]:
        return []

    def close(self):
        pass


# --------------------------------------------------------------------------- MuJoCo (sim2sim)
class MujocoBackend(MotorBackend):
    def __init__(self, spec: PolicySpec, xml: str | Path, key: str = "init", viewer: bool = False, realtime: bool = False,
                 key_callback=None):
        import mujoco

        self.mj = mujoco
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in spec.joint_names]
        aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in spec.joint_names]
        if min(jid) < 0 or min(aid) < 0:
            raise ValueError("MuJoCo joints/actuators must be named like the policy joints")
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.aid = np.array(aid)
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key)
        mujoco.mj_resetDataKeyframe(m, self.data, k)
        mujoco.mj_forward(m, self.data)
        n = spec.num_joints
        self._cmd = (self.read()[0], np.zeros(n), np.zeros(n), np.zeros(n))
        self.time = 0.0
        self.realtime = realtime
        self.viewer = mujoco.viewer.launch_passive(m, self.data, key_callback=key_callback) if viewer else None
        self._t_wall = time.perf_counter()

    def read(self):
        return self.data.qpos[self.qadr].copy(), self.data.qvel[self.vadr].copy()

    def command(self, q_target, kp, kd, tau_ff=None):
        n = self.spec.num_joints
        self._cmd = (np.asarray(q_target, float).copy(), np.broadcast_to(kp, (n,)).astype(float),
                     np.broadcast_to(kd, (n,)).astype(float), np.zeros(n) if tau_ff is None else np.asarray(tau_ff, float))

    def wait_tick(self, dt):
        q_t, kp, kd, tau_ff = self._cmd
        for _ in range(max(1, int(round(dt / self.model.opt.timestep)))):
            q, dq = self.read()
            self.data.ctrl[self.aid] = dc_motor_torque(self.spec, q_t, q, dq, tau_ff, kp, kd)
            self.mj.mj_step(self.model, self.data)
        self.time += dt
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            self.viewer.sync()
        if self.realtime:
            self._t_wall += dt
            rest = self._t_wall - time.perf_counter()
            if rest > 0:
                time.sleep(rest)
            else:
                self._t_wall = time.perf_counter()

    # ground truth, for evaluation only (the controller never uses it)
    def base_state(self):
        d = self.data
        w, x, y, z = d.qpos[3:7]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        v_b = R.T @ d.qvel[0:3]
        w_b = d.qvel[3:6]  # free joint angular velocity is in the body frame
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        return {"height": float(d.qpos[2]), "lin_vel_b": v_b, "ang_vel_b": w_b, "tilt_deg": tilt, "pos": d.qpos[0:3].copy()}

    def contacts(self):
        """Penetrating contacts as body-name pairs: (self contacts, non-foot bodies touching the ground)."""
        m, d, mj = self.model, self.data, self.mj
        name = lambda g: mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "world"  # noqa: E731
        self_c, ground = set(), set()
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0:
                continue
            a, b = name(c.geom1), name(c.geom2)
            if "world" in (a, b):
                other = b if a == "world" else a
                if not other.endswith("_foot"):
                    ground.add(other)
            else:
                self_c.add("|".join(sorted((a, b))))
        return self_c, ground

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


# --------------------------------------------------------------------------- real GO-M8010-6
class GoM8010Backend(MotorBackend):
    def __init__(self, spec: PolicySpec, hw: dict, sdk=None):
        if sdk is None:
            import unitree_actuator_sdk as sdk  # from github.com/unitreerobotics/unitree_actuator_sdk (python/ + lib/)
        self.sdk = sdk
        self.spec = spec
        mtype = sdk.MotorType.GO_M8010_6
        self.mtype = mtype
        self.gear = float(sdk.queryGearRatio(mtype))
        self.mode_foc = sdk.queryMotorMode(mtype, sdk.MotorMode.FOC)
        self.io_hz = float(hw.get("io_hz", 500.0))
        self.max_temp = float(hw["safety"]["max_temp_c"])
        self.timeout = float(hw["safety"]["comm_timeout_s"])
        joints = hw["joints"]
        n = spec.num_joints
        self.direction = np.array([float(joints[j]["direction"]) for j in spec.joint_names])
        self.offset = np.array([float(joints[j]["offset"]) for j in spec.joint_names])
        if not np.all(np.abs(self.direction) == 1.0):
            raise ValueError("direction must be +1 or -1")
        self._lock = threading.Lock()
        self._q = np.full(n, np.nan)
        self._dq = np.zeros(n)
        self._tau = np.zeros(n)
        self._temp = np.zeros(n)
        self._err = np.zeros(n, int)
        self._last_ok = np.zeros(n)
        self._bad_crc = np.zeros(n, int)
        self._cmd_q = np.zeros(n)
        self._cmd_kp = np.zeros(n)
        self._cmd_kd = np.zeros(n)
        self._cmd_tau = np.zeros(n)
        self._have_cmd = False
        self._stop = threading.Event()
        # one serial port + thread per bus
        self._buses = {}
        for i, name in enumerate(spec.joint_names):
            port = joints[name]["port"]
            self._buses.setdefault(port, []).append((i, int(joints[name]["id"])))
        self._threads = []
        for port, motors in self._buses.items():
            serial = sdk.SerialPort(port)
            cmds = [sdk.MotorCmd() for _ in motors]
            datas = [sdk.MotorData() for _ in motors]
            th = threading.Thread(target=self._io_loop, args=(serial, motors, cmds, datas), daemon=True, name=f"motor-io {port}")
            self._threads.append(th)
        self._t0 = time.perf_counter()
        for th in self._threads:
            th.start()
        self._next_tick = time.perf_counter()

    # joint <-> rotor conversions (also used by the tests)
    def joint_to_rotor_cmd(self, i, q, kp, kd, tau):
        g, s = self.gear, self.direction[i]
        return s * (q - self.offset[i]) * g, kp / g**2, kd / g**2, s * tau / g

    def rotor_to_joint_state(self, i, q_r, dq_r):
        g, s = self.gear, self.direction[i]
        return s * q_r / g + self.offset[i], s * dq_r / g

    def _io_loop(self, serial, motors, cmds, datas):
        period = 1.0 / self.io_hz
        nxt = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                c = (self._cmd_q.copy(), self._cmd_kp.copy(), self._cmd_kd.copy(), self._cmd_tau.copy(), self._have_cmd)
            for (i, mid), cmd, data in zip(motors, cmds, datas):
                cmd.motorType = self.mtype
                data.motorType = self.mtype
                cmd.mode = self.mode_foc
                cmd.id = mid
                if c[4]:
                    q_r, kp_r, kd_r, tau_r = self.joint_to_rotor_cmd(i, c[0][i], c[1][i], c[2][i], c[3][i])
                else:  # nothing commanded yet: zero torque, just read the state
                    q_r, kp_r, kd_r, tau_r = 0.0, 0.0, 0.0, 0.0
                cmd.q, cmd.dq, cmd.kp, cmd.kd, cmd.tau = float(q_r), 0.0, float(kp_r), float(kd_r), float(tau_r)
                serial.sendRecv(cmd, data)
                now = time.perf_counter()
                with self._lock:
                    if data.correct:
                        self._q[i], self._dq[i] = self.rotor_to_joint_state(i, data.q, data.dq)
                        self._tau[i] = data.tau
                        self._temp[i] = data.temp
                        self._err[i] = data.merror
                        self._last_ok[i] = now
                    else:
                        self._bad_crc[i] += 1
            nxt += period
            rest = nxt - time.perf_counter()
            if rest > 0:
                time.sleep(rest)
            else:
                nxt = time.perf_counter()

    def read(self):
        with self._lock:
            return self._q.copy(), self._dq.copy()

    def command(self, q_target, kp, kd, tau_ff=None):
        n = self.spec.num_joints
        with self._lock:
            self._cmd_q[:] = q_target
            self._cmd_kp[:] = np.broadcast_to(kp, (n,))
            self._cmd_kd[:] = np.broadcast_to(kd, (n,))
            self._cmd_tau[:] = 0.0 if tau_ff is None else tau_ff
            self._have_cmd = True

    def wait_tick(self, dt):
        self._next_tick += dt
        rest = self._next_tick - time.perf_counter()
        if rest > 0:
            time.sleep(rest)
        else:
            self._next_tick = time.perf_counter()

    def status(self):
        with self._lock:
            return {"q": self._q.copy(), "temp": self._temp.copy(), "merror": self._err.copy(),
                    "bad_crc": self._bad_crc.copy(), "age": time.perf_counter() - self._last_ok}

    def faults(self):
        s = self.status()
        out = []
        names = self.spec.joint_names
        for i, n in enumerate(names):
            if s["merror"][i] != 0:
                out.append(f"{n}: motor error {int(s['merror'][i])} (1 overheat, 2 overcurrent, 3 overvoltage, 4 encoder)")
            if s["temp"][i] >= self.max_temp:
                out.append(f"{n}: temperature {s['temp'][i]:.0f} C")
            if time.perf_counter() - self._t0 > self.timeout and s["age"][i] > self.timeout:
                out.append(f"{n}: no valid reply for {s['age'][i]:.3f} s")
        return out

    def close(self, damping_kd: float = 1.0, seconds: float = 1.0):
        q, _ = self.read()
        if np.all(np.isfinite(q)):
            self.command(q, 0.0, damping_kd)
            time.sleep(seconds)
        self.command(np.zeros(self.spec.num_joints), 0.0, 0.0)
        time.sleep(0.05)
        self._stop.set()
        for th in self._threads:
            th.join(timeout=1.0)
