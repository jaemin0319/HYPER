"""Velocity command sources: a Linux joystick (/dev/input/js*), the MuJoCo viewer keyboard, or a fixed script.

Every source returns (vx, vy, wz) in the base frame, clipped to the training command ranges, plus mode events
"stand", "walk", "damp" for the controller.
"""
from __future__ import annotations

import os
import struct
import threading

import numpy as np


class CommandSource:
    def velocity(self, t: float) -> np.ndarray:
        raise NotImplementedError

    def events(self, t: float) -> list[str]:
        return []


def _clip(cmd, ranges):
    lo = np.array([ranges["lin_vel_x"][0], ranges["lin_vel_y"][0], ranges["ang_vel_z"][0]])
    hi = np.array([ranges["lin_vel_x"][1], ranges["lin_vel_y"][1], ranges["ang_vel_z"][1]])
    return np.clip(cmd, lo, hi)


class ScriptedCommand(CommandSource):
    """segments: list of (duration_s, vx, vy, wz); starts walking after `stand_s`."""

    def __init__(self, segments, ranges, stand_s=2.0):
        self.segments = segments
        self.ranges = ranges
        self.stand_s = stand_s
        self.starts = np.cumsum([0.0] + [s[0] for s in segments])[:-1] + stand_s
        self.end = stand_s + sum(s[0] for s in segments)
        self._sent = set()

    def segment(self, t):
        for k, t0 in enumerate(self.starts):
            if t0 <= t < t0 + self.segments[k][0]:
                return k
        return None

    def velocity(self, t):
        k = self.segment(t)
        return np.zeros(3) if k is None else _clip(np.array(self.segments[k][1:], float), self.ranges)

    def events(self, t):
        out = []
        if "stand" not in self._sent:
            out.append("stand")
            self._sent.add("stand")
        if t >= self.stand_s - 1e-6:
            out.append("walk")  # repeated until the controller has finished standing
        return out


class LinuxJoystick(CommandSource):
    """Linux joystick API (struct js_event: u32 time, s16 value, u8 type, u8 number); no extra packages.

    cfg (hardware.json "joystick"): device, axes {vx: [index, sign], vy: [...], wz: [...]}, deadzone,
    buttons {stand: i, walk: i, damp: i}. Full stick deflection maps to the training command range edge.
    """

    def __init__(self, cfg: dict, ranges: dict):
        self.cfg = cfg
        self.ranges = ranges
        self.axes = {}
        self.buttons = {}
        self._pressed = []
        self._lock = threading.Lock()
        self.fd = os.open(cfg.get("device", "/dev/input/js0"), os.O_RDONLY)
        threading.Thread(target=self._loop, daemon=True, name="joystick").start()

    def _loop(self):
        fmt = "IhBB"
        size = struct.calcsize(fmt)
        while True:
            buf = os.read(self.fd, size)
            if len(buf) < size:
                continue
            _, value, etype, number = struct.unpack(fmt, buf)
            with self._lock:
                if etype & 0x02:
                    self.axes[number] = value / 32767.0
                elif etype & 0x01:
                    if value and not (etype & 0x80):
                        self._pressed.append(number)
                    self.buttons[number] = value

    def _axis(self, key):
        idx, sign = self.cfg["axes"][key]
        v = sign * self.axes.get(idx, 0.0)
        dz = self.cfg.get("deadzone", 0.1)
        return 0.0 if abs(v) < dz else (v - np.sign(v) * dz) / (1 - dz)

    def velocity(self, t):
        r = self.ranges
        with self._lock:
            ax = {k: self._axis(k) for k in ("vx", "vy", "wz")}
        scale = lambda v, rng: v * (rng[1] if v >= 0 else -rng[0])  # noqa: E731
        cmd = np.array([scale(ax["vx"], r["lin_vel_x"]), scale(ax["vy"], r["lin_vel_y"]), scale(ax["wz"], r["ang_vel_z"])])
        return _clip(cmd, r)

    def events(self, t):
        with self._lock:
            pressed, self._pressed = self._pressed, []
        names = {v: k for k, v in self.cfg["buttons"].items()}
        return [names[b] for b in pressed if b in names]


class KeyboardCommand(CommandSource):
    """MuJoCo viewer keys: W/S vx, A/D vy, Q/E wz (steps of `step`), SPACE zero, ENTER stand->walk, BACKSPACE damp."""

    def __init__(self, ranges, step=0.25):
        self.ranges = ranges
        self.step = step
        self.cmd = np.zeros(3)
        self._ev = ["stand"]

    def key_callback(self, keycode):
        k = chr(keycode) if 32 <= keycode < 127 else keycode
        d = {"W": (0, 1), "S": (0, -1), "A": (1, 1), "D": (1, -1), "Q": (2, 1), "E": (2, -1)}
        if k in d:
            i, s = d[k]
            self.cmd[i] += s * self.step
            self.cmd = _clip(self.cmd, self.ranges)
        elif k == " ":
            self.cmd[:] = 0
        elif keycode == 257:  # GLFW_KEY_ENTER
            self._ev.append("walk")
        elif keycode == 259:  # GLFW_KEY_BACKSPACE
            self._ev.append("damp")
        print(f"[key] command vx {self.cmd[0]:+.2f} vy {self.cmd[1]:+.2f} wz {self.cmd[2]:+.2f}", flush=True)

    def velocity(self, t):
        return self.cmd.copy()

    def events(self, t):
        ev, self._ev = self._ev, []
        return ev
