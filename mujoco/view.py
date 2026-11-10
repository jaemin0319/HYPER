"""Opens HYPER in the MuJoCo viewer, standing on Unitree Go2's control settings.

    python view.py            (from the mujoco folder of the HYPER package)

Control follows unitree_rl_lab's Go2 config (UNITREE_GO2_CFG, UnitreeActuatorCfg_Go2HV):
PD with stiffness 25 N·m/rad and damping 0.5 N·m·s/rad toward the init_state pose, then the torque-speed
limit of UnitreeActuator (X1, X2, Y1, Y2 below). Needs only `pip install mujoco`.
"""
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = Path(__file__).resolve().parent / "hyper_scene.xml"
STIFFNESS, DAMPING = 25.0, 0.5
X1, X2, Y1, Y2 = 13.5, 30.0, 20.2, 23.4      # rad/s, rad/s, N·m, N·m
FS, FD, VA = 0.0, 0.0, 0.01                   # Go2HV leaves the actuator friction model at zero


def torque_limit(tau, qd):
    """UnitreeActuator._clip_effort: Y1 when torque and speed share a sign, else Y2, falling linearly to zero
    between X1 and X2."""
    max_effort = np.where(qd * tau > 0, Y1, Y2)
    slope = -max_effort / (X2 - X1)
    fading = np.clip(slope * (np.abs(qd) - X1) + max_effort, 0.0, None)
    max_effort = np.where(np.abs(qd) < X1, max_effort, fading)
    return np.clip(tau, -max_effort, max_effort)


class Controller:
    def __init__(self, model, key="init"):
        self.joint_ids = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key)
        self.target = model.key_qpos[k][self.qpos_adr].copy()

    def __call__(self, model, data):
        q = data.qpos[self.qpos_adr]
        qd = data.qvel[self.dof_adr]
        tau = STIFFNESS * (self.target - q) - DAMPING * qd
        tau = torque_limit(tau, qd) - (FS * np.tanh(qd / VA) + FD * qd)
        data.ctrl[:] = tau
        return tau


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    key = sys.argv[1] if len(sys.argv) > 1 else "init"
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key))
    control = Controller(model, key)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            control(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
            rest = model.opt.timestep - (time.perf_counter() - start)
            if rest > 0:
                time.sleep(rest)


if __name__ == "__main__":
    main()
