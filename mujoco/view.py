"""Opens HYPER in the MuJoCo viewer, standing on Unitree Go2's control settings.

    python view.py            (from the mujoco folder of the HYPER package)

Control follows Isaac Lab's Go2 config (isaaclab_assets UNITREE_GO2_CFG, DCMotorCfg) with HYPER's motor,
Unitree GO-M8010-6 on every joint (calf belt driven 1:1): PD with stiffness 25 N·m/rad and damping
0.5 N·m·s/rad toward the init_state pose, then Isaac Lab's DCMotor four-quadrant torque-speed limit
(stall 23.7 N·m, no-load 30 rad/s, cap 23.7 N·m). Needs only `pip install mujoco`.
"""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = Path(__file__).resolve().parent / "hyper_scene.xml"
STIFFNESS, DAMPING = 25.0, 0.5
SATURATION_EFFORT, EFFORT_LIMIT, VELOCITY_LIMIT = 23.7, 23.7, 30.0   # GO-M8010-6 output: N·m, N·m, rad/s


def torque_limit(tau, qd):
    """isaaclab.actuators.DCMotor._clip_effort."""
    qd = np.clip(qd, -VELOCITY_LIMIT * (1 + EFFORT_LIMIT / SATURATION_EFFORT), VELOCITY_LIMIT * (1 + EFFORT_LIMIT / SATURATION_EFFORT))
    top = np.minimum(SATURATION_EFFORT * (1.0 - qd / VELOCITY_LIMIT), EFFORT_LIMIT)
    bottom = np.maximum(SATURATION_EFFORT * (-1.0 - qd / VELOCITY_LIMIT), -EFFORT_LIMIT)
    return np.clip(tau, bottom, top)


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
        tau = torque_limit(tau, qd)
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
