"""Isaac Lab articulation config for HYPER v1 (hyper_v1).

Every value below comes from unitree_rl_lab UNITREE_GO2_CFG
(https://github.com/unitreerobotics/unitree_rl_lab @ 4960b84, source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py, line 95):
init_state, the GO2HV actuator (unitree_actuators.UnitreeActuatorCfg_Go2HV: X1=13.5, X2=30, Y1=20.2, Y2=23.4),
stiffness/damping/friction, soft_joint_pos_limit_factor (0.9)
and joint_sdk_names. Joint limits, effort and velocity are in the URDF, copied from go2.usd.
Only the spawn differs: HYPER has a URDF, so it uses Unitree's own URDF spawn (UnitreeUrdfFileCfg).
Generated, not run: Isaac Lab was not installed where this file was made.
"""
import os

from isaaclab.assets.articulation import ArticulationCfg

from unitree_rl_lab.assets.robots import unitree_actuators
from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg, UnitreeUrdfFileCfg

HYPER_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "urdf", "hyper.urdf")

HYPER_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=HYPER_URDF,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={'.*R_hip_joint': -0.1, '.*L_hip_joint': 0.1, 'F[L,R]_thigh_joint': 0.8, 'R[L,R]_thigh_joint': 1.0, '.*_calf_joint': -1.5},
        joint_vel={'.*': 0.0},
    ),
    actuators={
        "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=['.*'],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
        ),
    },
    joint_sdk_names=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'],
)
