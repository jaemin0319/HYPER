"""HYPER v1 articulation for Isaac Lab, set up like Isaac Lab's UNITREE_GO2_CFG (isaaclab_assets/robots/unitree.py).

Differences from UNITREE_GO2_CFG, all from the hardware:
  - USD: usd/hyper.usd made from urdf/hyper.urdf by isaacsim/convert_urdf.py
  - motor: Unitree GO-M8010-6 on all 12 joints (gear 6.33, 23.7 N*m and 30 rad/s at the output). The calf
    motor sits on the thigh and drives the knee through a 1:1 belt, so the calf joint sees the same motor
    limits. Go2's DCMotorCfg values 23.5 / 23.5 / 30.0 become 23.7 / 23.7 / 30.0.
  - self collisions ON (Go2: off). HYPER's thighs reach the body housing inside the joint limits (CAD
    interference: front thigh back from ~1.40 rad, hip abduction outward from ~0.85 rad). A policy trained with
    self collisions off walked with the front thighs inside the body (found in MuJoCo sim2sim), which the real
    robot cannot do.
Everything else (spawn properties, init state, soft limits, PD gains 25 / 0.5, no friction) is Go2's.
"""
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

REPO_DIR = Path(__file__).resolve().parents[2]
HYPER_USD = REPO_DIR / "usd" / "hyper.usd"

# Unitree GO-M8010-6 (https://www.unitree.com/go1/motor): output side values.
GO_M8010_6_GEAR_RATIO = 6.33
GO_M8010_6_MAX_TORQUE = 23.7  # N*m
GO_M8010_6_MAX_SPEED = 30.0  # rad/s
CALF_BELT_RATIO = 1.0  # calf joint angle = calf motor output angle (motor on the thigh, belt 1:1)

HYPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(HYPER_USD),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=GO_M8010_6_MAX_TORQUE,
            saturation_effort=GO_M8010_6_MAX_TORQUE,
            velocity_limit=GO_M8010_6_MAX_SPEED * CALF_BELT_RATIO,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""HYPER v1 with GO-M8010-6 DC motor models (explicit PD, like Go2)."""
