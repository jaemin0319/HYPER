"""Velocity-tracking (joystick) tasks for HYPER: Isaac Lab's Unitree Go2 flat / rough tasks, unchanged except for
what HYPER's hardware has.

The configs subclass Isaac Lab's own Go2 configs, so terrain, commands, actions (joint position, scale 0.25,
default offset), sim (dt 0.005, decimation 4), events, rewards, terminations, curriculum and PPO settings are
Go2's by construction. Changes:

1. robot = HYPER_CFG (usd/hyper.usd, GO-M8010-6 DC motors).
2. policy (actor) observations = only what the robot measures or is given:
       velocity_commands (3, joystick) | joint_pos - default (12, motor encoders) |
       joint_vel (12, motor encoders)  | last action (12)
   Go2's base_lin_vel / base_ang_vel / projected_gravity (IMU, state estimator) and height_scan (lidar) are
   removed because HYPER has no IMU and no lidar; noise on joint_pos / joint_vel is Go2's.
3. critic observations = Go2's full policy observation set (sim only, never needed on the robot), so the
   value function sees what Go2's does (asymmetric actor-critic through rsl_rl obs_groups).
4. undesired_contacts back on for the thighs (Isaac Lab's base locomotion term, weight -1, 1 N; Go2 turns it
   off): with self collisions on (HYPER_CFG), a thigh touching the body or the ground is penalised instead of
   being used as a support.
5. base_contact termination counts only base-ground contact (a contact sensor on the base filtered to the
   terrain collider). Go2's term uses the base's net contact force, which with self collisions on also fires
   when a thigh touches the body; training then collapsed into 3-step episodes.
HYPER is blind: no IMU, no foot contact sensors, no lidar. Contact sensors are used for rewards/terminations in
simulation only, never as policy input.
"""
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg as Go2ObservationsCfg

from hyper_lab.robots import HYPER_CFG
from hyper_lab.tasks.velocity import mdp as hyper_mdp

# Order of the actor observation, used by the deployment code (deploy/hyper_deploy/observation.py).
POLICY_OBS_TERMS = ("velocity_commands", "joint_pos", "joint_vel", "actions")


@configclass
class HyperObservationsCfg:
    """Actor: motor feedback + joystick + last action. Critic: Go2's policy observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(Go2ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# terrain collider created by Isaac Lab's TerrainImporter ("plane": Nucleus default_environment.usd, "generator": mesh)
GROUND_COLLIDER = {"plane": "/World/ground/terrain/GroundPlane/CollisionPlane", "generator": "/World/ground/terrain/mesh"}


def _hyper(cfg):
    cfg.scene.robot = HYPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.base_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        filter_prim_paths_expr=[GROUND_COLLIDER[cfg.scene.terrain.terrain_type]],
        history_length=3,
        update_period=cfg.sim.dt,
    )
    cfg.terminations.base_contact = DoneTerm(
        func=hyper_mdp.illegal_filtered_contact,
        params={"sensor_cfg": SceneEntityCfg("base_ground_contact"), "threshold": 1.0},
    )
    cfg.rewards.undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )


@configclass
class HyperRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    observations: HyperObservationsCfg = HyperObservationsCfg()

    def __post_init__(self):
        super().__post_init__()  # Go2 rough settings
        _hyper(self)


@configclass
class HyperRoughEnvCfg_PLAY(HyperRoughEnvCfg):
    """Go2 rough PLAY changes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class HyperFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    observations: HyperObservationsCfg = HyperObservationsCfg()

    def __post_init__(self):
        super().__post_init__()  # Go2 flat settings (plane, no height scanner, flat rewards)
        _hyper(self)
        self.observations.critic.height_scan = None  # the flat task has no height scanner


@configclass
class HyperFlatEnvCfg_PLAY(HyperFlatEnvCfg):
    """Go2 flat PLAY changes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
