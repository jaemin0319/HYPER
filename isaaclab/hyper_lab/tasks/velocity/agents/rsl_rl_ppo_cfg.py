"""PPO settings: Isaac Lab's Go2 runners, with the actor/critic observation groups of the HYPER tasks."""
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
    UnitreeGo2RoughPPORunnerCfg,
)

OBS_GROUPS = {"policy": ["policy"], "critic": ["critic"]}


@configclass
class HyperRoughPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "hyper_rough"
        self.obs_groups = dict(OBS_GROUPS)


@configclass
class HyperFlatPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()  # Go2 flat: 300 iterations, [128, 128, 128]
        self.experiment_name = "hyper_flat"
        self.obs_groups = dict(OBS_GROUPS)
