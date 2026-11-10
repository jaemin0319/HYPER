"""Gym registration of the HYPER velocity tasks (same pattern as isaaclab_tasks .../config/go2/__init__.py)."""
import gymnasium as gym

from . import agents

_ENV = f"{__name__}.env_cfg"
_PPO = f"{agents.__name__}.rsl_rl_ppo_cfg"

for _terrain in ("Flat", "Rough"):
    for _play in ("", "-Play"):
        gym.register(
            id=f"Isaac-Velocity-{_terrain}-HYPER{_play}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{_ENV}:Hyper{_terrain}EnvCfg{'_PLAY' if _play else ''}",
                "rsl_rl_cfg_entry_point": f"{_PPO}:Hyper{_terrain}PPORunnerCfg",
            },
        )
