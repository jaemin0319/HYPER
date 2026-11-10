"""MDP terms HYPER adds to Isaac Lab's locomotion velocity terms."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_filtered_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Like mdp.illegal_contact, but on the contact forces between the sensor bodies and the sensor's filter prims
    (here: base vs ground). With self collisions on, the base's net contact force also holds thigh-on-body contacts,
    which are not a fall."""
    sensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.force_matrix_w_history  # (envs, history, bodies, filters, 3)
    return (torch.norm(f, dim=-1).amax(dim=1) > threshold).flatten(1).any(dim=1)
