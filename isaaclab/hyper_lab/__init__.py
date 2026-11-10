"""HYPER v1 for Isaac Lab 2.3: Isaac Lab's Unitree Go2 velocity tasks with HYPER's robot and sensors.

Importing this package only registers the gym tasks (string entry points), so it is safe before the
simulation app starts:

    Isaac-Velocity-Flat-HYPER-v0      Isaac-Velocity-Flat-HYPER-Play-v0
    Isaac-Velocity-Rough-HYPER-v0     Isaac-Velocity-Rough-HYPER-Play-v0
"""
from .tasks import velocity  # noqa: F401
