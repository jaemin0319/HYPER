"""Open usd/hyper_scene.usda in the Isaac Sim GUI and press Play (robot stands with the drive PD).

  python isaacsim/view.py            # GUI, plays immediately
  python isaacsim/view.py --paused   # GUI, press Play yourself

Without this script: Isaac Sim > File > Open > usd/hyper_scene.usda (or usd/hyper.usd for the bare
asset) > Play. Close the window to quit.
"""
import argparse
import os
from pathlib import Path

from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--scene", default=str(ROOT / "usd" / "hyper_scene.usda"))
ap.add_argument("--paused", action="store_true")
args = ap.parse_args()

app = SimulationApp({"headless": False, "width": 1600, "height": 1000})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402

omni.usd.get_context().open_stage(str(Path(args.scene).resolve()))
for _ in range(20):
    app.update()
set_camera_view(eye=[1.1, -1.0, 0.6], target=[0.0, 0.0, 0.25])
if not args.paused:
    omni.timeline.get_timeline_interface().play()
print(f"[view] {args.scene} opened ({'paused' if args.paused else 'playing'}). Close the window to quit.", flush=True)
while app.is_running():
    app.update()
os._exit(0)  # SimulationApp.close() can hang on this PC
