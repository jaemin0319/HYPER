"""Render usd/hyper_scene.usda headless with RTX after a few seconds of standing and save a PNG.

  python isaacsim/snapshot.py [--seconds 3] [--out isaacsim/images/standing_iso.png]
"""
import argparse
import os
from pathlib import Path

from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--scene", default=str(ROOT / "usd" / "hyper_scene.usda"))
ap.add_argument("--seconds", type=float, default=3.0)
ap.add_argument("--out", default=str(ROOT / "isaacsim" / "images" / "standing_iso.png"))
ap.add_argument("--eye", type=float, nargs=3, default=[1.1, -1.0, 0.6])
ap.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.25])
args = ap.parse_args()

app = SimulationApp({"headless": True, "width": 1280, "height": 800})
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402

omni.usd.get_context().open_stage(str(Path(args.scene).resolve()))
for _ in range(10):
    app.update()
cam = rep.create.camera(position=tuple(args.eye), look_at=tuple(args.target), focal_length=24.0)
rp = rep.create.render_product(cam, (1280, 800))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach(rp)
tl = omni.timeline.get_timeline_interface()
tl.play()
t0 = tl.get_current_time()
while tl.get_current_time() - t0 < args.seconds:
    app.update()
rep.orchestrator.step(rt_subframes=16, pause_timeline=True)
img = np.asarray(ann.get_data())[..., :3]
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(img).save(out)
print(f"[snapshot] t={tl.get_current_time() - t0:.2f}s -> {out}", flush=True)
os._exit(0)
