"""Convert urdf/hyper.urdf to a USD asset that Isaac Sim 5.1 / Isaac Lab 2.3 can load directly.

Run (Isaac Sim 5.1 + Isaac Lab 2.3 python):
  python isaacsim/convert_urdf.py            # writes usd/hyper.usd (+ usd/configuration/*)

Why a build copy of the URDF: the Isaac Sim URDF importer (2.4.30) names each visual prim after its
mesh file, and a name like `calf-foot.stl` gives `/visuals/FL_calf/calf-foot`, which is not a valid USD
prim name ('-' is not allowed). The import then stops with "Used null prim" and Isaac Lab still writes
an empty USD. The repo meshes were renamed to `calf_foot*` (2026-11-10), so nothing is renamed now; the
guard stays for future exports: meshes whose file name is not a USD identifier are hard-linked under
usd/_build/meshes with '-' -> '_' and the URDF copy (absolute mesh paths) points at them.

Import settings follow unitree_rl_lab's URDF spawn for Go2 (floating base, self collision on,
foot links kept through `dont_collapse`) and the README:
  - masses / inertia come from the URDF (link_density 0 -> no density recompute)
  - each per-part collision STL becomes its own convex hull (collider_type convex_hull)
  - joint drives: force-type position PD, stiffness 25 N*m/rad, damping 0.5 N*m*s/rad (GO2HV values),
    so the asset holds a pose when it is opened in plain Isaac Sim. Isaac Lab actuators overwrite these.
"""
import argparse
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default=str(ROOT / "urdf" / "hyper.urdf"))
parser.add_argument("--out", default=str(ROOT / "usd" / "hyper.usd"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True


def usd_safe_urdf(src: Path, build: Path) -> Path:
    """Copy the URDF into `build`, pointing meshes with non-identifier names at safe hard links."""
    if build.exists():
        shutil.rmtree(build)
    (build / "meshes").mkdir(parents=True)
    tree = ET.parse(src)
    renamed = {}
    for mesh in tree.iter("mesh"):
        f = (src.parent / mesh.get("filename")).resolve()
        if not f.is_file():
            sys.exit(f"[convert] missing mesh {f}")
        stem = f.stem
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stem):
            safe = re.sub(r"[^A-Za-z0-9_]", "_", stem)
            if safe[0].isdigit():
                safe = "_" + safe
            dst = build / "meshes" / f.relative_to(ROOT / "meshes").with_name(safe + f.suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.link(f, dst)
            renamed[str(f.relative_to(ROOT))] = str(dst.relative_to(ROOT))
            f = dst
        mesh.set("filename", str(f))
    out = build / src.name
    tree.write(out, encoding="utf-8", xml_declaration=True)
    for a, b in sorted(renamed.items()):
        print(f"[convert] mesh name fix: {a} -> {b}")
    return out


out = Path(args.out).resolve()
urdf = usd_safe_urdf(Path(args.urdf).resolve(), out.parent / "_build")

app = AppLauncher(args).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

cfg = UrdfConverterCfg(
    asset_path=str(urdf),
    usd_dir=str(out.parent),
    usd_file_name=out.name,
    force_usd_conversion=True,
    fix_base=False,
    merge_fixed_joints=True,  # only foot joints are fixed and they carry dont_collapse="true"
    self_collision=True,
    collider_type="convex_hull",
    link_density=0.0,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=25.0, damping=0.5),
    ),
)
conv = UrdfConverter(cfg)

# The importer makes each link's `visuals` instanceable, but binds the URDF colours from inside the
# instance prototype to /hyper_v1/Looks/material_*, i.e. outside the prototype. Renderers ignore such
# targets and fall back to the white DefaultMaterial of the STL, so the robot shows all white.
# De-instancing only the visuals (in the root layer) fixes it; collisions stay instanceable.
stage = Usd.Stage.Open(conv.usd_path)
stage.SetEditTarget(stage.GetRootLayer())
n_vis = 0
for prim in stage.GetDefaultPrim().GetChildren():
    vis = prim.GetChild("visuals")
    if vis and vis.IsInstanceable():
        vis.SetInstanceable(False)
        n_vis += 1
stage.GetRootLayer().Save()
print(f"[convert] visuals de-instanced on {n_vis} links (URDF material colours)", flush=True)

# Links without a URDF <visual> (the feet) get a `visuals` prim in hyper_base.usd that references a prim the
# importer never writes (hyper_physics.usd</visuals/FL_foot>), which warns on every load. Drop such references in
# the layer that authors them (a stronger layer cannot remove an arc from another layer stack).
from pxr import Sdf  # noqa: E402

n_ref = 0
for layer_file in sorted((out.parent / "configuration").glob("*.usd")):
    layer = Sdf.Layer.FindOrOpen(str(layer_file))
    dangling = []

    def _visit(path):
        spec = layer.GetPrimAtPath(path)
        if spec is None or not spec.hasReferences:
            return
        for ref in list(spec.referenceList.prependedItems):
            target = Sdf.Layer.FindOrOpen(layer.ComputeAbsolutePath(ref.assetPath)) if ref.assetPath else layer
            if target is None or (ref.primPath and target.GetPrimAtPath(ref.primPath) is None):
                dangling.append((spec, ref))

    layer.Traverse(Sdf.Path.absoluteRootPath, _visit)
    for spec, ref in dangling:
        spec.referenceList.prependedItems.remove(ref)
        n_ref += 1
    if dangling:
        layer.Save()
print(f"[convert] removed {n_ref} dangling references", flush=True)

# Isaac Lab does not raise when the importer fails, so check the result here.
stage = Usd.Stage.Open(conv.usd_path)
prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
n_body = sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims)
n_joint = sum(p.IsA(UsdPhysics.Joint) for p in prims)
ok = n_body == 17 and n_joint >= 16
print(f"[convert] {'OK' if ok else 'FAILED'}: {conv.usd_path} rigid bodies {n_body} (want 17), joints {n_joint} (want 16)", flush=True)
app.update()
os._exit(0 if ok else 1)  # SimulationApp.close() can hang on this PC
