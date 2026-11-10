"""Build usd/hyper_scene.usda: ground + lights + physics scene + HYPER (referenced from ./hyper.usd)
standing in the Isaac Lab init pose, so File > Open + Play in Isaac Sim works without any script.

  python isaacsim/make_scene.py

The robot asset (usd/hyper.usd) is not modified; the scene only adds overrides on top of the reference:
  - root at INIT_BASE_POS (0, 0, 0.4), like unitree_rl_lab's init_state
  - joint initial state (PhysicsJointStateAPI) and drive targets = init joint pose
  - link transforms = FK of the init pose, so the pre-Play view already shows that pose
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import urdf_model as um  # noqa: E402

from isaacsim import SimulationApp  # noqa: E402

import argparse  # noqa: E402

ROOT = um.ROOT
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "usd" / "hyper_scene.usda"))
ap.add_argument("--dome", type=float, default=120.0, help="dome light intensity")
ap.add_argument("--sun", type=float, default=600.0, help="distant light intensity")
args = ap.parse_args()
SCENE = Path(args.out).resolve()
app = SimulationApp({"headless": True})

from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

try:
    def set_pose(prim, M):
        """Write a 4x4 into the prim's existing translate/orient ops (the referenced asset authors them)."""
        ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
        if not ops:
            xf = UsdGeom.Xformable(prim)
            ops = {"xformOp:translate": xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble),
                   "xformOp:orient": xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)}
        if not {"xformOp:translate", "xformOp:orient"} <= set(ops) or len(set(ops) - {"xformOp:translate", "xformOp:orient", "xformOp:scale"}):
            raise RuntimeError(f"{prim.GetPath()}: unexpected xform ops {list(ops)}")
        t = ops["xformOp:translate"]
        t.Set(type(t.Get())(*M[:3, 3]) if t.Get() is not None else Gf.Vec3d(*M[:3, 3]))
        o = ops["xformOp:orient"]
        w, x, y, z = um.R_to_quat_wxyz(M[:3, :3])
        cur = o.Get()
        o.Set(type(cur)(w, type(cur.GetImaginary())(x, y, z)) if cur is not None else Gf.Quatd(w, x, y, z))


    links, joints = um.load()
    q0 = {j.name: um.init_joint_pos(j.name) for j in joints.values() if j.type == "revolute"}
    poses = um.fk(links, joints, q0)  # relative to the robot root prim

    stage = Usd.Stage.CreateNew(str(SCENE))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(1_000_000)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # physics: 200 Hz like unitree_rl_lab (sim dt 0.005), gravity -z
    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    px.CreateTimeStepsPerSecondAttr(200)

    # ground: infinite collision plane + 25 m visual quad, friction 1.0 (Isaac Lab terrain default)
    PhysicsSchemaTools.addGroundPlane(stage, "/World/groundPlane", "Z", 25.0, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.42, 0.44, 0.46))
    mat = UsdShade.Material.Define(stage, "/World/Physics/groundMaterial")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(1.0)
    pm.CreateDynamicFrictionAttr(1.0)
    pm.CreateRestitutionAttr(0.0)
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/groundPlane")):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/dome")
    dome.CreateIntensityAttr(args.dome)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/sun")
    sun.CreateIntensityAttr(args.sun)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(35, 0, 40))

    # robot
    robot = UsdGeom.Xform.Define(stage, "/World/HYPER")
    robot.GetPrim().GetReferences().AddReference("./" + os.path.relpath(ROOT / "usd" / "hyper.usd", SCENE.parent))
    set_pose(robot.GetPrim(), um.T(p=um.INIT_BASE_POS))

    jroot = "/World/HYPER/joints"
    for name, q in q0.items():
        jp = stage.GetPrimAtPath(f"{jroot}/{name}")
        if not jp.IsValid():
            sys.exit(f"[scene] joint prim missing: {jroot}/{name}")
        deg = float(np.degrees(q))
        jp.GetAttribute("state:angular:physics:position").Set(deg)
        jp.GetAttribute("state:angular:physics:velocity").Set(0.0)
        jp.GetAttribute("drive:angular:physics:targetPosition").Set(deg)

    for name, M in poses.items():
        set_pose(stage.GetPrimAtPath(f"/World/HYPER/{name}"), M)

    stage.GetRootLayer().documentation = "HYPER v1 in Isaac Sim: ground, lights, robot at the Isaac Lab init pose. Made by isaacsim/make_scene.py"
    stage.Save()
    print(f"[scene] wrote {SCENE}", flush=True)
except Exception:
    import traceback

    traceback.print_exc()
    os._exit(1)
os._exit(0)
