"""Dump the prim tree of a USD (types, applied APIs, variant selections, key physics attributes).

  python isaacsim/dump_usd.py [usd/hyper.usd] > outputs/logs/usd_tree.txt
"""
import os
import sys
from pathlib import Path

from isaacsim import SimulationApp

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "usd" / "hyper.usd").resolve()
app = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom  # noqa: E402

stage = Usd.Stage.Open(str(path))
out = []
out.append(f"file {path}")
out.append(f"defaultPrim {stage.GetDefaultPrim().GetPath()}  upAxis {UsdGeom.GetStageUpAxis(stage)}  metersPerUnit {UsdGeom.GetStageMetersPerUnit(stage)}")
out.append("layers: " + ", ".join(l.identifier.split('/')[-1] for l in stage.GetLayerStack()))
KEYS = ("physics:", "physxArticulation", "drive:", "physxJoint", "physxCollision", "physxConvex", "limit:", "state:")
for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    if prim.IsPseudoRoot():
        continue
    depth = len(prim.GetPath().pathString.split("/")) - 2
    vs = prim.GetVariantSets()
    vtxt = " ".join(f"{n}={vs.GetVariantSelection(n)}{vs.GetVariantSet(n).GetVariantNames()}" for n in vs.GetNames())
    apis = ",".join(prim.GetAppliedSchemas())
    line = f"{'  ' * depth}{prim.GetName()} <{prim.GetTypeName()}> [{apis}] {vtxt}"
    if prim.IsInstance():
        line += " (instance)"
    out.append(line)
    for a in prim.GetAttributes():
        n = a.GetName()
        if any(n.startswith(k) for k in KEYS) and a.HasAuthoredValue():
            v = a.Get()
            s = str(v)
            out.append(f"{'  ' * depth}    {n} = {s[:120]}")
    for r in prim.GetRelationships():
        if r.GetName().startswith("physics:") and r.GetTargets():
            out.append(f"{'  ' * depth}    {r.GetName()} -> {[str(t) for t in r.GetTargets()]}")
print("\n".join(out), flush=True)
os._exit(0)
