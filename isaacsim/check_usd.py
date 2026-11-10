"""Check usd/hyper.usd against urdf/hyper.urdf and run physics checks on usd/hyper_scene.usda.

  python isaacsim/check_usd.py          # writes isaacsim/check_result.json, prints a table

Static (asset only): stage units/axis, variant selection, bodies, mass/CoM/inertia tensor, joint frames,
axes, limits, effort, velocity, drive gains, collision shapes, mesh placement and scale (per-link AABB of
the USD meshes vs the STL vertices placed by the URDF origin), link transforms at q = 0, materials.
Physics (PhysX, 200 Hz, CPU): the scene's authored initial state, a 6 s stand with the drive PD
(25 / 0.5, same pass rules as mujoco/check_result.json), self-contact at zero/init/CAD poses, and a
one-joint-at-a-time sweep compared with MuJoCo's joint_sweep_contacts.
"""
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import urdf_model as um  # noqa: E402

from isaacsim import SimulationApp  # noqa: E402

ROOT = um.ROOT
ASSET = ROOT / "usd" / "hyper.usd"
SCENE = ROOT / "usd" / "hyper_scene.usda"
OUT = ROOT / "isaacsim" / "check_result.json"
KP, KD = 25.0, 0.5

app = SimulationApp({"headless": True})

from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

rows = []


def row(item, expected, measured, ok):
    rows.append({"item": item, "expected": expected, "measured": measured, "pass": bool(ok)})
    print(f"[{'PASS' if ok else 'FAIL'}] {item}: expected {expected}, measured {measured}", flush=True)


def gf_to_np(m):  # Gf.Matrix4d (row-vector convention) -> 4x4 column-vector matrix
    return np.array(m, float).T


def world_of(prim):
    return gf_to_np(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))


def aabb(pts):
    return np.vstack([pts.min(0), pts.max(0)])


def static_checks(links, joints):
    stage = Usd.Stage.Open(str(ASSET))
    root = stage.GetDefaultPrim()
    vs = root.GetVariantSets()
    row("stage: up axis, meters per unit, kg per unit", ["Z", 1.0, 1.0],
        [UsdGeom.GetStageUpAxis(stage), UsdGeom.GetStageMetersPerUnit(stage), UsdPhysics.GetStageKilogramsPerUnit(stage)],
        UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1.0 and UsdPhysics.GetStageKilogramsPerUnit(stage) == 1.0)
    sel = {n: vs.GetVariantSelection(n) for n in vs.GetNames()}
    row("default prim and variant selection", {"defaultPrim": "/hyper_v1", "Physics": "PhysX"},
        {"defaultPrim": str(root.GetPath()), **sel}, str(root.GetPath()) == "/hyper_v1" and sel.get("Physics") == "PhysX")

    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    bodies = {p.GetName(): p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    row("rigid bodies = URDF links", sorted(links), sorted(bodies), sorted(bodies) == sorted(links))
    arts = [str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    selfc = [p.GetAttribute("physxArticulation:enabledSelfCollisions").Get() for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    row("one articulation root on base, self collision on", [["/hyper_v1/base"], [True]], [arts, selfc], arts == ["/hyper_v1/base"] and selfc == [True])

    # mass properties
    em, ec, ei = 0.0, 0.0, 0.0
    total = 0.0
    for n, L in links.items():
        m = UsdPhysics.MassAPI(bodies[n])
        mass = m.GetMassAttr().Get()
        com = np.array(m.GetCenterOfMassAttr().Get(), float)
        d = np.array(m.GetDiagonalInertiaAttr().Get(), float)
        q = m.GetPrincipalAxesAttr().Get()
        R = um.quat_wxyz_to_R([q.GetReal(), *q.GetImaginary()])
        I = R @ np.diag(d) @ R.T
        total += mass
        em = max(em, abs(mass - L.mass) / L.mass)
        ec = max(ec, float(np.abs(com - L.com).max()))
        ei = max(ei, float(np.abs(I - L.inertia).max() / np.abs(L.inertia).max()))
    row("body masses (max relative error)", "< 1e-6", em, em < 1e-6)
    row("body centres of mass (max error m)", "< 1e-6", ec, ec < 1e-6)
    row("inertia tensor R diag R^T vs URDF (max error / max |I|)", "< 1e-4", ei, ei < 1e-4)
    urdf_total = sum(L.mass for L in links.values())
    row("total mass kg", urdf_total, total, abs(total - urdf_total) < 1e-4)

    # joints
    ujoints = {p.GetName(): p for p in prims if p.IsA(UsdPhysics.Joint)}
    row("joints = URDF joints (12 revolute + 4 fixed)", sorted(joints), sorted(ujoints), sorted(ujoints) == sorted(joints))
    bad = []
    for n, J in joints.items():
        p = ujoints[n]
        uj = UsdPhysics.Joint(p)
        b0 = [t.name for t in uj.GetBody0Rel().GetTargets()]
        b1 = [t.name for t in uj.GetBody1Rel().GetTargets()]
        pos0 = np.array(uj.GetLocalPos0Attr().Get(), float)
        pos1 = np.array(uj.GetLocalPos1Attr().Get(), float)
        r0, r1 = uj.GetLocalRot0Attr().Get(), uj.GetLocalRot1Attr().Get()
        R0 = um.quat_wxyz_to_R([r0.GetReal(), *r0.GetImaginary()])
        R1 = um.quat_wxyz_to_R([r1.GetReal(), *r1.GetImaginary()])
        # child frame in parent = T(pos0,R0) * T(pos1,R1)^-1 must equal the URDF origin
        M = um.T(R0, pos0) @ np.linalg.inv(um.T(R1, pos1))
        err = float(np.abs(M - J.origin).max())
        if b0 != [J.parent] or b1 != [J.child] or err > 1e-6:
            bad.append(f"{n}: bodies {b0}->{b1}, frame err {err:.2e}")
        if J.type == "revolute":
            if not p.IsA(UsdPhysics.RevoluteJoint):
                bad.append(f"{n}: not revolute")
                continue
            rj = UsdPhysics.RevoluteJoint(p)
            ax = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[rj.GetAxisAttr().Get()]
            ax_parent = R0 @ np.array(ax, float)  # USD axis is in the joint frame
            lo, hi = math.radians(rj.GetLowerLimitAttr().Get()), math.radians(rj.GetUpperLimitAttr().Get())
            drv = UsdPhysics.DriveAPI(p, "angular")
            eff = drv.GetMaxForceAttr().Get()
            kp = drv.GetStiffnessAttr().Get() * 180 / math.pi
            kd = drv.GetDampingAttr().Get() * 180 / math.pi
            vel = math.radians(p.GetAttribute("physxJoint:maxJointVelocity").Get())
            if np.abs(ax_parent - J.axis).max() > 1e-6:
                bad.append(f"{n}: axis {ax_parent} vs {J.axis}")
            if abs(lo - J.lower) > 1e-5 or abs(hi - J.upper) > 1e-5:
                bad.append(f"{n}: limits {lo:.6f},{hi:.6f} vs {J.lower},{J.upper}")
            if abs(eff - J.effort) > 1e-5 * J.effort or abs(vel - J.velocity) > 1e-5 * J.velocity:
                bad.append(f"{n}: effort/velocity {eff},{vel} vs {J.effort},{J.velocity}")
            if abs(kp - KP) > 1e-4 or abs(kd - KD) > 1e-6 or drv.GetTypeAttr().Get() != "force":
                bad.append(f"{n}: drive {drv.GetTypeAttr().Get()} kp {kp} kd {kd}")
        elif not p.IsA(UsdPhysics.FixedJoint):
            bad.append(f"{n}: not fixed")
    row("joint frames, bodies, axes, limits, effort, velocity, drive (force PD 25/0.5)", [], bad, not bad)
    world_joints = [n for n, p in ujoints.items() if not UsdPhysics.Joint(p).GetBody0Rel().GetTargets()]
    row("floating base (no joint to world)", [], world_joints, not world_joints)

    # link transforms at q = 0
    fk0 = um.fk(links, joints, {})
    err = max(float(np.abs(world_of(bodies[n]) - fk0[n]).max()) for n in links)
    row("link transforms in the asset = URDF FK at q = 0 (max error)", "< 1e-6", err, err < 1e-6)

    # collisions and mesh placement
    bad, worst_v, worst_c = [], 0.0, 0.0
    for n, L in links.items():
        lw = np.linalg.inv(world_of(bodies[n]))
        sub = list(Usd.PrimRange(bodies[n], Usd.TraverseInstanceProxies()))

        def pts_in_link(prim_list):
            out = []
            for mp in prim_list:
                if mp.IsA(UsdGeom.Mesh):
                    P = np.array(UsdGeom.Mesh(mp).GetPointsAttr().Get(), float)
                    M = lw @ world_of(mp)
                    out.append(P @ M[:3, :3].T + M[:3, 3])
            return np.vstack(out) if out else None

        vis_root = [p for p in sub if p.GetName() == "visuals" and p.GetParent() == bodies[n]]
        vis = [p for p in Usd.PrimRange(vis_root[0], Usd.TraverseInstanceProxies())] if vis_root else []
        got = pts_in_link(vis)
        exp_list = []
        for g in L.visuals:
            V = um.read_stl(g.filename) * g.scale
            exp_list.append(V @ g.origin[:3, :3].T + g.origin[:3, 3])
        if exp_list:
            if got is None:
                bad.append(f"{n}: no visual mesh")
            else:
                e = float(np.abs(aabb(got) - aabb(np.vstack(exp_list))).max())
                worst_v = max(worst_v, e)
        elif got is not None:
            bad.append(f"{n}: unexpected visual mesh")

        cols = [p for p in sub if p.HasAPI(UsdPhysics.CollisionAPI)]
        if len(cols) != len(L.collisions):
            bad.append(f"{n}: {len(cols)} colliders vs {len(L.collisions)} in URDF")
        for g in L.collisions:
            if g.kind == "sphere":
                sp = [p for p in cols if p.IsA(UsdGeom.Sphere)]
                if len(sp) != 1:
                    bad.append(f"{n}: sphere collider missing")
                    continue
                M = lw @ world_of(sp[0])
                r = UsdGeom.Sphere(sp[0]).GetRadiusAttr().Get() * np.cbrt(abs(np.linalg.det(M[:3, :3])))
                e = max(abs(r - g.radius), float(np.abs(M[:3, 3] - g.origin[:3, 3]).max()))
                worst_c = max(worst_c, e)
                continue
            name = g.name.replace("-", "_")
            cp = [p for p in cols if p.GetParent().GetName() == name or p.GetName() == name]
            if len(cp) != 1:
                bad.append(f"{n}: collider {name} not found")
                continue
            appr = UsdPhysics.MeshCollisionAPI(cp[0]).GetApproximationAttr().Get()
            if appr != "convexHull":
                bad.append(f"{n}/{name}: approximation {appr}")
            got = pts_in_link(list(Usd.PrimRange(cp[0], Usd.TraverseInstanceProxies())))
            V = um.read_stl(g.filename) * g.scale
            exp = V @ g.origin[:3, :3].T + g.origin[:3, 3]
            worst_c = max(worst_c, float(np.abs(aabb(got) - aabb(exp)).max()))
    row("colliders per link = URDF (convexHull meshes, foot sphere r 0.025)", [], bad, not bad)
    row("visual mesh AABB in link frame vs STL x URDF origin (max error m)", "< 1e-5", worst_v, worst_v < 1e-5)
    row("collision mesh AABB / sphere vs URDF (max error m)", "< 1e-5", worst_c, worst_c < 1e-5)

    # materials: renderers resolve bindings on the instance prototype, where a target outside the
    # prototype is dropped, so visual meshes must not be instance proxies and must resolve to the
    # URDF material (material_<name>), not the STL DefaultMaterial.
    mats, bad = {}, []
    for n, L in links.items():
        want = {f"material_{g.material}" for g in L.visuals if g.material}
        for p in Usd.PrimRange(bodies[n], Usd.TraverseInstanceProxies()):
            if p.IsA(UsdGeom.Mesh) and "/visuals/" in str(p.GetPath()):
                m, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
                mats[n] = str(m.GetPath()) if m else None
                if p.IsInstanceProxy():
                    bad.append(f"{n}: visual mesh is an instance proxy")
                if not m or m.GetPrim().GetName() not in want:
                    bad.append(f"{n}: bound {mats[n]}, want {sorted(want)}")
    row("visual meshes not instanced and bound to the URDF material", [], bad, not bad)
    return mats


def physics_checks(links, joints, mj):
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from omni.physx import get_physx_simulation_interface

    omni.usd.get_context().open_stage(str(SCENE))
    for _ in range(5):
        app.update()
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())  # contact reporting only for this check, scene file untouched
    for n in links:
        api = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(f"/World/HYPER/{n}"))
        api.CreateThresholdAttr(0.0)
    stage.SetEditTarget(stage.GetRootLayer())

    world = World(physics_dt=0.005, rendering_dt=0.005, stage_units_in_meters=1.0)
    robot = world.scene.add(SingleArticulation(prim_path="/World/HYPER", name="hyper"))
    world.reset()
    sim = get_physx_simulation_interface()
    dof = list(robot.dof_names)
    rev = [j.name for j in joints.values() if j.type == "revolute"]
    row("articulation DOFs = URDF revolute joints", sorted(rev), sorted(dof), sorted(dof) == sorted(rev))
    effort = np.array([joints[n].effort for n in dof])

    def link_of(path):
        s = str(path)
        if s.startswith("/World/HYPER/"):
            return s.split("/")[3]
        return "ground" if "groundPlane" in s else s

    def contacts():
        headers, data = sim.get_contact_report()
        out = {}
        for h in headers:
            if h.num_contact_data == 0:
                continue
            a = link_of(PhysicsSchemaTools.intToSdfPath(h.actor0))
            b = link_of(PhysicsSchemaTools.intToSdfPath(h.actor1))
            sep = min(data[i].separation for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data))
            k = "|".join(sorted((a, b)))
            out[k] = min(out.get(k, np.inf), float(sep))
        return out

    q_init = np.array([um.init_joint_pos(n) for n in dof])
    # 1) authored initial state of the scene file (what Play in the GUI starts from)
    q = robot.get_joint_positions()
    pos, quat = robot.get_world_pose()
    e = float(np.abs(q - q_init).max())
    row("scene start = Isaac Lab init state (joint err rad, base z m)", ["< 1e-3", 0.4], [e, float(pos[2])],
        e < 1e-3 and abs(pos[2] - 0.4) < 1e-3)

    # 2) stand 6 s with PD to the init pose
    robot.apply_action(ArticulationAction(joint_positions=q_init))
    contacts()
    steps, T_end = 1200, 6.0
    hist_z, tau_max, floor, self_min, finite = [], 0.0, {}, {}, True
    for k in range(steps):
        world.step(render=False)
        pos, quat = robot.get_world_pose()
        qj, qd = robot.get_joint_positions(), robot.get_joint_velocities()
        finite &= bool(np.all(np.isfinite(pos)) and np.all(np.isfinite(qj)))
        hist_z.append(float(pos[2]))
        c = contacts()
        if k >= steps - 400:  # last 2 s
            tau = np.clip(KP * (q_init - qj) - KD * qd, -effort, effort)
            tau_max = max(tau_max, float(np.abs(tau).max()))
            for key, sep in c.items():
                a, b = key.split("|")
                if "ground" in (a, b):
                    if sep < 1e-3:
                        other = b if a == "ground" else a
                        floor[other] = min(floor.get(other, np.inf), sep)
                else:
                    self_min[key] = min(self_min.get(key, np.inf), sep)
    w, x, y, z = quat
    up = um.quat_wxyz_to_R([w, x, y, z])[:, 2]
    tilt = math.degrees(math.acos(np.clip(up[2], -1, 1)))
    dz = max(hist_z[-400:]) - min(hist_z[-400:])
    st = mj["standing"]
    row("simulation stays finite", True, finite, finite)
    row("standing: base tilt deg after settling", "< 5", tilt, tilt < 5)
    row("standing: base height change in the last 2 s (m)", "< 0.001", dz, dz < 1e-3)
    feet = sorted(n for n in links if n.endswith("_foot"))
    row("standing: only feet touch the floor (sep < 1 mm)", feet, sorted(floor), sorted(floor) == feet)
    pen = {k: v for k, v in self_min.items() if v < 0}
    row("standing: no self-penetration", {}, pen, not pen)
    h = hist_z[-1]
    row("standing: base height vs MuJoCo (m)", f"{st['final_base_height_m']:.4f} +- 0.01", h, abs(h - st["final_base_height_m"]) < 0.01)
    qf = robot.get_joint_positions()
    mj_q = dict(zip([f"{l}_{j}_joint" for l in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")], st["final_qpos"][7:]))
    dq = float(max(abs(qf[i] - mj_q[n]) for i, n in enumerate(dof)))
    row("standing: joint angles vs MuJoCo final_qpos (max abs rad)", "< 0.05", dq, dq < 0.05)
    row("standing: max |PD torque| in the last 2 s (N*m, MuJoCo max_abs_torque for reference)",
        f"<= effort, MuJoCo {st['max_abs_torque']:.2f}", tau_max, tau_max < effort.min())
    standing = {"final_base_height_m": h, "tilt_deg": tilt, "height_change_last_2s_m": dz, "max_abs_pd_torque": tau_max,
                "floor_contacts_min_sep_m": floor, "final_joint_pos": dict(zip(dof, map(float, qf))),
                "base_pos": list(map(float, pos)), "base_quat_wxyz": list(map(float, quat))}

    # 3) self contact at fixed poses, floating at z = 1.5 (one step from the pose)
    def at_pose(qv):
        robot.set_world_pose(np.array([0.0, 0.0, 1.5]), np.array([1.0, 0.0, 0.0, 0.0]))
        robot.set_linear_velocity(np.zeros(3))
        robot.set_angular_velocity(np.zeros(3))
        robot.set_joint_positions(qv)
        robot.set_joint_velocities(np.zeros(len(dof)))
        robot.apply_action(ArticulationAction(joint_positions=qv))
        contacts()
        world.step(render=False)
        return {k: v for k, v in contacts().items() if v < 0 and "ground" not in k}

    pose_contacts = {}
    for pname, fn in (("zero", lambda n: 0.0), ("init", um.init_joint_pos), ("cad", um.cad_joint_pos)):
        pose_contacts[pname] = at_pose(np.array([fn(n) for n in dof]))
        row(f"no self-contact at {pname} pose", {}, pose_contacts[pname], not pose_contacts[pname])

    # 4) one joint at a time through its full range, others at the init pose
    sweep = {}
    for i, n in enumerate(dof):
        J = joints[n]
        grid = np.linspace(J.lower, J.upper, 201)
        hit = {}
        for a in grid:
            qv = q_init.copy()
            qv[i] = a
            for key in at_pose(qv):
                hit.setdefault(key, []).append(float(a))
        sweep[n] = {k: [round(min(v), 4), round(max(v), 4)] for k, v in hit.items()}
    # compare with MuJoCo (pair sets and interval ends)
    mj_sw = mj["joint_sweep_contacts"]
    diffs, worst = [], 0.0
    for n in dof:
        a = {"|".join(sorted(k.split("|"))): v for k, v in sweep[n].items()}
        b = {"|".join(sorted(k.split("|"))): v for k, v in mj_sw.get(n, {}).items()}
        if set(a) != set(b):
            diffs.append(f"{n}: PhysX only {sorted(set(a) - set(b))}, MuJoCo only {sorted(set(b) - set(a))}")
        for k in set(a) & set(b):
            worst = max(worst, abs(a[k][0] - b[k][0]), abs(a[k][1] - b[k][1]))
    row("joint sweep: same contact pairs as MuJoCo", [], diffs, not diffs)
    row("joint sweep: contact interval ends vs MuJoCo (max abs rad)", "< 0.1", worst, worst < 0.1)
    return {"standing": standing, "pose_contacts": pose_contacts, "joint_sweep_contacts": sweep}


result = {"asset": str(ASSET.relative_to(ROOT)), "scene": str(SCENE.relative_to(ROOT))}
code = 1
try:
    import isaacsim.core.version as _v  # noqa: F401

    links, joints = um.load()
    mj = json.loads((ROOT / "mujoco" / "check_result.json").read_text())
    result["materials"] = static_checks(links, joints)
    result.update(physics_checks(links, joints, mj))
    code = 0 if all(r["pass"] for r in rows) else 2
except Exception:
    traceback.print_exc()
result["rows"] = rows
result["all_pass"] = code == 0
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
print(f"[check] {sum(r['pass'] for r in rows)}/{len(rows)} pass -> {OUT}", flush=True)
os._exit(code)
