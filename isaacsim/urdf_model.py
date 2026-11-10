"""Minimal URDF reader + forward kinematics (numpy only), shared by the Isaac Sim scripts."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "urdf" / "hyper.urdf"

# unitree_rl_lab UNITREE_GO2_CFG init_state (isaaclab/hyper_cfg.py)
INIT_BASE_POS = (0.0, 0.0, 0.4)


def init_joint_pos(name: str) -> float:
    if name.endswith("_hip_joint"):
        return 0.1 if name[1] == "L" else -0.1
    if name.endswith("_thigh_joint"):
        return 0.8 if name[0] == "F" else 1.0
    if name.endswith("_calf_joint"):
        return -1.5
    raise KeyError(name)


# CAD assembly pose (README joint table, right legs mirror the hip sign)
_CAD = {"F": (0.0, 0.9442, -1.9148), "R": (0.0712, 0.8206, -1.7686)}


def cad_joint_pos(name: str) -> float:
    hip, thigh, calf = _CAD[name[0]]
    if name.endswith("_hip_joint"):
        return hip if name[1] == "L" else -hip
    return thigh if name.endswith("_thigh_joint") else calf


def rpy_to_R(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle_R(axis, q) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def R_to_quat_wxyz(R) -> tuple:
    R = np.asarray(R, float)
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return tuple(q / np.linalg.norm(q))


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def T(R=None, p=None) -> np.ndarray:
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    if p is not None:
        M[:3, 3] = p
    return M


def _vec(s, default="0 0 0"):
    return np.array([float(v) for v in (s or default).split()])


@dataclass
class Geom:
    kind: str  # mesh | sphere | box | cylinder
    origin: np.ndarray  # 4x4 in link frame
    filename: Path | None = None
    scale: np.ndarray | None = None
    radius: float | None = None
    name: str | None = None
    material: str | None = None


@dataclass
class Link:
    name: str
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # about com, link axes
    visuals: list = field(default_factory=list)
    collisions: list = field(default_factory=list)


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None


def _geoms(link_el, tag, base_dir):
    out = []
    for el in link_el.findall(tag):
        o = el.find("origin")
        M = T(rpy_to_R(_vec(o.get("rpy") if o is not None else None)), _vec(o.get("xyz") if o is not None else None))
        g = el.find("geometry")
        m, s = g.find("mesh"), g.find("sphere")
        mat = el.find("material")
        mat = mat.get("name") if mat is not None else None
        if m is not None:
            out.append(Geom("mesh", M, (base_dir / m.get("filename")).resolve(), _vec(m.get("scale"), "1 1 1"), name=el.get("name"), material=mat))
        elif s is not None:
            out.append(Geom("sphere", M, radius=float(s.get("radius")), name=el.get("name")))
        else:
            out.append(Geom(g[0].tag, M, name=el.get("name")))
    return out


def load(path: Path = URDF):
    root = ET.parse(path).getroot()
    links, joints = {}, {}
    for el in root.findall("link"):
        L = Link(el.get("name"))
        ine = el.find("inertial")
        if ine is not None:
            o = ine.find("origin")
            R = rpy_to_R(_vec(o.get("rpy") if o is not None else None))
            L.com = _vec(o.get("xyz") if o is not None else None)
            L.mass = float(ine.find("mass").get("value"))
            i = {k: float(v) for k, v in ine.find("inertia").attrib.items()}
            I = np.array([[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]])
            L.inertia = R @ I @ R.T
        L.visuals = _geoms(el, "visual", path.parent)
        L.collisions = _geoms(el, "collision", path.parent)
        links[L.name] = L
    for el in root.findall("joint"):
        o = el.find("origin")
        xyz, rpy = _vec(o.get("xyz")), _vec(o.get("rpy"))
        ax = el.find("axis")
        lim = el.find("limit")
        J = Joint(el.get("name"), el.get("type"), el.find("parent").get("link"), el.find("child").get("link"),
                  T(rpy_to_R(rpy), xyz), xyz, rpy, _vec(ax.get("xyz") if ax is not None else None, "1 0 0"))
        if lim is not None:
            for k in ("lower", "upper", "effort", "velocity"):
                if lim.get(k) is not None:
                    setattr(J, k, float(lim.get(k)))
        joints[J.name] = J
    return links, joints


def fk(links, joints, q: dict, base=np.eye(4)) -> dict:
    """World 4x4 of every link frame for joint positions q (missing joints = 0)."""
    child_of = {j.child: j for j in joints.values()}
    out = {}

    def pose(name):
        if name in out:
            return out[name]
        j = child_of.get(name)
        if j is None:
            out[name] = base
        else:
            M = pose(j.parent) @ j.origin
            if j.type in ("revolute", "continuous"):
                M = M @ T(axis_angle_R(j.axis, q.get(j.name, 0.0)))
            out[name] = M
        return out[name]

    for n in links:
        pose(n)
    return out


def read_stl(path: Path) -> np.ndarray:
    """Vertices (N,3) of a binary or ASCII STL."""
    data = path.read_bytes()
    n = int.from_bytes(data[80:84], "little") if len(data) >= 84 else 0
    if len(data) == 84 + 50 * n:
        rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
        return rec["v"].reshape(-1, 3).astype(float)
    pts = [list(map(float, ln.split()[1:4])) for ln in data.decode(errors="ignore").splitlines() if ln.strip().startswith("vertex")]
    return np.array(pts, float)
