"""Run the deployment controller on MuJoCo (sim2sim): same code as the real robot, only the motor backend differs.

    python deploy/scripts/run_sim2sim.py --policy deploy/policies/flat                  # scripted test, prints metrics
    python deploy/scripts/run_sim2sim.py --policy deploy/policies/flat --viewer --keyboard
        keys: W/S vx, A/D vy, Q/E yaw rate, SPACE zero, ENTER walk (after the 2 s stand), BACKSPACE damp
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy"))

from hyper_deploy.backends import MujocoBackend  # noqa: E402
from hyper_deploy.commands import KeyboardCommand, ScriptedCommand  # noqa: E402
from hyper_deploy.controller import Controller  # noqa: E402
from hyper_deploy.policy_io import MLPPolicy, PolicySpec  # noqa: E402

SCRIPT = [  # (seconds, vx, vy, wz), same commands as isaaclab/scripts/export_deploy.py
    (6, 0.0, 0.0, 0.0), (6, 0.5, 0.0, 0.0), (6, 1.0, 0.0, 0.0), (6, -0.5, 0.0, 0.0), (6, 0.0, 0.5, 0.0),
    (6, 0.0, -0.5, 0.0), (6, 0.0, 0.0, 1.0), (6, 0.0, 0.0, -1.0), (6, 0.5, 0.0, 0.5), (6, 0.7, 0.4, 0.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(REPO / "deploy" / "policies" / "flat"))
    ap.add_argument("--xml", default=str(REPO / "mujoco" / "hyper_scene.xml"))
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--keyboard", action="store_true", help="drive with the viewer keyboard instead of the script")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds skipped at the start of each segment")
    ap.add_argument("--friction", type=float, default=None, help="override sliding friction of every geom")
    ap.add_argument("--out", default=None, help="write metrics json here")
    args = ap.parse_args()

    pdir = Path(args.policy)
    spec = PolicySpec.load(pdir / "deploy.json")
    policy = MLPPolicy(pdir / "policy.npz")
    if args.keyboard:
        cmds = KeyboardCommand(spec.command_ranges)
        motors = MujocoBackend(spec, args.xml, viewer=True, realtime=True, key_callback=cmds.key_callback)
    else:
        cmds = ScriptedCommand(SCRIPT, spec.command_ranges, stand_s=2.0)
        motors = MujocoBackend(spec, args.xml, viewer=args.viewer, realtime=args.viewer)
    if args.friction is not None:
        motors.model.geom_friction[:, 0] = args.friction

    rows = []

    def log(t, state, cmd, q, dq, action):
        b = motors.base_state()  # ground truth, evaluation only
        sc, gc = motors.contacts()
        rows.append((t, state, cmd.copy(), b["lin_vel_b"][:2].copy(), b["ang_vel_b"][2], b["height"], b["tilt_deg"], sc, gc))

    ctl = Controller(spec, policy, motors, cmds, stand_s=2.0, log=log)
    t_end = float("inf") if args.keyboard else cmds.end
    try:
        while ctl.t < t_end:
            ctl.tick()
            if rows[-1][5] < 0.12 or rows[-1][6] > 60:
                print(f"[sim2sim] fell at t={ctl.t:.2f}s (height {rows[-1][5]:.3f} m, tilt {rows[-1][6]:.0f} deg)")
                break
    except KeyboardInterrupt:
        pass
    motors.close()
    if args.keyboard:
        return

    fell = rows[-1][5] < 0.12 or rows[-1][6] > 60
    seg = []
    for k, (dur, *c) in enumerate(SCRIPT):
        t0 = cmds.starts[k] + args.settle
        t1 = cmds.starts[k] + dur
        sel = [r for r in rows if t0 <= r[0] < t1 and r[1] == "walk"]
        if not sel:
            seg.append({"command": c, "samples": 0})
            continue
        v = np.array([[*r[3], r[4]] for r in sel])
        seg.append({"command": c, "mean_velocity": v.mean(0).round(3).tolist(),
                    "mean_abs_error": np.abs(v - np.array(c)).mean(0).round(3).tolist(),
                    "mean_height": round(float(np.mean([r[5] for r in sel])), 3),
                    "min_height": round(min(r[5] for r in sel), 3), "max_tilt_deg": round(max(r[6] for r in sel), 1),
                    "self_contact_frac": round(float(np.mean([bool(r[7]) for r in sel])), 3),
                    "nonfoot_ground_frac": round(float(np.mean([bool(r[8]) for r in sel])), 3),
                    "self_contact_pairs": sorted(set().union(*[r[7] for r in sel]))})
    res = {"engine": "mujoco", "xml": str(Path(args.xml).relative_to(REPO)) if Path(args.xml).is_relative_to(REPO) else args.xml,
           "policy": str(pdir), "friction": args.friction, "fell": fell, "duration_s": rows[-1][0], "segments": seg,
           "note": "controller input = motor q, dq + command only; velocities here are MuJoCo ground truth for scoring"}
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
