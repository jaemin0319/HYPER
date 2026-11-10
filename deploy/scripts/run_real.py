"""Run the policy on the real HYPER: 12 Unitree GO-M8010-6 motors (RS-485) + a Linux joystick. Only motor encoders
are read (HYPER has no IMU, no foot sensors, no lidar).

NOT TESTED ON HARDWARE YET. First runs: robot hung or on a stand, low-current power supply, hand on the e-stop.

    sudo -E python deploy/scripts/run_real.py --policy deploy/policies/flat --hardware deploy/config/hardware.json \
        --sdk /path/to/unitree_actuator_sdk/lib
    python deploy/scripts/run_real.py --dry-run          # no hardware: fake SDK motors + scripted commands

Joystick (hardware.json "joystick"): START stand up (2 s ramp to the default pose), A walk, B damp (final).
Ctrl+C: damping for 1 s, then zero torque.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy"))

from hyper_deploy.backends import GoM8010Backend  # noqa: E402
from hyper_deploy.commands import LinuxJoystick, ScriptedCommand  # noqa: E402
from hyper_deploy.controller import Controller  # noqa: E402
from hyper_deploy.policy_io import MLPPolicy, PolicySpec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(REPO / "deploy" / "policies" / "flat"))
    ap.add_argument("--hardware", default=str(REPO / "deploy" / "config" / "hardware.json"))
    ap.add_argument("--sdk", default=None, help="directory holding unitree_actuator_sdk*.so")
    ap.add_argument("--dry-run", action="store_true", help="fake motors (hyper_deploy.fake_unitree_sdk), scripted commands")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long (default: run until Ctrl+C)")
    args = ap.parse_args()

    pdir = Path(args.policy)
    spec = PolicySpec.load(pdir / "deploy.json")
    policy = MLPPolicy(pdir / "policy.npz")
    hw_path = Path(args.hardware)
    if not hw_path.exists() and args.dry_run:
        hw_path = REPO / "deploy" / "config" / "hardware.example.json"
    hw = json.loads(hw_path.read_text())
    if args.dry_run:
        from hyper_deploy import fake_unitree_sdk as sdk

        cmds = ScriptedCommand([(3, 0.5, 0.0, 0.0), (2, 0.0, 0.0, 0.5)], spec.command_ranges, stand_s=2.0)
        t_end = args.seconds or cmds.end
    else:
        if args.sdk:
            sys.path.insert(0, args.sdk)
        import unitree_actuator_sdk as sdk

        cmds = LinuxJoystick(hw["joystick"], spec.command_ranges)
        t_end = args.seconds or float("inf")

    motors = GoM8010Backend(spec, hw, sdk=sdk)
    t0 = time.perf_counter()
    while not np.all(np.isfinite(motors.read()[0])):
        if time.perf_counter() - t0 > 1.0:
            motors.close()
            sys.exit(f"[real] no state from all motors after 1 s: {motors.status()}")
        time.sleep(0.01)
    q, _ = motors.read()
    lim = spec.joint_pos_limits
    margin = hw["safety"]["joint_limit_margin_rad"]
    bad = [f"{n} {q[i]:+.3f}" for i, n in enumerate(spec.joint_names) if not (lim[i, 0] - margin <= q[i] <= lim[i, 1] + margin)]
    if bad and not args.dry_run:
        motors.close()
        sys.exit(f"[real] joints outside their limits at start (calibration / direction?): {bad}")
    print("[real] start pose:", dict(zip(spec.joint_names, np.round(q, 3).tolist())), flush=True)

    ctl = Controller(spec, policy, motors, cmds, stand_s=hw["safety"]["stand_seconds"], damp_kd=hw["safety"]["damp_kd"],
                     limit_margin=None if args.dry_run else margin)
    last_print = 0.0
    try:
        while ctl.t < t_end:
            ctl.tick()
            if ctl.t - last_print >= 1.0:
                last_print = ctl.t
                st = motors.status()
                print(f"[real] t={ctl.t:6.1f}s {ctl.state:7s} cmd={np.round(cmds.velocity(ctl.t), 2)} "
                      f"max_temp={st['temp'].max():.0f}C bad_crc={int(st['bad_crc'].sum())}", flush=True)
    except KeyboardInterrupt:
        print("[real] Ctrl+C", flush=True)
    finally:
        motors.close(damping_kd=hw["safety"]["damp_kd"])
    q, _ = motors.read()
    print(f"[real] stopped in state {ctl.state} {ctl.reason}; final pose", np.round(q, 3).tolist(), flush=True)


if __name__ == "__main__":
    main()
