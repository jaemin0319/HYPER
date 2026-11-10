"""Joint zero calibration and direction check for the GO-M8010-6 motors (motors stay at zero torque).

    sudo -E python deploy/scripts/calibrate.py --hardware deploy/config/hardware.json --sdk <sdk lib dir> --watch
        move each joint by hand: the printed joint angle must grow in the URDF direction
        (hip: +x, foot moves outward on the left legs; thigh/calf: +y, foot moves backward). Flip "direction" if not.
    sudo -E python deploy/scripts/calibrate.py --hardware deploy/config/hardware.json --sdk <sdk lib dir> --save
        with the robot held in hardware.json "calibration_pose": offset = pose - direction * motor output angle

If the motors report a position relative to power-up (not absolute over the 6.33 reduction), calibrate after every
power cycle before run_real.py.
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
from hyper_deploy.policy_io import PolicySpec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(REPO / "deploy" / "policies" / "flat"), help="for the joint list")
    ap.add_argument("--hardware", default=str(REPO / "deploy" / "config" / "hardware.json"))
    ap.add_argument("--sdk", default=None)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--fake", action="store_true", help="use the fake SDK (test)")
    args = ap.parse_args()
    spec = PolicySpec.load(Path(args.policy) / "deploy.json")
    hw_path = Path(args.hardware)
    hw = json.loads(hw_path.read_text())
    if args.fake:
        from hyper_deploy import fake_unitree_sdk as sdk
    else:
        if args.sdk:
            sys.path.insert(0, args.sdk)
        import unitree_actuator_sdk as sdk

    raw = json.loads(json.dumps(hw))
    for j in raw["joints"].values():
        j["offset"] = 0.0  # read direction * motor output angle
    motors = GoM8010Backend(spec, raw, sdk=sdk)
    try:
        time.sleep(0.3)
        if args.watch:
            while True:
                q, _ = motors.read()
                q_joint = q + np.array([hw["joints"][n]["offset"] for n in spec.joint_names])
                print(" ".join(f"{n[:-6]}={v:+.3f}" for n, v in zip(spec.joint_names, q_joint)), flush=True)
                time.sleep(0.2)
        samples = []
        for _ in range(50):
            samples.append(motors.read()[0])
            time.sleep(0.01)
        q_dir = np.median(np.array(samples), 0)
        pose = hw["calibration_pose"]
        offsets = {n: float(pose[n] - q_dir[i]) for i, n in enumerate(spec.joint_names)}
        for n in spec.joint_names:
            print(f"{n:16s} direction*motor {q_dir[spec.joint_names.index(n)]:+.4f}  pose {pose[n]:+.4f}  offset {offsets[n]:+.4f}")
        if args.save:
            for n, v in offsets.items():
                hw["joints"][n]["offset"] = round(v, 5)
            hw_path.write_text(json.dumps(hw, indent=1, ensure_ascii=False))
            print(f"[calibrate] offsets saved to {hw_path}")
    except KeyboardInterrupt:
        pass
    finally:
        motors.close(damping_kd=0.0, seconds=0.0)


if __name__ == "__main__":
    main()
