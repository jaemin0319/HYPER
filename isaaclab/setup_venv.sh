#!/usr/bin/env bash
# Create HYPER/.venv on top of an existing Isaac Sim 5.1 + Isaac Lab 2.3 python (system site packages visible),
# link Isaac Lab's task/RL/asset packages and add the RL packages, without changing the base environment.
#
#   ISAAC_PYTHON=/path/to/isaac/env/bin/python ISAACLAB_PATH=/path/to/IsaacLab bash isaaclab/setup_venv.sh
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
HYPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/mnt/isaac/miniforge3/envs/so101_sim/bin/python}"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/so101_sim_ws/external/IsaacLab}"

"$ISAAC_PYTHON" -c "import isaacsim, isaaclab" || { echo "ISAAC_PYTHON must have isaacsim 5.1 and isaaclab 2.3"; exit 1; }
[ -d "$HYPER_DIR/.venv" ] || "$ISAAC_PYTHON" -m venv --system-site-packages "$HYPER_DIR/.venv"
PY="$HYPER_DIR/.venv/bin/python"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# Isaac Lab extensions used by the tasks (source checkout, read only) + this repo's hyper_lab package
cat > "$SITE/hyper_isaaclab.pth" <<EOF
$ISAACLAB_PATH/source/isaaclab_tasks
$ISAACLAB_PATH/source/isaaclab_rl
$ISAACLAB_PATH/source/isaaclab_assets
$HYPER_DIR/isaaclab
EOF
# keep the base env's torch / numpy / pillow; rsl-rl-lib version is the one Isaac Lab 2.3 pins
"$PY" -m pip list 2>/dev/null | grep -i -E '^(torch|torchvision|numpy|typing.extensions|triton|packaging|pillow) ' \
  | awk '{print $1"=="$2}' > "$HYPER_DIR/.venv/constraints.txt"
echo 'protobuf<6' >> "$HYPER_DIR/.venv/constraints.txt"
"$PY" -m pip install -q -c "$HYPER_DIR/.venv/constraints.txt" "rsl-rl-lib==3.0.1" hydra-core tensorboard onnx "mujoco==3.17.0"
"$PY" -c "
import importlib.util as u, rsl_rl, hyper_lab, mujoco  # isaaclab_tasks / isaaclab_rl import omni, so only locate them here
assert all(u.find_spec(m) for m in ('isaaclab_tasks', 'isaaclab_rl', 'isaaclab_assets')), 'Isaac Lab packages not found'
print('[setup] ok, mujoco', mujoco.__version__)"
echo "[setup] done: source $HYPER_DIR/isaaclab/activate.sh"
