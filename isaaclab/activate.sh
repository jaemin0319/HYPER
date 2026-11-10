# source isaaclab/activate.sh   (after isaaclab/setup_venv.sh)
# HYPER_GPU=<index|uuid> picks the GPU (default: whatever CUDA_VISIBLE_DEVICES already is).
HYPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export HYPER_DIR
# shellcheck disable=SC1091
source "$HYPER_DIR/.venv/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${HYPER_GPU:-}" ]; then export CUDA_VISIBLE_DEVICES="$HYPER_GPU"; fi
echo "[hyper] python=$(command -v python) GPU=${CUDA_VISIBLE_DEVICES:-all}"
