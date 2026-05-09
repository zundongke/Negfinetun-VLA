#! /bin/bash
# Launcher for RISE-style imagined-rollout RL training (action-cond VLA + WM
# rollouts + RISE value-model bootstrap + DiffusionNFT loss).
# Companion runbook: docs/rise-rl-training.md
#
# Usage:
#   bash run_rise_rl.sh                                   # uses default config
#   bash run_rise_rl.sh libero_spatial_rise_actor_openpi_pi05
#
# All knobs (max_epochs, rollout_epoch, value_model_path, ckpt_path) live in the
# yaml; override on the CLI as `<key>=<value>` after the script name to retarget.
set -e

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export __EGL_VENDOR_LIBRARY_FILENAMES="${REPO_PATH}/10_nvidia.json"
export EGL_PLATFORM=device
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

: "${LIBERO_REPO_PATH:=/vePFS-East/comp_robot/shock/LIBERO}"
export LIBERO_REPO_PATH
export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:${PYTHONPATH:-}
export HYDRA_FULL_ERROR=1

CONFIG_NAME="${1:-libero_spatial_rise_actor_openpi_pi05}"
shift || true   # remaining args are passed verbatim as Hydra overrides

LOG_DIR="${REPO_PATH}/logs/rise-train-$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run.log"

CMD=(python "${SRC_FILE}"
     --config-path "${EMBODIED_PATH}/config/"
     --config-name "${CONFIG_NAME}"
     "runner.logger.log_path=${LOG_DIR}"
     "$@")

echo "${CMD[@]}" | tee "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
