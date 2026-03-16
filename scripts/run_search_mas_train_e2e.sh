#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/orchrl/config/search"

DEFAULT_CONFIG_NAME="search_mas_nosearch_external_5step_4x4_conservative"
DEFAULT_CUDA_VISIBLE_DEVICES="3,4,5"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_LOG_PATH="$REPO_ROOT/logs/search_mas_train_e2e_${TIMESTAMP}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
CONFIG_NAME="${CONFIG_NAME:-$DEFAULT_CONFIG_NAME}"
LOG_PATH="${LOG_PATH:-$DEFAULT_LOG_PATH}"

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$(dirname "$LOG_PATH")"

CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

if ! eval "$(CONFIG_DIR="$CONFIG_DIR" CONFIG_NAME="$CONFIG_NAME" python3 - <<'PY'
import os
import shlex
from hydra import compose, initialize_config_dir

config_dir = os.environ['CONFIG_DIR']
config_name = os.environ['CONFIG_NAME']
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name=config_name)

values = {
    'MAS_WORK_DIR': cfg.training.mate.mas_work_dir,
    'CONFIG_TEMPLATE_PATH': cfg.training.mate.config_template_path,
    'PROMPT_DATA_PATH': cfg.training.mate.prompt_loader.path,
    'MODEL_PATH_0': cfg.base_models.policy_0.path,
    'MODEL_PATH_1': cfg.base_models.policy_1.path,
    'MODEL_PATH_2': cfg.base_models.policy_2.path,
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"; then
  echo "[ERROR] Failed to resolve runtime paths from Hydra config: $CONFIG_NAME" >&2
  exit 1
fi

for required_dir in "$MAS_WORK_DIR"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "[ERROR] Required directory not found: $required_dir" >&2
    exit 1
  fi
done

for required_file in "$CONFIG_TEMPLATE_PATH" "$PROMPT_DATA_PATH" "$MODEL_PATH_0" "$MODEL_PATH_1" "$MODEL_PATH_2"; do
  if [[ ! -e "$required_file" ]]; then
    echo "[ERROR] Required path not found: $required_file" >&2
    exit 1
  fi
done

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

cd "$REPO_ROOT"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Config: $CONFIG_NAME"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] Log path: $LOG_PATH"
echo "[INFO] MAS work dir: $MAS_WORK_DIR"
echo "[INFO] Prompt data: $PROMPT_DATA_PATH"
echo "[INFO] Model paths: $MODEL_PATH_0 | $MODEL_PATH_1 | $MODEL_PATH_2"

python3 -m orchrl.trainer.train \
  --config-path "$CONFIG_DIR" \
  --config-name "$CONFIG_NAME" 2>&1 | tee "$LOG_PATH"
