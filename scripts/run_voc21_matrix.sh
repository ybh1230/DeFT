#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf-cache}"

cd "${PROJECT_DIR}"
python tools/run_eval_matrix.py \
  --workdir "${PROJECT_DIR}/SFP_ICCV" \
  --base-cmd "python eval.py --config ./configs/cfg_voc21.py" \
  --work-root "${PROJECT_DIR}/SFP_ICCV/work_logs/voc21_matrix"
