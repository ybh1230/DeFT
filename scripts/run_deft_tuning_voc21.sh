#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${PROJECT_DIR}/SFP_ICCV"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf-cache}"

cd "${RUNTIME_DIR}"

python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/tuning/k8 --deft --deft-knn 8
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/tuning/k16 --deft --deft-knn 16
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/tuning/soft --deft --deft-trace-strength 0.55 --deft-multiscale-strength 0.12
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/tuning/strong --deft --deft-trace-strength 0.90 --deft-multiscale-strength 0.25
