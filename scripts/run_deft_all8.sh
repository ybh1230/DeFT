#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${PROJECT_DIR}/SFP_ICCV"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf-cache}"

cd "${RUNTIME_DIR}"

run_one() {
  local name="$1"
  local cfg="$2"
  python eval.py \
    --config "./configs/${cfg}" \
    --work-dir "./work_logs/main_deft/${name}" \
    --deft
}

run_one "v21" "cfg_voc21.py"
run_one "pc60" "cfg_context60.py"
run_one "object" "cfg_coco_object.py"
run_one "v20" "cfg_voc20.py"
run_one "pc59" "cfg_context59.py"
run_one "stuff" "cfg_coco_stuff164k.py"
run_one "ade" "cfg_ade20k.py"
run_one "city" "cfg_city_scapes.py"
