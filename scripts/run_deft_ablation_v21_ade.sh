#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${PROJECT_DIR}/SFP_ICCV"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf-cache}"

cd "${RUNTIME_DIR}"

run_pair() {
  local tag="$1"
  shift
  for item in "v21:cfg_voc21.py" "ade:cfg_ade20k.py"; do
    local name="${item%%:*}"
    local cfg="${item##*:}"
    python eval.py \
      --config "./configs/${cfg}" \
      --work-dir "./work_logs/ablation_${tag}/${name}" \
      "$@"
  done
}

# Row 0: frozen runtime.
run_pair "runtime"

# Row 1: density-guided graph tracing, without local tracing or multi-scale recalibration.
run_pair "graph_only" --deft --deft-graph-weight 1.0 --deft-local-weight 0.0 --deft-multiscale-strength 0.0

# Row 2: graph + local tracing, without multi-scale recalibration.
run_pair "graph_local" --deft --deft-multiscale-strength 0.0

# Row 3: full DeFT.
run_pair "full_deft" --deft
