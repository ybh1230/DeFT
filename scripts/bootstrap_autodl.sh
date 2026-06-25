#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${PROJECT_DIR}/SFP_ICCV"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf-cache}"

mkdir -p /root/autodl-tmp/src /root/autodl-tmp/data "${HF_HOME}"

git config --global http.version HTTP/1.1
git config --global --add safe.directory '*'

echo "[DeFT] project: ${PROJECT_DIR}"
echo "[DeFT] runtime source: ${RUNTIME_DIR}"
echo "[DeFT] HF cache: ${HF_HOME}"

python -m pip install -U pip
python -m pip install -U huggingface_hub

if [ -f "${RUNTIME_DIR}/environment.yml" ]; then
  echo "[DeFT] environment.yml exists at ${RUNTIME_DIR}/environment.yml"
  echo "[DeFT] If the current Python env is empty, run:"
  echo "  conda env create -f ${RUNTIME_DIR}/environment.yml"
  echo "  conda activate deft"
fi

python -m py_compile \
  "${RUNTIME_DIR}/deft_segmentor.py" \
  "${RUNTIME_DIR}/eval.py" \
  "${RUNTIME_DIR}/deft/config.py" \
  "${RUNTIME_DIR}/deft/density_tracing.py" \
  "${RUNTIME_DIR}/deft/cli.py"

cat <<EOF

[DeFT] bootstrap finished.

Before evaluation, edit dataset roots in:
  ${RUNTIME_DIR}/configs/*.py

Example baseline:
  cd ${RUNTIME_DIR}
  python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_sfp

Example DeFT:
  cd ${RUNTIME_DIR}
  python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_deft --deft

Example matrix:
  cd ${PROJECT_DIR}
  python tools/run_eval_matrix.py \\
    --workdir ${RUNTIME_DIR} \\
    --base-cmd "python eval.py --config ./configs/cfg_voc21.py" \\
    --work-root ${RUNTIME_DIR}/work_logs/voc21_matrix

EOF
