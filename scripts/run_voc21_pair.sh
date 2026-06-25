#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${PROJECT_DIR}/SFP_ICCV"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf-cache}"

mkdir -p /root/autodl-tmp/data "${HF_HOME}"

if [ ! -d /root/autodl-tmp/data/VOCdevkit/VOC2012 ]; then
  cd /root/autodl-tmp/data
  wget -c -O VOCtrainval_11-May-2012.tar \
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  tar -xf VOCtrainval_11-May-2012.tar
fi

cd "${RUNTIME_DIR}"
python - <<'PY'
from pathlib import Path
cfg = Path("configs/cfg_voc21.py")
text = cfg.read_text()
text = text.replace(
    "data_root = './YOUR_PATH/VOCdevkit/VOC2012'",
    "data_root = '/root/autodl-tmp/data/VOCdevkit/VOC2012'",
)
cfg.write_text(text)
print("VOC21 data_root is ready")
PY

python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_runtime
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_deft --deft
