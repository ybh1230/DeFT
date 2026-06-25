# DeFT

**DeFT: Density-guided Feature Tracing** is a training-free open-vocabulary
semantic segmentation project. The method refines frozen visual tokens before
text-image similarity by estimating support density, tracing reliable evidence,
and restoring local structure with deterministic multi-scale recalibration.

The method name is **DeFT**. The repository includes a frozen OVSS runtime only
to make experiments reproducible on AutoDL; the proposed method is implemented
in `SFP_ICCV/deft/`.

## Layout

```text
DeFT/
  SFP_ICCV/
    eval.py                 # evaluation entry with --deft
    deft_segmentor.py       # runtime segmentor with DeFT inserted
    deft/                   # DeFT method implementation
    configs/                # dataset configs
    environment.yml         # recommended runtime environment
  paper/                    # AAAI-style LaTeX and Markdown draft
  experiments/              # ablation matrix
  results/                  # result log template
  tools/                    # result logging, matrix runner, map plotting
  scripts/                  # AutoDL setup and quick-run scripts
```

## AutoDL Quick Start

```bash
cd /
mkdir -p /root/autodl-tmp/src /root/autodl-tmp/hf-cache /root/autodl-tmp/data
cd /root/autodl-tmp/src

git config --global http.version HTTP/1.1
git config --global --add safe.directory '*'

git clone --depth 1 https://github.com/ybh1230/DeFT.git
cd DeFT

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache

bash scripts/bootstrap_autodl.sh
```

If GitHub clone fails:

```bash
cd /root/autodl-tmp/src
wget -O DeFT.zip https://codeload.github.com/ybh1230/DeFT/zip/refs/heads/main
unzip DeFT.zip
mv DeFT-main DeFT
cd DeFT
bash scripts/bootstrap_autodl.sh
```

## Clean Environment

Use a clean conda environment. This avoids the `mmcv-lite` / `mmcv._ext`
problem.

```bash
source /root/miniconda3/etc/profile.d/conda.sh

conda deactivate || true
conda env remove -n deft -y || true
conda create -n deft python=3.9 -y
conda activate deft

python -m pip install -U pip
pip install "setuptools<70" wheel

pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  ftfy regex tqdm numpy==1.26.4 yapf==0.40.1 \
  scikit-learn scikit-image opencv-python==4.10.0.84 \
  matplotlib pillow pycocotools fast-pytorch-kmeans \
  mmengine==0.8.4 mmsegmentation==1.1.1 huggingface_hub

pip uninstall -y mmcv mmcv-lite mmcv-full || true

pip install mmcv==2.0.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html \
  --only-binary mmcv
```

Check the environment:

```bash
python - <<'PY'
import torch, mmcv, mmengine, mmseg, ftfy
from mmcv.ops import sigmoid_focal_loss
print("env ok")
print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("mmcv:", mmcv.__version__)
print("mmengine:", mmengine.__version__)
print("mmseg:", mmseg.__version__)
PY
```

## Prepare VOC2012

```bash
cd /root/autodl-tmp/data

if [ ! -d /root/autodl-tmp/data/VOCdevkit/VOC2012 ]; then
  wget -c -O VOCtrainval_11-May-2012.tar \
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  tar -xf VOCtrainval_11-May-2012.tar
fi

cd /root/autodl-tmp/src/DeFT/SFP_ICCV
python - <<'PY'
from pathlib import Path
cfg = Path("configs/cfg_voc21.py")
text = cfg.read_text()
text = text.replace(
    "data_root = './YOUR_PATH/VOCdevkit/VOC2012'",
    "data_root = '/root/autodl-tmp/data/VOCdevkit/VOC2012'",
)
cfg.write_text(text)
print("updated", cfg)
PY
```

## Run

Baseline runtime:

```bash
cd /root/autodl-tmp/src/DeFT/SFP_ICCV
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_runtime
```

DeFT:

```bash
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_deft --deft
```

Run both in sequence:

```bash
cd /root/autodl-tmp/src/DeFT/SFP_ICCV && \
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_runtime && \
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_deft --deft
```

## Ablation Matrix

```bash
cd /root/autodl-tmp/src/DeFT
python tools/run_eval_matrix.py \
  --workdir ./SFP_ICCV \
  --base-cmd "python eval.py --config ./configs/cfg_voc21.py" \
  --work-root ./SFP_ICCV/work_logs/voc21_matrix
```

## Visualization Maps

```bash
cd /root/autodl-tmp/src/DeFT/SFP_ICCV
python eval.py \
  --config ./configs/cfg_voc21.py \
  --work-dir ./work_logs/voc21_deft_vis \
  --deft \
  --deft-return-maps \
  --deft-save-maps ./work_logs/voc21_deft_vis/deft_maps

cd /root/autodl-tmp/src/DeFT
python tools/plot_deft_maps.py \
  --input-dir ./SFP_ICCV/work_logs/voc21_deft_vis/deft_maps \
  --output-dir ./SFP_ICCV/work_logs/voc21_deft_vis/deft_map_png
```

## Current VOC21 Sanity Result

Initial VOC21 run on RTX 4090:

| Method | aAcc | mIoU | mAcc |
|---|---:|---:|---:|
| Frozen runtime | 89.65 | 64.33 | 85.14 |
| DeFT | 89.70 | 64.59 | 85.61 |

This confirms the code path works. Use the matrix and tuning commands for the
paper-scale results.
