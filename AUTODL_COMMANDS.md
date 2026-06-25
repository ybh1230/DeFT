# DeFT AutoDL Commands

## 1. Remove Old Folder and Clone

```bash
cd /
mkdir -p /root/autodl-tmp/src /root/autodl-tmp/hf-cache /root/autodl-tmp/data
cd /root/autodl-tmp/src

rm -rf DeFT
rm -rf DeFT-main
rm -f DeFT.zip

git config --global http.version HTTP/1.1
git config --global --add safe.directory '*'

git clone --depth 1 https://github.com/ybh1230/DeFT.git
cd DeFT
```

Fallback:

```bash
cd /root/autodl-tmp/src
rm -rf DeFT DeFT-main DeFT.zip
wget -O DeFT.zip https://codeload.github.com/ybh1230/DeFT/zip/refs/heads/main
unzip DeFT.zip
mv DeFT-main DeFT
cd DeFT
```

## 2. Environment Variables

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache
```

## 3. Bootstrap

```bash
bash scripts/bootstrap_autodl.sh
```

## 4. Clean Conda Environment

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

## 5. Prepare VOC2012

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

## 6. Run Runtime Baseline and DeFT

```bash
cd /root/autodl-tmp/src/DeFT/SFP_ICCV && \
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_runtime && \
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_deft --deft
```

## 7. Run Ablation Matrix

```bash
cd /root/autodl-tmp/src/DeFT
python tools/run_eval_matrix.py \
  --workdir ./SFP_ICCV \
  --base-cmd "python eval.py --config ./configs/cfg_voc21.py" \
  --work-root ./SFP_ICCV/work_logs/voc21_matrix
```
