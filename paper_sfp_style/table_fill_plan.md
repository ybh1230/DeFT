# Table Fill Plan

All commands assume:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate deft
cd /root/autodl-tmp/src/DeFT
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache
```

## Table 1: Main Quantitative Comparison

The first rows are copied as fixed comparison numbers from the SFP paper. Only
the final DeFT row is left blank in `main.tex`.

Run:

```bash
bash scripts/run_deft_all8.sh
```

Fill the DeFT row with the final `mIoU` from:

| Table column | Config | Log folder |
|---|---|---|
| V21 | `cfg_voc21.py` | `SFP_ICCV/work_logs/main_deft/v21` |
| PC60 | `cfg_context60.py` | `SFP_ICCV/work_logs/main_deft/pc60` |
| Object | `cfg_coco_object.py` | `SFP_ICCV/work_logs/main_deft/object` |
| V20 | `cfg_voc20.py` | `SFP_ICCV/work_logs/main_deft/v20` |
| PC59 | `cfg_context59.py` | `SFP_ICCV/work_logs/main_deft/pc59` |
| Stuff | `cfg_coco_stuff164k.py` | `SFP_ICCV/work_logs/main_deft/stuff` |
| ADE | `cfg_ade20k.py` | `SFP_ICCV/work_logs/main_deft/ade` |
| City | `cfg_city_scapes.py` | `SFP_ICCV/work_logs/main_deft/city` |

Collect logs:

```bash
python scripts/collect_mmseg_results.py \
  --work-root SFP_ICCV/work_logs/main_deft \
  --out results/main_deft_results.json
```

The paper table uses `mIoU`. `aAcc` and `mAcc` can be reported in supplementary
or kept in `results/main_deft_results.json`.

## Table 2: Ablation on DeFT Components

Run:

```bash
bash scripts/run_deft_ablation_v21_ade.sh
```

Fill `V21`, `ADE`, and `Avg.`:

```bash
python scripts/collect_mmseg_results.py \
  --work-root SFP_ICCV/work_logs \
  --out results/ablation_v21_ade_results.json
```

Rows:

| Row | Meaning | Command flags |
|---|---|---|
| 0 | Frozen runtime | no `--deft` |
| 1 | Graph tracing only | `--deft --deft-graph-weight 1.0 --deft-local-weight 0.0 --deft-multiscale-strength 0.0` |
| 2 | Graph + local tracing | `--deft --deft-multiscale-strength 0.0` |
| 3 | Full DeFT | `--deft` |

## Table 3: Sensitivity to Tracing Neighborhood

Run quick VOC21 tuning first:

```bash
bash scripts/run_deft_tuning_voc21.sh
```

For paper-scale sensitivity, rerun on V21 and ADE with:

```bash
cd SFP_ICCV
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/k_sens/v21_k4 --deft --deft-knn 4
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/k_sens/v21_k8 --deft --deft-knn 8
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/k_sens/v21_k12 --deft --deft-knn 12
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/k_sens/v21_k16 --deft --deft-knn 16
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/k_sens/v21_k24 --deft --deft-knn 24

python eval.py --config ./configs/cfg_ade20k.py --work-dir ./work_logs/k_sens/ade_k4 --deft --deft-knn 4
python eval.py --config ./configs/cfg_ade20k.py --work-dir ./work_logs/k_sens/ade_k8 --deft --deft-knn 8
python eval.py --config ./configs/cfg_ade20k.py --work-dir ./work_logs/k_sens/ade_k12 --deft --deft-knn 12
python eval.py --config ./configs/cfg_ade20k.py --work-dir ./work_logs/k_sens/ade_k16 --deft --deft-knn 16
python eval.py --config ./configs/cfg_ade20k.py --work-dir ./work_logs/k_sens/ade_k24 --deft --deft-knn 24
```

## Table 4: Different CLIP Backbones

Use the same table format as SFP. The current code defaults to `ViT-B/16`.
For `ViT-B/32` or `ViT-L/14`, change `clip_path` in `configs/base_config.py`
or make copied config files, then run V21 and ADE:

```bash
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/backbone/vitb16_v21 --deft
python eval.py --config ./configs/cfg_ade20k.py --work-dir ./work_logs/backbone/vitb16_ade --deft
```

## Qualitative Figures

Run:

```bash
cd SFP_ICCV
python eval.py \
  --config ./configs/cfg_voc21.py \
  --work-dir ./work_logs/voc21_deft_vis \
  --deft \
  --deft-return-maps \
  --deft-save-maps ./work_logs/voc21_deft_vis/deft_maps

cd ..
python tools/plot_deft_maps.py \
  --input-dir ./SFP_ICCV/work_logs/voc21_deft_vis/deft_maps \
  --output-dir ./SFP_ICCV/work_logs/voc21_deft_vis/deft_map_png
```

Use these PNGs for the paper's density-map and tracing-gate visualizations.
