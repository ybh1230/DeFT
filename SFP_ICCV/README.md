# DeFT Runtime

This folder contains the runnable evaluation runtime for **DeFT**.

Use `eval.py` directly:

```bash
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_runtime
python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_deft --deft
```

The DeFT method implementation is in:

```text
deft/
```

The main insertion point is in:

```text
deft_segmentor.py
```

Specifically, DeFT refines visual tokens after frozen visual feature extraction
and before text-image similarity.
