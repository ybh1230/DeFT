# Support Density Matters: Density-Guided Feature Tracing for Training-Free Open-Vocabulary Semantic Segmentation

## Abstract

Open-vocabulary semantic segmentation requires dense visual predictions from
vision-language models that are mainly trained with image-level supervision.
Training-free pipelines reuse frozen CLIP features and avoid dataset-specific
optimization, but dense transfer is still structurally fragile. We observe that
the decisive factor is not only semantic alignment, but also whether each patch
is sufficiently supported by nearby evidence on the visual token manifold.
Low-support patches often appear near object boundaries, thin structures, and
small regions.

We propose **DeFT**, a **De**nsity-guided **F**eature **T**racing framework.
DeFT estimates support density among frozen visual tokens, traces reliable
high-density evidence to low-support regions, and performs structure-conserving
multi-scale recalibration before text-image similarity. The method is fully
training-free and introduces no learnable parameters.

## 1. Introduction

Open-vocabulary semantic segmentation aims to segment arbitrary categories
described by natural language. CLIP provides strong image-text alignment, but
its patch features are not optimized for pixel-level prediction. Existing
training-free pipelines are attractive because they preserve vocabulary coverage
and avoid dataset-specific training.

However, direct dense transfer from CLIP has a structural weakness: confident
visual evidence is unevenly distributed. Some patches receive strong support
from semantically similar neighbors, while boundaries, thin parts, and small
objects often receive weak support. We call this **support density imbalance**.

Our key claim is:

> Support density matters for training-free open-vocabulary segmentation.

Instead of adding a trainable decoder, DeFT keeps the backbone frozen and
performs deterministic evidence tracing on visual tokens.

## Contributions

1. We identify **support density imbalance** as a key obstacle for training-free
   open-vocabulary semantic segmentation.
2. We propose **DeFT**, a deterministic feature tracing framework that estimates
   patch support density and propagates reliable evidence to low-support
   regions.
3. We introduce a structure-conserving multi-scale recalibration step that
   improves spatial coherence without changing CLIP's semantic space.
4. We provide comparisons, ablations, and visual analyses covering mIoU,
   boundary quality, small-object behavior, and runtime.

## 2. Method

Given frozen visual patch features `X` from a frozen vision-language runtime and text
features `T`, the baseline computes dense logits by cosine similarity. DeFT
refines `X` before this step:

```text
X_hat = DeFT(X)
logits = X_hat @ T.T
```

The method has four parts.

### 2.1 Support Density Estimation

DeFT builds a patch support graph using cosine similarity between normalized
visual tokens. For each token, it finds the top-k most similar neighbors and
computes a support density score from positive similarities.

Low-density tokens receive a larger density gate. These tokens usually
correspond to boundaries, thin structures, and small regions.

### 2.2 Density-Guided Evidence Tracing

Each token gathers two kinds of evidence:

- **Graph evidence** from semantically similar tokens.
- **Local evidence** from spatial neighborhoods.

The two are mixed by fixed weights. A density gate controls how much each token
is updated:

```text
updated token = original token * (1 - gate) + traced evidence * gate
```

High-density tokens are mostly preserved. Low-density tokens borrow reliable
evidence.

### 2.3 Structure-Conserving Multi-Scale Recalibration

After evidence tracing, DeFT applies deterministic multi-scale average pooling
residuals. The residual is injected more strongly into low-density regions,
which helps recover spatial continuity without learning parameters.

### 2.4 Text-Conditioned Confidence Rendering

The refined features are compared with the original text embeddings. An optional
small support-confidence prior can be used for rendering logits, but it does not
change the vocabulary or prompts.

## 3. Experiments

### Setup

- Runtime: frozen CLIP-based open-vocabulary segmentation pipeline.
- Insert position: after visual token extraction and before text-image
  similarity.
- Datasets: PASCAL VOC, PASCAL Context, COCO-Stuff, ADE20K.
- Metrics: mIoU, mAcc, boundary F-score, small-object mIoU, FPS, GPU memory.

### Main Comparison

| Method | Training-Free | VOC | Context | COCO-Stuff | ADE20K | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| CLIP baseline | Yes | -- | -- | -- | -- | -- |
| MaskCLIP-style baseline | Yes | -- | -- | -- | -- | -- |
| Frozen runtime | Yes | -- | -- | -- | -- | -- |
| **DeFT** | Yes | -- | -- | -- | -- | -- |

### Ablation

| Variant | mIoU | Boundary-F | Small mIoU |
|---|---:|---:|---:|
| Frozen runtime | -- | -- | -- |
| DeFT without graph evidence | -- | -- | -- |
| DeFT without local evidence | -- | -- | -- |
| DeFT without multi-scale recalibration | -- | -- | -- |
| DeFT full | -- | -- | -- |

### Visualizations

The paper should include:

- support density map,
- tracing gate map,
- token similarity graph before and after DeFT,
- class activation map before and after DeFT,
- boundary zoom-in examples.

## 4. Conclusion

DeFT argues that **support density matters** for training-free open-vocabulary
semantic segmentation. It improves frozen vision-language dense transfer by estimating
patch support density, tracing reliable evidence to sparse regions, and applying
structure-conserving multi-scale recalibration. The full method is deterministic,
plug-and-play, and training-free.
