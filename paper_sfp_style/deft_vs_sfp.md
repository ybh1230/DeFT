# DeFT vs. SFP: Strict Difference Table

This table is intentionally strict. In the paper, do not claim SFP's SOM, SAE,
or HAI as DeFT contributions. DeFT starts from the frozen runtime visual tokens
and performs density-guided structure recovery before text-image similarity.

| Aspect | SFP | DeFT |
|---|---|---|
| Core problem | Outlier propagation inside CLIP attention layers. | Support density imbalance in frozen visual tokens for dense prediction. |
| Central phrase | Feature purification matters. | Support density matters. |
| Motivation | Intermediate attention may emphasize irrelevant tokens and propagate them. | Boundary, thin-structure, and small-object regions often have weak visual support even when semantic alignment exists. |
| Main objective | Suppress propagated outliers and obtain cleaner semantic features. | Trace reliable high-density evidence to low-support regions and recover spatial structure. |
| Operation location | Inside the CLIP visual encoder across intermediate layers. | After frozen visual token extraction and before text-image similarity. |
| Main modules | SOM, SAE, HAI. | Support Density Estimation, Density-Guided Evidence Tracing, Structure-Conserving Recalibration. |
| Token criterion | Detect tokens with abnormal attention or feature statistics. | Estimate each token's support density from top-k visual neighbors. |
| Feature update | Replaces or mitigates selected tokens and integrates attention maps. | Uses a density gate to mix original tokens with graph and local evidence. |
| Spatial modeling | Uses hierarchical attention maps for object-centric representation. | Uses graph tracing plus deterministic multi-scale residual recalibration. |
| Learnable parameters | None in the training-free setting. | None. All DeFT operations are deterministic. |
| Training-free status | Training-free. | Training-free. |
| Visualization evidence | Outlier propagation and purified feature maps across layers. | Support density maps, tracing gates, and before/after segmentation maps. |
| Current implementation relation | Official released runtime. | Implemented as a plug-in on top of the frozen runtime for fair and reproducible evaluation. |
| What must not be claimed | DeFT should not re-claim SOM/SAE/HAI. | DeFT's novelty is support-density tracing and structure recovery. |

Recommended paper wording:

> Unlike prior feature purification that focuses on suppressing unreliable
> intermediate responses, DeFT studies the support density of frozen visual
> tokens and performs deterministic evidence tracing to recover low-support
> regions before text-image similarity.
