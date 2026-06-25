from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeFTConfig


TensorOrTuple = Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]


def _l2_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _infer_hw(num_tokens: int) -> Tuple[int, int]:
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(
            f"Cannot infer a square feature map from {num_tokens} tokens. "
            "Pass hw=(height, width) explicitly."
        )
    return side, side


def _tokens_to_map(tokens: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    bsz, num_tokens, channels = tokens.shape
    height, width = hw
    if height * width != num_tokens:
        raise ValueError(
            f"hw={hw} does not match token count {num_tokens}."
        )
    return tokens.transpose(1, 2).reshape(bsz, channels, height, width)


def _map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    return feature_map.flatten(2).transpose(1, 2).contiguous()


class DensityGuidedFeatureTracing(nn.Module):
    """Training-free structure recovery for frozen vision-language visual tokens.

    Input:
        * [B, N, C] tokens, optionally with hw=(H, W)
        * [B, C, H, W] feature maps

    Output:
        Refined features with the same shape. If config.return_maps=True,
        returns (features, maps), where maps contains density and gate maps for
        visualization.
    """

    def __init__(self, config: Optional[DeFTConfig] = None):
        super().__init__()
        self.config = config or DeFTConfig()

    def forward(
        self,
        x: torch.Tensor,
        hw: Optional[Tuple[int, int]] = None,
    ) -> TensorOrTuple:
        original_is_map = x.dim() == 4
        if original_is_map:
            hw = (x.shape[-2], x.shape[-1])
            tokens = _map_to_tokens(x)
        elif x.dim() == 3:
            tokens = x
            if hw is None:
                hw = _infer_hw(tokens.shape[1])
        else:
            raise ValueError("DeFT expects [B, N, C] tokens or [B, C, H, W] maps.")

        refined_tokens, maps = self.refine_tokens(tokens, hw)
        if original_is_map:
            refined = _tokens_to_map(refined_tokens, hw)
        else:
            refined = refined_tokens

        if self.config.return_maps:
            return refined, maps
        return refined

    @torch.no_grad()
    def refine_tokens(
        self,
        tokens: torch.Tensor,
        hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.config
        bsz, num_tokens, channels = tokens.shape
        k = max(1, min(cfg.knn, num_tokens - 1))

        norm_tokens = _l2_normalize(tokens, cfg.eps)
        sim = torch.bmm(norm_tokens, norm_tokens.transpose(1, 2))
        eye = torch.eye(num_tokens, device=tokens.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(eye, -1.0e4)

        knn_sim, knn_idx = sim.topk(k=k, dim=-1)
        positive_support = knn_sim.clamp_min(0.0)
        density = positive_support.mean(dim=-1)
        density_min = density.amin(dim=1, keepdim=True)
        density_max = density.amax(dim=1, keepdim=True)
        density_norm = (density - density_min) / (density_max - density_min + cfg.eps)

        gate = torch.sigmoid(
            (cfg.density_threshold - density_norm)
            / max(cfg.density_temperature, cfg.eps)
        )

        graph_agg = self._graph_aggregate(tokens, knn_idx, knn_sim)
        local_agg = self._local_aggregate(tokens, hw)
        traced = self._mix(tokens, graph_agg, local_agg, gate)
        traced = self._multiscale_recalibrate(traced, density_norm, hw)
        traced = _l2_normalize(traced, cfg.eps)

        height, width = hw
        maps = {
            "support_density": density_norm.reshape(bsz, 1, height, width),
            "trace_gate": gate.reshape(bsz, 1, height, width),
        }
        return traced, maps

    def _graph_aggregate(
        self,
        tokens: torch.Tensor,
        knn_idx: torch.Tensor,
        knn_sim: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        bsz, num_tokens, channels = tokens.shape
        gather_src = tokens.unsqueeze(1).expand(bsz, num_tokens, num_tokens, channels)
        gather_idx = knn_idx.unsqueeze(-1).expand(-1, -1, -1, channels)
        neighbors = torch.gather(gather_src, dim=2, index=gather_idx)
        weights = F.softmax(knn_sim / max(cfg.graph_temperature, cfg.eps), dim=-1)
        return (neighbors * weights.unsqueeze(-1)).sum(dim=2)

    def _local_aggregate(
        self,
        tokens: torch.Tensor,
        hw: Tuple[int, int],
    ) -> torch.Tensor:
        cfg = self.config
        feature_map = _tokens_to_map(tokens, hw)
        local_sum = torch.zeros_like(feature_map)
        for kernel_size in cfg.multiscale_kernels:
            padding = kernel_size // 2
            local_sum = local_sum + F.avg_pool2d(
                feature_map,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                count_include_pad=False,
            )
        local = local_sum / max(len(cfg.multiscale_kernels), 1)
        return _map_to_tokens(local)

    def _mix(
        self,
        tokens: torch.Tensor,
        graph_agg: torch.Tensor,
        local_agg: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        traced_context = (
            cfg.graph_weight * graph_agg + cfg.local_weight * local_agg
        ) / max(cfg.graph_weight + cfg.local_weight, cfg.eps)
        gate = gate.unsqueeze(-1) * cfg.trace_strength
        return tokens * (1.0 - gate) + traced_context * gate

    def _multiscale_recalibrate(
        self,
        tokens: torch.Tensor,
        density: torch.Tensor,
        hw: Tuple[int, int],
    ) -> torch.Tensor:
        cfg = self.config
        feature_map = _tokens_to_map(tokens, hw)
        density_map = density.reshape(tokens.shape[0], 1, hw[0], hw[1])
        sparse_region = 1.0 - density_map

        residual = torch.zeros_like(feature_map)
        for kernel_size in cfg.multiscale_kernels:
            padding = kernel_size // 2
            pooled = F.avg_pool2d(
                feature_map,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                count_include_pad=False,
            )
            residual = residual + (pooled - feature_map)
        residual = residual / max(len(cfg.multiscale_kernels), 1)
        feature_map = feature_map + cfg.multiscale_strength * sparse_region * residual
        return _map_to_tokens(feature_map)

    @torch.no_grad()
    def calibrate_logits(
        self,
        logits: torch.Tensor,
        density_map: torch.Tensor,
    ) -> torch.Tensor:
        """Optionally render logits with a deterministic support-confidence prior."""

        cfg = self.config
        if logits.dim() != 4:
            raise ValueError("logits must be [B, K, H, W].")
        if density_map.shape[-2:] != logits.shape[-2:]:
            density_map = F.interpolate(
                density_map,
                size=logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        confidence = 1.0 + cfg.confidence_strength * (density_map - density_map.mean())
        return logits * confidence


def deft_refine(
    x: torch.Tensor,
    hw: Optional[Tuple[int, int]] = None,
    config: Optional[DeFTConfig] = None,
) -> TensorOrTuple:
    """Functional entry point used by quick DeFT integrations."""

    return DensityGuidedFeatureTracing(config)(x, hw=hw)
