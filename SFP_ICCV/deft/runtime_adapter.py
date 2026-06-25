from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import DeFTConfig
from .density_tracing import DensityGuidedFeatureTracing


class DeFTRuntimeAdapter(nn.Module):
    """Wrapper for inserting DeFT into a frozen OVSS runtime.

    Preferred integration point:

        visual_tokens = visual_encoder(...)
        visual_tokens = adapter.refine_visual_features(visual_tokens, hw=(h, w))
        logits = text_image_similarity(visual_tokens, text_features)

    The adapter is optional. The main project integrates DeFT directly in
    ``deft_segmentor.py``.
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        config: Optional[DeFTConfig] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.deft = DensityGuidedFeatureTracing(config)

    @torch.no_grad()
    def refine_visual_features(
        self,
        visual_features: torch.Tensor,
        hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        return self.deft(visual_features, hw=hw)

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.base_model is None:
            raise RuntimeError(
                "DeFTRuntimeAdapter.forward requires a base_model. "
                "Use refine_visual_features for direct integration."
            )

        output = self.base_model(*args, **kwargs)
        hw = kwargs.get("hw", None)
        if isinstance(output, Dict):
            for key in ("visual_tokens", "image_features", "patch_tokens"):
                value = output.get(key)
                if isinstance(value, torch.Tensor) and value.dim() in (3, 4):
                    output[key] = self.refine_visual_features(value, hw=hw)
                    output["deft_enabled"] = True
                    return output
        return output
