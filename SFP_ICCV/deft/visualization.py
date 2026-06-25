from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


def save_trace_maps(maps: Dict[str, torch.Tensor], output_dir: str) -> None:
    """Save DeFT density/gate maps as .pt tensors for later visualization.

    Keeping the raw tensors avoids color-map choices during evaluation. Convert
    them to PNGs in notebooks or with the repository's existing visualization
    utilities.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, tensor in maps.items():
        torch.save(tensor.detach().cpu(), out / f"{name}.pt")
