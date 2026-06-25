from dataclasses import dataclass
from typing import Tuple


@dataclass
class DeFTConfig:
    """Configuration for the training-free DeFT refinement.

    DeFT has no learnable parameters. The constants below control deterministic
    graph aggregation and spatial smoothing. Keep them fixed for the main
    comparison, then report sensitivity in the appendix.
    """

    knn: int = 12
    density_threshold: float = 0.46
    density_temperature: float = 0.12
    trace_strength: float = 0.72
    graph_temperature: float = 0.07
    local_weight: float = 0.35
    graph_weight: float = 0.65
    multiscale_kernels: Tuple[int, ...] = (3, 5, 7)
    multiscale_strength: float = 0.18
    confidence_strength: float = 0.12
    eps: float = 1.0e-6
    return_maps: bool = False
