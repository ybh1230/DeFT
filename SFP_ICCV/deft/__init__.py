"""DeFT: density-guided feature tracing for training-free OVSS."""

from .config import DeFTConfig
from .density_tracing import DensityGuidedFeatureTracing, deft_refine
from .runtime_adapter import DeFTRuntimeAdapter
from .cli import add_deft_args, build_deft_config

__all__ = [
    "DeFTConfig",
    "DensityGuidedFeatureTracing",
    "DeFTRuntimeAdapter",
    "add_deft_args",
    "build_deft_config",
    "deft_refine",
]
