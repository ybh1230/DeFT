from __future__ import annotations

import argparse

from .config import DeFTConfig


def add_deft_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--deft", action="store_true", help="Enable DeFT refinement.")
    parser.add_argument("--deft-knn", type=int, default=12)
    parser.add_argument("--deft-density-threshold", type=float, default=0.46)
    parser.add_argument("--deft-density-temperature", type=float, default=0.12)
    parser.add_argument("--deft-trace-strength", type=float, default=0.72)
    parser.add_argument("--deft-graph-temperature", type=float, default=0.07)
    parser.add_argument("--deft-local-weight", type=float, default=0.35)
    parser.add_argument("--deft-graph-weight", type=float, default=0.65)
    parser.add_argument("--deft-multiscale-strength", type=float, default=0.18)
    parser.add_argument("--deft-confidence-strength", type=float, default=0.12)
    parser.add_argument("--deft-return-maps", action="store_true")
    parser.add_argument(
        "--deft-save-maps",
        default="",
        help="Directory for raw DeFT support-density and tracing-gate tensors.",
    )
    return parser


def build_deft_config(args: argparse.Namespace) -> DeFTConfig:
    return DeFTConfig(
        knn=args.deft_knn,
        density_threshold=args.deft_density_threshold,
        density_temperature=args.deft_density_temperature,
        trace_strength=args.deft_trace_strength,
        graph_temperature=args.deft_graph_temperature,
        local_weight=args.deft_local_weight,
        graph_weight=args.deft_graph_weight,
        multiscale_strength=args.deft_multiscale_strength,
        confidence_strength=args.deft_confidence_strength,
        return_maps=args.deft_return_maps,
    )
