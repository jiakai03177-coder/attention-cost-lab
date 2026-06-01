"""Estimate Transformer attention costs for inference planning."""

from .core import (
    AttentionConfig,
    AttentionEstimate,
    RooflineEstimate,
    RooflineHardware,
    estimate_attention,
    estimate_roofline,
)

__all__ = [
    "AttentionConfig",
    "AttentionEstimate",
    "RooflineEstimate",
    "RooflineHardware",
    "estimate_attention",
    "estimate_roofline",
]
