"""Estimate Transformer attention costs for inference planning."""

from .core import AttentionConfig, AttentionEstimate, estimate_attention

__all__ = ["AttentionConfig", "AttentionEstimate", "estimate_attention"]
