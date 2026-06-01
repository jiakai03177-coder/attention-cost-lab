from __future__ import annotations

from dataclasses import dataclass
from math import ceil


DTYPE_BYTES = {
    "fp32": 4,
    "float32": 4,
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "int8": 1,
    "fp8": 1,
}


PRESETS = {
    "llama2-7b": {
        "layers": 32,
        "hidden_size": 4096,
        "heads": 32,
        "kv_heads": 32,
    },
    "llama3-8b": {
        "layers": 32,
        "hidden_size": 4096,
        "heads": 32,
        "kv_heads": 8,
    },
    "mistral-7b": {
        "layers": 32,
        "hidden_size": 4096,
        "heads": 32,
        "kv_heads": 8,
    },
}


@dataclass(frozen=True)
class AttentionConfig:
    layers: int
    hidden_size: int
    heads: int
    seq_len: int
    batch_size: int = 1
    kv_heads: int | None = None
    dtype: str = "fp16"
    generated_tokens: int = 1

    @property
    def normalized_kv_heads(self) -> int:
        return self.kv_heads if self.kv_heads is not None else self.heads

    @property
    def dtype_bytes(self) -> int:
        try:
            return DTYPE_BYTES[self.dtype.lower()]
        except KeyError as exc:
            known = ", ".join(sorted(DTYPE_BYTES))
            raise ValueError(f"unknown dtype {self.dtype!r}; expected one of: {known}") from exc

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.heads != 0:
            raise ValueError("hidden_size must be divisible by heads")
        return self.hidden_size // self.heads

    def validate(self) -> None:
        values = {
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "heads": self.heads,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "kv_heads": self.normalized_kv_heads,
            "generated_tokens": self.generated_tokens,
        }
        for name, value in values.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.heads % self.normalized_kv_heads != 0:
            raise ValueError("heads must be divisible by kv_heads")
        _ = self.head_dim
        _ = self.dtype_bytes


@dataclass(frozen=True)
class AttentionEstimate:
    config: AttentionConfig
    head_dim: int
    group_size: int
    kv_cache_bytes: int
    kv_cache_growth_per_token_bytes: int
    prefill_attention_flops: int
    decode_attention_flops_per_token: int
    decode_attention_flops_total: int
    decode_kv_read_bytes_per_token: int
    arithmetic_intensity_flops_per_byte: float

    def to_rows(self) -> list[tuple[str, str]]:
        return [
            ("layers", str(self.config.layers)),
            ("hidden_size", str(self.config.hidden_size)),
            ("heads", str(self.config.heads)),
            ("kv_heads", str(self.config.normalized_kv_heads)),
            ("head_dim", str(self.head_dim)),
            ("GQA group size", str(self.group_size)),
            ("dtype", self.config.dtype),
            ("batch_size", str(self.config.batch_size)),
            ("seq_len", str(self.config.seq_len)),
            ("KV cache", format_bytes(self.kv_cache_bytes)),
            ("KV growth/token", format_bytes(self.kv_cache_growth_per_token_bytes)),
            ("prefill attention FLOPs", format_number(self.prefill_attention_flops)),
            ("decode FLOPs/token", format_number(self.decode_attention_flops_per_token)),
            ("decode KV read/token", format_bytes(self.decode_kv_read_bytes_per_token)),
            ("decode intensity", f"{self.arithmetic_intensity_flops_per_byte:.2f} FLOPs/byte"),
        ]


@dataclass(frozen=True)
class RooflineHardware:
    memory_bandwidth_gbps: float | None = None
    compute_tflops: float | None = None

    def validate(self) -> None:
        if self.memory_bandwidth_gbps is None and self.compute_tflops is None:
            raise ValueError("roofline estimates require memory bandwidth, compute throughput, or both")
        for name, value in {
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "compute_tflops": self.compute_tflops,
        }.items():
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0")

    @property
    def memory_bandwidth_bytes_per_second(self) -> float | None:
        if self.memory_bandwidth_gbps is None:
            return None
        return self.memory_bandwidth_gbps * 1_000_000_000

    @property
    def compute_flops_per_second(self) -> float | None:
        if self.compute_tflops is None:
            return None
        return self.compute_tflops * 1_000_000_000_000


@dataclass(frozen=True)
class RooflineEstimate:
    hardware: RooflineHardware
    decode_memory_seconds_per_step: float | None
    decode_compute_seconds_per_step: float | None
    decode_roofline_seconds_per_step: float
    decode_tokens_per_second: float
    bottleneck: str
    ridge_point_flops_per_byte: float | None

    def to_rows(self) -> list[tuple[str, str]]:
        rows = []
        if self.hardware.memory_bandwidth_gbps is not None:
            rows.append(("memory bandwidth", f"{self.hardware.memory_bandwidth_gbps:g} GB/s"))
        if self.hardware.compute_tflops is not None:
            rows.append(("compute throughput", f"{self.hardware.compute_tflops:g} TFLOP/s"))
        if self.ridge_point_flops_per_byte is not None:
            rows.append(("roofline ridge point", f"{self.ridge_point_flops_per_byte:.2f} FLOPs/byte"))
        if self.decode_memory_seconds_per_step is not None:
            rows.append(("decode memory lower bound", format_seconds(self.decode_memory_seconds_per_step)))
        if self.decode_compute_seconds_per_step is not None:
            rows.append(("decode compute lower bound", format_seconds(self.decode_compute_seconds_per_step)))
        rows.extend(
            [
                ("decode roofline time", format_seconds(self.decode_roofline_seconds_per_step)),
                ("decode roofline throughput", format_tokens_per_second(self.decode_tokens_per_second)),
                ("roofline bottleneck", self.bottleneck),
            ]
        )
        return rows


def estimate_attention(config: AttentionConfig) -> AttentionEstimate:
    config.validate()

    batch = config.batch_size
    layers = config.layers
    heads = config.heads
    kv_heads = config.normalized_kv_heads
    seq_len = config.seq_len
    head_dim = config.head_dim
    dtype_bytes = config.dtype_bytes

    kv_cache_bytes = batch * layers * seq_len * kv_heads * head_dim * 2 * dtype_bytes
    kv_cache_growth_per_token_bytes = batch * layers * kv_heads * head_dim * 2 * dtype_bytes

    # Attention core only: QK^T and softmax(V), each counted as multiply-add FLOPs.
    prefill_attention_flops = 4 * batch * layers * heads * (seq_len**2) * head_dim
    decode_attention_flops_per_token = 4 * batch * layers * heads * seq_len * head_dim
    decode_attention_flops_total = decode_attention_flops_per_token * config.generated_tokens

    decode_kv_read_bytes_per_token = batch * layers * seq_len * kv_heads * head_dim * 2 * dtype_bytes
    arithmetic_intensity = (
        decode_attention_flops_per_token / decode_kv_read_bytes_per_token
        if decode_kv_read_bytes_per_token
        else 0.0
    )

    return AttentionEstimate(
        config=config,
        head_dim=head_dim,
        group_size=ceil(heads / kv_heads),
        kv_cache_bytes=kv_cache_bytes,
        kv_cache_growth_per_token_bytes=kv_cache_growth_per_token_bytes,
        prefill_attention_flops=prefill_attention_flops,
        decode_attention_flops_per_token=decode_attention_flops_per_token,
        decode_attention_flops_total=decode_attention_flops_total,
        decode_kv_read_bytes_per_token=decode_kv_read_bytes_per_token,
        arithmetic_intensity_flops_per_byte=arithmetic_intensity,
    )


def estimate_roofline(estimate: AttentionEstimate, hardware: RooflineHardware) -> RooflineEstimate:
    hardware.validate()

    memory_seconds = None
    if hardware.memory_bandwidth_bytes_per_second is not None:
        memory_seconds = estimate.decode_kv_read_bytes_per_token / hardware.memory_bandwidth_bytes_per_second

    compute_seconds = None
    if hardware.compute_flops_per_second is not None:
        compute_seconds = estimate.decode_attention_flops_per_token / hardware.compute_flops_per_second

    candidates = [value for value in [memory_seconds, compute_seconds] if value is not None]
    roofline_seconds = max(candidates)
    if memory_seconds is not None and (compute_seconds is None or memory_seconds >= compute_seconds):
        bottleneck = "memory"
    else:
        bottleneck = "compute"

    ridge_point = None
    if (
        hardware.compute_flops_per_second is not None
        and hardware.memory_bandwidth_bytes_per_second is not None
    ):
        ridge_point = hardware.compute_flops_per_second / hardware.memory_bandwidth_bytes_per_second

    return RooflineEstimate(
        hardware=hardware,
        decode_memory_seconds_per_step=memory_seconds,
        decode_compute_seconds_per_step=compute_seconds,
        decode_roofline_seconds_per_step=roofline_seconds,
        decode_tokens_per_second=estimate.config.batch_size / roofline_seconds,
        bottleneck=bottleneck,
        ridge_point_flops_per_byte=ridge_point,
    )


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def format_number(value: int) -> str:
    units = ["", "K", "M", "B", "T", "P"]
    amount = float(value)
    for unit in units:
        if abs(amount) < 1000 or unit == units[-1]:
            return f"{amount:.2f}{unit}" if unit else str(int(amount))
        amount /= 1000
    return str(value)


def format_seconds(value: float) -> str:
    if value < 1e-6:
        return f"{value * 1e9:.2f} ns"
    if value < 1e-3:
        return f"{value * 1e6:.2f} us"
    if value < 1:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.2f} s"


def format_tokens_per_second(value: float) -> str:
    if value < 1000:
        return f"{value:.2f} tokens/s"
    return f"{format_number(round(value))} tokens/s"


def optimization_notes(estimate: AttentionEstimate) -> list[str]:
    notes = []
    cfg = estimate.config

    if cfg.normalized_kv_heads < cfg.heads:
        notes.append(
            f"GQA is active: {cfg.heads} query heads share {cfg.normalized_kv_heads} KV heads "
            f"({estimate.group_size}:1), reducing KV cache and decode reads."
        )
    else:
        notes.append("MHA is active: KV cache scales with every attention head.")

    if estimate.arithmetic_intensity_flops_per_byte < 10:
        notes.append(
            "Decode is likely memory-pressure sensitive because arithmetic intensity is low "
            "and each token rereads the KV cache."
        )
    else:
        notes.append("Decode has relatively high arithmetic intensity for the current shape.")

    if cfg.seq_len >= 8192:
        notes.append("Long context detected: prioritize KV cache layout, paging, and attention tiling.")
    elif cfg.seq_len >= 2048:
        notes.append("Medium/long context detected: KV cache bandwidth will matter during generation.")

    return notes
