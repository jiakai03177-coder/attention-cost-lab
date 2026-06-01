from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .core import AttentionConfig, PRESETS, estimate_attention, optimization_notes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attention-cost",
        description="Estimate Transformer attention FLOPs, KV cache size, and decode memory pressure.",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Load a common model shape.")
    parser.add_argument("--layers", type=int, help="Transformer layer count.")
    parser.add_argument("--hidden-size", type=int, help="Model hidden size.")
    parser.add_argument("--heads", type=int, help="Number of query attention heads.")
    parser.add_argument("--kv-heads", type=int, help="Number of key/value heads for GQA or MQA.")
    parser.add_argument("--seq-len", type=int, required=True, help="Context length in tokens.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. Defaults to 1.")
    parser.add_argument("--dtype", default="fp16", help="Activation/KV dtype. Defaults to fp16.")
    parser.add_argument("--generated-tokens", type=int, default=1, help="Decode tokens to estimate. Defaults to 1.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def config_from_args(args: argparse.Namespace) -> AttentionConfig:
    values = {}
    if args.preset:
        values.update(PRESETS[args.preset])

    for key, arg_name in [
        ("layers", "layers"),
        ("hidden_size", "hidden_size"),
        ("heads", "heads"),
        ("kv_heads", "kv_heads"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            values[key] = value

    missing = [key for key in ["layers", "hidden_size", "heads"] if key not in values]
    if missing:
        missing_args = ", ".join(f"--{key.replace('_', '-')}" for key in missing)
        raise SystemExit(f"missing required model shape: {missing_args} or --preset")

    return AttentionConfig(
        layers=values["layers"],
        hidden_size=values["hidden_size"],
        heads=values["heads"],
        kv_heads=values.get("kv_heads"),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dtype=args.dtype,
        generated_tokens=args.generated_tokens,
    )


def print_table(rows: list[tuple[str, str]]) -> None:
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    estimate = estimate_attention(config)

    if args.json:
        print(
            json.dumps(
                {
                    "config": asdict(config),
                    "estimate": {
                        "head_dim": estimate.head_dim,
                        "group_size": estimate.group_size,
                        "kv_cache_bytes": estimate.kv_cache_bytes,
                        "kv_cache_growth_per_token_bytes": estimate.kv_cache_growth_per_token_bytes,
                        "prefill_attention_flops": estimate.prefill_attention_flops,
                        "decode_attention_flops_per_token": estimate.decode_attention_flops_per_token,
                        "decode_attention_flops_total": estimate.decode_attention_flops_total,
                        "decode_kv_read_bytes_per_token": estimate.decode_kv_read_bytes_per_token,
                        "arithmetic_intensity_flops_per_byte": estimate.arithmetic_intensity_flops_per_byte,
                    },
                    "notes": optimization_notes(estimate),
                },
                indent=2,
            )
        )
        return 0

    print("Attention Cost Lab")
    print("==================")
    print_table(estimate.to_rows())
    print()
    print("Optimization notes:")
    for note in optimization_notes(estimate):
        print(f"- {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
