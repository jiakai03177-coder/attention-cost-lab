from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .core import AttentionConfig, PRESETS, estimate_attention, format_bytes, format_number, optimization_notes


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
    parser.add_argument("--seq-len", type=int, help="Context length in tokens.")
    parser.add_argument("--sweep", help="Comma-separated context lengths to compare, e.g. 1024,2048,4096,8192.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. Defaults to 1.")
    parser.add_argument("--dtype", default="fp16", help="Activation/KV dtype. Defaults to fp16.")
    parser.add_argument("--generated-tokens", type=int, default=1, help="Decode tokens to estimate. Defaults to 1.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--csv", help="Write estimates to a CSV file.")
    return parser


def parse_sweep_lengths(value: str) -> list[int]:
    lengths = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            length = int(stripped)
        except ValueError as exc:
            raise SystemExit(f"invalid sweep length: {stripped}") from exc
        if length < 1:
            raise SystemExit("--sweep lengths must be >= 1")
        lengths.append(length)

    if not lengths:
        raise SystemExit("--sweep must include at least one context length")
    return lengths


def config_from_args(args: argparse.Namespace) -> AttentionConfig:
    if args.seq_len is None and not args.sweep:
        raise SystemExit("missing required context length: --seq-len or --sweep")

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
        seq_len=args.seq_len if args.seq_len is not None else parse_sweep_lengths(args.sweep)[0],
        batch_size=args.batch_size,
        dtype=args.dtype,
        generated_tokens=args.generated_tokens,
    )


def config_with_seq_len(config: AttentionConfig, seq_len: int) -> AttentionConfig:
    return AttentionConfig(
        layers=config.layers,
        hidden_size=config.hidden_size,
        heads=config.heads,
        kv_heads=config.normalized_kv_heads,
        seq_len=seq_len,
        batch_size=config.batch_size,
        dtype=config.dtype,
        generated_tokens=config.generated_tokens,
    )


def print_table(rows: list[tuple[str, str]]) -> None:
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")


def estimate_payload(config: AttentionConfig) -> dict:
    estimate = estimate_attention(config)
    return {
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
    }


def csv_row(config: AttentionConfig) -> dict[str, int | float | str]:
    estimate = estimate_attention(config)
    return {
        "seq_len": config.seq_len,
        "batch_size": config.batch_size,
        "layers": config.layers,
        "hidden_size": config.hidden_size,
        "heads": config.heads,
        "kv_heads": config.normalized_kv_heads,
        "head_dim": estimate.head_dim,
        "dtype": config.dtype,
        "kv_cache_bytes": estimate.kv_cache_bytes,
        "kv_cache": format_bytes(estimate.kv_cache_bytes),
        "kv_cache_growth_per_token_bytes": estimate.kv_cache_growth_per_token_bytes,
        "prefill_attention_flops": estimate.prefill_attention_flops,
        "decode_attention_flops_per_token": estimate.decode_attention_flops_per_token,
        "decode_kv_read_bytes_per_token": estimate.decode_kv_read_bytes_per_token,
        "arithmetic_intensity_flops_per_byte": estimate.arithmetic_intensity_flops_per_byte,
    }


def write_csv(path: str, configs: list[AttentionConfig]) -> Path:
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [csv_row(config) for config in configs]
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def print_sweep_table(config: AttentionConfig, lengths: list[int]) -> None:
    rows = []
    for seq_len in lengths:
        estimate = estimate_attention(config_with_seq_len(config, seq_len))
        rows.append(
            [
                str(seq_len),
                format_bytes(estimate.kv_cache_bytes),
                format_bytes(estimate.kv_cache_growth_per_token_bytes),
                format_number(estimate.prefill_attention_flops),
                format_number(estimate.decode_attention_flops_per_token),
                format_bytes(estimate.decode_kv_read_bytes_per_token),
                f"{estimate.arithmetic_intensity_flops_per_byte:.2f}",
            ]
        )

    headers = [
        "seq_len",
        "KV cache",
        "KV/token",
        "prefill FLOPs",
        "decode FLOPs/token",
        "decode KV read/token",
        "FLOPs/byte",
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    print("Context Sweep")
    print("=============")
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    if args.sweep:
        lengths = parse_sweep_lengths(args.sweep)
        configs = [config_with_seq_len(config, seq_len) for seq_len in lengths]
        csv_path = write_csv(args.csv, configs) if args.csv else None

        if args.json:
            print(json.dumps({"sweep": [estimate_payload(item) for item in configs]}, indent=2))
            return 0

        print_sweep_table(config, lengths)
        if csv_path:
            print()
            print(f"CSV written to {csv_path}")
        return 0

    estimate = estimate_attention(config)
    csv_path = write_csv(args.csv, [config]) if args.csv else None

    if args.json:
        print(json.dumps(estimate_payload(config), indent=2))
        return 0

    print("Attention Cost Lab")
    print("==================")
    print_table(estimate.to_rows())
    print()
    print("Optimization notes:")
    for note in optimization_notes(estimate):
        print(f"- {note}")
    if csv_path:
        print()
        print(f"CSV written to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
