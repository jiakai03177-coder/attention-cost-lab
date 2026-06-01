from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .core import (
    AttentionConfig,
    PRESETS,
    RooflineHardware,
    estimate_attention,
    estimate_roofline,
    format_bytes,
    format_number,
    format_seconds,
    format_tokens_per_second,
    optimization_notes,
)


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
    parser.add_argument(
        "--memory-bandwidth-gbps",
        type=float,
        help="Sustained memory bandwidth in GB/s for roofline estimates.",
    )
    parser.add_argument(
        "--compute-tflops",
        type=float,
        help="Sustained attention compute throughput in TFLOP/s for roofline estimates.",
    )
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


def roofline_hardware_from_args(args: argparse.Namespace) -> RooflineHardware | None:
    if args.memory_bandwidth_gbps is None and args.compute_tflops is None:
        return None
    hardware = RooflineHardware(
        memory_bandwidth_gbps=args.memory_bandwidth_gbps,
        compute_tflops=args.compute_tflops,
    )
    try:
        hardware.validate()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return hardware


def print_table(rows: list[tuple[str, str]]) -> None:
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")


def estimate_payload(config: AttentionConfig, hardware: RooflineHardware | None = None) -> dict:
    estimate = estimate_attention(config)
    payload = {
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
    if hardware is not None:
        roofline = estimate_roofline(estimate, hardware)
        payload["roofline"] = {
            "memory_bandwidth_gbps": hardware.memory_bandwidth_gbps,
            "compute_tflops": hardware.compute_tflops,
            "decode_memory_seconds_per_step": roofline.decode_memory_seconds_per_step,
            "decode_compute_seconds_per_step": roofline.decode_compute_seconds_per_step,
            "decode_roofline_seconds_per_step": roofline.decode_roofline_seconds_per_step,
            "decode_tokens_per_second": roofline.decode_tokens_per_second,
            "bottleneck": roofline.bottleneck,
            "ridge_point_flops_per_byte": roofline.ridge_point_flops_per_byte,
        }
    return payload


def csv_row(config: AttentionConfig, hardware: RooflineHardware | None = None) -> dict[str, int | float | str | None]:
    estimate = estimate_attention(config)
    row = {
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
    if hardware is not None:
        roofline = estimate_roofline(estimate, hardware)
        row.update(
            {
                "memory_bandwidth_gbps": hardware.memory_bandwidth_gbps,
                "compute_tflops": hardware.compute_tflops,
                "decode_memory_seconds_per_step": roofline.decode_memory_seconds_per_step,
                "decode_compute_seconds_per_step": roofline.decode_compute_seconds_per_step,
                "decode_roofline_seconds_per_step": roofline.decode_roofline_seconds_per_step,
                "decode_tokens_per_second": roofline.decode_tokens_per_second,
                "roofline_bottleneck": roofline.bottleneck,
                "ridge_point_flops_per_byte": roofline.ridge_point_flops_per_byte,
            }
        )
    return row


def write_csv(path: str, configs: list[AttentionConfig], hardware: RooflineHardware | None = None) -> Path:
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [csv_row(config, hardware) for config in configs]
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def print_sweep_table(config: AttentionConfig, lengths: list[int], hardware: RooflineHardware | None = None) -> None:
    rows = []
    for seq_len in lengths:
        estimate = estimate_attention(config_with_seq_len(config, seq_len))
        row = [
            str(seq_len),
            format_bytes(estimate.kv_cache_bytes),
            format_bytes(estimate.kv_cache_growth_per_token_bytes),
            format_number(estimate.prefill_attention_flops),
            format_number(estimate.decode_attention_flops_per_token),
            format_bytes(estimate.decode_kv_read_bytes_per_token),
            f"{estimate.arithmetic_intensity_flops_per_byte:.2f}",
        ]
        if hardware is not None:
            roofline = estimate_roofline(estimate, hardware)
            row.extend(
                [
                    format_seconds(roofline.decode_roofline_seconds_per_step),
                    roofline.bottleneck,
                    format_tokens_per_second(roofline.decode_tokens_per_second),
                ]
            )
        rows.append(row)

    headers = [
        "seq_len",
        "KV cache",
        "KV/token",
        "prefill FLOPs",
        "decode FLOPs/token",
        "decode KV read/token",
        "FLOPs/byte",
    ]
    if hardware is not None:
        headers.extend(["decode roofline", "limit", "roofline throughput"])
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
    hardware = roofline_hardware_from_args(args)

    if args.sweep:
        lengths = parse_sweep_lengths(args.sweep)
        configs = [config_with_seq_len(config, seq_len) for seq_len in lengths]
        csv_path = write_csv(args.csv, configs, hardware) if args.csv else None

        if args.json:
            print(json.dumps({"sweep": [estimate_payload(item, hardware) for item in configs]}, indent=2))
            return 0

        print_sweep_table(config, lengths, hardware)
        if csv_path:
            print()
            print(f"CSV written to {csv_path}")
        return 0

    estimate = estimate_attention(config)
    csv_path = write_csv(args.csv, [config], hardware) if args.csv else None

    if args.json:
        print(json.dumps(estimate_payload(config, hardware), indent=2))
        return 0

    print("Attention Cost Lab")
    print("==================")
    print_table(estimate.to_rows())
    if hardware is not None:
        print()
        print("Roofline estimate:")
        print_table(estimate_roofline(estimate, hardware).to_rows())
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
