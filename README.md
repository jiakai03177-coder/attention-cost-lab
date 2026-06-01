# Attention Cost Lab

Estimate Transformer attention FLOPs, KV cache size, and decode memory pressure from model shape parameters.

This is a small CLI for quickly answering practical inference questions:

- How large is the KV cache at a given context length?
- How much does GQA reduce decode-time KV reads?
- How many attention FLOPs happen during prefill and decode?
- Is a shape likely to be memory-pressure sensitive during generation?
- What decode throughput does a simple hardware roofline imply?

## Demo

```bash
python -m attention_cost_lab.cli --preset llama3-8b --seq-len 8192
```

Example output:

```text
Attention Cost Lab
==================
layers                       32
hidden_size                  4096
heads                        32
kv_heads                     8
head_dim                     128
GQA group size               4
dtype                        fp16
batch_size                   1
seq_len                      8192
KV cache                     1.00 GiB
KV growth/token              128.00 KiB
prefill attention FLOPs      35.18T
decode FLOPs/token           4.29B
decode KV read/token         1.00 GiB
decode intensity             4.00 FLOPs/byte
```

## Install From GitHub

```bash
pip install git+https://github.com/jiakai03177-coder/attention-cost-lab.git
```

Local development:

```bash
git clone https://github.com/jiakai03177-coder/attention-cost-lab.git
cd attention-cost-lab
python -m pip install -e .
```

## Usage

Use a preset:

```bash
attention-cost --preset llama3-8b --seq-len 4096 --batch-size 2
```

Or pass a custom shape:

```bash
attention-cost \
  --layers 32 \
  --hidden-size 4096 \
  --heads 32 \
  --kv-heads 8 \
  --seq-len 8192 \
  --dtype fp16
```

JSON output:

```bash
attention-cost --preset mistral-7b --seq-len 8192 --json
```

Estimate a simple decode roofline from sustained bandwidth and compute:

```bash
attention-cost --preset llama3-8b --seq-len 8192 --memory-bandwidth-gbps 1000 --compute-tflops 100
```

Example roofline output:

```text
Roofline estimate:
memory bandwidth              1000 GB/s
compute throughput            100 TFLOP/s
roofline ridge point          100.00 FLOPs/byte
decode memory lower bound     1.07 ms
decode compute lower bound    42.95 us
decode roofline time          1.07 ms
decode roofline throughput    931.32 tokens/s
roofline bottleneck           memory
```

Compare multiple context lengths:

```bash
attention-cost --preset llama3-8b --sweep 1024,2048,4096,8192
```

Example sweep output:

```text
Context Sweep
=============
seq_len  KV cache     KV/token    prefill FLOPs  decode FLOPs/token  decode KV read/token  FLOPs/byte
-------  -----------  ----------  -------------  ------------------  --------------------  ----------
1024     128.00 MiB   128.00 KiB  549.76B        536.87M             128.00 MiB            4.00
2048     256.00 MiB   128.00 KiB  2.20T          1.07B               256.00 MiB            4.00
4096     512.00 MiB   128.00 KiB  8.80T          2.15B               512.00 MiB            4.00
8192     1.00 GiB     128.00 KiB  35.18T         4.29B               1.00 GiB              4.00
```

Export sweep data to CSV:

```bash
attention-cost --preset llama3-8b --sweep 1024,2048,4096,8192 --csv sweep.csv
```

CSV output includes raw byte/FLOP columns for plotting and formatted columns for quick inspection.
When roofline inputs are provided, CSV and JSON output include the roofline columns too.

## Model Presets

The presets are common shape shortcuts, not performance claims:

- `llama2-7b`
- `llama3-8b`
- `mistral-7b`

## Development

```bash
python -m unittest discover -s tests
python -m attention_cost_lab.cli --preset llama3-8b --seq-len 8192
```

## Roadmap

- ~~Add context length sweep comparisons.~~ Done in v0.2.0
- ~~Add CSV export for sweep results.~~ Done in v0.3.0
- ~~Add simple roofline estimates for memory bandwidth limits.~~ Done in v0.4.0
- Add SVG/Markdown report export.
- Add tensor-parallel and pipeline-parallel memory breakdowns.

## License

MIT
