# Attention Cost Lab

Estimate Transformer attention FLOPs, KV cache size, and decode memory pressure from model shape parameters.

This is a small CLI for quickly answering practical inference questions:

- How large is the KV cache at a given context length?
- How much does GQA reduce decode-time KV reads?
- How many attention FLOPs happen during prefill and decode?
- Is a shape likely to be memory-pressure sensitive during generation?

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

- Add CSV comparisons across context lengths.
- Add simple roofline estimates for memory bandwidth limits.
- Add SVG/Markdown report export.
- Add tensor-parallel and pipeline-parallel memory breakdowns.

## License

MIT
