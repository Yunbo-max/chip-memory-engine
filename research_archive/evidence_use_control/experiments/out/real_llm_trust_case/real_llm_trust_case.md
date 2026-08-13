# Real LLM Trust-Gated Case

Verdict: `PASS`

## Model

- Model path: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- GPU: NVIDIA GeForce RTX 4090 via CUDA_VISIBLE_DEVICES=2
- Peak CUDA allocation: 8414.1 MiB

## Learned Gate

- Validation AUROC: 0.8270
- Validation AUPRC: 0.7631
- Selected threshold: 0.930

## Held-Out LLM Generation Metrics

- Examples: 16
- Baseline mean F1: 0.7263
- Gated mean F1: 0.7357
- F1 delta: 0.0094
- Mean retrieval steps: 3.3750 -> 2.3125
- Relative effort reduction: 31.48%
- Answer preservation rate: 100.00%

## Interpretation

This is a real local LLM generation test. The gate is learned from real SQuAD2 answer-span labels, then used to shorten context before generation.
