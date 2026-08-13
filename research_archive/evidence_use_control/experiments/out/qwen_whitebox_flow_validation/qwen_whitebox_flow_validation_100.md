# Qwen White-Box Flow Validation

Verdict: `PASS`

- Examples: 100
- Baseline F1: 0.7911
- Flow-gated F1: 0.7768
- F1 delta: -0.0143
- Steps: 3.4100 -> 1.2800
- Effort reduction: 62.46%
- Peak CUDA allocation: 8414.1 MiB

This uses real Qwen3-4B internal attentions as a white-box flow signal. It is a bridge validation, not the exact 71120 contribution/relevance/calibration pipeline.
