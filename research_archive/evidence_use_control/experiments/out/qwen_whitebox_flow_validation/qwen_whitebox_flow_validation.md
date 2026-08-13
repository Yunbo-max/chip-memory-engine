# Qwen White-Box Flow Validation

Verdict: `PASS`

- Examples: 4
- Baseline F1: 0.8222
- Flow-gated F1: 0.8864
- F1 delta: 0.0641
- Steps: 3.2500 -> 1.0000
- Effort reduction: 69.23%
- Peak CUDA allocation: 8414.1 MiB

This uses real Qwen3-4B internal attentions as a white-box flow signal. It is a bridge validation, not the exact 71120 contribution/relevance/calibration pipeline.
