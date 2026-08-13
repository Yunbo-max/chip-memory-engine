# EvidenceUseGate-v3 Guarded Distillation

Verdict: `NO_TARGET_PASS`

- Runtime gate: calibrated attention-flow teacher with learned safety veto
- Efficiency teacher: calibrated Qwen attention-flow
- Guardrail: predicted F1 drop, v1 stop probability, noise risk, uncertainty
- Train/val/test examples: 96 / 40 / 100
- Train/val states: 355 / 160

## Pareto Sweep

| Risk budget | Gate F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.005 | 0.7665 | +0.0075 | 3.8800 -> 1.9700 | 49.23% | 11.00% | 89.00% | MISS_TARGET |
| 0.010 | 0.7665 | +0.0075 | 3.8800 -> 1.9700 | 49.23% | 11.00% | 89.00% | MISS_TARGET |
| 0.020 | 0.7665 | +0.0075 | 3.8800 -> 1.9700 | 49.23% | 11.00% | 89.00% | MISS_TARGET |
| 0.050 | 0.7665 | +0.0075 | 3.8800 -> 1.9700 | 49.23% | 11.00% | 89.00% | MISS_TARGET |

## Same-Split Attention Teacher

- F1 delta: -0.0070
- Effort reduction: 70.62%
- Wrong-stop rate: 16.00%

## Interpretation

v3 tests whether the learned gate can recover more of attention-flow's efficiency by treating safety as a veto instead of a mandatory low-risk condition. A target pass requires F1 delta >= -0.005, effort reduction >= 40%, and wrong-stop rate <= 5%.

- Peak CUDA allocation: 8414.1 MiB
