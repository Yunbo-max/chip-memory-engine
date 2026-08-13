# EvidenceUseGate-v3 Guarded Distillation

Verdict: `NO_TARGET_PASS`

- Runtime gate: calibrated attention-flow teacher with learned safety veto
- Efficiency teacher: calibrated Qwen attention-flow
- Guardrail: predicted F1 drop, v1 stop probability, noise risk, uncertainty
- Train/val/test examples: 4 / 2 / 2
- Train/val states: 18 / 9

## Pareto Sweep

| Risk budget | Gate F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.005 | 0.5152 | +0.0000 | 5.0000 -> 3.5000 | 30.00% | 0.00% | 100.00% | MISS_TARGET |
| 0.020 | 0.5152 | +0.0000 | 5.0000 -> 3.5000 | 30.00% | 0.00% | 100.00% | MISS_TARGET |

## Same-Split Attention Teacher

- F1 delta: -0.1818
- Effort reduction: 80.00%
- Wrong-stop rate: 50.00%

## Interpretation

v3 tests whether the learned gate can recover more of attention-flow's efficiency by treating safety as a veto instead of a mandatory low-risk condition. A target pass requires F1 delta >= -0.005, effort reduction >= 40%, and wrong-stop rate <= 5%.

- Peak CUDA allocation: 8414.1 MiB
