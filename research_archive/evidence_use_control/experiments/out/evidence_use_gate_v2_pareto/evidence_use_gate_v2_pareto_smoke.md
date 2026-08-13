# EvidenceUseGate-v2 Pareto

Verdict: `NO_TARGET_PASS`

- Runtime gate: controllable learned evidence-use policy
- Practical teacher: calibrated Qwen attention-flow
- Safety constraints: v1 conservative evidence labels, noise risk, uncertainty, predicted F1 drop
- Train/val/test examples: 4 / 2 / 2
- Train/val states: 18 / 9
- Attention teacher alpha/threshold: 0.60 / 0.510

## Same-Split Attention Teacher

- Baseline F1: 0.5152
- Attention-teacher F1: 0.3333
- F1 delta: -0.1818
- Steps: 5.0000 -> 1.0000
- Effort reduction: 80.00%

## Pareto Sweep

| Risk budget | Gate F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.000 | 0.5152 | +0.0000 | 5.0000 -> 3.5000 | 30.00% | 0.00% | 100.00% | MISS_TARGET |
| 0.010 | 0.5152 | +0.0000 | 5.0000 -> 3.5000 | 30.00% | 0.00% | 100.00% | MISS_TARGET |

## Interpretation

v2 tests the intended next step: learning a controllable safety-efficiency curve rather than one fixed threshold. A point passes the conservative target if F1 delta is at least -0.005 and effort reduction is at least 35%.

No noisy-distractor evaluation is included in this run; wrong-stop here is measured against the clean held-out top-k baseline.
- Peak CUDA allocation: 8414.1 MiB
