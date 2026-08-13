# EvidenceUseGate-v2 Pareto

Verdict: `NO_TARGET_PASS`

- Runtime gate: controllable learned evidence-use policy
- Practical teacher: calibrated Qwen attention-flow
- Safety constraints: v1 conservative evidence labels, noise risk, uncertainty, predicted F1 drop
- Train/val/test examples: 96 / 40 / 100
- Train/val states: 355 / 160
- Attention teacher alpha/threshold: 0.60 / 0.510

## Same-Split Attention Teacher

- Baseline F1: 0.7590
- Attention-teacher F1: 0.7520
- F1 delta: -0.0070
- Steps: 3.8800 -> 1.1400
- Effort reduction: 70.62%

## Pareto Sweep

| Risk budget | Gate F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.000 | 0.7730 | +0.0140 | 3.8800 -> 3.0500 | 21.39% | 2.00% | 98.00% | MISS_TARGET |
| 0.005 | 0.7683 | +0.0093 | 3.8800 -> 3.0300 | 21.91% | 3.00% | 97.00% | MISS_TARGET |
| 0.010 | 0.7607 | +0.0017 | 3.8800 -> 2.9400 | 24.23% | 5.00% | 95.00% | MISS_TARGET |
| 0.020 | 0.7607 | +0.0017 | 3.8800 -> 2.8000 | 27.84% | 5.00% | 95.00% | MISS_TARGET |

## Interpretation

v2 tests the intended next step: learning a controllable safety-efficiency curve rather than one fixed threshold. A point passes the conservative target if F1 delta is at least -0.005 and effort reduction is at least 35%.

No noisy-distractor evaluation is included in this run; wrong-stop here is measured against the clean held-out top-k baseline.
- Peak CUDA allocation: 8414.1 MiB
