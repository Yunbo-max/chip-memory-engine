# EvidenceUseGate Noise Robustness

Verdict: `NO_NOISE_ROBUSTNESS_TARGET_PASS`

- Status: clean-validation-selected policy tested on same held-out IDs with lexical distractor insertion
- Policy selection: train/validation are clean; no thresholds are selected on noisy test rows
- Distractor: one high lexical-overlap chunk from another SQuAD2 example inserted before clean retrieved chunks
- Train/val/test examples: 20 / 10 / 2
- GPU: NVIDIA GeForce RTX 4090 via CUDA_VISIBLE_DEVICES=1

## Selected Clean-Validation Policy

- Condition: `(predicted_f1_drop >= 0.000146) AND (uncertainty <= 0.413365)`
- Fallback source: `v2`
- Noisy test fallback count: 1 / 2

## Noise Stats

- Answer retained in noised chunks: 100.00%
- Mean distractor lexical overlap: 0.2344
- Mean distractor question recall: 0.6125

## Same-ID Noise Comparison

| Method | F1 | Delta vs clean full | Delta vs noisy full | Steps | Effort vs noisy full | Wrong-stop vs clean | Wrong-stop vs noisy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean full top-k | 0.3333 | +0.0000 | n/a | 3.5000 | 0.00% | 0.00% | n/a |
| Noisy full top-k | 0.3333 | +0.0000 | +0.0000 | 4.0000 | 0.00% | 0.00% | 0.00% |
| Calibrated attention-flow | 0.1818 | -0.1515 | -0.1515 | 4.0000 -> 2.0000 | 50.00% | 50.00% | 50.00% |
| EvidenceUseGate-v2 | 0.5000 | +0.1667 | +0.1667 | 4.0000 -> 2.0000 | 50.00% | 50.00% | 50.00% |
| EvidenceUseGate-v3 | 0.8333 | +0.5000 | +0.5000 | 4.0000 -> 3.0000 | 25.00% | 0.00% | 0.00% |
| Selective fallback | 0.5000 | +0.1667 | +0.1667 | 4.0000 -> 2.0000 | 50.00% | 50.00% | 50.00% |

## Interpretation

Selective fallback did not meet the proposed noise robustness criterion on this run. This means the current clean-selected fallback is not yet a reliable noise guardrail; the next version should train or validate with explicit hard distractor labels.
