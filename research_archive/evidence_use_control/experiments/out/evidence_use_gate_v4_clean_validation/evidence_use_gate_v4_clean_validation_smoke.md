# EvidenceUseGate-v4 Clean Validation

Verdict: `NO_CLEAN_TARGET_PASS`

- Status: clean validation-selected fallback policy
- Default policy: EvidenceUseGate-v3 guarded distill
- Fallback choices searched on validation only: v2 or full top-k
- Held-out test: one-shot evaluation on the same deterministic SQuAD2 IDs used by v1/v2/v3
- Train/val/test examples: 12 / 4 / 2
- GPU: NVIDIA GeForce RTX 4090 via CUDA_VISIBLE_DEVICES=0

## Selected Fallback Policy

- Condition: `uncertainty <= 0.247519`
- Fallback source: `v2`
- Validation fallback count: 1 / 4
- Test fallback count: 0 / 2

## Same-Slice Test Comparison

| Method | F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation |
|---|---:|---:|---:|---:|---:|---:|
| Calibrated attention-flow | 0.7857 | +0.0000 | 3.0000 -> 1.5000 | 50.00% | 0.00% | 100.00% |
| EvidenceUseGate-v2 | 0.7857 | +0.0000 | 3.0000 -> 3.0000 | 0.00% | 0.00% | 100.00% |
| EvidenceUseGate-v3 | 0.7857 | +0.0000 | 3.0000 -> 2.5000 | 16.67% | 0.00% | 100.00% |
| EvidenceUseGate-v4 clean | 0.7857 | +0.0000 | 3.0000 -> 2.5000 | 16.67% | 0.00% | 100.00% |

## Validation-Selected Candidate Policies

| Rank | Val pass | Fallback | Condition | Val F1 delta | Val effort | Val wrong-stop | Val fallbacks |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | yes | v2 | `uncertainty <= 0.247519` | +0.0000 | 38.89% | 0.00% | 1 |
| 2 | yes | full_topk | `uncertainty <= 0.247519` | +0.0000 | 38.89% | 0.00% | 1 |
| 3 | yes | v2 | `uncertainty <= 0.262169` | +0.0000 | 38.89% | 0.00% | 1 |
| 4 | yes | full_topk | `uncertainty <= 0.262169` | +0.0000 | 38.89% | 0.00% | 1 |
| 5 | yes | v2 | `uncertainty <= 0.276820` | +0.0000 | 38.89% | 0.00% | 1 |
| 6 | yes | full_topk | `uncertainty <= 0.276820` | +0.0000 | 38.89% | 0.00% | 1 |
| 7 | yes | v2 | `uncertainty <= 0.291470` | +0.0000 | 38.89% | 0.00% | 1 |
| 8 | yes | full_topk | `uncertainty <= 0.291470` | +0.0000 | 38.89% | 0.00% | 1 |
| 9 | yes | v2 | `uncertainty <= 0.306121` | +0.0000 | 38.89% | 0.00% | 1 |
| 10 | yes | full_topk | `uncertainty <= 0.306121` | +0.0000 | 38.89% | 0.00% | 1 |

## Interpretation

This is the clean test of whether the v4 fallback selector generalizes from validation to held-out test. It should be used as the defensible method result, while the earlier v4 selective-fallback file remains a post-hoc diagnostic.
