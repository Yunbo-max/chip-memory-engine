# EvidenceUseGate-v4 Clean Validation

Verdict: `NO_CLEAN_TARGET_PASS`

- Status: clean validation-selected fallback policy
- Default policy: EvidenceUseGate-v3 guarded distill
- Fallback choices searched on validation only: v2 or full top-k
- Held-out test: one-shot evaluation on the same deterministic SQuAD2 IDs used by v1/v2/v3
- Train/val/test examples: 96 / 40 / 100
- GPU: NVIDIA GeForce RTX 4090 via CUDA_VISIBLE_DEVICES=0

## Selected Fallback Policy

- Condition: `(uncertainty <= 0.333041) AND (fd_max_lexical_overlap_seen <= 0.243966)`
- Fallback source: `v2`
- Validation fallback count: 5 / 40
- Test fallback count: 20 / 100

## Same-Slice Test Comparison

| Method | F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation |
|---|---:|---:|---:|---:|---:|---:|
| Calibrated attention-flow | 0.7520 | -0.0070 | 3.8800 -> 1.1400 | 70.62% | 16.00% | 84.00% |
| EvidenceUseGate-v2 | 0.7607 | +0.0017 | 3.8800 -> 2.8000 | 27.84% | 5.00% | 95.00% |
| EvidenceUseGate-v3 | 0.7665 | +0.0075 | 3.8800 -> 1.9700 | 49.23% | 11.00% | 89.00% |
| EvidenceUseGate-v4 clean | 0.7820 | +0.0230 | 3.8800 -> 2.1700 | 44.07% | 7.00% | 93.00% |

## Safety Diagnostic

Using the same validation-selected risk condition but falling back to full top-k instead of v2 gives:

| Variant | F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation |
|---|---:|---:|---:|---:|---:|---:|
| Same risk condition + full top-k fallback | 0.7779 | +0.0188 | 3.8800 -> 2.4100 | 37.89% | 5.00% | 95.00% |

This is not the primary clean result because the clean run selected v2 fallback on validation. It is a diagnostic showing that the remaining miss is partly the fallback choice: the risk selector caught 6 of 11 v3 wrong-stops, while v2 fallback still left 2 wrong-stop cases that full top-k would avoid.

## Validation-Selected Candidate Policies

| Rank | Val pass | Fallback | Condition | Val F1 delta | Val effort | Val wrong-stop | Val fallbacks |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | yes | v2 | `(uncertainty <= 0.333041) AND (fd_max_lexical_overlap_seen <= 0.243966)` | +0.0502 | 52.50% | 5.00% | 5 |
| 2 | yes | v2 | `(uncertainty <= 0.333041) AND (fd_max_lexical_overlap_seen <= 0.257650)` | +0.0502 | 52.50% | 5.00% | 5 |
| 3 | yes | v2 | `(uncertainty <= 0.333041) AND (fd_max_lexical_overlap_seen <= 0.263860)` | +0.0502 | 52.50% | 5.00% | 5 |
| 4 | yes | v2 | `(uncertainty <= 0.333041) AND (fd_max_lexical_overlap_seen <= 0.268333)` | +0.0502 | 52.50% | 5.00% | 5 |
| 5 | yes | v2 | `(uncertainty <= 0.333041) AND (fd_max_lexical_overlap_seen <= 0.286411)` | +0.0502 | 52.50% | 5.00% | 5 |
| 6 | yes | v2 | `(uncertainty <= 0.333041) AND (fd_max_lexical_overlap_seen <= 0.401053)` | +0.0502 | 52.50% | 5.00% | 5 |
| 7 | yes | v2 | `(uncertainty <= 0.335837) AND (fd_max_lexical_overlap_seen <= 0.243966)` | +0.0502 | 52.50% | 5.00% | 6 |
| 8 | yes | v2 | `(uncertainty <= 0.335837) AND (fd_max_lexical_overlap_seen <= 0.257650)` | +0.0502 | 52.50% | 5.00% | 6 |
| 9 | yes | v2 | `(uncertainty <= 0.335837) AND (fd_max_lexical_overlap_seen <= 0.263860)` | +0.0502 | 52.50% | 5.00% | 6 |
| 10 | yes | v2 | `(uncertainty <= 0.335837) AND (fd_max_lexical_overlap_seen <= 0.268333)` | +0.0502 | 52.50% | 5.00% | 6 |

## Interpretation

This is the clean test of whether the v4 fallback selector generalizes from validation to held-out test. It should be used as the defensible method result, while the earlier v4 selective-fallback file remains a post-hoc diagnostic. The result is a near miss: F1 and effort pass, but wrong-stop is 7% rather than the target <=5%.
