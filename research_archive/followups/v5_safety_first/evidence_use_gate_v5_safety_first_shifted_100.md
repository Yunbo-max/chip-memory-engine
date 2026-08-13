# EvidenceUseGate-v4 Clean Validation

Verdict: `NO_CLEAN_TARGET_PASS`

- Status: clean validation-selected fallback policy
- Default policy: EvidenceUseGate-v3 guarded distill
- Fallback choices searched on validation only: v2 or full top-k
- Held-out test: one-shot evaluation on the same deterministic SQuAD2 IDs used by v1/v2/v3
- Train/val/test examples: 120 / 60 / 100
- GPU: NVIDIA GeForce RTX 4090 via CUDA_VISIBLE_DEVICES=1

## Selected Fallback Policy

- Condition: `(predicted_f1_drop >= 0.044516) OR (stop_prob <= 0.508173)`
- Fallback source: `full_topk`
- Validation fallback count: 12 / 60
- Test fallback count: 27 / 100

## Same-Slice Test Comparison

| Method | F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation |
|---|---:|---:|---:|---:|---:|---:|
| Calibrated attention-flow | 0.7719 | -0.0265 | 3.9200 -> 1.1700 | 70.15% | 16.00% | 84.00% |
| EvidenceUseGate-v2 | 0.8005 | +0.0021 | 3.9200 -> 3.2100 | 18.11% | 3.00% | 97.00% |
| EvidenceUseGate-v3 | 0.8235 | +0.0250 | 3.9200 -> 2.5100 | 35.97% | 3.00% | 97.00% |
| EvidenceUseGate-v4 clean | 0.8287 | +0.0303 | 3.9200 -> 2.6400 | 32.65% | 2.00% | 98.00% |

## Validation-Selected Candidate Policies

| Rank | Val pass | Fallback | Condition | Val F1 delta | Val effort | Val wrong-stop | Val fallbacks |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | yes | full_topk | `(predicted_f1_drop >= 0.044516) OR (stop_prob <= 0.508173)` | +0.0344 | 38.27% | 1.67% | 12 |
| 2 | yes | full_topk | `(predicted_f1_drop >= 0.044516) OR (stop_prob <= 0.538590)` | +0.0344 | 37.86% | 1.67% | 15 |
| 3 | yes | full_topk | `(predicted_f1_drop >= 0.044516) OR (fd_max_question_recall_seen <= 0.333333)` | +0.0344 | 37.45% | 1.67% | 16 |
| 4 | yes | full_topk | `(predicted_f1_drop >= 0.044516) OR (stop_prob <= 0.570619)` | +0.0344 | 37.04% | 1.67% | 17 |
| 5 | yes | full_topk | `(predicted_f1_drop >= 0.044516) OR (fd_max_question_recall_seen <= 0.372727)` | +0.0344 | 36.63% | 1.67% | 17 |
| 6 | yes | full_topk | `(predicted_f1_drop >= 0.044516) OR (stop_prob <= 0.601578)` | +0.0344 | 36.21% | 1.67% | 20 |
| 7 | yes | v2 | `(attention_teacher_stop_prob <= 0.983493) AND (fd_current_question_recall <= 0.818182)` | +0.0329 | 38.27% | 1.67% | 12 |
| 8 | yes | v2 | `(attention_teacher_stop_prob <= 0.983493) AND (fd_max_question_recall_seen <= 0.818182)` | +0.0329 | 38.27% | 1.67% | 12 |
| 9 | yes | v2 | `(attention_teacher_stop_prob <= 0.992280) AND (fd_current_question_recall <= 0.818182)` | +0.0329 | 38.27% | 1.67% | 13 |
| 10 | yes | v2 | `(attention_teacher_stop_prob <= 0.992280) AND (fd_max_question_recall_seen <= 0.818182)` | +0.0329 | 38.27% | 1.67% | 13 |

## Interpretation

This is the clean test of whether the v4 fallback selector generalizes from validation to held-out test. It should be used as the defensible method result, while the earlier v4 selective-fallback file remains a post-hoc diagnostic.
