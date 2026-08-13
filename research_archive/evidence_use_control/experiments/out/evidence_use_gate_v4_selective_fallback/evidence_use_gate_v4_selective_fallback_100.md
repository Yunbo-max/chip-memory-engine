# EvidenceUseGate-v4 Selective Fallback

Verdict: `PASS_DIAGNOSTIC_TARGET`

- Status: post-hoc diagnostic from saved v2/v3 100-example raw outputs
- Default policy: EvidenceUseGate-v3 guarded distill
- Efficiency teacher: calibrated Qwen attention-flow
- Safety veto signal: saved EvidenceUseGate heads at the v3 stop step
- Fallback target: v2 conservative policy or full top-k, selected by threshold search
- Clean validation note: thresholds were selected on this saved test table, so this is not an independent held-out validation

## Selected Policy

- Condition: `(expected_f1_if_stop <= 0.595431) OR (fd_current_question_recall <= 0.122917)`
- Fallback source: `v2`
- Fallback count: 25 / 100 (25.00%)

| Method | F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation |
|---|---:|---:|---:|---:|---:|---:|
| v2 fallback baseline | 0.7607 | +0.0017 | 3.8800 -> 2.8000 | 27.84% | 5.00% | 95.00% |
| v3 default | 0.7665 | +0.0075 | 3.8800 -> 1.9700 | 49.23% | 11.00% | 89.00% |
| v4 selected fallback | 0.7613 | +0.0023 | 3.8800 -> 2.2800 | 41.24% | 5.00% | 95.00% |
| full top-k | 0.7590 | +0.0000 | 3.8800 -> 3.8800 | 0.00% | 0.00% | 100.00% |

## Oracle Headroom

| Oracle fallback | F1 delta | Effort reduction | Wrong-stop | Fallback count |
|---|---:|---:|---:|---:|
| v3-wrong -> v2 | +0.0403 | 45.10% | 3.00% | 11 |
| v3-wrong -> full_topk | +0.0561 | 41.49% | 0.00% | 11 |

## Top Candidate Policies

| Rank | Pass | Fallback | Condition | F1 delta | Effort | Wrong-stop | Fallbacks |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | yes | v2 | `(expected_f1_if_stop <= 0.595431) OR (fd_current_question_recall <= 0.122917)` | +0.0023 | 41.24% | 5.00% | 25 |
| 2 | yes | full_topk | `(uncertainty <= 0.332270) AND (fd_max_lexical_overlap_seen <= 0.200000)` | +0.0188 | 40.98% | 5.00% | 14 |
| 3 | yes | full_topk | `(uncertainty <= 0.332270) AND (fd_max_lexical_overlap_seen <= 0.212030)` | +0.0188 | 40.98% | 5.00% | 14 |
| 4 | yes | full_topk | `(noise_risk >= 0.123677) AND (uncertainty <= 0.332270)` | +0.0280 | 40.46% | 5.00% | 14 |
| 5 | yes | full_topk | `(noise_risk >= 0.141030) AND (uncertainty <= 0.332270)` | +0.0280 | 40.46% | 5.00% | 14 |
| 6 | yes | full_topk | `(uncertainty <= 0.332270) AND (fd_max_lexical_overlap_seen <= 0.217391)` | +0.0188 | 40.46% | 5.00% | 16 |
| 7 | yes | full_topk | `(expected_f1_if_stop <= 0.595431) OR (fd_current_lexical_overlap <= 0.000000)` | +0.0016 | 40.46% | 5.00% | 17 |
| 8 | yes | full_topk | `(expected_f1_if_stop <= 0.595431) OR (fd_current_question_recall <= 0.000000)` | +0.0016 | 40.46% | 5.00% | 17 |
| 9 | yes | full_topk | `(uncertainty <= 0.332270) AND (stop_prob <= 0.696827)` | +0.0288 | 40.21% | 5.00% | 15 |
| 10 | yes | full_topk | `(uncertainty <= 0.332270) AND (stop_prob <= 0.715854)` | +0.0288 | 40.21% | 5.00% | 16 |

## Interpretation

v4 as a diagnostic reaches the requested operating point by keeping v3 for most examples and sending a small high-risk subset to a safer fallback policy. This supports the v4 direction, but it does not yet verify a new independent method because the fallback threshold was chosen post-hoc on the same 100 examples.

A clean v4 run should rebuild validation rows, choose the fallback thresholds on validation, then report this same table on held-out test IDs.
