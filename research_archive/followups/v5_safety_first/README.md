# EvidenceUseGate-v5 Safety-First Shifted Split

Status: preliminary clean v0 execution.

This run was created during the oral research memory mission as a prospective follow-up to the saved v4 clean-validation near miss.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 python3 /tf/notebooks/oral_research_memory_mission_2026_06_10/scripts/run_evidence_use_gate_v5_safety_first.py \
  --train-examples 120 \
  --val-examples 60 \
  --eval-examples 100 \
  --learn-limit 1536 \
  --reference-v3-json /tmp/no_v3_reference.json \
  --output-tag evidence_use_gate_v5_safety_first_shifted_100
```

## Method

- Default policy: EvidenceUseGate-v3 guarded distill.
- Fallback selector: validation-only threshold search.
- Selection rule: safety-first, prioritizing lower validation wrong-stop before validation effort.
- Fallback choices: v2 or full top-k.
- Selected policy: `(predicted_f1_drop >= 0.044516) OR (stop_prob <= 0.508173)`.
- Selected fallback: full top-k.
- Validation fallback count: 12 / 60.
- Test fallback count: 27 / 100.

## Test Results Recomputed From JSONL

| Method | F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation |
|---|---:|---:|---:|---:|---:|---:|
| Calibrated attention-flow | 0.7719 | -0.0265 | 3.9200 -> 1.1700 | 70.15% | 16.00% | 84.00% |
| EvidenceUseGate-v2 | 0.8005 | +0.0021 | 3.9200 -> 3.2100 | 18.11% | 3.00% | 97.00% |
| EvidenceUseGate-v3 | 0.8235 | +0.0250 | 3.9200 -> 2.5100 | 35.97% | 3.00% | 97.00% |
| EvidenceUseGate-v5 safety-first | 0.8287 | +0.0303 | 3.9200 -> 2.6400 | 32.65% | 2.00% | 98.00% |

## Verdict

`NO_CLEAN_TARGET_PASS` for v5 under the preset target because effort reduction is 32.65%, below the 35% threshold.

However, this is a useful positive diagnostic:

- v5 improves safety and quality over v3: wrong-stop 3% -> 2%, F1 +0.0250 -> +0.0303.
- v5 catches one v3 wrong-stop by routing it to full top-k.
- v5 gives up too much efficiency: 35.97% -> 32.65% effort reduction.
- On this shifted split, v3 itself meets the prototype target: F1 delta +0.0250, effort 35.97%, wrong-stop 3%.

Interpretation:

The safety-first selector works as a guardrail but is too conservative for the 35% effort target. The next policy should be Pareto-aware: keep the v5 safety fallback, but only trigger it when expected safety gain exceeds expected effort cost.

## Files

- Raw per-example outputs: `evidence_use_gate_v5_safety_first_shifted_100.jsonl`
- Full run JSON: `evidence_use_gate_v5_safety_first_shifted_100.json`
- Generated report: `evidence_use_gate_v5_safety_first_shifted_100.md`
- Recomputed metrics: `recomputed_metrics.json`
- Recompute script: `recompute_v4_clean_metrics.py`
