# Trust-Gated Retrieval Pilot Evaluation

Verdict: `PARTIAL_PASS_FULL_EVAL_INCOMPLETE`

## Criteria

- `one_71033_baseline_trace_converted`: `PASS`
  Evidence: Trace contains the shared schema fields from SKILL.md.
- `71120_trust_fields_same_step_granularity`: `PASS_STRUCTURAL_PROXY`
  Evidence: 3 steps and matching trust/simulatability/concentration arrays.
- `diagnostics_answer_and_effort_outputs`: `PASS`
  Evidence: ANLS/citation diagnostics and MADQA effort metrics executed.
- `fields_trace_back_to_chip_implementation_items`: `PASS`
  Evidence: Grounding map points to MADQA trace/metrics and 71120 flow/relevance/calibrator paths.
- `retrieved_chunks_aligned_with_real_contribution_relevance_features`: `FAIL`
  Evidence: Current run uses toy contribution matrices; real 71120 JSONL/PT outputs were not generated.
- `trust_supports_stop_continue_before_final_answer`: `PASS_STRUCTURAL_PROXY`
  Evidence: Stop step computed as 1.
- `effort_metrics_computable_from_logs`: `PASS`
  Evidence: Retrieval/tool steps and repeated-loop rate computed from search_history shape.
- `required_runtime_modules_grounded_to_code`: `PASS`
  Evidence: Runtime modules point to code-inspected chip implementation paths.

## Bottom Line

The pilot passes the schema, code-grounding, and diagnostic evaluation. It does not yet pass the full method evaluation, because retrieved chunks have not been aligned with real 71120 contribution and relevance outputs.

## Next Required Eval

Run real 71120 attribution/relevance/calibration on a real benchmark slice.

GPU constraint: CUDA_VISIBLE_DEVICES=2 and <=20 GiB memory cap

The benchmark redesign is now in:

- `experiments/real_benchmark_eval_design.md`
- `experiments/run_real_benchmark_eval.py`
- `experiments/out/real_benchmark/squad2/benchmark_trace.jsonl`
- `experiments/out/real_benchmark/squad2/real_benchmark_eval.json`

Current benchmark status:

- Dataset: SQuAD2 validation
- Examples prepared: 8
- Verdict: `NOT_READY_FOR_METHOD_VERDICT`
- Missing: real 71120 contribution/rank/path JSONL, baseline predictions, trust-gated predictions

Self-contained benchmark test of the stop/continue idea itself:

- Script: `experiments/run_self_contained_trust_benchmark.py`
- Dataset: SQuAD2 validation
- Evaluated answerable examples: 58 from the first 128 validation rows
- Baseline: fixed top-5 retrieved sentence chunks
- `oracle_evidence` policy verdict: `PASS`
  - Baseline mean token F1: 1.0000
  - Trust-gated mean token F1: 1.0000
  - Mean retrieval steps: 4.2414 -> 1.2069
  - Relative effort reduction: 71.54%
  - Report: `experiments/out/self_contained_benchmark/self_contained_trust_benchmark_oracle_evidence.md`
- `lexical_trust` policy verdict: `FAIL`
  - Baseline mean token F1: 1.0000
  - Trust-gated mean token F1: 0.8621
  - Mean retrieval steps: 4.2414 -> 1.0172
  - Relative effort reduction: 76.02%
  - Failure reason: the cheap lexical trust proxy stops too early on about 13.8% of examples.
  - Report: `experiments/out/self_contained_benchmark/self_contained_trust_benchmark_lexical_trust.md`

Interpretation: the method idea has enough benchmark headroom if the evidence/trust detector is good, but a naive lexical proxy is not good enough. The next required experiment is therefore specifically the real 71120 information-flow trust detector, not more schema work.

Learned real-data trust benchmark:

- Script: `experiments/run_learned_trust_benchmark.py`
- Dataset: SQuAD2 validation answerable examples from the first 1024 rows
- Learning target: whether a retrieved sentence chunk contains a gold answer span
- Model: logistic regression over real retrieval/evidence features
- Train/validation/test examples: 306 / 102 / 103
- Chunk classifier validation AUROC: 0.8278
- Chunk classifier validation AUPRC: 0.7879
- Selected stop threshold: 0.900
- Held-out verdict: `PASS`
  - Baseline mean token F1: 0.9903
  - Learned trust mean token F1: 0.9515
  - F1 delta: -0.0388
  - Mean retrieval steps: 3.5437 -> 2.2233
  - Relative effort reduction: 37.26%
  - Answer preservation rate: 96.12%
  - Report: `experiments/out/learned_trust_benchmark/learned_trust_benchmark.md`

Interpretation update: using real benchmark labels to learn the trust signal works in the self-contained setting. It passes the planned criterion of reducing effort while keeping answer quality within a 5-point drop. The remaining gap is replacing the lightweight learned evidence proxy with the paper-grounded 71120 information-flow features.

Real 4B LLM inference case for **Trust-Gated Retrieval Agent with Learned Evidence Proxy**:

- Script: `experiments/run_real_llm_trust_case.py`
- Model: Qwen/Qwen3-4B-Instruct-2507
- Local snapshot: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- GPU: GPU 2 only via `CUDA_VISIBLE_DEVICES=2`
- Peak CUDA allocation: 8414.1 MiB
- Held-out SQuAD2 generation examples: 16
- Gate: learned from real SQuAD2 answer-span labels
- Verdict: `PASS`
  - Baseline mean F1: 0.7263
  - Trust-gated mean F1: 0.7357
  - F1 delta: +0.0094
  - Mean retrieval steps: 3.3750 -> 2.3125
  - Relative effort reduction: 31.48%
  - Answer preservation rate: 100.00%
  - Report: `experiments/out/real_llm_trust_case/real_llm_trust_case.md`

Scaled 100-example real 4B LLM inference case:

- Script: `experiments/run_real_llm_trust_case.py`
- Model: Qwen/Qwen3-4B-Instruct-2507
- GPU: GPU 1 via `CUDA_VISIBLE_DEVICES=1`
- Held-out SQuAD2 answerable generation examples: 100
- Gate: learned from real SQuAD2 answer-span labels
- Verdict: `PASS`
  - Baseline mean F1: 0.8070
  - Trust-gated mean F1: 0.7950
  - F1 delta: -0.0121
  - Mean retrieval steps: 3.4100 -> 2.5800
  - Relative effort reduction: 24.34%
  - Answer preservation rate: 98.00%
  - Peak CUDA allocation: 8414.1 MiB
  - Report: `experiments/out/real_llm_trust_case/real_llm_trust_case_100.md`

Interpretation update: scaling from 16 to 100 examples preserves the basic result but weakens the headline. The method still passes under the 5-point F1-drop criterion: it saves about 24% retrieval/context effort with about a 1.2-point F1 drop. This is promising, but not yet a large-scale result.

Interpretation update: with a real 4B instruction model loaded on GPU 2, **Trust-Gated Retrieval Agent with Learned Evidence Proxy** reduced context/retrieval effort by about 31% and did not hurt answer quality on the first held-out generation slice.

Naming and scope:

- Verified method: **Trust-Gated Retrieval Agent with Learned Evidence Proxy**
- Target full method: **Trust-Gated Retrieval Agent with Information-Flow Trust Signal**
- Not claimed as verified: the 71120 paper method itself

The 71120 paper currently serves as the chips primitive/source module motivating a future replacement for the lightweight proxy. The current experiment does not use 71120 white-box contribution/relevance/calibration outputs.

Recommended project phrasing:

> We validate a trust-gated retrieval agent architecture using a lightweight learned evidence proxy. This serves as a first end-to-end proof of concept for the broader chip-composed method. Replacing the proxy with 71120-style information-flow trust is the next validation step.

Qwen white-box attention-flow bridge validation:

- Script: `experiments/run_qwen_whitebox_flow_validation.py`
- Model: Qwen/Qwen3-4B-Instruct-2507
- GPU: GPU 1 via `CUDA_VISIBLE_DEVICES=1`
- Signal: real Qwen3 internal attentions over retrieved context chunks
- Examples: 4 held-out SQuAD2 answerable examples
- Verdict: `PASS`
  - Baseline mean F1: 0.8222
  - Flow-gated mean F1: 0.8864
  - F1 delta: +0.0641
  - Mean retrieval steps: 3.2500 -> 1.0000
  - Relative effort reduction: 69.23%
  - Peak CUDA allocation: 8414.1 MiB
  - Report: `experiments/out/qwen_whitebox_flow_validation/qwen_whitebox_flow_validation.md`

Scope note: this is a white-box bridge validation using Qwen attention flow. It is closer to an internal trust signal than the learned proxy, but it is still not the exact 71120 contribution/relevance/calibration pipeline.

Calibrated Flow-Gated-Attention upgrade:

- Script: `experiments/run_qwen_calibrated_flow_trust_case.py`
- Model: Qwen/Qwen3-4B-Instruct-2507
- GPU: GPU 0 via `CUDA_VISIBLE_DEVICES=0`
- Signal: calibrated blend of Qwen internal attention-flow and retrieval relevance
- Calibration set: 24 SQuAD2 answerable examples
- Held-out generation examples: 32
- Calibration selected alpha: 0.60 internal-flow weight
- Calibration selected trust threshold: 0.510
- Verdict: `PASS`
  - Baseline mean F1: 0.8334
  - Calibrated-flow mean F1: 0.8114
  - F1 delta: -0.0220
  - Mean retrieval steps: 3.2500 -> 1.0938
  - Relative effort reduction: 66.35%
  - Answer preservation rate: 93.75%
  - Peak CUDA allocation: 8414.1 MiB
  - Report: `experiments/out/qwen_calibrated_flow_trust_case/qwen_calibrated_flow_trust_case_32.md`

Interpretation update: this is the first stronger gate that asks whether Qwen internally attends to the retrieved chunks, rather than only whether chunks lexically match the question. It saves much more context than the TF-IDF learned proxy, but it also introduces a larger F1 drop than the 100-example proxy run. This supports the upgrade direction while showing the need for contribution/relevance calibration and noise robustness tests.

Scaled 100-example Calibrated Flow-Gated-Attention run:

- Script: `experiments/run_qwen_calibrated_flow_trust_case.py`
- Model: Qwen/Qwen3-4B-Instruct-2507
- GPU: GPU 0 via `CUDA_VISIBLE_DEVICES=0`
- Signal: calibrated blend of Qwen internal attention-flow and retrieval relevance
- Calibration set: 24 SQuAD2 answerable examples
- Held-out generation examples: 100
- Calibration selected alpha: 0.60 internal-flow weight
- Calibration selected trust threshold: 0.510
- Verdict: `PASS`
  - Baseline mean F1: 0.7830
  - Calibrated-flow mean F1: 0.7825
  - F1 delta: -0.0005
  - Mean retrieval steps: 3.3700 -> 1.1100
  - Relative effort reduction: 67.06%
  - Answer preservation rate: 93.00%
  - Peak CUDA allocation: 8414.1 MiB
  - Report: `experiments/out/qwen_calibrated_flow_trust_case/qwen_calibrated_flow_trust_case_100.md`

Interpretation update: at 100 examples, the calibrated attention-flow gate preserves aggregate F1 almost exactly while reducing retrieval/context effort by about two thirds. This is a stronger result than the 32-example run and is now the best evidence that an internal-flow signal can improve the trust-gated retrieval architecture. It remains an attention-flow bridge, not yet the full 71120 contribution/relevance/calibration implementation.

Unified 100-example comparison:

| Method | Final Answerer | Trust Signal | Examples | Baseline F1 | Method F1 | F1 Delta | Baseline Steps | Method Steps | Effort Reduction | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Oracle evidence upper bound | Extractive oracle | Gold answer span in retrieved chunk | 100 | 0.9900 | 1.0000 | +0.0100 | 3.8500 | 1.3300 | 65.45% | Pass |
| Lexical trust proxy | Extractive oracle | TF-IDF lexical trust heuristic | 100 | 0.9900 | 0.8200 | -0.1700 | 3.8500 | 1.0100 | 73.77% | Fail |
| Learned evidence proxy | Extractive oracle | LogisticRegression over retrieval/evidence features | 100 | 0.9900 | 0.9500 | -0.0400 | 3.5700 | 2.2200 | 37.82% | Pass |
| Learned evidence proxy + Qwen | Qwen3-4B-Instruct | LogisticRegression over retrieval/evidence features | 100 | 0.8070 | 0.7950 | -0.0121 | 3.4100 | 2.5800 | 24.34% | Pass |
| Raw Qwen attention-flow | Qwen3-4B-Instruct | Qwen internal attention-flow threshold | 100 | 0.7911 | 0.7768 | -0.0143 | 3.4100 | 1.2800 | 62.46% | Pass |
| Calibrated Qwen attention-flow | Qwen3-4B-Instruct | Qwen internal attention-flow + retrieval relevance calibration | 100 | 0.7830 | 0.7825 | -0.0005 | 3.3700 | 1.1100 | 67.06% | Best current |
| 71120-style contribution-flow | LLaMA-2-7B-Chat | 71120 contribution matrices + retrieval relevance | 100 | 0.7181 | 0.7102 | -0.0079 | 4.0000 | 2.8300 | 29.25% | Mechanistic pass |
| EvidenceUseGate-v0 | Qwen3-4B-Instruct | learned gate from attention/retrieval state, teacher from counterfactual probes | 100 | 0.7786 | 0.7326 | -0.0460 | 3.8000 | 1.5000 | 60.53% | Fail |
| EvidenceUseGate-v1 conservative | Qwen3-4B-Instruct | multi-head learned gate with conservative counterfactual teacher and hard negatives | 100 | 0.7590 | 0.7675 | +0.0085 | 3.8800 | 3.3200 | 14.43% | Safety pass / conservative |
| EvidenceUseGate-v2 Pareto, budget 0.020 | Qwen3-4B-Instruct | calibrated attention-flow distillation with v1 safety constraints | 100 | 0.7590 | 0.7607 | +0.0017 | 3.8800 | 2.8000 | 27.84% | Pareto curve / target miss |
| EvidenceUseGate-v3 guarded distill | Qwen3-4B-Instruct | calibrated attention-flow teacher with learned safety veto | 100 | 0.7590 | 0.7665 | +0.0075 | 3.8800 | 1.9700 | 49.23% | Efficiency pass / safety fail |
| EvidenceUseGate-v4 selective fallback | Qwen3-4B-Instruct | v3 efficient policy with safety-head fallback to v2 | 100 | 0.7590 | 0.7613 | +0.0023 | 3.8800 | 2.2800 | 41.24% | Diagnostic pass / post-hoc |
| EvidenceUseGate-v4 clean validation | Qwen3-4B-Instruct | validation-selected v3 policy with safety fallback to v2 | 100 | 0.7590 | 0.7820 | +0.0230 | 3.8800 | 2.1700 | 44.07% | Clean near miss, wrong-stop 7% |

Unified comparison report: `experiments/out/unified_100_example_comparison.md`

71120-style contribution-flow pilot:

- Script: `experiments/run_llama_contribution_flow_trust_case.py`
- Source mechanism: local `71120_RAG-information-flow/proposed/Ours/llama/llama.py`
- Model: local LLaMA-2-7B-Chat checkpoint
- GPU: GPU 3 via `CUDA_VISIBLE_DEVICES=3`
- Signal: 71120 contribution matrices blended with retrieval relevance
- Held-out generation examples: 100
- Threshold: 0.650
- Verdict: `PASS`
  - Baseline mean F1: 0.7181
  - Contribution-flow mean F1: 0.7102
  - F1 delta: -0.0079
  - Mean retrieval steps: 4.0000 -> 2.8300
  - Relative effort reduction: 29.25%
  - Answer preservation rate: 96.00%
  - Peak CUDA allocation: 13102.6 MiB
  - Report: `experiments/out/llama_contribution_flow_trust_case/llama_contribution_flow_trust_case_100_t065.md`

Scope note: this is the first run using the real 71120 contribution-matrix code path. It is closer to the target information-flow method than attention-flow, but it is still not a strict paper reproduction because the exact LLaMA-3.2/Gemma checkpoints and full XGBoost calibrator over saved relevance layouts were not available in the local cache.

EvidenceUseGate-v0 learned gate:

- Script: `experiments/run_evidence_use_gate_v0.py`
- Model: Qwen/Qwen3-4B-Instruct-2507
- GPU: GPU 0 via `CUDA_VISIBLE_DEVICES=0`
- Runtime signal: learned gradient boosting gate over cheap Qwen attention/retrieval state features
- Teacher signal: answer sufficiency plus counterfactual drop probes
- Train/validation/test examples: 80 / 32 / 100
- Train/validation states: 312 / 107
- Teacher validation AUROC: 0.6181
- Teacher validation AUPRC: 0.7177
- Selected threshold: 0.660
- Verdict: `FAIL`
  - Baseline mean F1: 0.7786
  - EvidenceUseGate mean F1: 0.7326
  - F1 delta: -0.0460
  - Mean retrieval steps: 3.8000 -> 1.5000
  - Relative effort reduction: 60.53%
  - Answer preservation rate: 88.00%
  - Peak CUDA allocation: 8414.1 MiB
  - Report: `experiments/out/evidence_use_gate_v0/evidence_use_gate_v0_100.md`

Interpretation update: EvidenceUseGate-v0 implements the intended learned-gate structure rather than a fixed attention/relevance rule, but the first teacher/policy is too aggressive. The validation teacher classifier is weak out of sample, so the gate learns to stop early on attention patterns that do not reliably preserve answer quality. This is a useful negative result: the next version needs a stronger teacher, a noise-risk head, and contribution-flow teacher features where the 71120 attributor is model-compatible.

EvidenceUseGate-v1 conservative learned gate:

- Script: `experiments/run_evidence_use_gate_v1_conservative.py`
- Model: Qwen/Qwen3-4B-Instruct-2507
- GPU: GPU 1 via `CUDA_VISIBLE_DEVICES=1`
- Runtime gate: multi-head learned controller over cheap Qwen attention/retrieval state features
- Heads: stop probability, expected F1 if stop, continue value, noise risk, uncertainty
- Teacher signal: conservative answer sufficiency plus counterfactual drop and attention/relevance alignment
- Hard negatives: high-attention wrong chunks, lexical/relevance bait, high-relevance low-contribution cases, and later-answer cases
- Train/validation/test examples: 96 / 40 / 100
- Train/validation states: 355 / 160
- Stop-head validation AUROC: 0.5563
- Noise-head validation AUROC: 0.6530
- Stop threshold: 0.250
- Noise threshold: 0.150
- Uncertainty threshold: 0.400
- Verdict: `FAIL_TARGET_BUT_QUALITY_PASS`
  - Baseline mean F1: 0.7590
  - EvidenceUseGate-v1 mean F1: 0.7675
  - F1 delta: +0.0085
  - Mean retrieval steps: 3.8800 -> 3.3200
  - Relative effort reduction: 14.43%
  - Answer preservation rate: 100.00%
  - Wrong-stop rate: 0.00%
  - Peak CUDA allocation: 8414.1 MiB
  - Report: `experiments/out/evidence_use_gate_v1_conservative/evidence_use_gate_v1_conservative_100.md`

Interpretation update: EvidenceUseGate-v1 fixes the main v0 failure. It uses a conservative teacher and multi-head risk outputs, so it no longer over-stops on the clean 100-example held-out set; aggregate F1 even improves slightly. However, it does not meet the next-version target because effort reduction is only 14.43%, not 35-45%, and the teacher AUROC is below the requested 0.75 target. The right conclusion is not simply that v1 failed: it fixed safety but lost efficiency.

Current method spectrum:

- EvidenceUseGate-v0: too aggressive, with 60.53% effort reduction but a -0.0460 F1 delta.
- EvidenceUseGate-v1 conservative: too cautious, with +0.0085 F1 delta, 0% wrong-stop rate, 100% answer preservation, but only 14.43% effort reduction.
- Calibrated Qwen attention-flow: current best practical cost-quality tradeoff, with -0.0005 F1 delta and 67.06% effort reduction.

EvidenceUseGate-v2 target: learn a controllable Pareto policy rather than a single fixed stop threshold. The controller exposes a risk budget:

```text
risk_budget = 0.000  -> most conservative tested policy
risk_budget = 0.005  -> near-zero-drop policy
risk_budget = 0.010  -> balanced policy
risk_budget = 0.020  -> most aggressive tested policy
```

The v2 evaluation reports the full F1-effort Pareto curve. The implementation distills calibrated Qwen attention-flow as the empirical stop-action teacher, then uses v1-style conservative safety heads as constraints to override high-risk stops. The target claim is: learned evidence-use control can trace a better Pareto frontier than fixed calibrated attention-flow. The first v2 run improves over v1 but does not yet prove that stronger claim.

EvidenceUseGate-v2 Pareto learned gate:

- Script: `experiments/run_evidence_use_gate_v2_pareto.py`
- Model: Qwen/Qwen3-4B-Instruct-2507
- GPU: GPU 0 via `CUDA_VISIBLE_DEVICES=0`
- Runtime gate: risk-budget-controlled learned evidence-use policy
- Practical teacher: calibrated Qwen attention-flow, alpha 0.60, threshold 0.510
- Safety constraints: v1 conservative evidence labels, noise risk, uncertainty, predicted F1 drop
- Train/validation/test examples: 96 / 40 / 100
- Train/validation states: 355 / 160
- Verdict: `NO_TARGET_PASS`
  - Same-split attention teacher: F1 0.7590 -> 0.7520, steps 3.8800 -> 1.1400, effort reduction 70.62%, wrong-stop rate 16.00%
  - Best v2 effort point, risk budget 0.020: F1 0.7590 -> 0.7607, F1 delta +0.0017, steps 3.8800 -> 2.8000, effort reduction 27.84%, wrong-stop rate 5.00%
  - Peak CUDA allocation: 8414.1 MiB
  - Report: `experiments/out/evidence_use_gate_v2_pareto/evidence_use_gate_v2_pareto_100.md`

V2 Pareto sweep:

| Risk budget | Gate F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.7730 | +0.0140 | 3.8800 -> 3.0500 | 21.39% | 2.00% | 98.00% |
| 0.005 | 0.7683 | +0.0093 | 3.8800 -> 3.0300 | 21.91% | 3.00% | 97.00% |
| 0.010 | 0.7607 | +0.0017 | 3.8800 -> 2.9400 | 24.23% | 5.00% | 95.00% |
| 0.020 | 0.7607 | +0.0017 | 3.8800 -> 2.8000 | 27.84% | 5.00% | 95.00% |

Interpretation update: EvidenceUseGate-v2 does what it was designed to do structurally: it exposes a controllable F1-effort curve and improves over v1's 14.43% effort reduction while preserving F1. It does not yet meet the planned target, because the best point reaches 27.84% effort reduction rather than 35%+. The same-split attention teacher is much more aggressive at 70.62% effort reduction but has a -0.0070 F1 delta and 16% wrong-stop rate, so v2 is correctly adding safety but still too cautious.

EvidenceUseGate-v3 guarded distillation:

- Script: `experiments/run_evidence_use_gate_v3_guarded_distill.py`
- Model: Qwen/Qwen3-4B-Instruct-2507
- GPU: GPU 0 via `CUDA_VISIBLE_DEVICES=0`
- Runtime gate: calibrated attention-flow teacher with learned safety veto
- Efficiency teacher: calibrated Qwen attention-flow, alpha 0.60, threshold 0.510
- Guardrail: predicted F1 drop, v1 stop probability, noise risk, uncertainty
- Train/validation/test examples: 96 / 40 / 100
- Train/validation states: 355 / 160
- Verdict: `NO_TARGET_PASS`
  - Baseline mean F1: 0.7590
  - EvidenceUseGate-v3 mean F1: 0.7665
  - F1 delta: +0.0075
  - Mean retrieval steps: 3.8800 -> 1.9700
  - Relative effort reduction: 49.23%
  - Answer preservation rate: 89.00%
  - Wrong-stop rate: 11.00%
  - Peak CUDA allocation: 8414.1 MiB
  - Report: `experiments/out/evidence_use_gate_v3_guarded_distill/evidence_use_gate_v3_guarded_distill_100.md`

Interpretation update: EvidenceUseGate-v3 solves v2's efficiency problem but reopens the safety problem. It reaches the desired 40-50% effort range while preserving aggregate F1, but wrong-stop rises to 11%, above the 5% target. The current learned-gate frontier is therefore:

| Method | F1 delta | Effort reduction | Wrong-stop |
|---|---:|---:|---:|
| v1 conservative | +0.0085 | 14.43% | 0.00% |
| v2 Pareto best | +0.0017 | 27.84% | 5.00% |
| v3 guarded distill | +0.0075 | 49.23% | 11.00% |
| v4 selective fallback diagnostic | +0.0023 | 41.24% | 5.00% |
| v4 clean validation | +0.0230 | 44.07% | 7.00% |

EvidenceUseGate-v4 selective fallback diagnostic:

- Script: `experiments/run_evidence_use_gate_v4_selective_fallback.py`
- Runtime used for this diagnostic: no new GPU inference; recomputed from saved v2/v3 raw 100-example outputs
- Same exact question IDs: confirmed by script before evaluation
- Default policy: v3 guarded distill
- Fallback source: v2 budget 0.020
- Selected fallback condition: `expected_f1_if_stop <= 0.595431 OR fd_current_question_recall <= 0.122917`
- Calibration status: post-hoc on the saved 100-example table, not an independent held-out validation
- Verdict: `PASS_DIAGNOSTIC_TARGET`
  - Baseline mean F1: 0.7590
  - EvidenceUseGate-v4 diagnostic mean F1: 0.7613
  - F1 delta: +0.0023
  - Mean retrieval steps: 3.8800 -> 2.2800
  - Relative effort reduction: 41.24%
  - Answer preservation rate: 95.00%
  - Wrong-stop rate: 5.00%
  - Fallback count: 25 / 100
  - Report: `experiments/out/evidence_use_gate_v4_selective_fallback/evidence_use_gate_v4_selective_fallback_100.md`

Interpretation update: v4 shows the intended combination is plausible: v3 supplies efficiency, and selective fallback recovers the requested safety target on the saved table. However, this is diagnostic because the threshold was chosen post-hoc. The next clean validation should rebuild train/validation/test state rows, select the fallback rule only on validation, then report once on the same held-out question IDs.

EvidenceUseGate-v4 clean validation:

- Script: `experiments/run_evidence_use_gate_v4_clean_validation.py`
- Model: Qwen/Qwen3-4B-Instruct-2507
- GPU: GPU 0 via `CUDA_VISIBLE_DEVICES=0`
- Runtime gate: v3 guarded-distill default with validation-selected selective fallback
- Fallback source selected in the clean run: v2 budget 0.020
- Selected validation-only fallback condition: `uncertainty <= 0.333041 AND fd_max_lexical_overlap_seen <= 0.243966`
- Same exact test IDs as v1/v2/v3: confirmed by script against the saved v3 artifact
- Verdict: `NO_CLEAN_TARGET_PASS`
  - Baseline mean F1: 0.7590
  - EvidenceUseGate-v4 clean mean F1: 0.7820
  - F1 delta: +0.0230
  - Mean retrieval steps: 3.8800 -> 2.1700
  - Relative effort reduction: 44.07%
  - Answer preservation rate: 93.00%
  - Wrong-stop rate: 7.00%
  - Fallback count: 20 / 100
  - Peak CUDA allocation: 8414.1 MiB
  - Report: `experiments/out/evidence_use_gate_v4_clean_validation/evidence_use_gate_v4_clean_validation_100.md`
  - Raw rows: `experiments/out/evidence_use_gate_v4_clean_validation/evidence_use_gate_v4_clean_validation_100.jsonl`
  - Recompute script: `experiments/out/evidence_use_gate_v4_clean_validation/recompute_v4_clean_metrics.py`

Interpretation update: clean v4 is a near miss, not a final stop. It preserves and improves F1 while reaching the effort target, and reduces v3 wrong-stop from 11% to 7%, but the planned endpoint requires <=5%. The same validation-selected risk condition with full top-k fallback gives a diagnostic point of +0.0188 F1 delta, 37.89% effort reduction, and 5% wrong-stop. That suggests the next clean variant should use a safety-first fallback choice for high-risk rows before running the noisy-distractor test.

Candidate command shape:

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/run_real_benchmark_eval.py run-info-flow --dataset squad2 --model-path <local_llama_model> --i-block 1
```
