# EvidenceUseGate-v2 Pareto Roadmap

## Current Diagnosis

The learned-gate experiments show a safety-efficiency tradeoff:

| Method | Main Behavior | F1 Delta | Effort Reduction | Diagnosis |
|---|---:|---:|---:|---|
| EvidenceUseGate-v0 | aggressive learned stop | -0.0460 | 60.53% | over-stops |
| EvidenceUseGate-v1 conservative | conservative learned stop | +0.0085 | 14.43% | safe but under-saves |
| EvidenceUseGate-v2 Pareto, budget 0.020 | risk-budget learned stop | +0.0017 | 27.84% | improves v1 but misses effort target |
| EvidenceUseGate-v3 guarded distill | attention teacher plus safety veto | +0.0075 | 49.23% | efficient but wrong-stop too high |
| EvidenceUseGate-v4 selective fallback | v3 default plus fallback to v2 | +0.0023 | 41.24% | diagnostic target pass, post-hoc |
| EvidenceUseGate-v4 clean validation | validation-selected v3 plus fallback to v2 | +0.0230 | 44.07% | clean near miss, wrong-stop 7% |
| Calibrated Qwen attention-flow | fixed calibrated stop | -0.0005 | 67.06% | best current operating point |

EvidenceUseGate-v1 is not a simple failure. It fixed v0's safety problem: wrong-stop rate is 0% and answer preservation is 100%. Its weakness is efficiency.

The first EvidenceUseGate-v2 run confirms the Pareto framing. It improves the learned-gate frontier from 14.43% to 27.84% effort reduction while preserving F1, but it still does not reach the 35% effort target.

## First v2 Run

| Risk Budget | F1 Delta | Effort Reduction | Wrong-Stop Rate | Answer Preservation |
|---:|---:|---:|---:|---:|
| 0.000 | +0.0140 | 21.39% | 2.00% | 98.00% |
| 0.005 | +0.0093 | 21.91% | 3.00% | 97.00% |
| 0.010 | +0.0017 | 24.23% | 5.00% | 95.00% |
| 0.020 | +0.0017 | 27.84% | 5.00% | 95.00% |

Same-split calibrated attention teacher:

- F1 delta: -0.0070
- Effort reduction: 70.62%
- Wrong-stop rate: 16.00%

Interpretation: v2 correctly blocks many unsafe attention-teacher early stops, but the safety constraints remain too conservative to reach the target efficiency.

## First v3 Run

| Risk Budget | F1 Delta | Effort Reduction | Wrong-Stop Rate | Answer Preservation |
|---:|---:|---:|---:|---:|
| 0.005 | +0.0075 | 49.23% | 11.00% | 89.00% |
| 0.010 | +0.0075 | 49.23% | 11.00% | 89.00% |
| 0.020 | +0.0075 | 49.23% | 11.00% | 89.00% |
| 0.050 | +0.0075 | 49.23% | 11.00% | 89.00% |

Interpretation: v3 recovers the desired efficiency range by treating safety as a veto rather than a mandatory low-risk condition. Aggregate F1 is safe, but per-example wrong-stop rate is too high. This means the next bottleneck is not efficiency; it is wrong-stop prediction.

## v4 Diagnostic

| Policy | F1 Delta | Effort Reduction | Wrong-Stop Rate | Fallback Count |
|---|---:|---:|---:|---:|
| v2 fallback baseline | +0.0017 | 27.84% | 5.00% | 100 |
| v3 default | +0.0075 | 49.23% | 11.00% | 0 |
| v4 selective fallback | +0.0023 | 41.24% | 5.00% | 25 |

Selected post-hoc fallback condition:

```text
expected_f1_if_stop <= 0.595431
or fd_current_question_recall <= 0.122917
```

Interpretation: selective fallback has enough signal to combine v3's efficiency with v2's safety target on the saved 100-example table. This is diagnostic, not a clean validation, because the fallback threshold was selected post-hoc on the same test rows. A clean v4 must rebuild validation rows, choose this fallback rule on validation, and evaluate once on the held-out test IDs.

## Clean v4 Run

| Method | F1 Delta | Effort Reduction | Wrong-Stop Rate | Answer Preservation |
|---|---:|---:|---:|---:|
| Calibrated attention-flow | -0.0070 | 70.62% | 16.00% | 84.00% |
| v2 Pareto best | +0.0017 | 27.84% | 5.00% | 95.00% |
| v3 guarded distill | +0.0075 | 49.23% | 11.00% | 89.00% |
| v4 clean validation | +0.0230 | 44.07% | 7.00% | 93.00% |

Selected validation-only fallback rule:

```text
if uncertainty <= 0.333041
and fd_max_lexical_overlap_seen <= 0.243966:
    fallback to v2
else:
    use v3
```

Interpretation: clean v4 improves the learned-gate frontier and beats the F1/effort target, but it is not the endpoint yet because wrong-stop is 7%, above the 5% requirement. It caught 6 of 11 v3 wrong-stops. A diagnostic with the same validation-selected risk condition but full top-k fallback would reach +0.0188 F1 delta, 37.89% effort reduction, and 5% wrong-stop, suggesting the next variant should use a safety-first full fallback for the riskiest bucket.

## Current Learned-Gate Frontier

| Method | F1 Delta | Effort Reduction | Wrong-Stop Rate |
|---|---:|---:|---:|
| v1 conservative | +0.0085 | 14.43% | 0.00% |
| v2 Pareto best | +0.0017 | 27.84% | 5.00% |
| v3 guarded distill | +0.0075 | 49.23% | 11.00% |
| v4 selective fallback diagnostic | +0.0023 | 41.24% | 5.00% |
| v4 clean validation | +0.0230 | 44.07% | 7.00% |

The v4 diagnostic implements the intended selective fallback:

```text
if wrong_stop_risk is high:
    use v2 conservative policy or full top-k
else:
    use v3 guarded-distill step-2 policy
```

The remaining research requirement is clean calibration: choose the fallback rule on validation, not on the final 100-example report table.

## v2 Goal

EvidenceUseGate-v2 should not be a single stop threshold. It should be a controllable policy:

```text
risk_budget -> stop policy
```

Example operating points:

```text
risk_budget = 0.0  -> v1-like conservative
risk_budget = 0.5  -> balanced
risk_budget = 1.0  -> attention-flow-like aggressive
```

The target evaluation is a Pareto curve:

```text
F1 delta vs effort reduction
```

The stronger claim is:

> A learned evidence-use gate can trace a better cost-quality Pareto frontier than fixed calibrated attention-flow.

## Training Design

Use calibrated Qwen attention-flow as the practical teacher, because it is currently the best empirical method.

Use v1 conservative evidence-use labels as safety constraints:

- expected F1 if stop
- noise risk
- uncertainty
- hard negative labels
- counterfactual evidence-use labels

The student learns:

```text
teacher_action = calibrated_attention_flow_stop
safety_labels = conservative teacher + hard negatives
student = learned stop policy with safety override
```

At runtime:

```text
stop if expected_f1_drop <= risk_budget
        and expected_effort_saved is high
        and safety_override is false
```

## Required Experiment

Run the same 100-example held-out SQuAD2 slice and sweep risk budgets:

| Risk Budget | Expected Behavior |
|---:|---|
| 0.000 | v1-like safety |
| 0.005 | near-zero F1 drop |
| 0.010 | balanced quality/cost |
| 0.020 | aggressive effort saving |

Report:

- mean baseline F1
- mean gate F1
- F1 delta
- mean baseline steps
- mean gate steps
- effort reduction
- wrong-stop rate
- answer preservation
- noisy-distractor wrong-stop rate

Target:

- show at least one operating point with F1 delta >= -0.005 and effort reduction >= 35%
- show a Pareto point that dominates or matches calibrated Qwen attention-flow on quality at comparable effort
- show lower wrong-stop rate than calibrated attention-flow under noisy distractors
