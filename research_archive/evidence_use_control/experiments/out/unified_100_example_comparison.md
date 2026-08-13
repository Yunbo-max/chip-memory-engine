# Unified 100-Example Method Comparison

All rows below use 100 evaluated SQuAD2 answerable examples. The self-contained rows are retrieval/evidence simulations without final LLM generation. The Qwen rows use Qwen3-4B-Instruct final answer generation.

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

## Interpretation

The 100-example comparison supports the upgrade path:

1. A pure lexical trust proxy saves effort but fails accuracy.
2. A learned evidence proxy is safer, but saves less effort.
3. Raw Qwen attention-flow saves much more effort, but still loses about 1.4 F1 points.
4. Calibrating Qwen attention-flow with retrieval relevance is the best current result: about 67% effort reduction with essentially unchanged aggregate F1.
5. The 71120-style contribution-flow row uses the real contribution-matrix code path and passes on 100 examples, but with lower effort savings than calibrated Qwen attention-flow.
6. EvidenceUseGate-v0 implements the right learned-gate structure, but its first counterfactual-teacher version overfits and stops too early on held-out examples.
7. EvidenceUseGate-v1 conservative fixes the main v0 failure: it preserves answer quality and records zero wrong stops on the clean held-out set. It is too cautious, saving only 14.43% effort.
8. EvidenceUseGate-v2 Pareto distills calibrated attention-flow but keeps v1-style safety constraints. It produces a real risk-budget curve and improves the best learned-gate effort point to 27.84% while preserving F1, but it still misses the 35% effort target.
9. EvidenceUseGate-v3 changes safety from a mandatory low-risk condition into a veto on an attention-flow teacher. It reaches the desired efficiency band at 49.23% effort reduction and preserves aggregate F1, but wrong-stop rises to 11%, above the 5% target.
10. EvidenceUseGate-v4 selective fallback keeps v3 as the efficient default and falls back to v2 for high-risk stops. On the saved 100-example table it reaches +0.0023 F1 delta, 41.24% effort reduction, and 5% wrong-stop rate, but the fallback threshold is post-hoc on the same table and therefore diagnostic rather than a clean held-out validation.
11. EvidenceUseGate-v4 clean validation chooses the fallback rule on validation only and evaluates once on the same 100 held-out IDs. It passes F1 and effort targets with +0.0230 F1 delta and 44.07% effort reduction, but misses the strict safety target with 7% wrong-stop rate.

The contribution-flow row is closer to 71120 mechanistically because it uses contribution matrices, but it is still not a strict paper reproduction: the local run uses LLaMA-2-7B-Chat rather than the exact LLaMA-3.2/Gemma checkpoints and uses a simple relevance blend rather than the full XGBoost calibrator trained on the paper's saved relevance layouts.

EvidenceUseGate-v0 and v1 together are important design results: v0 is too aggressive, saving 60.53% effort but losing 4.6 F1 points; v1 fixes safety, reaching 0% wrong-stop rate and 100% answer preservation, but saves only 14.43% effort. Calibrated Qwen attention-flow is currently the sweet spot: it preserves quality and saves 67.06% effort.

EvidenceUseGate-v2 confirms that the right object is a controllable tradeoff curve rather than another single threshold:

```text
risk_budget = 0.0  -> v1-like conservative policy
risk_budget = 0.5  -> balanced policy
risk_budget = 1.0  -> attention-flow-like aggressive policy
```

The first v2 run improves the learned-gate frontier but does not dominate calibrated attention-flow:

| Risk budget | F1 delta | Effort reduction | Wrong-stop rate |
|---:|---:|---:|---:|
| 0.000 | +0.0140 | 21.39% | 2.00% |
| 0.005 | +0.0093 | 21.91% | 3.00% |
| 0.010 | +0.0017 | 24.23% | 5.00% |
| 0.020 | +0.0017 | 27.84% | 5.00% |

This is progress over v1's 14.43% effort reduction, but it is not yet the target operating region.

The v2 run used calibrated Qwen attention-flow as the practical teacher and v1 conservative safety heads as constraints. In other words, it learned when to imitate attention-flow's early stop and when to override it because noise risk, uncertainty, or conservative evidence-use labels said the stop was unsafe.

EvidenceUseGate-v3 shows the other side of the tradeoff. Treating safety as a veto recovers much more efficiency:

| Method | F1 delta | Effort reduction | Wrong-stop rate |
|---|---:|---:|---:|
| EvidenceUseGate-v2 best | +0.0017 | 27.84% | 5.00% |
| EvidenceUseGate-v3 guarded distill | +0.0075 | 49.23% | 11.00% |
| EvidenceUseGate-v4 selective fallback diagnostic | +0.0023 | 41.24% | 5.00% |
| EvidenceUseGate-v4 clean validation | +0.0230 | 44.07% | 7.00% |

This means v3 solves v2's efficiency problem but reopens the safety problem. The v4 diagnostic shows that selective fallback can recover the requested operating point on the saved table. The clean v4 run confirms that the validation-selected selector improves the learned-gate frontier, but it does not yet satisfy the strict endpoint because wrong-stop is 7% rather than <=5%.

The clean v4 error analysis points to the next small fix: the validation-selected risk detector caught 6 of 11 v3 wrong stops. Using the same validation-selected risk condition with full top-k fallback instead of v2 fallback would give a diagnostic test point of +0.0188 F1 delta, 37.89% effort reduction, and 5% wrong-stop. That secondary point is not the primary clean result, but it suggests the remaining bottleneck is fallback choice as much as risk detection.
