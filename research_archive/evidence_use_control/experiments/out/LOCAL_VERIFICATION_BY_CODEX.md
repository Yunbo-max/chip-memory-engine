# Local Verification By Codex

## Verified Numeric Result

The aggregate numbers were recomputed directly from:

`experiments/out/verification_bundle/per_example_results.jsonl`

without relying on the summary file.

Recomputed values:

- `n = 16`
- `baseline_mean_f1 = 0.7263392857142857`
- `trust_gated_mean_f1 = 0.7357142857142858`
- `f1_delta = +0.009375`
- `baseline_mean_steps = 3.375`
- `trust_gated_mean_steps = 2.3125`
- `relative_effort_reduction = 0.3148148148148148`

Therefore this statement is supported:

> On this 16-example held-out SQuAD2 slice, Qwen3-4B-Instruct with trust-gated retrieval reduced retrieval/context effort by about 31.48% and did not reduce F1.

More precise method name:

> Trust-Gated Retrieval Agent with Learned Evidence Proxy

## Scope Limitation

This is **not** a full verification of the 71120 information-flow method.

The trust gate in this experiment uses:

- TF-IDF sparse retrieval features
- LogisticRegression trust/evidence model
- SQuAD2 answer-span labels
- threshold `0.93`

It does **not** use real 71120 white-box contribution, relevance, or calibration outputs.

Correct interpretation:

> This verifies that the trust-gated retrieval composition idea has an initial positive real 4B LLM inference result. It does not yet verify that replacing the lightweight learned proxy with the 71120 information-flow trust signal preserves the gain.

Recommended project phrasing:

> We validate a trust-gated retrieval agent architecture using a lightweight learned evidence proxy. This serves as a first end-to-end proof of concept for the broader chip-composed method. Replacing the proxy with 71120-style information-flow trust is the next validation step.
