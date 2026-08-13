# Self-Contained Trust Benchmark

Verdict: `FAIL`

## Setup

- Dataset: SQuAD2 validation
- Examples: 58
- Baseline: inspect fixed top-5 retrieved chunks
- Trust policy: lexical_trust
- Trust-gated: stop when trust >= 0.62 for lexical policy, or when evidence is found for oracle policy

## Metrics

- Baseline mean token F1: 1.0000
- Trust-gated mean token F1: 0.8621
- Token F1 delta: -0.1379
- Baseline mean steps: 4.2414
- Trust-gated mean steps: 1.0172
- Mean effort reduction: 3.2241
- Relative effort reduction: 76.02%
- Answer preservation rate: 86.21%

## Interpretation

This evaluates the stop/continue policy itself on real benchmark examples. The oracle_evidence policy is an upper bound for a perfect evidence/trust detector; lexical_trust is the cheap deployable proxy. Neither claims to reproduce 71120 white-box transformer attribution.
