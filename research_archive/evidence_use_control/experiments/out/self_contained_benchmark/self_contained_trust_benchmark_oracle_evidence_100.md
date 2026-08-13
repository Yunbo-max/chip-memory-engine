# Self-Contained Trust Benchmark

Verdict: `PASS`

## Setup

- Dataset: SQuAD2 validation
- Examples: 100
- Baseline: inspect fixed top-5 retrieved chunks
- Trust policy: oracle_evidence
- Trust-gated: stop when trust >= 0.62 for lexical policy, or when evidence is found for oracle policy

## Metrics

- Baseline mean token F1: 0.9900
- Trust-gated mean token F1: 1.0000
- Token F1 delta: 0.0100
- Baseline mean steps: 3.8500
- Trust-gated mean steps: 1.3300
- Mean effort reduction: 2.5200
- Relative effort reduction: 65.45%
- Answer preservation rate: 100.00%

## Interpretation

This evaluates the stop/continue policy itself on real benchmark examples. The oracle_evidence policy is an upper bound for a perfect evidence/trust detector; lexical_trust is the cheap deployable proxy. Neither claims to reproduce 71120 white-box transformer attribution.
