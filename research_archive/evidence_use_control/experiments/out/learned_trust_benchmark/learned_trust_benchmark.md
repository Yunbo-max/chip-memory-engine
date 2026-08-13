# Learned Trust Benchmark

Verdict: `PASS`

## Data

- Dataset: SQuAD2 validation
- Train examples: 307
- Validation examples: 102
- Test examples: 103
- Chunk labels learned from real answer spans in sentence chunks

## Learned Trust Model

- Model: logistic regression over retrieval/evidence features
- Chunk classifier AUROC: 0.8278
- Chunk classifier AUPRC: 0.7879
- Selected threshold: 0.900

## Validation Policy Metrics

- Baseline F1: 0.9902
- Learned trust F1: 0.9412
- Mean steps: 3.5000 -> 2.1863
- Effort reduction: 37.54%

## Held-Out Test Metrics

- Baseline F1: 0.9903
- Learned trust F1: 0.9515
- F1 delta: -0.0388
- Mean steps: 3.5437 -> 2.2233
- Effort reduction: 37.26%
- Answer preservation rate: 96.12%

## Interpretation

This learns the stop signal from real benchmark data. It is still a lightweight evidence proxy rather than the 71120 white-box information-flow signal, but unlike the earlier lexical threshold it is trained and tuned on real labels.
