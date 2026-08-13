# Qwen Calibrated Flow Trust Case

Verdict: `PASS`

- Validation examples for calibration: 24
- Held-out generation examples: 100
- Alpha internal-flow weight: 0.60
- Trust threshold: 0.510
- Baseline F1: 0.7830
- Calibrated-flow F1: 0.7825
- F1 delta: -0.0005
- Steps: 3.3700 -> 1.1100
- Effort reduction: 67.06%
- Answer preservation rate: 93.00%
- Peak CUDA allocation: 8414.1 MiB

This upgrades the gate from TF-IDF-only learned proxy to a calibrated blend of Qwen internal attention-flow and retrieval relevance. It is still a bridge toward, not an exact reproduction of, 71120 information-flow contribution/relevance calibration.
