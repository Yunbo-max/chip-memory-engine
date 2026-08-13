# EvidenceUseGate-v1 Conservative

Verdict: `FAIL`

- Runtime gate: multi-head learned controller
- Heads: stop probability, expected F1 if stop, continue value, noise risk, uncertainty
- Teacher: conservative answer sufficiency + counterfactual drop + attention/relevance alignment
- Hard negatives: high-attention wrong, lexical/relevance bait, later-answer cases
- Train/val/test examples: 96 / 40 / 100
- Train/val states: 355 / 160
- Stop-head val AUROC: 0.5563
- Noise-head val AUROC: 0.6530
- Stop threshold: 0.250
- Noise threshold: 0.150
- Uncertainty threshold: 0.400
- Baseline F1: 0.7590
- EvidenceUseGate-v1 F1: 0.7675
- F1 delta: 0.0085
- Steps: 3.8800 -> 3.3200
- Effort reduction: 14.43%
- Answer preservation rate: 100.00%
- Wrong-stop rate: 0.00%
- Peak CUDA allocation: 8414.1 MiB

This version tests whether a conservative learned evidence-use controller can preserve quality better than v0. It does not use contribution-flow at runtime.
