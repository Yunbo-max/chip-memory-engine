# EvidenceUseGate-v1 Conservative

Verdict: `FAIL`

- Runtime gate: multi-head learned controller
- Heads: stop probability, expected F1 if stop, continue value, noise risk, uncertainty
- Teacher: conservative answer sufficiency + counterfactual drop + attention/relevance alignment
- Hard negatives: high-attention wrong, lexical/relevance bait, later-answer cases
- Train/val/test examples: 4 / 2 / 3
- Train/val states: 20 / 7
- Stop-head val AUROC: 0.8333
- Noise-head val AUROC: 0.8333
- Stop threshold: 0.250
- Noise threshold: 0.050
- Uncertainty threshold: 0.300
- Baseline F1: 0.7879
- EvidenceUseGate-v1 F1: 0.7879
- F1 delta: 0.0000
- Steps: 5.0000 -> 5.0000
- Effort reduction: 0.00%
- Answer preservation rate: 100.00%
- Wrong-stop rate: 0.00%
- Peak CUDA allocation: 8414.1 MiB

This version tests whether a conservative learned evidence-use controller can preserve quality better than v0. It does not use contribution-flow at runtime.
