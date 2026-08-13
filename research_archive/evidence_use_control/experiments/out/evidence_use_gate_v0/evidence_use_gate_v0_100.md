# EvidenceUseGate-v0

Verdict: `FAIL`

- Model: Qwen3-4B-Instruct
- Runtime gate: learned gradient_boosting over cheap attention/retrieval state features
- Teacher: answer sufficiency + counterfactual drop probes
- Train/val/test examples: 80 / 32 / 100
- Train/val states: 312 / 107
- Teacher val AUROC: 0.6181
- Teacher val AUPRC: 0.7177
- Selected stop threshold: 0.660
- Baseline F1: 0.7786
- EvidenceUseGate F1: 0.7326
- F1 delta: -0.0460
- Steps: 3.8000 -> 1.5000
- Effort reduction: 60.53%
- Answer preservation rate: 88.00%
- Peak CUDA allocation: 8414.1 MiB

This is not a hand-written attention + relevance rule. The gate learns stop probability from runtime state features, with labels produced by expensive teacher probes. Contribution-flow remains a separate mechanistic teacher candidate because the available 71120 implementation is LLaMA/Gemma-specific.
