# Trust-Gated Retrieval Agent Artifact Bundle

This bundle contains the real Qwen3-4B inference result, learned trust benchmark, calibrated attention-flow runs, 71120-style contribution-flow pilot, EvidenceUseGate learned-controller runs, verification files, raw per-example outputs, scripts, and environment info.

Verified method: **Trust-Gated Retrieval Agent with Learned Evidence Proxy**.

Best 100-example practical result: calibrated Qwen attention-flow, baseline F1 0.7830, method F1 0.7825, retrieval steps 3.3700 -> 1.1100, effort reduction 67.06%.

Latest clean learned-gate result: EvidenceUseGate-v4 clean validation, baseline F1 0.7590, method F1 0.7820, retrieval steps 3.8800 -> 2.1700, effort reduction 44.07%, wrong-stop rate 7.00%. This passes the F1 and effort targets but misses the strict safety endpoint of <=5% wrong-stop rate.

Learned-gate diagnostic: v1 fixed safety but saved only 14.43%; v2 reached 27.84% effort with 5% wrong-stop; v3 reached 49.23% effort but wrong-stop rose to 11%; post-hoc v4 selectively falls back from v3 to v2 and reaches 41.24% effort with 5% wrong-stop; clean v4 reaches 44.07% effort with 7% wrong-stop. Calibrated Qwen attention-flow remains the best clean practical method at 67.06% effort reduction with almost unchanged F1, while clean v4 is the strongest learned controller so far but not the final endpoint.

Scope note: this bundle does not verify the 71120 paper method itself. The current gate is a lightweight learned evidence proxy. The target full method is **Trust-Gated Retrieval Agent with Information-Flow Trust Signal**, where the proxy is replaced by 71120-style white-box contribution/relevance/calibration.

Start with:

- `experiments/out/real_llm_trust_case/real_llm_trust_case.md`
- `experiments/out/unified_100_example_comparison.md`
- `experiments/out/evidence_use_gate_v4_clean_validation/evidence_use_gate_v4_clean_validation_100.md`
- `experiments/out/evidence_use_gate_v4_selective_fallback/evidence_use_gate_v4_selective_fallback_100.md`
- `experiments/out/evidence_use_gate_v3_guarded_distill/evidence_use_gate_v3_guarded_distill_100.md`
- `experiments/out/evidence_use_gate_v2_pareto/evidence_use_gate_v2_pareto_100.md`
- `experiments/out/evidence_use_gate_v1_conservative/evidence_use_gate_v1_conservative_100.md`
- `experiments/out/evidence_use_gate_v2_pareto_roadmap.md`
- `experiments/out/verification_bundle/per_example_results.jsonl`
- `experiments/out/verification_bundle/recompute_metrics.py`
