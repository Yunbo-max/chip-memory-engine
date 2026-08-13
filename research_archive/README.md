# Research archive

## Why this is in the repository

This directory preserves the experiment trail that informed the Chip Memory verifier, evidence-use controller, and selective-fallback roadmap. It includes failures and near misses, not only the strongest result.

The evidence-use experiments are **adjacent design evidence**. They evaluate retrieval stopping and trust on SQuAD2; they do not directly evaluate the Chip Memory engine, the L1/L2/L3 representation, multi-agent task success, or a controlled comparison with G-Memory.

## Contents

| Path | Contents |
|---|---|
| `evidence_use_control/` | Historical artifact bundle: experiment runners, Markdown reports, JSON/JSONL outputs, commands, and recomputation scripts through clean v4 |
| `followups/v5_safety_first/` | Shifted-split safety-first outputs and recomputation files |
| `followups/noise_robustness/` | Two-example lexical-distractor smoke output and recomputation files |
| `followups/scripts/` | Historical v5 and noise runner scripts |
| `BIG_CHIP_SYNTHESIS.md` | Broader hypothesis synthesis that motivated an evidence-controlled agent direction |
| `REPRODUCIBILITY.md` | What can be recomputed now and what external assets are still required |

Nested ZIP files from the old artifact bundle are intentionally excluded because their extracted contents are already present.

## Preservation policy

- Historical reports and raw outputs are copied without rewriting their verdicts.
- Environment-specific commands and paths are retained as provenance.
- Later documents may correct the interpretation, but should not silently edit an old result.
- The main normalized interpretation lives in [`docs/EXPERIMENTAL_EXPERIENCE.md`](../docs/EXPERIMENTAL_EXPERIENCE.md).
- Known failures and resulting decisions live in [`docs/FAILURE_AND_DECISION_LOG.md`](../docs/FAILURE_AND_DECISION_LOG.md).

## Important naming note

The shifted safety-first artifact was run through the v4 clean-validation implementation and its generated report still begins with “EvidenceUseGate-v4 Clean Validation.” The follow-up README identifies it as the v5 safety-first shifted-split experiment. The archive preserves both facts rather than rewriting the generated file.

## Licensing and data note

The archive contains derived benchmark examples and model outputs from the local research workspace. It does not include model weights, the external information-flow repository, or the complete paper-Chip corpus. Before making this repository public or redistributing raw benchmark rows, review the licenses and terms of the underlying datasets and external code.
