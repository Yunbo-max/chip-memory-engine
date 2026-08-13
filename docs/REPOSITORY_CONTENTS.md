# Repository contents

## Executable package

| Path | Purpose |
|---|---|
| `src/chip_memory/loader.py` | Load and normalize classic and extended Chip variants |
| `src/chip_memory/retriever.py` | Candidate ranking, L1/L2/L3 retrieval, and graph expansion |
| `src/chip_memory/projector.py` | Role-conditioned selection under a context budget |
| `src/chip_memory/verifier.py` | Negative, conditional, and incompatibility findings |
| `src/chip_memory/runtime.py` | Append-only runtime events and bounded feedback priors |
| `src/chip_memory/engine.py` | High-level multi-agent memory lifecycle |
| `src/chip_memory/cli.py` | Validation, indexing, querying, and lifecycle commands |

## Data contracts

| Path | Purpose |
|---|---|
| `schema/chip.schema.json` | Permissive compatibility contract validated on the 502-Chip corpus |
| `schema/multiagent_ontology.yaml` | Proposed multi-agent node, relation, and event vocabulary |
| `docs/CHIP_SCHEMA_V10.md` | Classic four-pin Chip schema reference |
| `docs/EXTENDED_CHIP_SCHEMA_REFERENCE.md` | Extended action/research-memory schema reference |
| `docs/CHIP_MEMORY_METHOD_SPEC.md` | Original portable Chip Memory method description |

## Examples and tests

| Path | Purpose |
|---|---|
| `examples/chips/` | Synthetic classic and extended Chips |
| `examples/demo.py` | Complete retrieval and lifecycle example |
| `tests/` | Loader, retrieval, runtime, and CLI regression tests |
| `tools/verify_experience_archive.py` | Standard-library recomputation of key archived results |
| `.github/workflows/tests.yml` | Python 3.10–3.12 continuous integration |

## Design and research documentation

| Path | Purpose |
|---|---|
| `CHIPS_vs_GMemory_multiagent_design.md` | Detailed architectural comparison and proposed research method |
| `docs/ARCHITECTURE.md` | Implemented system architecture |
| `docs/GMEMORY_LESSONS.md` | Code-level lessons from the official G-Memory implementation |
| `docs/MULTIAGENT_INTEGRATION.md` | Framework-neutral agent integration |
| `docs/SCHEMA_AND_PROVENANCE.md` | Source/runtime authority and schema policy |
| `docs/EVALUATION_PLAN.md` | Baselines, metrics, ablations, and claims to test |
| `docs/EXPERIMENTAL_EXPERIENCE.md` | Detailed successes, failures, metrics, and lessons from repeated experiments |
| `docs/FAILURE_AND_DECISION_LOG.md` | Traceable failure-to-design decision ledger |
| `docs/FULL_CORPUS_SMOKE_REPORT.md` | Results from 502 real paper Chips |
| `docs/LIMITATIONS_AND_ROADMAP.md` | Known limitations and future work |
| `docs/CONVERSATION.md` | Conversation record that led to the design and implementation |
| `THIRD_PARTY_NOTES.md` | Upstream sources and independent-implementation boundary |

## Historical research archive

| Path | Purpose |
|---|---|
| `research_archive/evidence_use_control/` | Preserved v0-v4 runners, reports, commands, and raw outputs |
| `research_archive/followups/v5_safety_first/` | Shifted-split safety-first follow-up |
| `research_archive/followups/noise_robustness/` | Failed two-example lexical-noise smoke diagnostic |
| `research_archive/REPRODUCIBILITY.md` | Saved-output recomputation and full-run dependency boundary |

The archive is adjacent evidence for controller design. It is not a direct benchmark of the Chip Memory engine.

## Excluded material

The repository does not duplicate the 502-paper Chip corpus, paper PDFs, model weights, external repositories, live runtime logs, or generated indexes. Historical experiment outputs are included in `research_archive/` for traceability; their underlying dataset and external-code licenses must be reviewed before public redistribution.
