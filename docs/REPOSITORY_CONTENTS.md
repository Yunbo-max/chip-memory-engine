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
| `docs/FULL_CORPUS_SMOKE_REPORT.md` | Results from 502 real paper Chips |
| `docs/LIMITATIONS_AND_ROADMAP.md` | Known limitations and future work |
| `docs/CONVERSATION.md` | Conversation record that led to the design and implementation |
| `THIRD_PARTY_NOTES.md` | Upstream sources and independent-implementation boundary |

## Excluded material

The repository does not duplicate the 502-paper Chip corpus, paper PDFs, external repositories, runtime logs, or generated indexes. Those assets are large, may have separate licenses, and are supplied as external paths when running the engine.

