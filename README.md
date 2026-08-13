# Chip Memory Engine

Chip Memory Engine is a runnable reference implementation for grounded retrieval over paper Chips. It keeps **one immutable JSON Chip per paper**, retrieves the paper's **L1 structures, L2 mechanisms, and L3 events**, projects a bounded connected subgraph for an agent role, checks negative and conditional evidence, and records runtime outcomes in a separate append-only log.

The package is an independent implementation. It learns from the useful lifecycle boundary in multi-agent memory systems—initialize, retrieve, observe, finish, and apply feedback—without copying G-Memory source code or replacing the Chip representation with G-Memory's interaction/query/insight hierarchy.

## What is included

- Compatibility loader for current classic and extended Chip variants.
- Immutable `ChipStore` with duplicate-ID rejection.
- Dependency-free BM25 candidate and item indexes.
- Staged L1/L2/L3 retrieval.
- One-hop expansion through shared graph participants.
- Deterministic planner, critic, executor, verifier, and researcher projections.
- Compatibility, condition, limitation, and negative-result findings.
- Source-file and evidence-anchor citations.
- G-Memory-like multi-agent lifecycle API.
- Append-only JSONL runtime events and smoothed feedback priors.
- CLI for validation, index construction, querying, demonstrations, and runtime audits.
- Illustrative classic and extended Chips.
- Automated tests.

## What it deliberately does not do

- It never writes into a paper Chip.
- It does not promote an agent observation into a paper claim.
- It does not claim that lexical retrieval is the final retrieval algorithm.
- It does not use an LLM to silently rewrite retrieved evidence.
- It does not copy G-Memory code.
- It does not claim superiority without controlled experiments.

## Installation

Python 3.10 or later is required. The runtime has no third-party dependencies.

```bash
git clone https://github.com/Yunbo-max/chip-memory-engine.git
cd chip-memory-engine
python3 -m pip install -e .
```

For development tests:

```bash
python3 -m pip install pytest
pytest
```

It can also run without installation:

```bash
PYTHONPATH=src python3 -m chip_memory --help
```

## Quick start with the bundled examples

Validate the Chips:

```bash
chip-memory validate --chips examples/chips
```

Retrieve a critic-specific subgraph:

```bash
chip-memory query \
  --chips examples/chips \
  --query "Choose a debate method under a one-round communication budget" \
  --role critic \
  --layers L1 L2 L3 \
  --token-budget 1600 \
  --format markdown
```

Build a normalized, inspectable snapshot:

```bash
chip-memory build-index \
  --chips examples/chips \
  --output outputs/index.json
```

Exercise the full runtime lifecycle:

```bash
chip-memory demo-lifecycle \
  --chips examples/chips \
  --query "Design a grounded planner-critic team" \
  --role planner \
  --runtime outputs/runtime

chip-memory runtime-summary --runtime outputs/runtime
```

## Query an external paper-Chip bank

The 502-paper corpus used for the smoke report is not duplicated in this repository. Point the engine at one or more existing Chip directories:

```bash
chip-memory query \
  --chips /path/to/iclr/chips \
  --chips /path/to/icml/chips \
  --chips /path/to/cvpr/chips \
  --query "multi-agent verification with negative evidence and limited communication" \
  --role verifier \
  --token-budget 4000 \
  --format markdown \
  --output chip_projection.md
```

## Python API

```python
from chip_memory import ChipMemoryEngine

engine = ChipMemoryEngine.from_paths(
    ["path/to/chips"],
    runtime_path="outputs/runtime",
)

engine.init_task_context(
    "Design a planner-critic system under a small communication budget",
    agent_roles=["planner", "critic", "verifier"],
)

result = engine.retrieve_memory(
    role="critic",
    layers=["L1", "L2", "L3"],
    candidate_limit=8,
    total_hit_limit=24,
    token_budget=3000,
)

for hit in result.projection.hits:
    print(hit.item.layer.value, hit.item.kind, hit.item.text)

engine.add_agent_node(
    "critic-1",
    "The retrieved L3 result conflicts with the available budget.",
    role="critic",
)
engine.move_memory_state(
    "reject incompatible mechanism",
    "selected a lower-cost method",
    reward=True,
)
engine.save_task_context(True, "Negative-evidence gate prevented bad transfer")
engine.backward(True)
```

## Retrieval pipeline

```text
task + role + memory budget
        |
        v
candidate paper ranking using all pin and graph text
        |
        v
L1/L2/L3 item scoring inside candidate Chips
        |
        v
one-hop participant-based graph expansion
        |
        v
role weighting and context-budget projection
        |
        v
compatibility + negative-evidence findings
        |
        v
grounded role-specific subgraph with citations
```

Candidate ranking may use all semantic pin text, but returned graph items preserve their actual layer:

- **L1:** nodes and `Structure` events.
- **L2:** typed binary edges.
- **L3:** reified events and result hyperedges.

## Runtime data separation

Paper truth and runtime experience have different authorities:

```text
chips/*.chip.json
    immutable, source-grounded paper knowledge

runtime/usage_events.jsonl
    context_started
    retrieval
    agent_event
    state_transition
    task_completed
    backward
```

Feedback priors are derived from completed contexts using Beta-smoothed success rates. They are deliberately bounded so repeated success can adjust ranking but cannot overwhelm lexical and structural relevance.

## Main modules

| Module | Responsibility |
|---|---|
| `loader.py` | Discover, validate, normalize, and preserve Chip variants |
| `types.py` | Stable typed objects and serialization |
| `text.py` | Tokenization, context estimation, and BM25 |
| `retriever.py` | Candidate ranking, layer scoring, and graph expansion |
| `projector.py` | Role-conditioned bounded subgraphs |
| `verifier.py` | Explicit negative, incompatible, and conditional evidence |
| `runtime.py` | Append-only observations and feedback priors |
| `engine.py` | High-level multi-agent lifecycle |
| `cli.py` | Reproducible command-line operations |

## Documentation map

- [Chips vs. G-Memory design](CHIPS_vs_GMemory_multiagent_design.md)
- [Architecture](docs/ARCHITECTURE.md)
- [G-Memory lessons and code mapping](docs/GMEMORY_LESSONS.md)
- [Schema and provenance](docs/SCHEMA_AND_PROVENANCE.md)
- [Chip Memory method specification](docs/CHIP_MEMORY_METHOD_SPEC.md)
- [Classic Chip schema v10](docs/CHIP_SCHEMA_V10.md)
- [Extended Chip schema reference](docs/EXTENDED_CHIP_SCHEMA_REFERENCE.md)
- [Multi-agent integration](docs/MULTIAGENT_INTEGRATION.md)
- [Evaluation plan](docs/EVALUATION_PLAN.md)
- [Full-corpus smoke report](docs/FULL_CORPUS_SMOKE_REPORT.md)
- [Limitations and roadmap](docs/LIMITATIONS_AND_ROADMAP.md)
- [Conversation record](docs/CONVERSATION.md)
- [Repository contents](docs/REPOSITORY_CONTENTS.md)
- [Third-party notes](THIRD_PARTY_NOTES.md)

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and research-integrity requirements.

## Status

This is a research-grade reference prototype. It is sufficient for deterministic experiments and agent-framework integration, but the publication claims in the evaluation document still need to be tested. See the roadmap before treating it as a production memory service.
