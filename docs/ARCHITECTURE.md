# Architecture

## 1. Design objective

The engine makes paper Chips operational for retrieval and multi-agent use while preserving their defining representation:

- one source paper corresponds to one immutable Chip;
- pins expose stable semantic regions of the paper;
- L1 represents containment and components;
- L2 represents typed binary mechanisms;
- L3 represents multi-party events and results;
- every runtime recommendation remains traceable to its source Chip.

The engine does not turn paper Chips into mutable agent memories. Runtime experience is stored beside the paper bank and references the Chip items that were used.

## 2. System boundaries

```text
                        SOURCE-GROUNDED ZONE
 paper PDFs/repos ──> paper Chips ──> normalized read-only views
                              |                 |
                              |                 v
                              |          lexical + graph index
                              |                 |
                              +-----------------+
                                                |
                                                v
 task ──> candidate Chips ──> L1/L2/L3 hits ──> verifier
                                                |
                                                v
                                      role-specific projection
                                                |
                        RUNTIME ZONE            v
 agents <──────────────────────────── grounded subgraph
   |
   v
append-only events ──> outcome-derived bounded feedback priors
```

The arrow from runtime feedback returns to ranking only. It never points into the source paper files.

## 3. Loading and normalization

`ChipLoader` supports two families observed in the repository:

1. Classic graph containers using `method`, `gap`, `evaluation`, and `result`.
2. Extended paper pins such as `problem_gap`, `method_mechanism`, `evaluation_validation`, `experimental_setting`, `result_outcome`, `implementation`, and `reuse_transfer`, often with top-level graph arrays.

The loader recursively discovers `nodes`, `edges`, and `events` arrays. It creates a derived `KnowledgeItem` for each graph object:

| Source object | Runtime layer |
|---|---|
| Node | L1 |
| `Structure` event | L1 |
| Edge | L2 |
| Other event | L3 |
| Result hyperedge | L3 |

Pin contents contribute to paper-level candidate ranking, but they are not mislabeled as L1/L2/L3 graph objects.

No normalization function mutates the parsed `raw` mapping or writes to the source path.

## 4. Candidate ranking

The current reference backend uses dependency-free BM25. Each paper-level document combines:

- title and metadata;
- semantic pin contents;
- footprint;
- normalized graph-item text.

The candidate stage is deliberately broad. Its purpose is to identify a small paper set before more expensive graph-level scoring.

The backend interface can later be replaced by dense or hybrid retrieval without changing `ChipRecord`, `KnowledgeItem`, or the lifecycle API.

## 5. Layer scoring

Within candidate papers, the engine scores only items from the requested layers. The default is all three.

Per-layer limits prevent a paper with many nodes from crowding out events. A total-hit limit caps the intermediate result before context projection.

The reference score is:

```text
item score = lexical relevance
           + bounded runtime feedback
           + role preference
           + optional graph-expansion bonus
```

The output exposes every score component so an experiment can audit why an item ranked highly.

## 6. Graph expansion

Lexical matching produces seed items. The engine then expands one hop through shared graph references inside the same paper.

Example:

```text
query matches L1 node "Critic"
    |
    +── L2 edge: Critic CRITIQUES Proposal
    |
    +── L3 event: Debate{critic, proposal, evidence, outcome}
```

This recovers operational context that may not repeat the words in the task query. Expansion never crosses papers through an unverified string equality. Cross-paper reasoning should use canonical bus references or explicitly typed cross-Chip links.

## 7. Role projection

The projector is deterministic. It does not ask an LLM to paraphrase evidence. It combines preferences over pins, layers, and role keywords:

- planner: gaps, structure, dependencies, and transfer constraints;
- critic: failures, risks, contradictions, costs, and negative results;
- executor: method flow, training, code, settings, and procedures;
- verifier: evaluation, metrics, baselines, results, and evidence;
- researcher: balanced coverage.

The token budget applies to injected memory context, excluding the task text already held by the caller. The projector attempts to include one item per requested layer before filling remaining capacity by score.

## 8. Verification

The verifier is intentionally conservative. It recognizes explicit phrases indicating:

- incompatibility or failure;
- negative results, limitations, risk, degradation, or added cost;
- requirements, assumptions, conditions, and budgets;
- missing event-level evidence anchors.

Findings are `block`, `warning`, or `info`. A block is not a universal logical proof that transfer is impossible; it means retrieved source content explicitly describes an incompatible or failed condition and the downstream agent must resolve it.

## 9. Runtime lifecycle

The high-level lifecycle mirrors useful integration points common to multi-agent memory systems:

```python
init_task_context(...)
retrieve_memory(...)
add_agent_node(...)
move_memory_state(...)
save_task_context(...)
backward(...)
```

The similarity is at the interface level. The underlying content remains Chip-native and source-grounded.

## 10. Feedback

Runtime events associate retrieved item IDs and paper IDs with completed task outcomes. The derived prior is:

```text
prior = (successes + 1) / (trials + 2) - 0.5
```

This Beta(1,1)-smoothed value is bounded between approximately `-0.5` and `+0.5`. Additional coefficients in retrieval keep it subordinate to task relevance.

This is usage adaptation, not truth revision. A successful execution means an item was useful in one context; it does not establish the paper claim beyond its source evidence.

## 11. Persistence

The reference system uses:

- source `.chip.json` files as the authority;
- optional normalized JSON snapshots as rebuildable artifacts;
- JSONL as the append-only runtime authority;
- in-memory BM25 and graph-reference indexes as rebuildable state.

A service deployment could replace in-memory indexing with SQLite, PostgreSQL, a graph database, or a vector store while retaining the same provenance boundary.

