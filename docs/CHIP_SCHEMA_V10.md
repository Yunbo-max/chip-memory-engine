# Chip Schema v10 — Final

A "chip" is a self-contained knowledge unit for one paper. The schema is identical for static chips (papers, theorems) and episodic chips (agent memories) — only the distribution of L3 event kinds differs.

---

## Top-level chip structure

```jsonc
{
  "chip_id": "ICLR2026_{openreview_id}_{slug}",
  "chip_type": "static-paper",            // or "episodic", "theorem", "codebase-module"
  "chip_metadata": {
    "created_at":    "<ISO timestamp>",
    "version":       1,
    "supersedes":    [],                   // chips this one supersedes
    "superseded_by": null,
    "source":        "ICLR 2026 oral",
    "paper":         "openreview-id"
  },

  // 4 first-class entities (always present, depth varies)
  "method":      { ...inner_kg... },       // paper's introduced method
  "gap":         { ...inner_kg... },       // gaps the paper addresses
  "evaluation":  { ...inner_kg... },       // eval procedure(s) used
  "result":      { ...result hyperedges... },  // bridge: methods × evaluation

  // Cross-chip references (Bus)
  "footprint": {
    "primitives":   [...refs to bus],
    "scenarios":    [...refs to bus],
    "failure_modes":[...refs to bus],
    "datasets":     [...refs to bus],
    "external_methods": [...refs to chips/bus],
    "baselines":    [...refs to chips/bus]
  }
}
```

---

## The 4 entities each have a common inner-KG sub-schema

```jsonc
"<entity>": {
  "id": "...",
  "label": "...",
  "kind": "Method|Gap|Evaluation|Result",
  "nodes": [
    { "id": "...", "kind": "...", "label": "...", "props": {...} }
  ],
  "edges": [
    { "src": "...", "dst": "...", "rel": "...", "layer": "L2" }
  ],
  "events": [
    { "id": "...", "kind": "Structure|Training|Forward|GapBinding|EvalPipeline|Temporal|...", ... }
  ]
}
```

---

## Node kinds (used inside any entity's inner KG)

### Universal / leaf (mostly come from / link to Bus)
- `Primitive` — reusable named building block (Transformer, GRPO, LLM-as-Judge, Exact-Match, …)
- `Hyperparameter` — a specific value (n_layers=12, lr=1e-4, T=1000)
- `Choice` — a categorical decision point (noise_schedule ∈ {linear, cosine})
- `Dataset` — named benchmark/dataset
- `Scenario` — a task/context where a method runs or fails (multi-hop QA, long-context)
- `FailureMode` — typed failure (slow-inference, insufficient-recall, noise-sensitive)
- `Significance` — qualitative result label (SOTA, competitive, marginal, negative)
- `ResultType` — taxonomy of result (absolute-score, comparative, scaling-law, ablation, efficiency)

### Method-internal
- `Module` — paper-named functional part (Embedder, Encoder, Critic)
- `Input` / `Output` — data endpoints
- `Objective` — loss / reward / training target

### Gap-internal
- `Gap` (the gap node itself; pairs with one GapBinding event)

### Evaluation-internal
- (re-uses `Module` for pipeline stages)

---

## Event kinds (L3 reified hyperedges — multi-way bindings)

| Event kind | Used in | What it binds |
|---|---|---|
| `Structure` | Method, Gap, Evaluation | `{parent, children: [...]}` — containment (replaces L1 edges) |
| `Training` | Method | `{loss, updates: [...], is_joint}` — joint optimization |
| `Forward` | Method | `{consumes: [...], produces: [...], step}` — multi-input forward step |
| `Gating` | Method | `{router, routes_to: [...], condition}` — MoE / conditional compute |
| `GapBinding` | Gap | `{affected_subject, condition, failure_mode, evidence_datasets, strength}` |
| `EvalPipeline` | Evaluation | `{dataset, judge, metric, aggregator, hyperparams}` |
| `Temporal` ⭐ | Any entity (mostly episodic) | `{fact, valid_from, valid_to, event_time, invalidated_by, confidence}` |

### TemporalEvent spec

```jsonc
{
  "id": "te_<...>",
  "kind": "Temporal",
  "fact": {
    "subject":   "<node-id or event-id>",
    "predicate": "<verb>",
    "object":    "<node-id or value>"
  },
  "valid_from":     "<ISO>",        // when the fact starts being true
  "valid_to":       "<ISO|null>",   // null = currently valid
  "event_time":     "<ISO|null>",   // when the real-world event occurred (vs when observed)
  "invalidated_by": "<te-id|null>", // pointer to the TemporalEvent that succeeded this one
  "confidence":     0.0-1.0
}
```

For static paper chips, `Temporal` events are rare (a paper claim is timeless within its scope). They appear when:
- A paper's claim is explicitly conditioned on a model version ("on GPT-4-0613, …")
- A paper's data is time-sliced ("during 2023-Q3 markets, …")
- Episodic memory in agent settings

---

## Result entity specifics (the bridge)

A `Result` is uniquely structured: it's mostly a hyperedge entity that connects N Methods × 1 Evaluation. Internal nodes are minimal.

```jsonc
"result": {
  "results": [
    {
      "id": "r1",
      "kind": "Result",
      "methods":     ["Q-RAG", "RMT", "Titans"],   // 1 = absolute, ≥2 = comparison
      "evaluation":  "long-context-accuracy-eval",
      "scope":       "10M tokens",
      "outcome":     "Q-RAG outperforms",
      "significance": "Significance:SOTA",
      "result_type":  "ResultType:Absolute-SOTA",
      "numerical":    "...",                       // verbatim numbers if abstract reports
      "props":        { ... }
    }
  ]
}
```

---

## Bus (shared across all chips)

```
/Users/yunbo/Documents/future/memory_bank/
├── bus/
│   ├── primitives.json
│   ├── scenarios.json
│   ├── failure_modes.json
│   ├── datasets.json
│   ├── significance.json
│   ├── result_types.json
│   └── temporal_index.json     # cross-chip temporal index (mostly empty for static chips)
├── chips/
│   └── {chip_id}.json
└── memory_bank.graph.json      # consolidated full graph
```
