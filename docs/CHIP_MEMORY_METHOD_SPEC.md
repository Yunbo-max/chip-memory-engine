# Chip-Memory: A Modular Knowledge-Representation Architecture

> **Portable spec.** This document is self-contained and tool-agnostic — any LLM (Claude, ChatGPT, Gemini) can read it and apply the methodology. Claude-Code-specific execution notes are isolated in §10.

---

## 0. One-sentence summary

Represent any body of knowledge (a paper, a theorem, an agent memory, a codebase module) as a **chip**: a self-contained unit with a standardized 4-pin interface, an internal multi-layer graph, that connects to other chips through a shared **bus** of reusable primitives — exactly like integrated-circuit design.

---

## 1. The metaphor: knowledge as chip design

| Chip design | Chip-Memory |
|---|---|
| Transistor | a single graph node (one entity instance) |
| Triple (subject, predicate, object) | the atomic unit of knowledge — everything reduces to this |
| Standard cell / logic gate | a **Primitive** (Transformer, GRPO, LLM-as-Judge, Exact-Match…) — reused across chips |
| IP block / macro | one of the 4 first-class **entities** (Method/Gap/Evaluation/Result) with its inner graph |
| Chip / chiplet | one **knowledge unit** = one paper/theorem/episode (4 entities + footprint) |
| Bus / interconnect | the **shared leaf nodes** every chip references |
| Die / system-on-chip | the **memory bank** = all chips wired through the bus |
| Reuse rate of a cell | in-degree of a Primitive node = how many chips use it (a research-intelligence signal) |
| Tape-out | the consolidated `memory_bank.graph.json` |

**Key property — uniformity:** every chip has the *same* internal structure regardless of domain. A physics paper, an RL paper, and an agent memory all use identical machinery. This is what distinguishes chip-memory from heterogeneous memory systems (which use different schemas for different memory types).

---

## 2. The 4 first-class entities (the "pins")

Every chip exposes exactly four standardized entities. Each is itself a small graph (an "inner KG").

| Entity | Captures | Analogy |
|---|---|---|
| **Method** | the named methodology + its internal architecture | "how it works" |
| **Gap** | the limitations in prior work that motivate the chip | "why it exists" |
| **Evaluation** | the procedure used to measure (pipeline, judge, metric) | "how it's measured" |
| **Result** | findings that bridge N Methods × 1 Evaluation | "what was found" |

**Result is special — it is a hyperedge, not a container.** It binds a set of methods to one evaluation:
- `methods = [X]` → an absolute result ("X achieves SOTA on D")
- `methods = [X, Y, Z]` → a comparison ("X beats Y, Z on D")

---

## 3. Inner vs Outer

- **Inner** = the structure *inside* one entity of one chip (its components, data flow, equations). Private to the chip.
- **Outer** = nodes *shared across chips* — the bus. When two chips both use "Transformer", they point to the **same** Primitive node.

This separation is what makes the graph navigable at conference scale: you can ask "which chips contain a Transformer?" by looking at one bus node's in-edges, instead of string-matching across thousands of chips.

---

## 4. The three layers (L1 / L2 / L3)

Within any entity's inner KG, relations fall into three layers. **They are orthogonal — each expresses something the others cannot.**

| Layer | Expresses | Structure | Example |
|---|---|---|---|
| **L1** | containment ("X is part of Y") | hierarchy / tree | "Value Head is part of RL Agent" |
| **L2** | data/causal flow ("X feeds Y") | directed graph (DAG or cyclic) | "Embedder feeds latent to RL Agent" |
| **L3** | multi-way binding ("loss jointly updates A, B, C") | reified hyperedge | "TD-loss updates {Embedder, Value-Head}" |

**Crucial design decision:** L1 (containment) is implemented *via L3*, not via separate edges. A containment relation is just a `Structure` event: `{parent, children: [...]}`. This collapses the whole system to **one mechanism** (see §5).

**Why L3 (reification):** binary edges lose binding. "loss → A" and "loss → B" as two edges don't say whether it's one joint update or two independent ones. An L3 event node binds all participants into one fact, preserving that information for later querying.

---

## 5. The unified mechanism — everything is one of three things

A chip's inner KG (for each entity) is exactly:

```
nodes  : the entities that exist
edges  : binary relations (L2 data/causal flow), src → dst with a `rel` label
events : reified hyperedges (L3) — multi-way bindings, including L1 containment
```

That's it. No special "containment edges", no nested JSON, no separate metric nodes. L1 = `Structure` events; L2 = edges; L3 = all events. Temporal, gating, training, evaluation-pipeline — all are event *kinds*.

### Node kinds

**Reusable / bus (shared across chips):**
- `Primitive` — building block (Transformer, GRPO, UNet, LLM-as-Judge, Exact-Match…)
- `Hyperparameter` — a specific value (n_layers=12, lr=1e-4, T=1000)
- `Choice` — a categorical decision point (noise_schedule ∈ {linear, cosine})
- `Dataset` — named benchmark/dataset
- `Scenario` — a task/context (multi-hop QA, long-context)
- `FailureMode` — a typed failure (slow-inference, insufficient-recall)
- `Significance` — result label (SOTA, competitive, marginal, negative)
- `ResultType` — result taxonomy (absolute, comparative, scaling, ablation, efficiency)

**Method-internal:** `Module` (paper-named part), `Input`, `Output`, `Objective`
**Gap-internal:** `Gap` (each pairs with one `GapBinding` event)

### Event kinds (L3 reified hyperedges)

| Kind | Used in | Binds |
|---|---|---|
| `Structure` | any entity | `{parent, children:[...]}` — containment (replaces L1 edges) |
| `Training` | Method | `{loss, updates:[...], is_joint}` — joint optimization |
| `Forward` | Method | `{consumes:[...], produces:[...], step}` — multi-input forward |
| `Gating` | Method | `{router, routes_to:[...], condition}` — conditional compute |
| `GapBinding` | Gap | `{affected_subject, condition, failure_mode, evidence_datasets, strength}` |
| `EvalPipeline` | Evaluation | `{dataset, judge, metric, aggregator, hyperparams}` |
| `Temporal` | any (mostly episodic) | `{fact, valid_from, valid_to, event_time, invalidated_by, confidence}` |

### Temporal handling

Time is **not a special layer** — it is one event kind. Static chips (papers, theorems) rarely use `Temporal`. Episodic chips (agent memory) use it heavily. Bi-temporal model: transaction time (chip metadata `created_at`), validity time (`valid_from`/`valid_to`), event time (`event_time`). This is how chip-memory subsumes temporal-KG systems (Zep/Graphiti) without making everything time-stamped.

---

## 6. The chip JSON schema (v10)

```jsonc
{
  "chip_id": "<unique id>",
  "chip_type": "static-paper",          // or episodic | theorem | codebase-module
  "chip_metadata": {
    "created_at": "<ISO>", "version": 1,
    "supersedes": [], "superseded_by": null,
    "source": "...", "extraction_depth": "full-text|abstract"
  },

  // each of the 4 entities has: nodes[] + edges[] + events[]
  "method":     { "id","label","kind":"Method","nodes":[…],"edges":[…],"events":[…] },
  "gap":        { "id","kind":"Gap","nodes":[…],"edges":[…],"events":[…] },
  "evaluation": { "id","kind":"Evaluation","nodes":[…],"edges":[…],"events":[…] },
  "result":     { "results":[ {…hyperedge…} ] },

  // bus references — everything reusable this chip touches
  "footprint": {
    "primitives":[…], "scenarios":[…], "failure_modes":[…],
    "datasets":[…], "external_methods":[…], "baselines":[…]
  }
}
```

### Node format
```jsonc
{"id":"…", "kind":"Module|Primitive|Hyperparameter|Choice|Input|Output|Objective|Gap|Scenario|FailureMode|…",
 "label":"…", "parent":"<id|null, optional>", "props":{…}}
```

### Edge format (L2)
```jsonc
{"src":"…", "dst":"…", "rel":"feeds|implemented_by|modifies|parameterizes|causes|alternative_to|…", "layer":"L2"}
```

### Event format (L3)
```jsonc
{"id":"…", "kind":"Structure|Training|Forward|Gating|GapBinding|EvalPipeline|Temporal", …kind-specific fields…}
```

### Result hyperedge format
```jsonc
{"id":"r1", "kind":"Result",
 "methods":["<this-method>","<baseline-1>",…],   // 1=absolute, ≥2=comparison
 "evaluation":"<eval-id>",
 "scope":"…", "outcome":"…",
 "significance":"Significance:SOTA|Competitive|Marginal|Negative",
 "result_type":"ResultType:Absolute|Comparative|Scaling|Ablation|Efficiency",
 "numerical":"<verbatim numbers if reported>"}
```

---

## 7. Extraction procedure (paper → chip)

Given a source document, produce one chip:

1. **Method** — find the named method. List 5-50 Module nodes (named components). For each, if the underlying technique is named, add a `Primitive` node + `implemented_by` edge. Add L2 edges for data flow. Add one `Structure` event listing all top-level modules. Add `Training`/`Forward`/`Gating` events only if the text is explicit about joint training / multi-input / routing. Add `Hyperparameter` nodes with values from the experiments/appendix.
2. **Gap** — find 2-8 gaps about prior work. For each, a `GapBinding` event with affected_subject + condition (`Scenario`) + failure_mode (`FailureMode`) + evidence_datasets. Add L2 `causes`/`alternative_to` edges between gaps. Add a `Structure` event if some gaps are sub-gaps of a meta-gap.
3. **Evaluation** — one `EvalPipeline` event per dataset/protocol (dataset + judge + metric + aggregator + hyperparams). If the chip *proposes* a novel evaluation, expand its inner KG with `Module` pipeline stages.
4. **Result** — one `Result` hyperedge per headline finding. `methods` = the chip's method + baselines. `numerical` = verbatim numbers from tables.
5. **Footprint** — list every reusable leaf-node name (canonical capitalization). These feed the bus.

**Rules:** extract VERBATIM; never fabricate names/numbers; leave fields empty if absent; depth scales with extraction source (abstract → shallow chip; full-text → deep chip; the schema is identical).

---

## 8. Bus aggregation (chips → shared graph)

After all chips exist:

1. Collect every `footprint.*` name across all chips.
2. **Canonicalize** each leaf type: normalize strings, embed, cluster at similarity ≥ 0.92, pick most-frequent variant as canonical, LLM-verify ambiguous clusters. (e.g., "DPO" = "Direct Preference Optimization".)
3. Write `bus/{primitives,scenarios,failure_modes,datasets,significance,result_types}.json`.
4. **Special merge:** if an inner `Primitive` is itself a chip's `Method` (e.g., "Transformer" used as a building block), link them — this enables "which chips build on method X?" queries.
5. **Do NOT** auto-merge inner `Component` nodes across chips (Q-RAG's Embedder ≠ another paper's Embedder — different instances).

---

## 9. Memory-bank assembly + the queries it unlocks

Build `memory_bank.graph.json`: all chip nodes + all bus nodes + edges/events, with chips linked to bus nodes via footprint references.

**Footprint-based retrieval:** each chip's footprint is its "signature". Retrieval = footprint intersection (find chips whose bus-references overlap), which is symbolic and explainable — unlike vector-similarity retrieval.

**Queries that flat KGs / PapersWithCode cannot answer:**
- "Which chips have a `Training` event with `is_joint=true` (joint multi-module training)?" — needs L3.
- "Which Primitives are used in ≥5 chips as building blocks?" — reuse-rate signal.
- "Which methods use Transformer as denoiser AND beat UNet-denoiser on long-context?" — needs inner-KG + Result hyperedge.
- "What `FailureMode`s are reported under `Scenario:Long-Context`?" — needs Gap decomposition.
- "Cross-pollination gaps: a method-family applied in domain A but never tested on domain B's datasets." — gap finder.

---

## 10. Execution (Claude Code specific — skip if using another tool)

- One sub-agent per chip; Sonnet 4.6 for bulk (Opus 4.7 deeper but hits session limits under heavy parallel load). Dispatch in waves of ~12.
- Each agent reads: this spec + the paper full text + a reference chip (for depth calibration); writes `<chip_id>.chip.json`; returns a one-line stat summary.
- Aggregate bus + build graph with a Python pass (networkx + pyvis for visualization).
- Directory layout:
  ```
  memory_bank/
  ├── chips/<chip_id>.chip.json
  ├── bus/{primitives,scenarios,failure_modes,datasets,…}.json
  └── memory_bank.graph.json
  ```

---

## 11. Positioning (for paper-writing)

This is a **modular knowledge-representation architecture**, comparable to agent-memory systems (Mem0, Zep/Graphiti, MIRIX, MemoryOS). Its distinctive properties versus those:
- **Uniform internal structure** across all memory units (others are heterogeneous or flat).
- **Standardized 4-pin interface** + **bus** (others have no pin/bus abstraction).
- **Inner-KG decomposition down to primitive + hyperparameter level** (others stop at entity-relation triples).
- **Result-as-hyperedge** bridging methods × evaluation (others store results flat).

It is expressible in RDF (reification + named graphs) but is a *prescriptive design pattern* within that space — the way Star Schema is a pattern within relational DBs. To publish: implement a reference instantiation, demonstrate queries prior frameworks cannot answer, and (ideally) show cross-domain generality (papers + one other domain).
