# Limitations and roadmap

## Current limitations

### Lexical retrieval

BM25 is deterministic and dependency-free but misses semantic paraphrases. Graph expansion helps only after at least one lexical seed is found.

### Heuristic pin assignment

Some existing Chips use global graph arrays without an explicit pin on every object. The loader assigns pins from object kinds. This affects role weighting but not the original layer or payload.

### Heuristic verification

The verifier recognizes explicit linguistic signals. It does not prove logical compatibility or compare numerical constraints symbolically.

### No canonical entity resolution

Cross-paper primitives are not merged automatically. Unsafe string merging can conflate paper-private modules. A future bus resolver should preserve aliases, confidence, and review status.

### Single active task per engine

The high-level engine mirrors a simple agent-memory lifecycle and holds one active `TaskContext`. Concurrent services need session isolation.

### Local event store

JSONL is transparent and robust for experiments, but not a distributed transactional database.

### Feedback confounding

A successful task does not prove every retrieved item was helpful. Current feedback attributes bounded credit to projected items. More precise systems should capture which agent actually consumed or cited each item.

### Source quality

The engine preserves Chip content; it cannot guarantee that extraction was correct. Extraction verification and source anchoring remain upstream responsibilities.

## Roadmap

### Phase 1: harden the data contract

- choose one canonical schema version;
- require per-object pin labels;
- require source digests and evidence anchors;
- add migration tools for older variants;
- publish ontology conformance tests.

### Phase 2: hybrid retrieval

- add pluggable embedding backend;
- fuse sparse, dense, bus, and graph scores;
- add query decomposition by role and requested layer;
- calibrate scores on relevance judgments.

### Phase 3: stronger verification

- parse numerical and categorical constraints;
- represent requirements and conflicts as typed L3 compatibility events;
- compare task constraints with event conditions;
- add contradiction clusters across papers;
- distinguish absence of evidence from evidence of absence.

### Phase 4: precise runtime credit

- record memory read and citation events per agent;
- apply credit only to consumed items;
- model task-family-specific usefulness;
- add decay and distribution-shift detection;
- audit ranking changes after every feedback event.

### Phase 5: framework adapters

- AutoGen adapter;
- LangGraph adapter;
- custom DAG-team adapter;
- OpenAI Agents SDK adapter;
- service API with per-session task contexts.

### Phase 6: publication evaluation

- implement all baselines;
- freeze benchmark tasks and relevance judgments;
- run registered ablations;
- release traceable results and failure cases;
- avoid a general “better than G-Memory” claim unless the controlled evidence supports it.

