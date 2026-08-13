# Schema and provenance

## 1. Source authority

The source `.chip.json` is authoritative. Runtime normalization is derived and rebuildable.

Every `KnowledgeItem` retains:

- `chip_id`;
- source path;
- source URL when present;
- graph-local ID;
- layer;
- semantic pin;
- kind;
- original payload;
- graph references;
- evidence anchors.

## 2. Supported schema families

The compatibility loader accepts paper Chips with:

- `chip_id` or `id`;
- graph arrays at the top level or nested inside graph containers;
- `source/target` or `src/dst` edge endpoints;
- `kind`, `rel`, `relation`, or `type` relation labels;
- classic or extended semantic pins;
- result hyperedges in `result.results` or `result_outcome.results`.

The included JSON Schema checks a stable structural subset. The Python validator provides operational normalization and warnings.

## 3. Layer semantics

The loader does not infer L2 or L3 claims from prose. It maps explicitly represented objects:

- nodes become L1 entities;
- `Structure` events become L1 containment units;
- edges become L2 relations;
- non-structure events and results become L3 events.

Pin prose helps rank a paper, but the engine does not present prose alone as a graph mechanism.

## 4. Evidence

Event-level anchors are collected from fields including `evidence`, `section`, `page`, `table`, `figure`, `citation`, `source`, and `numerical`.

If an L3 event lacks such fields, retrieval still cites the Chip file but emits an informational finding that the event lacks a local evidence anchor.

## 5. Runtime schema

Each JSONL line contains:

```json
{
  "event_id": "uuid",
  "timestamp": "ISO-8601",
  "event_type": "retrieval",
  "context_id": "uuid",
  "payload": {}
}
```

Defined event types are:

- `context_started`;
- `retrieval`;
- `agent_event`;
- `state_transition`;
- `task_completed`;
- `backward`.

Applications may add fields inside `payload`. Existing events should never be edited in place.

## 6. Multi-agent vocabulary

`schema/multiagent_ontology.yaml` proposes node, relation, and event types for papers describing multi-agent methods. It is an extension vocabulary, not a different hierarchy.

For example:

```yaml
kind: CollaborationEpisode
participants:
  - planner
  - critic
  - executor
condition: "two communication rounds"
outcome: "verified plan"
evidence: "Section 4.2, Table 3"
```

This remains an L3 event inside the appropriate paper Chip.

## 7. Versioning recommendation

Future Chip producers should include:

```json
{
  "schema_version": "11.0",
  "chip_metadata": {
    "version": 1,
    "created_at": "ISO-8601",
    "source_digest": "sha256:...",
    "extractor": "...",
    "verification_status": "..."
  }
}
```

Schema migration should create a new derived representation or explicitly version the Chip. It should never silently rewrite historical evidence.

