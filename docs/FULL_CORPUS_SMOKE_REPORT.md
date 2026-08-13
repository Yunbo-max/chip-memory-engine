# Full-corpus smoke report

## Scope

The reference engine was tested against the three main paper-Chip banks in the parent repository:

| Bank | Chips |
|---|---:|
| ICLR 2026 oral | 224 |
| ICML 2026 oral | 137 |
| CVPR 2026 oral | 141 |
| **Total** | **502** |

The compact mirror banks were excluded to avoid duplicate paper IDs.

## Environment

- Python: 3.10.9
- Engine: 0.1.0
- Date: 2026-08-02
- Retrieval backend: dependency-free in-memory BM25
- Measurement: one local process; timings are smoke measurements, not benchmark claims

## Automated tests

```text
21 passed in 0.31s
```

The tests cover:

- classic and extended Chip loading;
- singleton and object graph compatibility;
- L1/L2/L3 mapping;
- graph participant references;
- immutable source files;
- malformed JSON;
- duplicate Chip IDs;
- schema loading;
- layer filters;
- negative-transfer blocks;
- role-conditioned ranking;
- context budgets;
- graph expansion;
- full runtime lifecycle;
- JSONL validity;
- feedback priors;
- active-context safety;
- runtime directory handling;
- validation, indexing, and retrieval CLI commands.

## Operational loader result

```json
{
  "files": 502,
  "valid": 502,
  "invalid": 0,
  "warnings": 415
}
```

Normalized graph totals:

| Runtime layer | Items |
|---|---:|
| L1 | 17,886 |
| L2 | 6,256 |
| L3 | 9,505 |
| **Total** | **33,647** |

Observed warnings:

| Warning | Count |
|---|---:|
| No HTTP(S) source URL | 349 |
| No L2 items | 61 |
| No L1 items | 3 |
| No L3 items | 2 |

These are source-bank quality warnings rather than parser failures. A local source path is still retained when an HTTP URL is absent.

Loader timing:

```text
elapsed: 2.33 seconds
maximum resident memory: 18,332 KiB
```

## JSON Schema result

All 502 Chips pass the included permissive compatibility schema:

```json
{
  "files": 502,
  "schema_valid": 502,
  "schema_invalid": 0
}
```

The audit found schema drift worth preserving:

- classic and extended pin families coexist;
- event participants may be flat arrays or role-keyed objects;
- historical graph collections may be arrays, singleton objects, or ID-keyed objects.

The loader normalizes these forms without rewriting the source Chips.

## Index snapshot

The `build-index` command produced a complete normalized snapshot:

```json
{
  "schema": "chip-memory-runtime-index-v1",
  "chip_count": 502,
  "item_count": 33647,
  "catalog_count": 502,
  "items_count": 33647
}
```

Smoke measurements:

```text
output size: 38 MiB
elapsed: 5.31 seconds
maximum resident memory: 413,700 KiB
```

The snapshot is rebuildable and is not the source authority.

## Real retrieval query

Query:

```text
multi-agent verification using negative evidence under limited communication budget
```

Role and limits:

```text
role: verifier
candidate papers: 12
total intermediate hit limit: 30
projection budget: 3,000 estimated tokens
```

Result:

```json
{
  "candidate_chips": 12,
  "allowed_items": 888,
  "lexical_seeds": 214,
  "graph_expansion_candidates": 82,
  "projected_hits": 24,
  "projected_layers": {
    "L1": 13,
    "L2": 1,
    "L3": 10
  },
  "findings": {
    "warning": 2,
    "info": 10
  }
}
```

Retrieval timing:

```text
elapsed: 4.51 seconds
maximum resident memory: 355,072 KiB
```

This run prompted a retrieval improvement: graph expansion now explicitly preserves requested-layer coverage when a connected lower-scoring layer would otherwise be crowded out.

## Runtime lifecycle result

The 502-Chip bank was also exercised through:

1. context initialization;
2. retrieval;
3. one agent event;
4. one state transition;
5. successful completion;
6. backward feedback;
7. runtime summary reconstruction.

Result:

```json
{
  "contexts": 1,
  "completed_contexts_with_retrieval": 1,
  "event_counts": {
    "context_started": 1,
    "retrieval": 1,
    "agent_event": 1,
    "state_transition": 1,
    "task_completed": 1,
    "backward": 1
  },
  "scored_chips": 5,
  "scored_items": 20
}
```

Lifecycle timing:

```text
elapsed: 4.60 seconds
maximum resident memory: 355,128 KiB
```

This run exposed and fixed a path-handling bug for existing runtime directories whose names contain dot suffixes.

## Interpretation

The smoke tests establish that the implementation:

- loads the complete target corpus;
- preserves all source files;
- supports observed schema variants;
- produces L1/L2/L3 retrieval on real Chips;
- exposes compatibility findings;
- creates a complete append-only lifecycle trace;
- can build a portable normalized snapshot.

They do **not** establish that Chip Memory outperforms G-Memory or other baselines. That requires the controlled task evaluation described in `EVALUATION_PLAN.md`.

