# Lessons from the G-Memory implementation

## Scope of the code review

The design was informed by inspection of the official G-Memory repository and its runtime integration. No G-Memory source code is included in this package.

Official sources:

- <https://github.com/bingreeky/GMemory>
- <https://arxiv.org/abs/2506.07398>
- <https://papers.neurips.cc/paper_files/paper/2025/file/136a45cd9b841bf785625709a19c6508-Paper-Conference.pdf>

The inspected repository revision was `7b581c51d993bd600df14691d101d7e601040cc6`.

## What the implementation actually does

G-Memory's runtime representation is split across several stores:

- a Chroma collection stores task text and serialized task metadata;
- agent messages form NetworkX directed graphs inside a chain of states;
- the chain is serialized into task metadata;
- a pickled NetworkX graph connects related task queries;
- insight rules live in JSON with scores and positive/negative task associations;
- prompts extract key steps, diagnose failure, edit insight rules, merge rules, rank successful trajectories, and project rules to agent roles.

The lifecycle is the most reusable architectural lesson:

1. initialize a task context;
2. retrieve prior material;
3. inject role-relevant memory;
4. record agent messages and environment transitions;
5. save the completed task with an outcome;
6. use feedback to change future retrieval.

## What this package adapts

| G-Memory lesson | Chip Memory adaptation |
|---|---|
| Stable lifecycle API | `ChipMemoryBase` and `ChipMemoryEngine` |
| Task-time retrieval | Candidate paper and L1/L2/L3 item retrieval |
| Role projection | Deterministic role-specific subgraph projection |
| Interaction observation | Append-only typed agent and state events |
| Outcome feedback | Bounded usage priors over cited paper items |
| Persistent memory | Immutable Chips plus separate JSONL runtime events |

## What this package intentionally changes

### Representation

G-Memory organizes runtime history as interactions, queries, and insights. Chip Memory keeps the paper's own representation: structure, binary mechanism, and multi-party event.

### Retrieval unit

G-Memory primarily returns trajectories and textual insight rules. Chip Memory returns graph items with paper and evidence provenance.

### Projection

The G-Memory implementation uses an LLM prompt to rewrite insights for a role. The reference Chip projector reorders and selects source items without paraphrasing them. This avoids adding unsupported content during projection and creates a deterministic baseline.

### Updates

G-Memory changes its insight list as experience accumulates. Chip Memory never changes paper-derived claims. It records whether retrieved items were used in successful or failed executions and applies only a bounded retrieval prior.

### Negative evidence

G-Memory stores failed trajectories and uses them while updating insight rules. Chip Memory additionally elevates explicit negative results, incompatibilities, risks, and conditions already present in source papers.

## Code-level cautions learned from the comparison

- Conceptual graphs do not require one physical graph database; persistence should be selected by access pattern.
- Runtime observations need an explicit authority boundary from source knowledge.
- LLM-generated rule edits are flexible but difficult to test and reproduce.
- Role projection should be evaluated independently of retrieval.
- Success feedback can cause self-reinforcement, so its weight must be bounded.
- Failed examples should be visible to downstream decision logic, not only used during background summarization.
- A memory paper needs dedicated tests for its memory module, not only benchmark scripts.

## Licensing boundary

No license file was observed at the reviewed G-Memory repository revision. This package therefore uses an independently written implementation and references only public architectural ideas and public interfaces. Obtain explicit permission or a clear license before copying any upstream source.

