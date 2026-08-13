# Experimental experience book

## Purpose and scope

This document records what the project learned from repeated implementation, retrieval, trust-gating, and safety experiments. It is intentionally stricter than a progress summary: successful runs, failed runs, post-hoc diagnostics, and unresolved questions are all retained.

Two evidence tracks must not be conflated:

1. **Direct Chip Memory evidence** covers loading, normalizing, retrieving, projecting, and recording runtime use of real paper Chips.
2. **Adjacent evidence-control evidence** covers SQuAD2 retrieval and early-stop experiments. Those experiments informed the verifier and fallback design, but they are not a direct benchmark of this Chip Memory engine and do not establish superiority over G-Memory.

The historical scripts, reports, raw per-example outputs, and recomputation utilities are preserved under [`research_archive/`](../research_archive/README.md). The archive is included so that negative findings cannot disappear when the method changes.

## Executive conclusions

The accumulated evidence supports eight practical conclusions.

1. **Keep one paper as one immutable Chip.** Paper authority should not be mixed with agent observations or learned usage rules.
2. **Keep L1, L2, and L3 distinct.** A useful answer often needs structure, mechanism, and conditional empirical evidence together; one layer cannot safely stand in for the others.
3. **Similarity is a candidate signal, not a use decision.** The lexical-trust experiment saved 73.77% of retrieval effort but lost 0.17 token F1. High overlap can be bait.
4. **Aggregate answer quality is not a sufficient safety measure.** The v3 gate improved aggregate F1 while producing an 11% wrong-stop rate.
5. **A conservative fallback is necessary.** Clean v4 recovered a useful efficiency/quality point, but its 7% wrong-stop rate still missed the 5% endpoint.
6. **Safety and efficiency form a Pareto frontier.** The shifted v5 run reduced wrong stops to 2%, but effort savings fell to 32.65%, below the preset 35% target.
7. **Thresholds must be selected on validation only.** The original v4 post-hoc result was useful diagnostically but was not a defensible held-out result.
8. **Noise must be part of training or validation.** A two-example lexical-distractor smoke test failed; a clean-selected gate was not automatically a noise guardrail.

The result is not “use the most aggressive gate.” The result is a layered memory system in which retrieval proposes evidence, a role-specific projector limits it, a verifier exposes conditions and negative evidence, and a fallback controller chooses whether to use, reject, or expand the evidence.

## Evidence labels used here

| Label | Meaning |
|---|---|
| Measured | Recomputed or directly reported from saved machine-readable output |
| Smoke-tested | Executed successfully, but not a controlled comparative benchmark |
| Diagnostic | Useful for design, but selected or inspected post-hoc |
| Preliminary | Small, shifted, or incomplete result that needs replication |
| Proposed | Not yet empirically established |

A historical report's `PASS` means only that the run met that report's local acceptance rule. It does not mean “no accuracy loss,” “production safe,” or “better than every baseline.”

## Direct Chip Memory experience

### Full-corpus compatibility

The engine was exercised on 502 real paper Chips from three non-duplicated banks.

| Bank | Chips |
|---|---:|
| ICLR 2026 oral | 224 |
| ICML 2026 oral | 137 |
| CVPR 2026 oral | 141 |
| Total | 502 |

The operational loader accepted all 502 files and normalized 33,647 graph items:

| Layer | Meaning in this engine | Items |
|---|---|---:|
| L1 | Nodes and `Structure` events | 17,886 |
| L2 | Typed binary relations | 6,256 |
| L3 | Reified events and result hyperedges | 9,505 |

The loader emitted 415 non-fatal source-quality warnings:

| Warning | Count |
|---|---:|
| No HTTP(S) source URL | 349 |
| No L2 items | 61 |
| No L1 items | 3 |
| No L3 items | 2 |

What this established:

- classic and extended pin families coexist in the real corpus;
- participant collections can be arrays or role-keyed objects;
- graph collections can be arrays, singleton objects, or ID-keyed objects;
- missing a layer is a data-quality condition, not necessarily a parse failure;
- local provenance must be retained when an HTTP URL is unavailable;
- mirrored banks must not be loaded together without duplicate-ID detection.

What it did not establish:

- that all source extractions are factually correct;
- that all 415 warnings are harmless for every downstream task;
- that a permissive compatibility schema should become the final canonical authoring schema.

### Index and retrieval smoke measurements

The normalized snapshot contained all 502 Chips and 33,647 items. It was 38 MiB, built in 5.31 seconds, and reached roughly 413.7 MiB maximum resident memory in the measured local process.

For the query:

```text
multi-agent verification using negative evidence under limited communication budget
```

with role `verifier`, 12 candidate papers, a 30-hit intermediate limit, and a 3,000-token projection budget, the engine observed:

| Stage | Count |
|---|---:|
| Candidate Chips | 12 |
| Allowed items | 888 |
| Lexical seeds | 214 |
| Graph-expansion candidates | 82 |
| Projected hits | 24 |
| Projected L1 / L2 / L3 | 13 / 1 / 10 |
| Warning / information findings | 2 / 10 |

The retrieval took 4.51 seconds and roughly 355.1 MiB maximum resident memory in that local smoke run. These are engineering measurements, not latency benchmarks: there was one process, one query, no warm/cold protocol, and no competing systems.

The run exposed an important failure. Graph-connected, lower-scoring layers could be crowded out after seed selection. The retriever and projector now preserve requested-layer coverage when qualifying items exist. This is why layer coverage is a tested invariant rather than a presentation preference.

### Runtime lifecycle experience

The complete lifecycle was exercised on the 502-Chip bank:

1. initialize context;
2. retrieve memory;
3. record one agent event;
4. record one state transition;
5. complete the task;
6. apply backward feedback;
7. reconstruct the runtime summary.

The resulting append-only trace had one event of every expected type and produced bounded priors for five Chips and twenty items. The run also exposed a path bug: an existing runtime directory with a dot in its name was initially liable to be treated as a file. The runtime now checks whether an existing path is a directory before interpreting its suffix.

Other development corrections are preserved as design rules:

- role bonuses must not be added once in retrieval and then added again during projection;
- a memory token budget applies to injected memory, not to the task text the caller already owns;
- feedback must be derived only from completed contexts that actually retrieved the item;
- source Chips must remain byte-for-byte unchanged throughout the lifecycle;
- a second active task context must be rejected until session isolation exists.

The direct implementation details and exact smoke outputs are in [the full-corpus report](FULL_CORPUS_SMOKE_REPORT.md) and [the failure log](FAILURE_AND_DECISION_LOG.md).

## Adjacent evidence-use experiments

### Why these experiments matter to Chips

A Chip retriever can surface a relevant paper and still cause negative transfer. The agent may stop reading too early, trust a high-overlap but incompatible result, omit the condition under which a mechanism worked, or treat a source-grounded observation as universal. The evidence-use experiments study this downstream decision: **when is the currently retrieved evidence sufficient to use, and when should the system continue or fall back?**

They therefore inform a Chip verifier/controller, especially its treatment of L3 outcomes, conditions, counterevidence, and runtime uncertainty. They do not test the three-layer Chip representation itself.

### Metric definitions

- **Token F1** is answer overlap on the evaluated SQuAD2 answerable slice.
- **Steps** are retrieved chunks inspected before answer generation.
- **Relative effort reduction** is `(baseline steps - method steps) / baseline steps`.
- **Wrong stop** in the v2-v5 family means the stopped answer F1 is more than 0.05 below the corresponding full-context baseline for that example.
- **Answer preservation** means the stopped answer F1 is at least the corresponding baseline F1.

Different rows used different models, calibration splits, or shifted example splits. They should not be ranked as if they were all one controlled tournament. The same-slice v1-v4 rows are the cleanest progression for studying the controller trade-off.

### Experiment ledger

| Experiment | n | Baseline F1 | Method F1 | F1 delta | Steps | Effort saved | Wrong stop | Main lesson |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Oracle evidence upper bound | 100 | 0.9900 | 1.0000 | +0.0100 | 3.85 -> 1.33 | 65.45% | not reported | A good sufficiency detector has large headroom |
| Lexical trust proxy | 100 | 0.9900 | 0.8200 | -0.1700 | 3.85 -> 1.01 | 73.77% | not reported | Overlap alone stops dangerously early |
| Learned evidence proxy | 100 | 0.9900 | 0.9500 | -0.0400 | 3.57 -> 2.22 | 37.82% | not reported | Learning improves safety, but a local “pass” still permits loss |
| Learned proxy + Qwen | 100 | 0.8070 | 0.7950 | -0.0121 | 3.41 -> 2.58 | 24.34% | not reported | Proxy benefit shrinks with real generation |
| Raw Qwen attention-flow | 100 | 0.7911 | 0.7768 | -0.0143 | 3.41 -> 1.28 | 62.46% | not reported | Internal flow is efficient but not sufficient alone |
| Calibrated Qwen flow | 100 | 0.7830 | 0.7825 | -0.0005 | 3.37 -> 1.11 | 67.06% | not reported | Flow plus retrieval relevance was the best practical clean bridge result |
| LLaMA contribution-flow | 100 | 0.7181 | 0.7102 | -0.0079 | 4.00 -> 2.83 | 29.25% | not reported | Real contribution matrices worked, but this was not an exact paper reproduction |
| EvidenceUseGate-v0 | 100 | 0.7786 | 0.7326 | -0.0460 | 3.80 -> 1.50 | 60.53% | not reported | Aggressive learned gate over-stopped |
| EvidenceUseGate-v1 conservative | 100 | 0.7590 | 0.7675 | +0.0085 | 3.88 -> 3.32 | 14.43% | 0% | Safety was recovered by becoming too cautious |
| EvidenceUseGate-v2, risk 0.020 | 100 | 0.7590 | 0.7607 | +0.0017 | 3.88 -> 2.80 | 27.84% | 5% | Pareto control helped but missed 35% effort target |
| EvidenceUseGate-v3 | 100 | 0.7590 | 0.7665 | +0.0075 | 3.88 -> 1.97 | 49.23% | 11% | Aggregate F1 hid unsafe per-example stops |
| EvidenceUseGate-v4 post-hoc | 100 | 0.7590 | 0.7613 | +0.0023 | 3.88 -> 2.28 | 41.24% | 5% | Selective fallback looked promising, but selection leaked test information |
| EvidenceUseGate-v4 clean | 100 | 0.7590 | 0.7820 | +0.0230 | 3.88 -> 2.17 | 44.07% | 7% | Clean selection passed F1/effort but missed strict safety |
| EvidenceUseGate-v5 safety-first, shifted split | 100 | 0.7984 | 0.8287 | +0.0303 | 3.92 -> 2.64 | 32.65% | 2% | Safety improved, but conservative fallback missed effort target |
| Lexical-noise selective fallback smoke | 2 | 0.3333 noisy full | 0.5000 | +0.1667 | 4.00 -> 2.00 | 50.00% | 50% | The clean-selected fallback was worse than v3 on this tiny noisy slice |

The final row is deliberately not generalized: `n=2` is a smoke diagnostic, not a robustness estimate.

### Structural and recomputation checks

The first composition pilot used toy contribution matrices. It passed trace conversion and alignment checks, but it did not run full attribution. That result supports interface compatibility only.

A later 16-example Qwen slice was independently recomputed from saved JSONL rather than from its summary:

| Metric | Baseline | Trust-gated |
|---|---:|---:|
| Mean F1 | 0.726339 | 0.735714 |
| Mean steps | 3.3750 | 2.3125 |
| Relative effort reduction | n/a | 31.48% |

This supports an initial end-to-end learned-proxy result. It still does not validate the exact external information-flow method.

### The iteration sequence and what changed

#### Oracle and lexical policies

The oracle run established that early stopping could save substantial work if evidence sufficiency were known. The lexical run then showed that a cheap similarity threshold was not an adequate approximation: it achieved even larger savings by discarding necessary context.

Decision: retrieval relevance and evidence sufficiency must be separate features.

#### Learned evidence proxy

A logistic regression over retrieval/evidence features reached chunk-classifier AUROC 0.8278 and AUPRC 0.7879, then saved 37.82% of extractive effort with a 0.04 F1 drop. In the 100-example Qwen generation run, it saved 24.34% with a 0.0121 F1 drop.

Decision: use learned cheap features as a controller input, not as unqualified proof that the answer is supported.

#### Internal-flow teachers

Raw Qwen attention-flow produced high savings but a visible F1 loss. Calibrating it with retrieval relevance produced the best clean bridge point: 67.06% effort reduction with a -0.0005 F1 delta. The LLaMA run used a real contribution-matrix path and saved 29.25%, but used LLaMA-2-7B-Chat and a simple relevance blend rather than the exact paper checkpoints and full calibrator.

Decision: expensive or white-box signals are promising teachers. They are not automatically reliable runtime authorities, and mechanistic similarity must not be described as exact reproduction.

#### EvidenceUseGate-v0 to v3

v0 implemented the desired “expensive teacher, cheap runtime gate” shape but overfit and stopped too early. v1 added conservative labels, multiple prediction heads, and hard negatives. It eliminated wrong stops on the clean slice at the cost of low savings. v2 exposed a controllable risk curve but remained below the effort target. v3 inverted the relationship: attention-flow proposed efficient stops and learned safety features acted as a veto. This recovered efficiency but raised wrong stops to 11%.

Decision: quality, effort, and wrong-stop rate must be co-primary. A controller cannot pass based on mean F1 alone.

#### Selective fallback v4

The first v4 selected a fallback rule on the same 100-example table used for reporting. It met the desired point, but only as a diagnostic. The clean v4 selected thresholds on validation and evaluated once on held-out test IDs. It achieved +0.0230 F1 and 44.07% effort reduction, but 7% wrong stops.

The clean run also showed that the risk selector caught six of eleven v3 wrong stops. Applying the same risk condition with full top-k fallback gave a secondary diagnostic point of +0.0188 F1, 37.89% effort reduction, and 5% wrong stops. Because validation selected v2 rather than full top-k as the fallback, this secondary point is not the primary clean result.

Decision: separate the risk detector from the fallback choice. Validate both without changing the held-out decision after seeing test outcomes.

#### Safety-first shifted split v5

The v5 wrapper changed selection priority to safety first and used a shifted 120/60/100 split. It chose full top-k fallback for 27 of 100 test examples. Wrong stops fell to 2% and F1 improved, but effort savings fell to 32.65%. On this shifted split, v3 itself reached 35.97% effort savings and 3% wrong stops, illustrating split sensitivity.

Decision: report the frontier and its uncertainty, not one permanent “best” threshold. The next selector should estimate expected safety gain against fallback cost.

#### Noise robustness

The noise runner inserted a high lexical-overlap chunk from another example before the clean chunks. Thresholds were selected on clean validation, not the noisy test rows. Only two test examples completed in the preserved smoke run. Selective fallback produced a 50% wrong-stop rate while v3 produced 0% on those two examples.

Decision: explicitly include distractors, incompatible conditions, and adversarial source text in train/validation. Clean calibration is not evidence of robustness.

## What these experiences change in the Chip method

### 1. Retrieve an evidence packet, not a pile of snippets

For each candidate mechanism, the projected memory should contain a connected packet:

- L1 component or structural unit;
- L2 relation explaining how the component operates;
- L3 outcome, condition, failure, or evaluation event;
- source and evidence anchor;
- explicit missing-evidence flags;
- compatibility findings for the current task.

The packet can be incomplete, but incompleteness must be visible. A high L2 score without supporting L3 evidence should not silently become an empirical claim.

### 2. Add an evidence-use decision after projection

The current deterministic verifier identifies explicit language. The next controller should return one of:

```text
USE       evidence is relevant, compatible, and sufficiently anchored
CONTINUE  current packet is promising but incomplete
FALLBACK  risk is high; retrieve more or use the full grounded packet
REJECT    explicit incompatibility or negative evidence blocks transfer
ABSTAIN   the Chip bank does not support the requested claim
```

Its features should include retrieval score progression, layer coverage, evidence-anchor coverage, condition match, negative-result presence, cross-paper agreement, role, and bounded runtime history. Internal model-flow features may be optional teachers or advanced backends, not mandatory package dependencies.

### 3. Make full grounded retrieval the safe fallback

For paper Chips, “full top-k” should become “the fullest context-bounded connected packet with conditions and counterevidence.” A fallback must add missing authority, not merely add more text.

### 4. Use the three layers differently by role

| Role | Primary need | Required guardrail |
|---|---|---|
| Planner | L1 decomposition plus L2 dependencies | Must see constraints before choosing a mechanism |
| Critic | L3 failures, risks, limitations, and incompatible conditions | Must be able to block transfer |
| Executor | L2 procedure plus implementation-bearing L1/L3 items | Must not convert a conceptual relation into undocumented code |
| Verifier | L3 evidence, baselines, ablations, and anchors | Must distinguish reported fact from runtime inference |
| Researcher | Balanced L1/L2/L3 packet and cross-paper alternatives | Must surface disagreement and missing evidence |

Role projection is a view over one paper Chip, not a new role-authored truth. Every role must be able to trace a returned item back to the same immutable source.

### 5. Keep runtime experience in a separate authority plane

The engine's JSONL runtime plane can record:

- which role received which item;
- whether the role read or cited it;
- which condition or warning changed the decision;
- whether fallback was triggered;
- task success and reviewer feedback;
- distribution-shift or noise tags.

Only consumed or cited items should eventually receive credit. Success of a task does not prove that every retrieved item helped. Feedback must stay bounded and reversible by rebuilding priors from the event log.

### 6. Preserve failure memories without promoting them to paper truth

An agent failure can improve future routing, but it remains a runtime observation. It may say “this item was harmful for tasks with constraint C,” not “the paper's method is invalid.” Promotion to a reusable cross-task rule requires its own provenance, confidence, validation set, and review state.

## Multi-agent architecture recommended by the evidence

```text
immutable paper Chips
        |
        v
candidate ranking -> L1/L2/L3 retrieval -> graph expansion
        |
        v
role projector -> connected evidence packet per agent
        |
        v
evidence-use controller
  | USE       | CONTINUE/FALLBACK       | REJECT/ABSTAIN
  v           v                         v
agent acts   retrieve/verify more      block unsupported transfer
        \       |                       /
         \      v                      /
          append-only multi-agent runtime events
                        |
                        v
             bounded, auditable priors
```

Recommended coordination protocol:

1. The planner requests structure and candidate mechanisms.
2. The critic requests negative results and violated conditions for those same candidate items.
3. The executor requests only implementable procedures whose conditions survived criticism.
4. The verifier checks citations, layer completeness, and whether the conclusion exceeds the L3 evidence.
5. A coordinator stops only if no blocking finding remains and the evidence-use controller accepts the packet.
6. Each role's actual reads, citations, decisions, and fallbacks are appended to runtime memory.

This supports multi-agent work without converting Chips into G-Memory's interaction/query/insight hierarchy. The paper is still the unit of source memory; the agent trace is a separate runtime graph.

## Same problem and different problem families

### Closest validated direction

The closest direction is multi-agent retrieval and evidence control: deciding which paper mechanism to expose, whether its conditions match, whether evidence is sufficient, and whether another retrieval/verifier step is justified.

### Promising but unvalidated transfers

The same primitives can be evaluated on other problems:

- **tool agents:** L1 tool/component inventory, L2 preconditions and transitions, L3 execution outcomes;
- **preference optimization:** L1 objectives/data, L2 optimization mechanism, L3 failure/safety trade-offs;
- **VLM or speech agents:** L1 modality modules, L2 fusion/alignment mechanisms, L3 perception or robustness events;
- **code agents:** L1 repository components, L2 dependencies, L3 tests, failures, and environment events;
- **security agents:** L1 attack surface, L2 exploit chain, L3 observed success/failure under a sandbox condition.

These are method hypotheses. Each domain needs a typed vocabulary, grounded Chips, domain-specific compatibility checks, and controlled evaluation before a transfer claim is justified.

## Next experiment program

### Stage A: retrieval judgments

Create a frozen, blinded set of `(query, role, chip_id, item_id)` judgments with relevance, layer necessity, compatibility, evidence sufficiency, and rationale. Compare flat BM25, whole-Chip retrieval, L1-only, L1+L2, full L1/L2/L3, graph expansion, and role projection.

Primary questions:

- Does L3 improve negative-evidence recall?
- Does preserving layer coverage improve sufficiency without excessive context?
- Does role projection improve role relevance without hiding required warnings?

### Stage B: paired negative transfer

For each method, construct two surface-similar tasks: one compatible and one that violates a documented condition. Measure retrieval, warning recall, rejection accuracy, unsupported application, and final task success.

### Stage C: multi-agent end-to-end

Use identical base models, seeds, task order, context budgets, and communication budgets for:

1. no memory;
2. flat chunks;
3. whole Chips;
4. Chips with three-layer retrieval;
5. Chips with role projection;
6. Chips with verification/fallback;
7. G-Memory;
8. a hybrid of paper Chips and runtime trajectories.

Report task success, unsupported-claim rate, negative-transfer rate, context tokens, model calls, fallbacks, and citation accuracy. Do not tune the Chip system on the held-out tasks while leaving baselines frozen.

### Stage D: controller frontier

Predeclare thresholds and report a curve over:

- answer/task quality;
- context or tool-step savings;
- wrong-use or wrong-stop rate;
- fallback rate;
- abstention coverage;
- calibration error.

Evaluate cheap lexical/retrieval features, learned evidence features, optional internal-flow teachers, and full grounded fallback. Select on validation and evaluate the final policy once.

### Stage E: robustness and feedback

Add high-overlap distractors, instruction-like source text, contradictory papers, missing layers, misleading early task successes, and distribution shifts. Verify that the event-derived prior remains bounded and can recover after misleading outcomes.

## Claim ledger

### Supported now

- The reference loader accepted 502 real Chips and normalized 33,647 L1/L2/L3 items.
- The engine performed deterministic layer-aware retrieval, graph expansion, role projection, compatibility checks, and an append-only lifecycle on that corpus.
- The archived evidence-control experiments demonstrate a recurring safety/efficiency trade-off and show that lexical similarity alone is unsafe for early stopping.
- The clean v4 and shifted v5 results show that selective fallback is promising but not yet a stable endpoint.

### Not supported yet

- Chip Memory outperforms G-Memory.
- Three-layer retrieval improves end-to-end multi-agent task success.
- The current heuristic verifier proves logical or numerical compatibility.
- The two-example noise run estimates robustness.
- The archived contribution-flow run exactly reproduces the external information-flow paper.
- Runtime success priors identify causal usefulness of individual memory items.

## Practical rule for future work

Every new result should ship with the command, split definition, configuration, machine-readable per-example output, recomputation code, acceptance rule, failure analysis, and a statement of what it does **not** prove. If a run fails, retain it. The failure sequence is part of the method.
