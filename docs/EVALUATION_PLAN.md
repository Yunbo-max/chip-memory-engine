# Evaluation plan

## 1. Central hypothesis

Role-conditioned retrieval over grounded L1/L2/L3 paper Chips will reduce unsupported memory use and negative transfer relative to flat paper retrieval, while preserving or improving task success. Separately labeled runtime feedback will improve repeated-task retrieval without contaminating paper-derived claims.

## 2. Systems

1. No long-term memory.
2. Flat paper-chunk BM25.
3. Flat paper-chunk dense retrieval.
4. Whole-Chip retrieval.
5. Chip L1 only.
6. Chip L1 + L2.
7. Chip L1 + L2 + L3.
8. Full Chip with role projection.
9. Full Chip with role projection and verification findings.
10. Full Chip with bounded runtime feedback.
11. G-Memory.
12. Hybrid runtime trajectories plus paper-Chip retrieval.

All systems should use the same base model, task order, communication budget, and context budget.

## 3. Task families

- repeated tasks from one benchmark;
- paraphrased tasks with identical constraints;
- similar-looking tasks with incompatible constraints;
- cross-domain mechanism transfer;
- tasks with explicit negative results in the source bank;
- tasks with incomplete evidence;
- adversarial source text containing instruction-like content;
- tasks requiring different information for planner, critic, executor, and verifier.

## 4. Metrics

### Outcome

- task success;
- reward;
- repair success after a failed step;
- time or environment steps to completion.

### Memory quality

- relevant-item precision and recall;
- relevant-paper recall;
- role relevance;
- condition match;
- negative-evidence recall;
- contradiction detection;
- unsupported memory-derived claim rate;
- source-citation accuracy;
- abstention quality when evidence is insufficient.

### Efficiency

- injected context tokens;
- retrieval latency;
- agent-model calls;
- total cost;
- index build time and size.

### Safety and integrity

- negative-transfer rate;
- paper/runtime contamination rate;
- prompt-injection following rate;
- incorrect feedback amplification;
- reproducibility across repeated runs.

## 5. Required ablations

- remove L2;
- remove L3;
- remove graph expansion;
- remove role weighting;
- give every role the same projection;
- remove evidence anchors;
- remove verifier findings;
- remove negative-result items;
- replace BM25 with embedding-only retrieval;
- remove runtime feedback;
- increase feedback weight until self-reinforcement appears;
- mix runtime observations into source Chips to quantify contamination.

## 6. Retrieval judgments

Create a blinded relevance set containing:

- query;
- agent role;
- paper ID;
- graph item ID;
- layer;
- relevance grade;
- compatibility grade;
- evidence sufficiency;
- annotator rationale.

At least two annotators should judge high-stakes transfer cases. Report agreement and adjudication policy.

## 7. Negative-transfer test construction

Construct paired tasks:

```text
Task A: surface-similar and compatible with method M
Task B: surface-similar but violates M's compute/data/communication assumption
```

Measure whether each system retrieves M, whether it surfaces the violated condition, and whether the agent applies or rejects M.

## 8. Feedback evaluation

Use controlled task sequences:

1. cold start with no runtime events;
2. repeated compatible tasks;
3. distribution shift;
4. deliberately misleading early outcomes;
5. recovery after corrected evidence.

Report both immediate performance and ranking drift. The bounded prior should adapt slowly and recover from misleading outcomes.

## 9. Statistical reporting

- use paired comparisons on identical tasks and seeds;
- report confidence intervals, not only means;
- separate retrieval variance from model-generation variance;
- correct for multiple comparisons when evaluating many variants;
- retain all failed runs and predeclare exclusion rules;
- publish per-task memory traces where licensing permits.

## 10. Success criteria for the prototype

The first prototype is successful if it demonstrates:

- valid loading of the full target Chip bank;
- lower unsupported-claim and negative-transfer rates than flat retrieval;
- useful differentiation between role projections;
- no paper-file mutation;
- fully traceable runtime events;
- competitive task success at equal or lower context cost.

## 11. Controller acceptance criteria learned from prior experiments

The archived evidence-use sequence showed that mean F1 alone can conceal harmful early stops. Any learned evidence-use controller must therefore report a multi-objective frontier containing:

- task quality or answer F1;
- context/retrieval/tool effort;
- wrong-use or wrong-stop rate;
- answer preservation;
- fallback and abstention rates;
- calibration under clean, distractor, and incompatible-condition settings.

Thresholds and fallback source must be selected on validation only. The selected policy is evaluated once on held-out test IDs. Post-hoc searches are allowed only as clearly labeled diagnostics for the next experiment.

The earlier retrieval studies used provisional targets of F1 delta >= -0.005, effort reduction >= 35%, and wrong-stop <= 5%. Those values are useful historical reference points, not universal thresholds for every multi-agent benchmark. A new benchmark must predeclare task-appropriate targets.

## 12. Minimum artifact bundle

Every reported run should retain:

- exact task/example IDs and split construction;
- executable command and complete configuration;
- environment and model identifiers;
- per-example machine-readable outputs;
- independent metric recomputation code;
- acceptance rule fixed before test;
- failure analysis and exclusions;
- explicit statement distinguishing direct Chip evidence from adjacent controller evidence.

See [the experience book](EXPERIMENTAL_EXPERIENCE.md) for the experiment history that motivated these requirements.
