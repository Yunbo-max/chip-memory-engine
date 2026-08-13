# Failure and decision log

## Purpose

This log turns implementation and experiment failures into explicit engineering constraints. “Fixed” means a regression test or current code path covers the observed case. “Open” means the project has only a mitigation or a proposed experiment.

## Data and schema failures

| ID | Failure | Evidence | Decision | Status |
|---|---|---|---|---|
| DATA-001 | Assuming one pin family rejects real Chips | Classic and extended pins coexist in 502 files | Accept both at load time; converge only through versioned migration | Mitigated |
| DATA-002 | Assuming graph collections are arrays misses historical objects | Singleton and ID-keyed graph objects were observed | Normalize array, singleton object, and ID-keyed object forms | Fixed/tested |
| DATA-003 | Assuming event participants are a flat list loses role-keyed participants | Both participant shapes occur | Walk nested maps/lists when resolving known node references | Fixed/tested |
| DATA-004 | Treating a missing layer as a fatal parse error discards usable records | 61 Chips lacked L2; 3 lacked L1; 2 lacked L3 | Load with a visible warning; let queries and evaluations judge sufficiency | Fixed/tested |
| DATA-005 | Assuming every paper has an HTTP URL loses provenance | 349 Chips lacked HTTP(S) URLs | Retain absolute source path and emit warning | Fixed/tested |
| DATA-006 | Loading compact mirrors with main banks creates duplicate paper IDs | Corpus audit found overlapping banks | Reject duplicate `chip_id`; select non-overlapping source banks | Fixed/tested |
| DATA-007 | A permissive compatibility schema can hide authoring drift | All 502 pass, but shapes vary substantially | Keep compatibility schema permissive; design a stricter future canonical schema and migrator | Open |
| DATA-008 | A valid Chip may still contain an incorrect extraction | Structural validation does not compare against the paper | Require evidence anchors, source digests, and extraction review states | Open |

## Retrieval and projection failures

| ID | Failure | Evidence | Decision | Status |
|---|---|---|---|---|
| RET-001 | High-scoring connected items can crowd out a requested layer | Full-corpus query motivated a coverage correction | Preserve requested-layer coverage during expansion and projection | Fixed/tested |
| RET-002 | Role preference was at risk of being counted twice | Retrieval and projection both call the role scorer | Projection applies only `bonus - existing_bonus` | Fixed in code |
| RET-003 | Counting task text against the memory budget underfills memory | Task text is already owned by the caller | Budget only injected memory plus projection overhead | Fixed in code/tested |
| RET-004 | Lexical candidate retrieval misses semantic paraphrases | BM25 has no semantic encoder | Keep BM25 as deterministic baseline; add a pluggable dense/hybrid backend | Open |
| RET-005 | Out-of-vocabulary queries still need deterministic behavior | Sparse search can return no candidates | Return deterministic zero-score candidates and expose an OOV diagnostic | Mitigated |
| RET-006 | Graph expansion helps only after a seed is found | One-hop expansion cannot recover a completely missed paper | Add dense/bus candidate retrieval and query decomposition | Open |
| RET-007 | Heuristic pin assignment can misclassify unusual kinds | Some objects lack explicit pin labels | Preserve original payload and expose inferred pin; require pins in future schema | Open |
| RET-008 | More context is not automatically safer | Lexical-trust and noise experiments show bait and irrelevant context | Fallback must retrieve a connected grounded packet with conditions/counterevidence | Open design |

## Runtime and feedback failures

| ID | Failure | Evidence | Decision | Status |
|---|---|---|---|---|
| RUN-001 | A directory named like `runtime.session` can be mistaken for a file | Full-corpus lifecycle exposed the suffix bug | Existing directories always map to `usage_events.jsonl` inside them | Fixed/tested |
| RUN-002 | Updating paper Chips with agent experience contaminates source truth | Architectural review and provenance invariant | Paper Chips are immutable; all runtime experience is append-only JSONL | Fixed/tested |
| RUN-003 | Successful tasks can over-credit every retrieved item | Current outcome is task-level, not causal | Bound priors; next record per-agent reads/citations and credit only consumption | Partly mitigated |
| RUN-004 | Unfinished retrieval contexts can pollute feedback | Retrieval without a completed outcome has no label | Derive priors only from completed contexts with retrieval events | Fixed/tested |
| RUN-005 | One engine instance cannot safely host concurrent active tasks | Current engine stores one `current_task_context` | Reject a second active context; add session isolation for services | Fixed limitation |
| RUN-006 | A partial final JSONL write can break reconstruction | Process interruption can leave a bad tail | Tolerate invalid partial lines during iteration; retain append-only writes and fsync | Mitigated |
| RUN-007 | Positive feedback can self-reinforce a bad early retrieval | Beta smoothing alone does not detect distribution shift | Bound score contribution; evaluate misleading-history and recovery sequences | Open |

## Evidence-use experiment failures

| ID | Failure | Measured observation | Decision | Status |
|---|---|---|---|---|
| EXP-001 | Structural composition was mistaken for full attribution evidence | Toy contribution pilot used negligible GPU memory and no real attribution | Label it structural proxy only | Corrected claim |
| EXP-002 | Lexical similarity triggered premature stopping | 73.77% effort saved, but F1 fell by 0.1700 | Never use lexical overlap as the sole trust signal | Closed design choice |
| EXP-003 | A lightweight learned proxy retained nontrivial quality loss | 37.82% effort saved with -0.0400 extractive F1 | Treat it as controller input/bridge, not proof of support | Open improvement |
| EXP-004 | Raw internal attention was too aggressive | 62.46% effort saved with -0.0143 F1 | Calibrate with external relevance and safety features | Mitigated |
| EXP-005 | v0 overfit its teacher and over-stopped | 60.53% effort saved, -0.0460 F1, 88% preservation | Add hard negatives, conservative labels, multiple heads | Addressed by v1 |
| EXP-006 | v1 recovered safety by becoming too conservative | 0% wrong stop, but only 14.43% effort saved | Learn a controllable risk/efficiency curve | Addressed by v2 |
| EXP-007 | v2 missed the desired efficiency region | Best shown point saved 27.84% with 5% wrong stop | Use efficient teacher with learned veto | Addressed by v3 |
| EXP-008 | v3's mean F1 hid per-example risk | +0.0075 F1 and 49.23% effort, but 11% wrong stop | Make wrong-stop a co-primary endpoint | Open frontier |
| EXP-009 | First v4 selected its threshold on the test table | 41.24% effort and 5% wrong stop were post-hoc | Retain as diagnostic; rerun with validation-only selection | Addressed by clean v4 |
| EXP-010 | Clean v4 missed strict safety | 44.07% effort, +0.0230 F1, 7% wrong stop | Separate risk detection from fallback choice | Open |
| EXP-011 | Safety-first v5 missed the effort target | 2% wrong stop and +0.0303 F1, but 32.65% effort | Optimize expected safety gain per fallback cost | Open |
| EXP-012 | Clean calibration did not guarantee noise robustness | Selective fallback had 50% wrong stop on `n=2`; v3 had 0% | Train/validate with explicit distractors; rerun at adequate sample size | Open/preliminary |
| EXP-013 | Cross-run “best method” claims ignored split/model changes | Baselines vary across experiment families | Compare only controlled same-slice rows; report frontiers and uncertainty | Ongoing policy |
| EXP-014 | Environment-specific model and external-repository paths impair replay | Historical scripts use local cache and `/tf/notebooks` paths | Preserve exact scripts, document dependencies, add portable future runners | Open archive work |

## Research-integrity decisions

### Failed runs remain first-class artifacts

v0, conservative v1, no-target v2/v3, post-hoc v4, clean v4 near miss, v5 target miss, and the two-example noise failure are all retained. A later positive result must not overwrite them.

### Three endpoints are mandatory

A controller result must report:

1. answer or task quality;
2. effort/context/tool cost;
3. wrong-use or wrong-stop rate.

Fallback and abstention rates should also be reported. Mean task quality cannot compensate silently for concentrated harmful errors.

### Test-time choices are frozen

Feature construction, thresholds, fallback source, target constraints, and exclusion rules must be selected before evaluating the held-out test set. Post-hoc results can guide the next experiment but must be labeled diagnostic.

### Mechanistic resemblance is not reproduction

The archived LLaMA run uses a real contribution-matrix code path, but not the exact checkpoints and calibrator of the external paper. The Qwen attention runs are bridge experiments. Neither should be named as a strict reproduction.

### Engine smoke tests are not comparative evidence

The 502-Chip run demonstrates compatibility and operability. It does not show that Chips improve task outcomes or beat G-Memory. Those claims remain gated on the controlled plan in [EVALUATION_PLAN.md](EVALUATION_PLAN.md).

## Required regression coverage

Any future change touching a logged failure should add or retain a test for:

- source immutability;
- duplicate IDs;
- schema collection variants;
- layer mapping and requested-layer filters;
- layer coverage under expansion and budgets;
- role-dependent ordering without duplicate bonus;
- negative/incompatible evidence;
- runtime path interpretation and JSONL validity;
- completed-context-only priors;
- archived metric recomputation and artifact existence.
