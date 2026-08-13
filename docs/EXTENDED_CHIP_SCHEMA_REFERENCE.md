# General Memory Chip Schema Reference

## Stable Pins

Use these six pins across domains:

| Pin | Product/business | Content/script | Robot/work rule | Code/research |
|---|---|---|---|---|
| `problem_gap` | user pain, bottleneck, market/job failure | attention gap, audience confusion, weak hook | unsafe/inefficient behavior, missing state handling | design problem, prior limitation |
| `method_mechanism` | product mechanism, workflow, policy | hook, scene, plot, claim, editing pattern | state-action rule, planner, safety policy | architecture, method, module |
| `evaluation_validation` | KPI, A/B test, cohort, review checklist | retention, CTR, comments, brand fit review | simulation, log review, operator checklist | tests, benchmarks, metrics |
| `result_outcome` | adoption, retention, cost, failure | engagement, conversion, audience reaction | success/failure, safety, latency | behavior, performance, findings |
| `grounding_implementation` | PRD, owner, dashboard, SOP, decision log | script, source clip, shot list, platform data | SOP, policy file, code, sensor/log trace | file, symbol, table, artifact |
| `reuse_transfer` | playbook, product pattern, assumption | reusable format, narrative move, audience condition | operational rule, safety boundary | implementation/research pattern |

The old four pins map directly:

- `gap` -> `problem_gap`
- `method` -> `method_mechanism`
- `evaluation` -> `evaluation_validation`
- `result` -> `result_outcome`

Add the two extra pins when the goal is reasoning or action:

- `grounding_implementation`
- `reuse_transfer`

## Three Layers

The three layers are representational, not domain-specific.

| Layer | Meaning | Store as |
|---|---|---|
| `L1` | containment and hierarchy | `Structure` event |
| `L2` | binary flow, cause, implementation, tradeoff | edge with `layer: "L2"` |
| `L3` | multi-way binding with conditions | event |

Examples:

- `L1`: meta-gap contains sub-gap; method contains module; module contains primitive
- `L2`: primitive feeds module; gap causes failure mode; code symbol implements primitive
- `L3`: compatibility under assumptions; result binding; eval pipeline; training update; grounding claim

## Reusable Unit Boundary

Define a reusable unit as a transferable mechanism, pattern, rule, or process step:

1. It is nameable.
2. It recurs across sources or could recur in future work.
3. It has a role, input/output, or constraint that can transfer.

Avoid:

- too coarse: `business strategy`, `viral video`, `robot intelligence`, `AI product`
- too fine: one word choice, one button color, one config flag, one sensor threshold unless it is central

Prefer:

- `Activation Hook`
- `Friction Removal Step`
- `Audience Objection Handling`
- `Scene-to-Claim Bridge`
- `State-Action Safety Rule`
- `Escalation Rule`
- `Review Checklist`
- `Evidence Anchor`

## Recommended Extensions

### Reusable Unit Card

```jsonc
{
  "id": "unit_escalation_rule",
  "kind": "Primitive",
  "label": "Escalation Rule",
  "props": {
    "abstraction_level": "transferable_unit",
    "role": "safety_or_quality_control",
    "inputs": ["current_state", "risk_signal", "operator_goal"],
    "outputs": ["continue", "stop", "fallback", "ask_for_review"],
    "assumptions": ["risk signal is observable before action"],
    "lifecycle": "execution",
    "mitigates_failure_modes": ["unsafe_action", "low_quality_output"],
    "introduces_failure_modes": ["over-escalation", "slower_execution"],
    "evidence": ["log review", "A/B test", "operator feedback"]
  }
}
```

### Compatibility Event

```jsonc
{
  "id": "compat_escalation_review",
  "kind": "Compatibility",
  "relation": "depends_on",
  "participants": ["unit_escalation_rule", "unit_review_checklist"],
  "condition": "a reviewer or automated check is available before irreversible action",
  "scenario": "robot_operation_or_content_publish",
  "failure_mode": "escalation_without_resolution_path",
  "evidence_source": "workflow_test",
  "confidence": 0.8
}
```

Relations:

- `depends_on`
- `requires`
- `conflicts_with`
- `complements`
- `substitutes`
- `adds_cost`
- `mitigates`
- `introduces_risk`

### Code Anchor

```jsonc
{
  "id": "anchor_escalation_sop",
  "kind": "ProcessAnchor",
  "target": "unit_escalation_rule",
  "path": "ops/safety_sop.md",
  "owner": "operations_lead",
  "anchor_type": "rule",
  "lifecycle": "execution",
  "inputs": ["state", "risk_signal"],
  "outputs": ["fallback_action", "review_ticket"],
  "dependencies": ["review_checklist", "incident_log"],
  "grounding_status": "partial"
}
```

Grounding status:

- `exact`: code implements the method as described
- `partial`: code implements a simplified or adapted version
- `conceptual`: code only demonstrates the idea
- `missing`: no implementation anchor found

## Minimum Useful Chip

A chip is useful for action reasoning only if it has:

1. At least one meta-gap and one sub-gap or measurable failure mode.
2. A proposed action/mechanism separated from the current or previous approach when comparison matters.
3. Reusable units with roles.
4. At least one compatibility relation or explicit assumption.
5. Validation protocol with check, metric/review criterion, scenario, and aggregator/decision rule.
6. Result binding: action(s) x validation x outcome.
7. Grounding anchor, or an explicit `missing` status.
8. A reuse note explaining when the mechanism transfers and when it should not.

## Concrete Pin Requirements

### `problem_gap`

Required:

- one main problem, goal, or rule
- 2-6 sub-problems, sub-goals, situations, or failure classes
- typed `FailureMode` nodes
- `Scenario` nodes
- at least one `GapBinding` event
- one residual gap or limitation if known

Good L2 relations:

- `contains_subgap`
- `causes`
- `specializes`
- `is_evidence_for`
- `leaves_residual_gap`

### `method_mechanism`

Required:

- proposed `Method`, `Action`, `Rule`, `Workflow`, or `ContentPattern` node
- explicit current approach / previous behavior / baseline nodes when comparison matters
- `Module`, `Step`, or `Primitive` decomposition
- reusable unit decomposition
- execution/process/content flow event when applicable
- action-vs-current-state delta

Delta relations:

- `adds_primitive`
- `replaces_primitive`
- `retains_module`
- `removes_condition`
- `changes_signal`
- `changes_lifecycle`
- `trades_off`

### `evaluation_validation`

Required:

- source/scenario
- sample, cohort, segment, test case, or review set
- comparison set when available
- metric, checklist, judge, or observable signal
- evaluator/reviewer/owner
- aggregator or decision rule
- raw artifact or explicit missing status
- falsification threshold when designing new tests

### `result_outcome`

Required result fields:

- `actions` or `methods`: include proposed action/mechanism and compared current approach when available
- `evaluation`: exact eval id
- `scope`: dataset/split/model/setting
- `outcome`: claim
- `numerical`: numbers or raw outcome
- `result_type`: comparative, diagnostic, transfer, robustness, efficiency, negative, tradeoff, engagement, safety, quality, or adoption

### `grounding_implementation`

Required:

- at least one anchor node or explicit missing anchor
- target primitive/method/eval/result
- lifecycle
- input/output or interface
- grounding status

Anchor kinds:

- `CodeAnchor`
- `TextAnchor`
- `ProcessAnchor`
- `DataAnchor`
- `ArtifactAnchor`

### `reuse_transfer`

Required:

- reusable pattern
- assumptions
- invalid conditions
- compatibility events
- cost/risk events
- transfer boundaries

## Domain-Specific Extensions

Keep general chips domain-neutral. Put specialized rules in child skills:

- dynamic action-feedback improvement -> `skills/general-practice-memory/`
- research method evolution -> `skills/research/02research-practice-memory/references/method_evolution_templates.md`
- research reasoning -> `skills/research/02research-practice-memory/references/research_reasoning_protocol.md`
- paper extraction -> `skills/research/01research-paper-memory/references/`
