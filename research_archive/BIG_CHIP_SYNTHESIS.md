# Big-Chip Synthesis

Status: draft from local ICLR/ICML chips.

Source decks:
- `/tf/notebooks/oral_research_memory_mission_2026_06_10/inventories/paper_inventory.jsonl`
- `/tf/notebooks/oral_research_memory_mission_2026_06_10/inventories/high_value_target_cards.md`
- `/tf/notebooks/oral_research_memory_mission_2026_06_10/inventories/external_oral_scope_inventory.md`

Scope note:
- Local ICLR coverage has 223 chips labeled as ICLR 2026 oral.
- External ICLR mirror coverage has 224 oral records; 223 match local chips and 1 is a recorded gap.
- Official ICML coverage has 168 oral records; 7 match local chips and 161 are recorded gaps. Among ICML, 84 are heuristic LLM/VLM/speech candidates and 77 of those are recorded gaps rather than chips.

## Big Picture

The strongest common gap across the LLM/VLM/speech oral-memory set is not raw model capability. It is control: when a 7B-class model is placed in a long-context, retrieval, tool-use, multimodal, or preference-optimization loop, the system needs a cheap but trustworthy controller that knows when evidence is sufficient, when context/tool outputs are noise, when reasoning has drifted, and when extra compute is worth paying for.

The local papers repeatedly attack this from different angles:

- `Q-RAG`, `MemAgent`, `LoongRL`, `Strategic Navigation`, and the previous retrieval experiments frame long-context success as adaptive evidence acquisition rather than fixed top-k context stuffing.
- `Information Flow Reveals When to Trust Language Models`, `Verifying CoT via Computational Graph`, `Hallucination Begins Where Saliency Drops`, `Reliable Weak-to-Strong Monitoring`, and reward-hacking detectors show that internal model-use signals can expose failure modes that output-only metrics miss.
- `AgentFlow`, `AgentGym-RL`, `MedAgentGym`, `SimuHome`, `RedTeamCUA`, and `Agent Data Protocol` show that tool-use agents need trajectory-level control and safety evaluation, not just stronger final-answer models.
- `SSPO`, `TI-DPO`, `SafeDPO`, `TROLL`, `P-GenRM`, `Omni-Reward`, and `Nash Preference Optimization` show that preference/reward signals are abundant but noisy; the useful object is often a localized or calibrated control signal, not a monolithic scalar reward.
- `ThinKV`, `ThreadWeaver`, `Hierarchical Speculative Decoding`, `Diffusion LMs Know the Answer Before Decoding`, `In-Place TTT`, and `Mamba-3` show a parallel compute-control theme: decide which tokens, threads, drafts, cache entries, or updates are worth preserving.
- `UALM`, `WAVE`, `EmotionThinker`, `VibeVoice`, and `Latent Speech-Text Transformer` show that speech/multimodal models face the same control issue with denser, noisier evidence channels.
- Newly parsed official ICML oral gaps sharpen three clusters: security evaluation for autonomous agents (`SandboxEscapeBench`, `Jailbreak Foundry`, finetuning-activated adversarial behavior), VLM perception-credit assignment (`Bad Seeing or Bad Thinking`, `Agent0-VL`, `DOUBT`), and mechanistic/dynamic computation control (`Mechanistic Data Attribution`, `FlashTrace`, `Program-of-Layers`).

## Transferable Primitives

| Primitive | Source Cluster | Reusable Mechanism | 7B Feasibility |
|---|---|---|---|
| Evidence sufficiency gate | Q-RAG, 71120, MemAgent, LoongRL | Stop/continue based on whether retrieved context is actually used and enough for answer quality | High: can run on 4B/7B RAG slices |
| Internal-use verifier | 71120, CRV, saliency-drop, reward-hacking effort | Use attention/contribution/saliency/trace features to distinguish true evidence use from lexical/position noise | Medium-high: attention is cheap; contribution is teacher/fallback |
| Selective fallback controller | v4 retrieval diagnostics, weak-to-strong monitoring, RedTeamCUA | Let efficient policy act by default, but route risky states to conservative policy/full context | High: can be evaluated post-hoc then cleanly validated |
| Trajectory belief monitor | T3 belief deviation, AgentFlow, AgentGym-RL, Strategic Navigation | Detect when multi-turn agent state drifts away from task evidence and truncate/replan | Medium: needs agent traces but can start on synthetic/document QA |
| Localized preference signal | TI-DPO, SSPO, SafeDPO, TROLL | Move from sequence-level reward to token/span/step-level correction with conservative safety constraints | Medium: training cost is higher, but LoRA 7B possible |
| Compute allocator | ThinKV, ThreadWeaver, speculative decoding, diffusion early answer | Allocate cache/threads/drafts/denoising steps by predicted marginal value | Medium: implementation touches generation internals |
| Multimodal evidence-risk signal | VLM saliency, UALM/WAVE/EmotionThinker | Detect whether visual/audio evidence supports the generated answer or only anchors a hallucinated explanation | Medium: needs VLM/audio model and benchmark data |
| Security stress-test harness | ICML SandboxEscapeBench, Jailbreak Foundry, ICLR FAB | Turn evolving LLM-agent attacks into reproducible tests for tool-use and post-training safety | Medium: some tasks may need sandboxing/code execution but no paid API is inherent |
| Dynamic computation program | Program-of-Layers, ThinKV, ThreadWeaver, FlashTrace | Predict when a different layer/cache/thread/attribution path is safer or cheaper for the current input | Medium-low: implementation can touch model internals |

## Main Synthesis

The papers point toward a general method family:

**Evidence-Controlled 7B Agents**

At each step, the model receives a state: retrieved chunks, observations, tool outputs, visual/audio segments, prior reasoning, cache/compute traces, and candidate answer. A controller predicts four things:

- expected answer quality if stopping now
- risk that the model is using irrelevant or adversarial evidence
- uncertainty / instability under small perturbations
- marginal value of another retrieval/tool/compute step

The controller then chooses one of: stop, continue, drop/reorder evidence, fallback to full context, branch another thread, or ask a verifier. This unifies adaptive RAG, long-context memory, agent tool use, reward-safety monitoring, and efficient inference.

After adding the official ICML oral inventory, this framing still holds, but the best external validation pressure becomes clearer: a controller must also survive adversarial and multimodal settings. The strongest missing ICML chips point to three stress tests for the same architecture:

- security: sandbox escape, jailbreak reproduction, finetuning-triggered hidden behavior
- multimodal perception: distinguish wrong visual perception from wrong reasoning
- interpretability: use multi-token attribution and training-origin attribution as teachers for cheap runtime gates

The best 7B research opportunity is to make this controller learned and validated, but not naive. Prior diagnostics from the retrieval bundle already showed:

- naive learned stop gates can over-stop
- conservative evidence-use gates preserve quality but save less effort
- calibrated attention-flow can be a high-efficiency teacher
- selective fallback can recover safety if thresholds are chosen cleanly on validation

That aligns directly with the oral-paper pattern: learn cheap runtime control from stronger but more expensive teachers such as contribution flow, counterfactual deletion, trace verification, saliency shifts, or conservative reward labels.

## Best Gap To Attack

**Gap:** 7B LLM/VLM agents are evaluated mostly by final answer, while the system has no reliable low-cost state controller for deciding whether current evidence is sufficient and safe.

Why this is attractive:

- It is cross-paper: retrieval, long-context, agents, preference, and VLM hallucination all need it.
- It is feasible: v0 can use saved RAG outputs, SQuAD2/HotpotQA, Qwen/Qwen-VL 7B, and cheap attention features.
- It is novel enough: the contribution is the control framing and selective safety fallback, not a single hand-written threshold.
- It can produce quick positive or diagnostic results: F1/EM vs effort, wrong-stop rate, noise robustness, fallback frequency.
- It can be extended naturally to the new ICML gaps: replace text chunks with tool traces, image regions, or attack modules, while retaining the same evidence-use/fallback structure.

## Strongest Method Hypothesis

**Flow-Gated Retrieval/Agent Control with Learned Selective Safety Fallback**

Use an efficient teacher policy, such as calibrated attention-flow or retrieval-score progression, for high step savings. Train a small evidence-use/safety model from conservative labels, contribution/counterfactual teachers, or hard negatives. At inference, the efficient policy proposes early stop; the safety model either accepts or routes to a conservative fallback/full-context answer.

This is stronger than a rule stack because the learned component is not "attention + relevance + contribution" by hand. It learns when cheap signals are trustworthy and when they are bait.

## Validation Standard

Minimum defendable prototype:

- fixed IDs
- thresholds selected only on validation
- one-shot held-out test
- raw JSONL and recompute script
- compare fixed top-k, calibrated attention-flow, conservative learned gate, aggressive learned gate, selective fallback
- noise distractor setting

Target:

- clean F1 delta >= -0.005
- effort reduction >= 35%
- wrong-stop <= 5%
- in noise setting, selective fallback has lower wrong-stop and >= F1 than aggressive policy
