#!/usr/bin/env python3
"""EvidenceUseGate-v3 guarded distillation.

v2 proved that a learned gate can expose a risk-budget Pareto curve, but it was
still conservative: F1 stayed safe while effort reduction topped out at 27.84%.

v3 changes the control rule. Instead of requiring every safety head to be low
before stopping, it treats calibrated Qwen attention-flow as the high-efficiency
teacher and lets the learned safety heads veto only clearly risky early stops.

Goal:

- effort reduction >= 40-50%;
- F1 delta >= -0.005;
- wrong-stop rate <= 5%.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = BUNDLE_ROOT / "experiments/out/evidence_use_gate_v3_guarded_distill"
V2_SCRIPT = BUNDLE_ROOT / "experiments/run_evidence_use_gate_v2_pareto.py"
DEFAULT_MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"


def guarded_stop_condition(pred: dict[str, float], step: int, total_steps: int, policy: dict[str, Any]) -> bool:
    teacher_ready = pred["attention_teacher_stop_prob"] >= policy["teacher_stop_threshold"]
    if not teacher_ready:
        return False

    drop_block = (
        pred["predicted_f1_drop"] > policy["drop_threshold"]
        and pred["stop_prob"] < policy["safe_stop_threshold"]
    )
    noise_block = (
        pred["noise_risk"] > policy["noise_threshold"]
        and pred["uncertainty"] > policy["uncertainty_threshold"]
    )
    continue_advantage = pred["continue_value"] - pred["expected_f1_if_stop"]
    quality_block = (
        continue_advantage > policy["quality_margin"]
        and pred["stop_prob"] < policy["safe_stop_threshold"]
    )
    min_step_block = step < policy["min_stop_step"]
    return not (drop_block or noise_block or quality_block or min_step_block)


def evaluate_policy(
    v2,
    val_summaries: list[dict[str, Any]],
    rows_by_id: dict[str, list[dict[str, Any]]],
    pred_by_id: dict[str, list[dict[str, float]]],
    policy: dict[str, Any],
) -> dict[str, float]:
    baseline_f1 = []
    gate_f1 = []
    baseline_steps = []
    gate_steps = []
    wrong_stops = []
    teacher_matches = []
    for summary in val_summaries:
        qid = summary["question_id"]
        rows = rows_by_id[qid]
        preds = pred_by_id[qid]
        stop = len(rows)
        for idx, pred in enumerate(preds, start=1):
            if guarded_stop_condition(pred, idx, len(rows), policy):
                stop = idx
                break
        full_f1 = float(summary["f1_by_step"][-1])
        selected_f1 = float(summary["f1_by_step"][stop - 1])
        teacher_step = int(rows[0]["attention_teacher_stop_step"])
        baseline_f1.append(full_f1)
        gate_f1.append(selected_f1)
        baseline_steps.append(len(rows))
        gate_steps.append(stop)
        wrong_stops.append(1.0 if selected_f1 < full_f1 - 0.05 else 0.0)
        teacher_matches.append(1.0 if stop == teacher_step else 0.0)
    b_f1 = float(np.mean(baseline_f1)) if baseline_f1 else 0.0
    g_f1 = float(np.mean(gate_f1)) if gate_f1 else 0.0
    b_steps = float(np.mean(baseline_steps)) if baseline_steps else 0.0
    g_steps = float(np.mean(gate_steps)) if gate_steps else 0.0
    return {
        "baseline_f1": b_f1,
        "gate_f1": g_f1,
        "f1_delta": g_f1 - b_f1,
        "baseline_steps": b_steps,
        "gate_steps": g_steps,
        "effort_reduction": b_steps - g_steps,
        "relative_effort_reduction": (b_steps - g_steps) / b_steps if b_steps else 0.0,
        "wrong_stop_rate": float(np.mean(wrong_stops)) if wrong_stops else 0.0,
        "attention_teacher_exact_match_rate": float(np.mean(teacher_matches)) if teacher_matches else 0.0,
    }


def choose_policies(
    v2,
    v1,
    heads: dict[str, Any],
    val_summaries: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    risk_budgets: list[float],
    max_f1_drop: float,
    max_wrong_stop: float,
) -> list[dict[str, Any]]:
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in val_rows:
        rows_by_id.setdefault(row["question_id"], []).append(row)
    pred_by_id = {
        qid: [v2.predict_v2(v1, heads, row["features"]) for row in rows]
        for qid, rows in rows_by_id.items()
    }

    policies = []
    for budget in risk_budgets:
        best = None
        relaxed_best = None
        for teacher_threshold in np.linspace(0.05, 0.65, 13):
            for drop_threshold in np.linspace(0.04, min(0.40, 0.10 + budget * 4.0), 13):
                for noise_threshold in np.linspace(0.30, 0.95, 14):
                    for uncertainty_threshold in np.linspace(0.20, 0.65, 10):
                        for safe_stop_threshold in [0.05, 0.15, 0.30, 0.45, 0.60]:
                            for min_stop_step in [1, 2]:
                                policy = {
                                    "risk_budget": float(budget),
                                    "teacher_stop_threshold": float(teacher_threshold),
                                    "drop_threshold": float(drop_threshold),
                                    "noise_threshold": float(noise_threshold),
                                    "uncertainty_threshold": float(uncertainty_threshold),
                                    "safe_stop_threshold": float(safe_stop_threshold),
                                    "quality_margin": float(max(max_f1_drop, budget)),
                                    "min_stop_step": int(min_stop_step),
                                }
                                metrics = evaluate_policy(v2, val_summaries, rows_by_id, pred_by_id, policy)
                                if metrics["effort_reduction"] <= 0:
                                    continue
                                score = (
                                    metrics["relative_effort_reduction"]
                                    + 0.15 * metrics["gate_f1"]
                                    - 0.55 * metrics["wrong_stop_rate"]
                                    - 0.10 * max(0.0, -metrics["f1_delta"])
                                )
                                candidate = {
                                    **policy,
                                    **{f"val_{k}": v for k, v in metrics.items()},
                                    "selection_score": float(score),
                                }
                                if metrics["f1_delta"] >= -max_f1_drop and metrics["wrong_stop_rate"] <= max_wrong_stop:
                                    if best is None or candidate["selection_score"] > best["selection_score"]:
                                        best = candidate
                                elif metrics["f1_delta"] >= -budget and metrics["wrong_stop_rate"] <= max(0.10, max_wrong_stop):
                                    if relaxed_best is None or candidate["selection_score"] > relaxed_best["selection_score"]:
                                        relaxed_best = candidate
        if best is None:
            best = relaxed_best
        if best is None:
            fallback = {
                "risk_budget": float(budget),
                "teacher_stop_threshold": 0.95,
                "drop_threshold": 0.04,
                "noise_threshold": 0.30,
                "uncertainty_threshold": 0.20,
                "safe_stop_threshold": 0.60,
                "quality_margin": float(max_f1_drop),
                "min_stop_step": 2,
            }
            metrics = evaluate_policy(v2, val_summaries, rows_by_id, pred_by_id, fallback)
            best = {**fallback, **{f"val_{k}": v for k, v in metrics.items()}, "selection_score": -1.0}
        policies.append(best)
    return policies


def aggregate_budget_rows(rows: list[dict[str, Any]], budget_key: str) -> dict[str, float]:
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    baseline_f1 = mean([row["baseline_f1"] for row in rows])
    gate_f1 = mean([row["budget_results"][budget_key]["f1"] for row in rows])
    baseline_steps = mean([row["baseline_steps"] for row in rows])
    gate_steps = mean([row["budget_results"][budget_key]["steps"] for row in rows])
    return {
        "n": len(rows),
        "baseline_mean_f1": baseline_f1,
        "gate_mean_f1": gate_f1,
        "f1_delta": gate_f1 - baseline_f1,
        "baseline_mean_steps": baseline_steps,
        "gate_mean_steps": gate_steps,
        "mean_effort_reduction": baseline_steps - gate_steps,
        "relative_effort_reduction": (baseline_steps - gate_steps) / baseline_steps if baseline_steps else 0.0,
        "answer_preservation_rate": mean([1.0 if row["budget_results"][budget_key]["f1"] >= row["baseline_f1"] else 0.0 for row in rows]),
        "wrong_stop_rate": mean([1.0 if row["budget_results"][budget_key]["f1"] < row["baseline_f1"] - 0.05 else 0.0 for row in rows]),
    }


def run_on_test(
    v0,
    v1,
    v2,
    qcal,
    heads: dict[str, Any],
    policies: list[dict[str, Any]],
    learned,
    flow,
    tokenizer,
    model,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    for idx, ex in enumerate(examples, start=1):
        ranked = v0.build_ranked_chunks(learned, ex, args.fixed_k)
        chunks = ranked["ranked_chunks"]
        retrieval_scores = ranked["retrieval_scores"]

        state_records = []
        for step in range(1, len(chunks) + 1):
            seen = chunks[:step]
            attn = v0.score_prefix_attention(flow, tokenizer, model, ex["question"], seen, args.max_length)
            attn_scores = [float(x) for x in attn["chunk_scores"]]
            features, debug = v0.feature_vector(ex["question"], seen, retrieval_scores[:step], attn_scores, args.fixed_k)
            pred = v2.predict_v2(v1, heads, features)
            state_records.append({"step": step, "attention_scores_seen": attn_scores, "feature_debug": debug, **pred})

        budget_stops: dict[str, int] = {}
        for policy in policies:
            key = f"{policy['risk_budget']:.3f}"
            stop = len(chunks)
            for record in state_records:
                if guarded_stop_condition(record, int(record["step"]), len(chunks), policy):
                    stop = int(record["step"])
                    break
            budget_stops[key] = stop

        attention_step, attention_scored = v2.attention_teacher_step(
            qcal,
            learned,
            flow,
            tokenizer,
            model,
            ex,
            args.fixed_k,
            args.max_length,
            args.teacher_alpha,
            args.teacher_threshold,
        )

        answer_by_step: dict[int, str] = {}
        needed_steps = set(budget_stops.values()) | {len(chunks), attention_step}
        for stop in sorted(needed_steps):
            answer_by_step[stop] = v0.generate(
                flow,
                tokenizer,
                model,
                ex["question"],
                chunks[:stop],
                args.max_new_tokens,
                args.max_length,
            )

        baseline_answer = answer_by_step[len(chunks)]
        baseline_f1 = learned.token_f1(baseline_answer, ex["answers"])
        budget_results = {}
        for policy in policies:
            key = f"{policy['risk_budget']:.3f}"
            stop = budget_stops[key]
            answer = answer_by_step[stop]
            budget_results[key] = {
                "steps": stop,
                "answer": answer,
                "f1": learned.token_f1(answer, ex["answers"]),
                "policy": policy,
            }

        attention_answer = answer_by_step[attention_step]
        row = {
            "question_id": ex["id"],
            "question": ex["question"],
            "answers": ex["answers"],
            "baseline_answer": baseline_answer,
            "baseline_f1": baseline_f1,
            "baseline_steps": len(chunks),
            "budget_results": budget_results,
            "attention_teacher_answer": attention_answer,
            "attention_teacher_f1": learned.token_f1(attention_answer, ex["answers"]),
            "attention_teacher_steps": attention_step,
            "attention_teacher_combined_scores_by_step": [float(x) for x in attention_scored["combined_scores"]],
            "attention_teacher_labels_by_step": [int(x) for x in attention_scored["labels"]],
            "state_records": state_records,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "phase": "test",
                    "idx": idx,
                    "question_id": ex["id"],
                    "baseline_f1": baseline_f1,
                    "attention_teacher_f1": row["attention_teacher_f1"],
                    "budget_steps": budget_stops,
                }
            ),
            flush=True,
        )
    return rows


def write_report(result: dict[str, Any], md_path: Path) -> None:
    lines = [
        "# EvidenceUseGate-v3 Guarded Distillation",
        "",
        f"Verdict: `{result['verdict']}`",
        "",
        "- Runtime gate: calibrated attention-flow teacher with learned safety veto",
        "- Efficiency teacher: calibrated Qwen attention-flow",
        "- Guardrail: predicted F1 drop, v1 stop probability, noise risk, uncertainty",
        f"- Train/val/test examples: {result['splits']['train_examples']} / {result['splits']['val_examples']} / {result['splits']['test_examples']}",
        f"- Train/val states: {result['splits']['train_states']} / {result['splits']['val_states']}",
        "",
        "## Pareto Sweep",
        "",
        "| Risk budget | Gate F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation | Verdict |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for point in result["pareto_curve"]:
        agg = point["test"]
        lines.append(
            f"| {point['risk_budget']:.3f} | {agg['gate_mean_f1']:.4f} | {agg['f1_delta']:+.4f} | "
            f"{agg['baseline_mean_steps']:.4f} -> {agg['gate_mean_steps']:.4f} | "
            f"{agg['relative_effort_reduction']:.2%} | {agg['wrong_stop_rate']:.2%} | "
            f"{agg['answer_preservation_rate']:.2%} | {point['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Same-Split Attention Teacher",
            "",
            f"- F1 delta: {result['attention_teacher_aggregate']['f1_delta']:+.4f}",
            f"- Effort reduction: {result['attention_teacher_aggregate']['relative_effort_reduction']:.2%}",
            f"- Wrong-stop rate: {result['attention_teacher_aggregate']['wrong_stop_rate']:.2%}",
            "",
            "## Interpretation",
            "",
            "v3 tests whether the learned gate can recover more of attention-flow's efficiency by treating safety as a veto instead of a mandatory low-risk condition. A target pass requires F1 delta >= -0.005, effort reduction >= 40%, and wrong-stop rate <= 5%.",
            "",
            f"- Peak CUDA allocation: {result['max_memory_allocated_mib']:.1f} MiB",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--learn-limit", type=int, default=1536)
    parser.add_argument("--train-examples", type=int, default=96)
    parser.add_argument("--val-examples", type=int, default=40)
    parser.add_argument("--eval-examples", type=int, default=100)
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument("--risk-budgets", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--max-f1-drop", type=float, default=0.005)
    parser.add_argument("--max-wrong-stop", type=float, default=0.05)
    parser.add_argument("--f1-epsilon", type=float, default=0.03)
    parser.add_argument("--min-good-f1", type=float, default=0.55)
    parser.add_argument("--min-counterfactual-drop", type=float, default=0.05)
    parser.add_argument("--min-alignment", type=float, default=0.10)
    parser.add_argument("--min-attention-concentration", type=float, default=0.70)
    parser.add_argument("--teacher-alpha", type=float, default=0.60)
    parser.add_argument("--teacher-threshold", type=float, default=0.51)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--output-tag", default="evidence_use_gate_v3_guarded_distill_100")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    risk_budgets = [float(x.strip()) for x in args.risk_budgets.split(",") if x.strip()]
    v2 = importlib.util.spec_from_file_location("evidence_use_gate_v2", V2_SCRIPT)
    if v2 is None or v2.loader is None:
        raise RuntimeError(f"Cannot load {V2_SCRIPT}")
    v2_module = importlib.util.module_from_spec(v2)
    v2.loader.exec_module(v2_module)

    gpu = v2_module.configure_gpu()
    v0 = v2_module.load_module("evidence_use_gate_v0", v2_module.V0_SCRIPT)
    v1 = v2_module.load_module("evidence_use_gate_v1", v2_module.V1_SCRIPT)
    qcal = v2_module.load_module("qwen_calibrated_flow", v2_module.QCAL_SCRIPT)
    learned = v0.load_module("learned_trust", v0.LEARNED_SCRIPT)
    flow = v0.load_module("qwen_flow", v0.FLOW_SCRIPT)

    examples = learned.load_answerable(args.learn_limit)
    rng = np.random.default_rng(20260610)
    examples = [examples[int(i)] for i in rng.permutation(len(examples))]
    train_examples = examples[: args.train_examples]
    val_examples = examples[args.train_examples : args.train_examples + args.val_examples]
    test_start = args.train_examples + args.val_examples
    test_examples = examples[test_start : test_start + args.eval_examples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
    )
    model.eval()

    train_rows, train_summaries, train_attention_teacher = v2_module.build_teacher_rows(
        v0, v1, qcal, learned, flow, tokenizer, model, train_examples, args, "teacher_train"
    )
    val_rows, val_summaries, val_attention_teacher = v2_module.build_teacher_rows(
        v0, v1, qcal, learned, flow, tokenizer, model, val_examples, args, "teacher_val"
    )

    heads, train_metrics = v2_module.fit_v2_heads(v1, train_rows)
    validation_metrics = v2_module.validation_head_metrics(v1, heads, val_rows)
    policies = choose_policies(
        v2_module,
        v1,
        heads,
        val_summaries,
        val_rows,
        risk_budgets,
        args.max_f1_drop,
        args.max_wrong_stop,
    )
    test_rows = run_on_test(
        v0,
        v1,
        v2_module,
        qcal,
        heads,
        policies,
        learned,
        flow,
        tokenizer,
        model,
        test_examples,
        args,
    )

    curve = []
    best_target_budget = None
    for policy in policies:
        key = f"{policy['risk_budget']:.3f}"
        agg = aggregate_budget_rows(test_rows, key)
        target_pass = (
            agg["f1_delta"] >= -args.max_f1_drop
            and agg["relative_effort_reduction"] >= 0.40
            and agg["wrong_stop_rate"] <= args.max_wrong_stop
        )
        curve.append(
            {
                "risk_budget": policy["risk_budget"],
                "policy": policy,
                "test": agg,
                "verdict": "PASS_TARGET" if target_pass else "MISS_TARGET",
            }
        )
        if target_pass and best_target_budget is None:
            best_target_budget = policy["risk_budget"]

    attention_teacher_aggregate = v2_module.aggregate_attention_teacher(test_rows)
    result = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "config": vars(args),
        "gpu": gpu,
        "splits": {
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "test_examples": len(test_examples),
            "train_states": len(train_rows),
            "val_states": len(val_rows),
        },
        "teacher": {
            "alpha": args.teacher_alpha,
            "threshold": args.teacher_threshold,
            "source": "calibrated Qwen attention-flow",
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "policies": policies,
        "pareto_curve": curve,
        "best_target_budget": best_target_budget,
        "attention_teacher_aggregate": attention_teacher_aggregate,
        "verdict": "PASS" if best_target_budget is not None else "NO_TARGET_PASS",
        "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 3),
        "train_attention_teacher_records": train_attention_teacher,
        "val_attention_teacher_records": val_attention_teacher,
        "test_rows": test_rows,
        "note": "v3 uses calibrated attention-flow as a high-efficiency teacher and learned safety heads as vetoes.",
    }
    json_path = OUT_DIR / f"{args.output_tag}.json"
    md_path = OUT_DIR / f"{args.output_tag}.md"
    commands_path = OUT_DIR / "commands.txt"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, md_path)
    commands_path.write_text(
        "CUDA_VISIBLE_DEVICES="
        + os.environ.get("CUDA_VISIBLE_DEVICES", "")
        + " python "
        + str(Path(__file__).resolve())
        + " "
        + " ".join(
            [
                f"--train-examples {args.train_examples}",
                f"--val-examples {args.val_examples}",
                f"--eval-examples {args.eval_examples}",
                f"--learn-limit {args.learn_limit}",
                f"--risk-budgets {args.risk_budgets}",
                f"--max-f1-drop {args.max_f1_drop}",
                f"--max-wrong-stop {args.max_wrong_stop}",
                f"--teacher-alpha {args.teacher_alpha}",
                f"--teacher-threshold {args.teacher_threshold}",
                f"--output-tag {args.output_tag}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "json": str(json_path),
                "report": str(md_path),
                "verdict": result["verdict"],
                "best_target_budget": best_target_budget,
                "pareto_curve": curve,
                "attention_teacher_aggregate": attention_teacher_aggregate,
                "validation_metrics": validation_metrics,
                "max_memory_allocated_mib": result["max_memory_allocated_mib"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
