#!/usr/bin/env python3
"""EvidenceUseGate-v2 Pareto policy.

v0 learned an aggressive stop policy and over-stopped. v1 made the teacher
conservative and fixed safety, but saved too little effort. v2 makes the stop
controller controllable: a risk budget selects an operating point on an
F1-effort curve.

Training signal:

- calibrated Qwen attention-flow provides the practical stop-action teacher;
- v1 conservative labels, noise labels, uncertainty, and predicted F1 drop
  provide safety constraints;
- validation chooses thresholds separately for each risk budget.

Runtime signal remains cheap Qwen attention/retrieval state features. The
counterfactual probes are used only for teacher labels during training.
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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = BUNDLE_ROOT / "experiments/out/evidence_use_gate_v2_pareto"
V0_SCRIPT = BUNDLE_ROOT / "experiments/run_evidence_use_gate_v0.py"
V1_SCRIPT = BUNDLE_ROOT / "experiments/run_evidence_use_gate_v1_conservative.py"
QCAL_SCRIPT = BUNDLE_ROOT / "experiments/run_qwen_calibrated_flow_trust_case.py"
DEFAULT_MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
GPU_MEMORY_CAP_GIB = 20.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_gpu() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or "," in visible:
        raise RuntimeError("Run with exactly one visible GPU, e.g. CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
    torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_MEMORY_CAP_GIB / total_gib), 0)
    return {
        "visible_cuda_devices": visible,
        "device_name": props.name,
        "memory_cap_gib": GPU_MEMORY_CAP_GIB,
        "total_memory_gib": round(total_gib, 3),
    }


def attention_teacher_step(
    qcal,
    learned,
    flow,
    tokenizer,
    model,
    ex: dict[str, Any],
    fixed_k: int,
    max_length: int,
    alpha: float,
    threshold: float,
) -> tuple[int, dict[str, Any]]:
    scored = qcal.score_example(learned, flow, tokenizer, model, ex, fixed_k, max_length)
    scores = qcal.combined_scores(scored, alpha)
    step = qcal.stop_step(scores, threshold)
    scored["combined_scores"] = scores
    scored["teacher_step"] = step
    return step, scored


def annotate_v2_rows(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    attention_stop_step: int,
    f1_epsilon: float,
) -> list[dict[str, Any]]:
    topk_f1 = float(summary["f1_by_step"][-1]) if summary["f1_by_step"] else 0.0
    best_f1 = float(max(summary["f1_by_step"])) if summary["f1_by_step"] else 0.0
    out_rows = []
    for row in rows:
        step = int(row["step"])
        f1 = float(row["f1"])
        teacher_stop_ok = int(step >= attention_stop_step)
        actual_drop = max(0.0, topk_f1 - f1)
        best_drop = max(0.0, best_f1 - f1)

        # Keep the imitation target as the calibrated attention-flow action,
        # but downweight unsafe positives so the safety heads can override them.
        weight = 1.0
        if teacher_stop_ok:
            weight = 1.25
        if row.get("conservative_stop_label", 0):
            weight = max(weight, 1.5)
        if row.get("noise_label", 0):
            weight = 2.25
        if teacher_stop_ok and row.get("noise_label", 0):
            weight = 0.75
        if teacher_stop_ok and actual_drop <= f1_epsilon:
            weight = max(weight, 1.75)

        new_row = dict(row)
        new_row.update(
            {
                "attention_teacher_stop_step": attention_stop_step,
                "attention_teacher_stop_label": teacher_stop_ok,
                "actual_f1_drop_vs_topk": actual_drop,
                "actual_f1_drop_vs_best": best_drop,
                "sample_weight_v2": weight,
            }
        )
        out_rows.append(new_row)
    return out_rows


def build_teacher_rows(
    v0,
    v1,
    qcal,
    learned,
    flow,
    tokenizer,
    model,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    phase: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    attention_teacher_records: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples, start=1):
        rows, summary = v0.teacher_rows_for_example(
            learned,
            flow,
            tokenizer,
            model,
            ex,
            args.fixed_k,
            args.max_new_tokens,
            args.max_length,
            args.f1_epsilon,
            args.min_good_f1,
            args.min_counterfactual_drop,
        )
        rows = v1.conservative_relabel(
            rows,
            summary,
            args.fixed_k,
            args.f1_epsilon,
            args.min_good_f1,
            args.min_counterfactual_drop,
            args.min_alignment,
            args.min_attention_concentration,
        )
        teacher_step, scored = attention_teacher_step(
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
        rows = annotate_v2_rows(rows, summary, teacher_step, args.f1_epsilon)
        all_rows.extend(rows)
        summaries.append(summary)
        attention_teacher_records.append(
            {
                "question_id": ex["id"],
                "teacher_step": teacher_step,
                "combined_scores_by_step": [float(x) for x in scored["combined_scores"]],
                "flow_scores_by_step": [float(x) for x in scored["flow_scores"]],
                "retrieval_scores_by_step": [float(x) for x in scored["retrieval_scores"]],
                "labels_by_step": [int(x) for x in scored["labels"]],
            }
        )
        print(
            json.dumps(
                {
                    "phase": phase,
                    "idx": idx,
                    "question_id": ex["id"],
                    "states": len(rows),
                    "attention_teacher_step": teacher_step,
                }
            ),
            flush=True,
        )
    return all_rows, summaries, attention_teacher_records


def fit_v2_heads(v1, train_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    safety_heads, safety_metrics = v1.fit_heads(train_rows)

    x = np.asarray([row["features"] for row in train_rows], dtype=np.float32)
    y_teacher = np.asarray([row["attention_teacher_stop_label"] for row in train_rows], dtype=np.int64)
    y_drop = np.asarray([row["actual_f1_drop_vs_topk"] for row in train_rows], dtype=np.float32)
    weights = np.asarray([row["sample_weight_v2"] for row in train_rows], dtype=np.float32)

    imitation_head = GradientBoostingClassifier(
        n_estimators=180,
        learning_rate=0.035,
        max_depth=2,
        random_state=31,
    )
    drop_head = GradientBoostingRegressor(
        n_estimators=180,
        learning_rate=0.03,
        max_depth=2,
        random_state=32,
    )
    imitation_head.fit(x, y_teacher, sample_weight=weights)
    drop_head.fit(x, y_drop, sample_weight=weights)

    teacher_prob = imitation_head.predict_proba(x)[:, 1] if len(set(y_teacher.tolist())) > 1 else np.zeros(len(y_teacher))
    distill_metrics = {
        "train_attention_teacher_positive_rate": float(np.mean(y_teacher)) if len(y_teacher) else 0.0,
        "train_attention_teacher_auroc": float(roc_auc_score(y_teacher, teacher_prob)) if len(set(y_teacher.tolist())) > 1 else None,
        "train_attention_teacher_auprc": float(average_precision_score(y_teacher, teacher_prob)) if len(set(y_teacher.tolist())) > 1 else None,
        "train_mean_actual_f1_drop": float(np.mean(y_drop)) if len(y_drop) else 0.0,
    }
    heads = {
        **safety_heads,
        "attention_teacher": imitation_head,
        "drop": drop_head,
    }
    return heads, {"safety": safety_metrics, "distill": distill_metrics}


def predict_v2(v1, heads: dict[str, Any], features: list[float]) -> dict[str, float]:
    base = v1.predict_heads(heads, features)
    x = np.asarray([features], dtype=np.float32)
    base["attention_teacher_stop_prob"] = float(heads["attention_teacher"].predict_proba(x)[0, 1])
    base["predicted_f1_drop"] = float(np.clip(heads["drop"].predict(x)[0], 0.0, 1.0))
    base["expected_f1_drop"] = float(max(0.0, base["continue_value"] - base["expected_f1_if_stop"]))
    return base


def validation_head_metrics(v1, heads: dict[str, Any], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    safety = v1.val_head_metrics(heads, val_rows)
    x_rows = [row["features"] for row in val_rows]
    y_teacher = np.asarray([row["attention_teacher_stop_label"] for row in val_rows], dtype=np.int64)
    teacher_probs = np.asarray([predict_v2(v1, heads, x)["attention_teacher_stop_prob"] for x in x_rows], dtype=np.float32)
    drops = np.asarray([row["actual_f1_drop_vs_topk"] for row in val_rows], dtype=np.float32)
    pred_drops = np.asarray([predict_v2(v1, heads, x)["predicted_f1_drop"] for x in x_rows], dtype=np.float32)
    drop_mae = float(np.mean(np.abs(drops - pred_drops))) if len(drops) else 0.0
    return {
        **safety,
        "val_attention_teacher_positive_rate": float(np.mean(y_teacher)) if len(y_teacher) else 0.0,
        "val_attention_teacher_auroc": float(roc_auc_score(y_teacher, teacher_probs)) if len(set(y_teacher.tolist())) > 1 else None,
        "val_attention_teacher_auprc": float(average_precision_score(y_teacher, teacher_probs)) if len(set(y_teacher.tolist())) > 1 else None,
        "val_drop_mae": drop_mae,
        "val_mean_actual_f1_drop": float(np.mean(drops)) if len(drops) else 0.0,
    }


def stop_condition(pred: dict[str, float], step: int, total_steps: int, policy: dict[str, Any]) -> bool:
    stop_utility = pred["expected_f1_if_stop"] - policy["utility_lambda"] * step
    continue_utility = pred["continue_value"] - policy["utility_lambda"] * min(step + 1, total_steps)
    return bool(
        pred["attention_teacher_stop_prob"] >= policy["teacher_stop_threshold"]
        and pred["predicted_f1_drop"] <= policy["drop_threshold"]
        and pred["noise_risk"] <= policy["noise_threshold"]
        and pred["uncertainty"] <= policy["uncertainty_threshold"]
        and stop_utility >= continue_utility - policy["risk_budget"]
    )


def evaluate_policy_on_val(
    v1,
    heads: dict[str, Any],
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
            if stop_condition(pred, idx, len(rows), policy):
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
    v1,
    heads: dict[str, Any],
    val_summaries: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    risk_budgets: list[float],
    utility_lambda: float,
) -> list[dict[str, Any]]:
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in val_rows:
        rows_by_id.setdefault(row["question_id"], []).append(row)
    pred_by_id = {
        qid: [predict_v2(v1, heads, row["features"]) for row in rows]
        for qid, rows in rows_by_id.items()
    }

    policies = []
    for budget in risk_budgets:
        best = None
        for teacher_threshold in np.linspace(0.10, 0.90, 17):
            for drop_threshold in np.linspace(max(0.0, budget), min(0.20, budget + 0.12), 13):
                for noise_threshold in np.linspace(0.10, 0.80, 15):
                    for uncertainty_threshold in np.linspace(0.10, 0.50, 9):
                        policy = {
                            "risk_budget": float(budget),
                            "teacher_stop_threshold": float(teacher_threshold),
                            "drop_threshold": float(drop_threshold),
                            "noise_threshold": float(noise_threshold),
                            "uncertainty_threshold": float(uncertainty_threshold),
                            "utility_lambda": float(utility_lambda),
                        }
                        metrics = evaluate_policy_on_val(v1, heads, val_summaries, rows_by_id, pred_by_id, policy)
                        if metrics["effort_reduction"] <= 0:
                            continue
                        if metrics["f1_delta"] < -float(budget):
                            continue
                        score = (
                            metrics["relative_effort_reduction"]
                            + 0.25 * metrics["gate_f1"]
                            - 0.35 * metrics["wrong_stop_rate"]
                            - 0.05 * abs(metrics["f1_delta"])
                        )
                        candidate = {**policy, **{f"val_{k}": v for k, v in metrics.items()}, "selection_score": float(score)}
                        if best is None or candidate["selection_score"] > best["selection_score"]:
                            best = candidate
        if best is None:
            fallback = {
                "risk_budget": float(budget),
                "teacher_stop_threshold": 0.95,
                "drop_threshold": float(budget),
                "noise_threshold": 0.10,
                "uncertainty_threshold": 0.10,
                "utility_lambda": float(utility_lambda),
            }
            metrics = evaluate_policy_on_val(v1, heads, val_summaries, rows_by_id, pred_by_id, fallback)
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


def aggregate_attention_teacher(rows: list[dict[str, Any]]) -> dict[str, float]:
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    baseline_f1 = mean([row["baseline_f1"] for row in rows])
    teacher_f1 = mean([row["attention_teacher_f1"] for row in rows])
    baseline_steps = mean([row["baseline_steps"] for row in rows])
    teacher_steps = mean([row["attention_teacher_steps"] for row in rows])
    return {
        "n": len(rows),
        "baseline_mean_f1": baseline_f1,
        "attention_teacher_mean_f1": teacher_f1,
        "f1_delta": teacher_f1 - baseline_f1,
        "baseline_mean_steps": baseline_steps,
        "attention_teacher_mean_steps": teacher_steps,
        "mean_effort_reduction": baseline_steps - teacher_steps,
        "relative_effort_reduction": (baseline_steps - teacher_steps) / baseline_steps if baseline_steps else 0.0,
        "answer_preservation_rate": mean([1.0 if row["attention_teacher_f1"] >= row["baseline_f1"] else 0.0 for row in rows]),
        "wrong_stop_rate": mean([1.0 if row["attention_teacher_f1"] < row["baseline_f1"] - 0.05 else 0.0 for row in rows]),
    }


def run_on_test(
    v0,
    v1,
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
            pred = predict_v2(v1, heads, features)
            state_records.append(
                {
                    "step": step,
                    "attention_scores_seen": attn_scores,
                    "feature_debug": debug,
                    **pred,
                }
            )

        budget_stops: dict[str, int] = {}
        for policy in policies:
            key = f"{policy['risk_budget']:.3f}"
            stop = len(chunks)
            for record in state_records:
                if stop_condition(record, int(record["step"]), len(chunks), policy):
                    stop = int(record["step"])
                    break
            budget_stops[key] = stop

        attention_step, attention_scored = attention_teacher_step(
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
    curve = result["pareto_curve"]
    best_target = result["best_target_budget"]
    lines = [
        "# EvidenceUseGate-v2 Pareto",
        "",
        f"Verdict: `{result['verdict']}`",
        "",
        "- Runtime gate: controllable learned evidence-use policy",
        "- Practical teacher: calibrated Qwen attention-flow",
        "- Safety constraints: v1 conservative evidence labels, noise risk, uncertainty, predicted F1 drop",
        f"- Train/val/test examples: {result['splits']['train_examples']} / {result['splits']['val_examples']} / {result['splits']['test_examples']}",
        f"- Train/val states: {result['splits']['train_states']} / {result['splits']['val_states']}",
        f"- Attention teacher alpha/threshold: {result['teacher']['alpha']:.2f} / {result['teacher']['threshold']:.3f}",
        "",
        "## Same-Split Attention Teacher",
        "",
        f"- Baseline F1: {result['attention_teacher_aggregate']['baseline_mean_f1']:.4f}",
        f"- Attention-teacher F1: {result['attention_teacher_aggregate']['attention_teacher_mean_f1']:.4f}",
        f"- F1 delta: {result['attention_teacher_aggregate']['f1_delta']:.4f}",
        f"- Steps: {result['attention_teacher_aggregate']['baseline_mean_steps']:.4f} -> {result['attention_teacher_aggregate']['attention_teacher_mean_steps']:.4f}",
        f"- Effort reduction: {result['attention_teacher_aggregate']['relative_effort_reduction']:.2%}",
        "",
        "## Pareto Sweep",
        "",
        "| Risk budget | Gate F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation | Verdict |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for point in curve:
        agg = point["test"]
        verdict = "target" if point["risk_budget"] == best_target else point["verdict"]
        lines.append(
            f"| {point['risk_budget']:.3f} | {agg['gate_mean_f1']:.4f} | {agg['f1_delta']:+.4f} | "
            f"{agg['baseline_mean_steps']:.4f} -> {agg['gate_mean_steps']:.4f} | "
            f"{agg['relative_effort_reduction']:.2%} | {agg['wrong_stop_rate']:.2%} | "
            f"{agg['answer_preservation_rate']:.2%} | {verdict} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "v2 tests the intended next step: learning a controllable safety-efficiency curve rather than one fixed threshold. A point passes the conservative target if F1 delta is at least -0.005 and effort reduction is at least 35%.",
            "",
            "No noisy-distractor evaluation is included in this run; wrong-stop here is measured against the clean held-out top-k baseline.",
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
    parser.add_argument("--risk-budgets", default="0.0,0.005,0.01,0.02")
    parser.add_argument("--utility-lambda", type=float, default=0.012)
    parser.add_argument("--f1-epsilon", type=float, default=0.03)
    parser.add_argument("--min-good-f1", type=float, default=0.55)
    parser.add_argument("--min-counterfactual-drop", type=float, default=0.05)
    parser.add_argument("--min-alignment", type=float, default=0.10)
    parser.add_argument("--min-attention-concentration", type=float, default=0.70)
    parser.add_argument("--teacher-alpha", type=float, default=0.60)
    parser.add_argument("--teacher-threshold", type=float, default=0.51)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--output-tag", default="evidence_use_gate_v2_pareto_100")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    risk_budgets = [float(x.strip()) for x in args.risk_budgets.split(",") if x.strip()]
    gpu = configure_gpu()
    v0 = load_module("evidence_use_gate_v0", V0_SCRIPT)
    v1 = load_module("evidence_use_gate_v1", V1_SCRIPT)
    qcal = load_module("qwen_calibrated_flow", QCAL_SCRIPT)
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

    train_rows, train_summaries, train_attention_teacher = build_teacher_rows(
        v0, v1, qcal, learned, flow, tokenizer, model, train_examples, args, "teacher_train"
    )
    val_rows, val_summaries, val_attention_teacher = build_teacher_rows(
        v0, v1, qcal, learned, flow, tokenizer, model, val_examples, args, "teacher_val"
    )

    heads, train_metrics = fit_v2_heads(v1, train_rows)
    validation_metrics = validation_head_metrics(v1, heads, val_rows)
    policies = choose_policies(v1, heads, val_summaries, val_rows, risk_budgets, args.utility_lambda)
    test_rows = run_on_test(
        v0,
        v1,
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
        target_pass = agg["f1_delta"] >= -0.005 and agg["relative_effort_reduction"] >= 0.35
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

    attention_teacher_aggregate = aggregate_attention_teacher(test_rows)
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
        "note": "v2 learns a risk-budget-controlled Pareto policy by distilling calibrated Qwen attention-flow with v1 safety constraints. No noisy-distractor evaluation is included.",
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
                f"--utility-lambda {args.utility_lambda}",
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
