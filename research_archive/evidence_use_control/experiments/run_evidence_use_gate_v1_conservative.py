#!/usr/bin/env python3
"""EvidenceUseGate-v1-conservative.

v0 showed that "learned gate" is not magic: it saved effort but over-stopped.
This version changes the controller in three ways:

1. Conservative teacher labels:
   stop is positive only when prefix F1 is close to top-k/full-prefix F1,
   the attention/relevance state is aligned, and the most-attended evidence is
   causally useful under a counterfactual drop probe.

2. Multi-head prediction:
   stop_prob, expected_f1_if_stop, continue_value, noise_risk, uncertainty.

3. Hard negatives:
   high-attention wrong prefixes, lexical/relevance bait with no
   counterfactual evidence use, and cases where the answer only appears later
   receive larger negative weight.

Runtime still uses cheap Qwen attention/retrieval state features only. The
expensive counterfactual probes are teacher labels, not inference-time calls.
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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = BUNDLE_ROOT / "experiments/out/evidence_use_gate_v1_conservative"
V0_SCRIPT = BUNDLE_ROOT / "experiments/run_evidence_use_gate_v0.py"
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
        raise RuntimeError("Run with exactly one visible GPU, e.g. CUDA_VISIBLE_DEVICES=1")
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


def conservative_relabel(
    v0_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    fixed_k: int,
    f1_epsilon: float,
    min_good_f1: float,
    min_counterfactual_drop: float,
    min_alignment: float,
    min_attention_concentration: float,
) -> list[dict[str, Any]]:
    """Replace v0's permissive labels with conservative multi-head targets."""
    f1_by_step = [float(x) for x in summary["f1_by_step"]]
    best_topk_f1 = float(f1_by_step[-1]) if f1_by_step else 0.0
    best_overall_f1 = float(max(f1_by_step)) if f1_by_step else 0.0
    relabeled = []
    for idx, row in enumerate(v0_rows):
        step = int(row["step"])
        f1 = float(row["f1"])
        future = f1_by_step[idx + 1 :] if idx + 1 < len(f1_by_step) else [f1]
        best_future = float(max(future)) if future else f1
        debug = row["feature_debug"]
        alignment = float(debug.get("attention_relevance_alignment", 0.0))
        attention_conc = float(debug.get("attention_concentration", 0.0))
        attention_margin = float(debug.get("attention_margin", 0.0))
        lexical = float(debug.get("max_lexical_overlap_seen", 0.0))
        recall = float(debug.get("max_question_recall_seen", 0.0))
        cf_drop = float(row["counterfactual_drop_delta"])

        quality_ok = (
            f1 >= min_good_f1
            and f1 >= best_topk_f1 - f1_epsilon
            and f1 >= best_overall_f1 - f1_epsilon
        )
        causal_ok = cf_drop >= min_counterfactual_drop
        alignment_ok = alignment >= min_alignment or attention_conc >= min_attention_concentration
        conservative_stop = int(quality_ok and causal_ok and alignment_ok)

        answer_later = best_future > f1 + f1_epsilon
        high_attention_wrong = attention_margin >= 0.45 and f1 < best_topk_f1 - f1_epsilon
        lexical_bait = (lexical >= 0.22 or recall >= 0.55) and (not causal_ok) and f1 < min_good_f1
        relevance_without_use = alignment >= min_alignment and (not causal_ok) and f1 < best_topk_f1 - f1_epsilon
        noise_label = int(high_attention_wrong or lexical_bait or relevance_without_use or answer_later)

        weight = 1.0
        if conservative_stop:
            weight = 1.35
        if noise_label:
            weight = 2.75
        if row["teacher_label"] == 1 and conservative_stop == 0:
            weight = max(weight, 2.0)
        if step == len(v0_rows) and conservative_stop:
            weight = max(weight, 1.75)

        out = dict(row)
        out.update(
            {
                "conservative_stop_label": conservative_stop,
                "noise_label": noise_label,
                "answer_later_label": int(answer_later),
                "hard_negative_reasons": {
                    "high_attention_wrong": bool(high_attention_wrong),
                    "lexical_bait": bool(lexical_bait),
                    "relevance_without_counterfactual_use": bool(relevance_without_use),
                    "answer_appears_later": bool(answer_later),
                },
                "expected_f1_target": f1,
                "continue_value_target": best_future,
                "sample_weight_v1": weight,
                "best_topk_f1": best_topk_f1,
            }
        )
        relabeled.append(out)
    return relabeled


def fit_heads(train_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    x = np.asarray([row["features"] for row in train_rows], dtype=np.float32)
    y_stop = np.asarray([row["conservative_stop_label"] for row in train_rows], dtype=np.int64)
    y_noise = np.asarray([row["noise_label"] for row in train_rows], dtype=np.int64)
    y_f1 = np.asarray([row["expected_f1_target"] for row in train_rows], dtype=np.float32)
    y_continue = np.asarray([row["continue_value_target"] for row in train_rows], dtype=np.float32)
    w = np.asarray([row["sample_weight_v1"] for row in train_rows], dtype=np.float32)

    stop_head = GradientBoostingClassifier(
        n_estimators=160,
        learning_rate=0.03,
        max_depth=2,
        random_state=17,
    )
    f1_head = GradientBoostingRegressor(
        n_estimators=180,
        learning_rate=0.025,
        max_depth=2,
        random_state=18,
    )
    continue_head = GradientBoostingRegressor(
        n_estimators=180,
        learning_rate=0.025,
        max_depth=2,
        random_state=19,
    )
    noise_head = GradientBoostingClassifier(
        n_estimators=160,
        learning_rate=0.03,
        max_depth=2,
        random_state=20,
    )
    uncertainty_forest = RandomForestClassifier(
        n_estimators=80,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=21,
        n_jobs=-1,
    )

    stop_head.fit(x, y_stop, sample_weight=w)
    f1_head.fit(x, y_f1, sample_weight=w)
    continue_head.fit(x, y_continue, sample_weight=w)
    noise_head.fit(x, y_noise, sample_weight=w)
    uncertainty_forest.fit(x, y_stop, sample_weight=w)

    train_stop_prob = stop_head.predict_proba(x)[:, 1] if len(set(y_stop.tolist())) > 1 else np.zeros(len(y_stop))
    train_noise_prob = noise_head.predict_proba(x)[:, 1] if len(set(y_noise.tolist())) > 1 else np.zeros(len(y_noise))
    metrics = {
        "train_states": int(len(train_rows)),
        "stop_positive_rate": float(np.mean(y_stop)) if len(y_stop) else 0.0,
        "noise_positive_rate": float(np.mean(y_noise)) if len(y_noise) else 0.0,
        "train_stop_auroc": float(roc_auc_score(y_stop, train_stop_prob)) if len(set(y_stop.tolist())) > 1 else None,
        "train_stop_auprc": float(average_precision_score(y_stop, train_stop_prob)) if len(set(y_stop.tolist())) > 1 else None,
        "train_noise_auroc": float(roc_auc_score(y_noise, train_noise_prob)) if len(set(y_noise.tolist())) > 1 else None,
        "train_noise_auprc": float(average_precision_score(y_noise, train_noise_prob)) if len(set(y_noise.tolist())) > 1 else None,
    }
    heads = {
        "stop": stop_head,
        "f1": f1_head,
        "continue": continue_head,
        "noise": noise_head,
        "uncertainty_forest": uncertainty_forest,
    }
    return heads, metrics


def predict_heads(heads: dict[str, Any], features: list[float]) -> dict[str, float]:
    x = np.asarray([features], dtype=np.float32)
    stop_prob = float(heads["stop"].predict_proba(x)[0, 1])
    expected_f1 = float(np.clip(heads["f1"].predict(x)[0], 0.0, 1.0))
    continue_value = float(np.clip(heads["continue"].predict(x)[0], 0.0, 1.0))
    noise_risk = float(heads["noise"].predict_proba(x)[0, 1])
    tree_probs = []
    forest = heads["uncertainty_forest"]
    for tree in forest.estimators_:
        proba = tree.predict_proba(x)[0]
        classes = list(tree.classes_)
        tree_probs.append(float(proba[classes.index(1)]) if 1 in classes else 0.0)
    uncertainty = float(np.std(tree_probs)) if tree_probs else abs(stop_prob - 0.5) * -1 + 0.5
    return {
        "stop_prob": stop_prob,
        "expected_f1_if_stop": expected_f1,
        "continue_value": continue_value,
        "noise_risk": noise_risk,
        "uncertainty": uncertainty,
    }


def evaluate_policy_on_teacher(
    heads: dict[str, Any],
    summaries: list[dict[str, Any]],
    rows_by_id: dict[str, list[dict[str, Any]]],
    stop_threshold: float,
    noise_threshold: float,
    uncertainty_threshold: float,
    utility_lambda: float,
) -> dict[str, float]:
    baseline_f1 = []
    gate_f1 = []
    baseline_steps = []
    gate_steps = []
    wrong_stops = []
    for summary in summaries:
        state_rows = rows_by_id[summary["question_id"]]
        stop = len(state_rows)
        state_preds = []
        for idx, row in enumerate(state_rows, start=1):
            pred = predict_heads(heads, row["features"])
            state_preds.append(pred)
            stop_utility = pred["expected_f1_if_stop"] - utility_lambda * idx
            continue_utility = pred["continue_value"] - utility_lambda * min(idx + 1, len(state_rows))
            if (
                pred["stop_prob"] >= stop_threshold
                and pred["noise_risk"] <= noise_threshold
                and pred["uncertainty"] <= uncertainty_threshold
                and stop_utility >= continue_utility
            ):
                stop = idx
                break
        full_f1 = float(summary["f1_by_step"][-1])
        selected_f1 = float(summary["f1_by_step"][stop - 1])
        baseline_f1.append(full_f1)
        gate_f1.append(selected_f1)
        baseline_steps.append(len(state_rows))
        gate_steps.append(stop)
        wrong_stops.append(1.0 if selected_f1 < full_f1 - 0.05 else 0.0)
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
    }


def choose_policy(
    heads: dict[str, Any],
    val_summaries: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    max_f1_drop: float,
    utility_lambda: float,
) -> dict[str, Any]:
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in val_rows:
        rows_by_id.setdefault(row["question_id"], []).append(row)
    pred_by_id: dict[str, list[dict[str, float]]] = {
        qid: [predict_heads(heads, row["features"]) for row in rows]
        for qid, rows in rows_by_id.items()
    }

    def eval_cached(
        stop_threshold: float,
        noise_threshold: float,
        uncertainty_threshold: float,
    ) -> dict[str, float]:
        baseline_f1 = []
        gate_f1 = []
        baseline_steps = []
        gate_steps = []
        wrong_stops = []
        for summary in val_summaries:
            qid = summary["question_id"]
            preds = pred_by_id[qid]
            stop = len(preds)
            for idx, pred in enumerate(preds, start=1):
                stop_utility = pred["expected_f1_if_stop"] - utility_lambda * idx
                continue_utility = pred["continue_value"] - utility_lambda * min(idx + 1, len(preds))
                if (
                    pred["stop_prob"] >= stop_threshold
                    and pred["noise_risk"] <= noise_threshold
                    and pred["uncertainty"] <= uncertainty_threshold
                    and stop_utility >= continue_utility
                ):
                    stop = idx
                    break
            full_f1 = float(summary["f1_by_step"][-1])
            selected_f1 = float(summary["f1_by_step"][stop - 1])
            baseline_f1.append(full_f1)
            gate_f1.append(selected_f1)
            baseline_steps.append(len(preds))
            gate_steps.append(stop)
            wrong_stops.append(1.0 if selected_f1 < full_f1 - 0.05 else 0.0)
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
        }

    best = None
    for stop_threshold in np.linspace(0.25, 0.95, 15):
        for noise_threshold in np.linspace(0.05, 0.55, 11):
            for uncertainty_threshold in np.linspace(0.10, 0.45, 8):
                metrics = eval_cached(float(stop_threshold), float(noise_threshold), float(uncertainty_threshold))
                if metrics["effort_reduction"] <= 0 or metrics["f1_delta"] < -max_f1_drop:
                    continue
                score = (
                    metrics["gate_f1"]
                    - utility_lambda * metrics["gate_steps"]
                    - 0.06 * metrics["wrong_stop_rate"]
                )
                candidate = {
                    "stop_threshold": float(stop_threshold),
                    "noise_threshold": float(noise_threshold),
                    "uncertainty_threshold": float(uncertainty_threshold),
                    "utility_lambda": utility_lambda,
                    "selection_score": float(score),
                    **{f"val_{k}": v for k, v in metrics.items()},
                }
                if best is None or candidate["selection_score"] > best["selection_score"]:
                    best = candidate
    if best is None:
        best = {
            "stop_threshold": 0.95,
            "noise_threshold": 0.05,
            "uncertainty_threshold": 0.10,
            "utility_lambda": utility_lambda,
            "selection_score": -1.0,
            "val_baseline_f1": 0.0,
            "val_gate_f1": 0.0,
            "val_f1_delta": -1.0,
            "val_baseline_steps": 0.0,
            "val_gate_steps": 0.0,
            "val_effort_reduction": 0.0,
            "val_relative_effort_reduction": 0.0,
            "val_wrong_stop_rate": 1.0,
        }
    return best


def val_head_metrics(heads: dict[str, Any], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    x_rows = [row["features"] for row in val_rows]
    y_stop = np.asarray([row["conservative_stop_label"] for row in val_rows], dtype=np.int64)
    y_noise = np.asarray([row["noise_label"] for row in val_rows], dtype=np.int64)
    stop_probs = np.asarray([predict_heads(heads, x)["stop_prob"] for x in x_rows], dtype=np.float32)
    noise_probs = np.asarray([predict_heads(heads, x)["noise_risk"] for x in x_rows], dtype=np.float32)
    return {
        "val_states": int(len(val_rows)),
        "val_stop_positive_rate": float(np.mean(y_stop)) if len(y_stop) else 0.0,
        "val_noise_positive_rate": float(np.mean(y_noise)) if len(y_noise) else 0.0,
        "val_stop_auroc": float(roc_auc_score(y_stop, stop_probs)) if len(set(y_stop.tolist())) > 1 else None,
        "val_stop_auprc": float(average_precision_score(y_stop, stop_probs)) if len(set(y_stop.tolist())) > 1 else None,
        "val_noise_auroc": float(roc_auc_score(y_noise, noise_probs)) if len(set(y_noise.tolist())) > 1 else None,
        "val_noise_auprc": float(average_precision_score(y_noise, noise_probs)) if len(set(y_noise.tolist())) > 1 else None,
    }


def run_on_test(
    v0,
    heads: dict[str, Any],
    policy: dict[str, Any],
    learned,
    flow,
    tokenizer,
    model,
    examples: list[dict[str, Any]],
    fixed_k: int,
    max_new_tokens: int,
    max_length: int,
) -> list[dict[str, Any]]:
    rows = []
    for idx, ex in enumerate(examples, start=1):
        ranked = v0.build_ranked_chunks(learned, ex, fixed_k)
        chunks = ranked["ranked_chunks"]
        retrieval_scores = ranked["retrieval_scores"]
        stop = len(chunks)
        state_records = []
        for step in range(1, len(chunks) + 1):
            seen = chunks[:step]
            attn = v0.score_prefix_attention(flow, tokenizer, model, ex["question"], seen, max_length)
            attn_scores = [float(x) for x in attn["chunk_scores"]]
            features, debug = v0.feature_vector(ex["question"], seen, retrieval_scores[:step], attn_scores, fixed_k)
            pred = predict_heads(heads, features)
            stop_utility = pred["expected_f1_if_stop"] - policy["utility_lambda"] * step
            continue_utility = pred["continue_value"] - policy["utility_lambda"] * min(step + 1, len(chunks))
            record = {
                "step": step,
                "attention_scores_seen": attn_scores,
                "feature_debug": debug,
                **pred,
                "stop_utility": stop_utility,
                "continue_utility": continue_utility,
            }
            state_records.append(record)
            if (
                pred["stop_prob"] >= policy["stop_threshold"]
                and pred["noise_risk"] <= policy["noise_threshold"]
                and pred["uncertainty"] <= policy["uncertainty_threshold"]
                and stop_utility >= continue_utility
            ):
                stop = step
                break
        baseline_answer = v0.generate(flow, tokenizer, model, ex["question"], chunks, max_new_tokens, max_length)
        gate_answer = v0.generate(flow, tokenizer, model, ex["question"], chunks[:stop], max_new_tokens, max_length)
        row = {
            "question_id": ex["id"],
            "question": ex["question"],
            "answers": ex["answers"],
            "baseline_answer": baseline_answer,
            "evidence_use_gate_v1_answer": gate_answer,
            "baseline_f1": learned.token_f1(baseline_answer, ex["answers"]),
            "evidence_use_gate_v1_f1": learned.token_f1(gate_answer, ex["answers"]),
            "baseline_steps": len(chunks),
            "evidence_use_gate_v1_steps": stop,
            "state_records": state_records,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "phase": "test",
                    "idx": idx,
                    "question_id": ex["id"],
                    "baseline_f1": row["baseline_f1"],
                    "evidence_use_gate_v1_f1": row["evidence_use_gate_v1_f1"],
                    "baseline_steps": row["baseline_steps"],
                    "evidence_use_gate_v1_steps": row["evidence_use_gate_v1_steps"],
                }
            ),
            flush=True,
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    baseline_f1 = mean([r["baseline_f1"] for r in rows])
    gate_f1 = mean([r["evidence_use_gate_v1_f1"] for r in rows])
    baseline_steps = mean([r["baseline_steps"] for r in rows])
    gate_steps = mean([r["evidence_use_gate_v1_steps"] for r in rows])
    return {
        "n": len(rows),
        "baseline_mean_f1": baseline_f1,
        "evidence_use_gate_v1_mean_f1": gate_f1,
        "f1_delta": gate_f1 - baseline_f1,
        "baseline_mean_steps": baseline_steps,
        "evidence_use_gate_v1_mean_steps": gate_steps,
        "mean_effort_reduction": baseline_steps - gate_steps,
        "relative_effort_reduction": (baseline_steps - gate_steps) / baseline_steps if baseline_steps else 0.0,
        "answer_preservation_rate": mean([1.0 if r["evidence_use_gate_v1_f1"] >= r["baseline_f1"] else 0.0 for r in rows]),
        "wrong_stop_rate": mean([1.0 if r["evidence_use_gate_v1_f1"] < r["baseline_f1"] - 0.05 else 0.0 for r in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--learn-limit", type=int, default=1536)
    parser.add_argument("--train-examples", type=int, default=96)
    parser.add_argument("--val-examples", type=int, default=40)
    parser.add_argument("--eval-examples", type=int, default=100)
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument("--max-f1-drop", type=float, default=0.005)
    parser.add_argument("--utility-lambda", type=float, default=0.012)
    parser.add_argument("--f1-epsilon", type=float, default=0.03)
    parser.add_argument("--min-good-f1", type=float, default=0.55)
    parser.add_argument("--min-counterfactual-drop", type=float, default=0.05)
    parser.add_argument("--min-alignment", type=float, default=0.10)
    parser.add_argument("--min-attention-concentration", type=float, default=0.70)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--output-tag", default="evidence_use_gate_v1_conservative_100")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu = configure_gpu()
    v0 = load_module("evidence_use_gate_v0", V0_SCRIPT)
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

    train_rows: list[dict[str, Any]] = []
    train_summaries = []
    for idx, ex in enumerate(train_examples, start=1):
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
        rows = conservative_relabel(
            rows,
            summary,
            args.fixed_k,
            args.f1_epsilon,
            args.min_good_f1,
            args.min_counterfactual_drop,
            args.min_alignment,
            args.min_attention_concentration,
        )
        train_rows.extend(rows)
        train_summaries.append(summary)
        print(json.dumps({"phase": "teacher_train", "idx": idx, "question_id": ex["id"], "states": len(rows)}), flush=True)

    val_rows: list[dict[str, Any]] = []
    val_summaries = []
    for idx, ex in enumerate(val_examples, start=1):
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
        rows = conservative_relabel(
            rows,
            summary,
            args.fixed_k,
            args.f1_epsilon,
            args.min_good_f1,
            args.min_counterfactual_drop,
            args.min_alignment,
            args.min_attention_concentration,
        )
        val_rows.extend(rows)
        val_summaries.append(summary)
        print(json.dumps({"phase": "teacher_val", "idx": idx, "question_id": ex["id"], "states": len(rows)}), flush=True)

    heads, train_metrics = fit_heads(train_rows)
    validation_metrics = val_head_metrics(heads, val_rows)
    policy = choose_policy(heads, val_summaries, val_rows, args.max_f1_drop, args.utility_lambda)
    test_rows = run_on_test(
        v0,
        heads,
        policy,
        learned,
        flow,
        tokenizer,
        model,
        test_examples,
        args.fixed_k,
        args.max_new_tokens,
        args.max_length,
    )
    agg = aggregate(test_rows)
    verdict = "PASS" if agg["f1_delta"] >= -args.max_f1_drop and agg["relative_effort_reduction"] >= 0.35 else "FAIL"
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
        "teacher_label_definition": {
            "stop_positive": "prefix F1 close to top-k/best F1, counterfactual drop positive, and attention/relevance alignment or concentration high",
            "hard_negatives": "high attention wrong chunk, lexical/relevance bait with no counterfactual use, answer appears later",
            "runtime_contribution_calls": 0,
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "policy": policy,
        "aggregate": agg,
        "verdict": verdict,
        "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 3),
        "test_rows": test_rows,
        "note": "Conservative multi-head learned EvidenceUseGate. Runtime uses cheap Qwen attention/retrieval features; counterfactual probes supervise teacher labels only.",
    }
    json_path = OUT_DIR / f"{args.output_tag}.json"
    md_path = OUT_DIR / f"{args.output_tag}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# EvidenceUseGate-v1 Conservative",
                "",
                f"Verdict: `{verdict}`",
                "",
                "- Runtime gate: multi-head learned controller",
                "- Heads: stop probability, expected F1 if stop, continue value, noise risk, uncertainty",
                "- Teacher: conservative answer sufficiency + counterfactual drop + attention/relevance alignment",
                "- Hard negatives: high-attention wrong, lexical/relevance bait, later-answer cases",
                f"- Train/val/test examples: {len(train_examples)} / {len(val_examples)} / {len(test_examples)}",
                f"- Train/val states: {len(train_rows)} / {len(val_rows)}",
                f"- Stop-head val AUROC: {validation_metrics['val_stop_auroc']:.4f}" if validation_metrics["val_stop_auroc"] is not None else "- Stop-head val AUROC: n/a",
                f"- Noise-head val AUROC: {validation_metrics['val_noise_auroc']:.4f}" if validation_metrics["val_noise_auroc"] is not None else "- Noise-head val AUROC: n/a",
                f"- Stop threshold: {policy['stop_threshold']:.3f}",
                f"- Noise threshold: {policy['noise_threshold']:.3f}",
                f"- Uncertainty threshold: {policy['uncertainty_threshold']:.3f}",
                f"- Baseline F1: {agg['baseline_mean_f1']:.4f}",
                f"- EvidenceUseGate-v1 F1: {agg['evidence_use_gate_v1_mean_f1']:.4f}",
                f"- F1 delta: {agg['f1_delta']:.4f}",
                f"- Steps: {agg['baseline_mean_steps']:.4f} -> {agg['evidence_use_gate_v1_mean_steps']:.4f}",
                f"- Effort reduction: {agg['relative_effort_reduction']:.2%}",
                f"- Answer preservation rate: {agg['answer_preservation_rate']:.2%}",
                f"- Wrong-stop rate: {agg['wrong_stop_rate']:.2%}",
                f"- Peak CUDA allocation: {result['max_memory_allocated_mib']:.1f} MiB",
                "",
                "This version tests whether a conservative learned evidence-use controller can preserve quality better than v0. It does not use contribution-flow at runtime.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path), "report": str(md_path), "verdict": verdict, "aggregate": agg, "policy": policy, "validation_metrics": validation_metrics, "max_memory_allocated_mib": result["max_memory_allocated_mib"]}, indent=2))


if __name__ == "__main__":
    main()
