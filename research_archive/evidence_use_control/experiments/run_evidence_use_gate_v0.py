#!/usr/bin/env python3
"""EvidenceUseGate-v0: learn when Qwen attention is trustworthy.

This is the next step after calibrated attention-flow. The runtime gate is not
a hand-written blend of attention/relevance/contribution. Instead, it learns a
stop policy from richer per-step state features.

Teacher labels are generated with expensive answer-sufficiency and
counterfactual-drop probes:

- Is the answer with chunks[1:t] already as good as the best prefix/full answer?
- Does removing the highest-attended chunk hurt the answer?

At test time the gate uses only cheap runtime features: retrieval scores,
Qwen attention-flow behavior, chunk position/length, lexical overlap, entropy,
concentration, and attention-relevance alignment.

This v0 uses counterfactual teachers. The local 71120 contribution code is
LLaMA/Gemma-specific, so it is kept as a separate mechanistic experiment rather
than mixed into the Qwen teacher labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = BUNDLE_ROOT / "experiments/out/evidence_use_gate_v0"
LEARNED_SCRIPT = BUNDLE_ROOT / "experiments/run_learned_trust_benchmark.py"
FLOW_SCRIPT = BUNDLE_ROOT / "experiments/run_qwen_whitebox_flow_validation.py"
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


def toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def lexical_features(question: str, chunk: str) -> tuple[float, float]:
    q = set(toks(question))
    c = set(toks(chunk))
    if not q:
        return 0.0, 0.0
    overlap = len(q & c) / max(1, len(q | c))
    recall = len(q & c) / max(1, len(q))
    return float(overlap), float(recall)


def safe_entropy(values: list[float]) -> float:
    arr = np.asarray([max(0.0, v) for v in values], dtype=np.float64)
    total = float(arr.sum())
    if total <= 1e-12 or len(arr) <= 1:
        return 0.0
    p = arr / total
    ent = -float(np.sum([x * math.log(x + 1e-12) for x in p]))
    return ent / math.log(len(arr))


def minmax(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return []
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [float((x - lo) / (hi - lo)) for x in arr]


def build_ranked_chunks(learned, ex: dict[str, Any], fixed_k: int) -> dict[str, Any]:
    chunks = learned.split_sentences(ex["context"])
    order, retrieval_scores = learned.rank_chunks(ex["question"], chunks)
    order = order[: min(fixed_k, len(order))]
    return {
        "chunks": chunks,
        "indices": order,
        "ranked_chunks": [chunks[i] for i in order],
        "retrieval_scores": [float(retrieval_scores[i]) for i in order],
    }


def answer_prompt(flow, question: str, chunks: list[str]) -> str:
    prompt, _ = flow.make_prompt(question, chunks)
    return prompt


def generate(flow, tokenizer, model, question: str, chunks: list[str], max_new_tokens: int, max_length: int) -> str:
    prompt = answer_prompt(flow, question, chunks)
    return flow.generate_answer(tokenizer, model, prompt, max_new_tokens, max_length)


def feature_vector(
    question: str,
    seen_chunks: list[str],
    retrieval_scores_seen: list[float],
    attention_scores_seen: list[float],
    fixed_k: int,
) -> tuple[list[float], dict[str, Any]]:
    step = len(seen_chunks)
    retrieval_norm = minmax(retrieval_scores_seen)
    attn_norm = minmax(attention_scores_seen)
    current_retrieval = retrieval_norm[-1] if retrieval_norm else 0.0
    current_attention = attn_norm[-1] if attn_norm else 0.0
    max_retrieval = max(retrieval_norm) if retrieval_norm else 0.0
    max_attention = max(attn_norm) if attn_norm else 0.0
    mean_retrieval = float(np.mean(retrieval_norm)) if retrieval_norm else 0.0
    mean_attention = float(np.mean(attn_norm)) if attn_norm else 0.0
    sorted_retrieval = sorted(retrieval_norm, reverse=True)
    sorted_attention = sorted(attn_norm, reverse=True)
    retrieval_margin = sorted_retrieval[0] - sorted_retrieval[1] if len(sorted_retrieval) > 1 else sorted_retrieval[0] if sorted_retrieval else 0.0
    attention_margin = sorted_attention[0] - sorted_attention[1] if len(sorted_attention) > 1 else sorted_attention[0] if sorted_attention else 0.0
    retrieval_entropy = safe_entropy(retrieval_norm)
    attention_entropy = safe_entropy(attn_norm)
    retrieval_concentration = max_retrieval / (sum(max(0.0, x) for x in retrieval_norm) + 1e-9)
    attention_concentration = max_attention / (sum(max(0.0, x) for x in attn_norm) + 1e-9)
    if len(retrieval_norm) > 1 and np.std(retrieval_norm) > 1e-9 and np.std(attn_norm) > 1e-9:
        alignment = float(np.corrcoef(retrieval_norm, attn_norm)[0, 1])
    else:
        alignment = 0.0
    current_overlap, current_recall = lexical_features(question, seen_chunks[-1])
    max_overlap = 0.0
    max_recall = 0.0
    for chunk in seen_chunks:
        overlap, recall = lexical_features(question, chunk)
        max_overlap = max(max_overlap, overlap)
        max_recall = max(max_recall, recall)
    chunk_lengths = [len(toks(chunk)) for chunk in seen_chunks]
    current_len = chunk_lengths[-1] if chunk_lengths else 0
    cumulative_len = sum(chunk_lengths)

    features = [
        step / fixed_k,
        1.0 / step,
        current_retrieval,
        max_retrieval,
        mean_retrieval,
        retrieval_margin,
        retrieval_entropy,
        retrieval_concentration,
        current_attention,
        max_attention,
        mean_attention,
        attention_margin,
        attention_entropy,
        attention_concentration,
        alignment,
        current_overlap,
        current_recall,
        max_overlap,
        max_recall,
        current_len / 128.0,
        cumulative_len / 512.0,
    ]
    names = [
        "step_frac",
        "inverse_step",
        "current_retrieval",
        "max_retrieval",
        "mean_retrieval",
        "retrieval_margin",
        "retrieval_entropy",
        "retrieval_concentration",
        "current_attention",
        "max_attention",
        "mean_attention",
        "attention_margin",
        "attention_entropy",
        "attention_concentration",
        "attention_relevance_alignment",
        "current_lexical_overlap",
        "current_question_recall",
        "max_lexical_overlap_seen",
        "max_question_recall_seen",
        "current_chunk_len_128",
        "cumulative_chunk_len_512",
    ]
    debug = {name: float(value) for name, value in zip(names, features)}
    return features, debug


def score_prefix_attention(flow, tokenizer, model, question: str, chunks: list[str], max_length: int) -> dict[str, Any]:
    return flow.attention_flow_scores(tokenizer, model, question, chunks, max_length)


def teacher_rows_for_example(
    learned,
    flow,
    tokenizer,
    model,
    ex: dict[str, Any],
    fixed_k: int,
    max_new_tokens: int,
    max_length: int,
    f1_epsilon: float,
    min_good_f1: float,
    min_counterfactual_drop: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranked = build_ranked_chunks(learned, ex, fixed_k)
    chunks = ranked["ranked_chunks"]
    retrieval_scores = ranked["retrieval_scores"]
    answers_by_step = []
    f1_by_step = []
    attention_by_step = []
    features_by_step = []
    feature_debug_by_step = []
    counterfactual_drop_by_step = []
    drop_answer_by_step = []

    for step in range(1, len(chunks) + 1):
        seen = chunks[:step]
        attn = score_prefix_attention(flow, tokenizer, model, ex["question"], seen, max_length)
        attention_scores = [float(x) for x in attn["chunk_scores"]]
        vec, debug = feature_vector(ex["question"], seen, retrieval_scores[:step], attention_scores, fixed_k)
        answer = generate(flow, tokenizer, model, ex["question"], seen, max_new_tokens, max_length)
        f1 = learned.token_f1(answer, ex["answers"])

        top_idx = int(np.argmax(attention_scores)) if attention_scores else 0
        dropped = [chunk for idx, chunk in enumerate(seen) if idx != top_idx]
        drop_answer = generate(flow, tokenizer, model, ex["question"], dropped, max_new_tokens, max_length)
        drop_f1 = learned.token_f1(drop_answer, ex["answers"])
        cf_delta = max(0.0, float(f1 - drop_f1))

        attention_by_step.append(attention_scores)
        features_by_step.append(vec)
        feature_debug_by_step.append(debug)
        answers_by_step.append(answer)
        f1_by_step.append(float(f1))
        counterfactual_drop_by_step.append(cf_delta)
        drop_answer_by_step.append(drop_answer)

    best_future_f1_by_step = []
    for idx in range(len(f1_by_step)):
        best_future_f1_by_step.append(max(f1_by_step[idx:]))
    best_overall_f1 = max(f1_by_step) if f1_by_step else 0.0
    rows = []
    for idx, vec in enumerate(features_by_step):
        answer_sufficient = (
            f1_by_step[idx] >= min_good_f1
            and f1_by_step[idx] >= best_overall_f1 - f1_epsilon
            and f1_by_step[idx] >= best_future_f1_by_step[idx] - f1_epsilon
        )
        evidence_used = counterfactual_drop_by_step[idx] >= min_counterfactual_drop
        # Counterfactual drop is a strong teacher signal, but for one-chunk
        # contexts a correct answer can remain stable due parametric memory.
        # We keep those as lower-confidence positives via sample_weight.
        label = 1 if answer_sufficient and (evidence_used or idx == 0) else 0
        weight = 1.0
        if answer_sufficient and not evidence_used:
            weight = 0.55
        if not answer_sufficient and counterfactual_drop_by_step[idx] > 0:
            weight = 1.25
        rows.append(
            {
                "question_id": ex["id"],
                "step": idx + 1,
                "features": vec,
                "feature_debug": feature_debug_by_step[idx],
                "teacher_label": label,
                "teacher_weight": weight,
                "answer": answers_by_step[idx],
                "f1": f1_by_step[idx],
                "best_overall_f1": best_overall_f1,
                "counterfactual_drop_delta": counterfactual_drop_by_step[idx],
                "counterfactual_drop_answer": drop_answer_by_step[idx],
                "attention_scores_seen": attention_by_step[idx],
            }
        )
    summary = {
        "question_id": ex["id"],
        "question": ex["question"],
        "answers": ex["answers"],
        "ranked_chunks": chunks,
        "retrieval_scores": retrieval_scores,
        "answers_by_step": answers_by_step,
        "f1_by_step": f1_by_step,
        "counterfactual_drop_by_step": counterfactual_drop_by_step,
        "best_overall_f1": best_overall_f1,
    }
    return rows, summary


def fit_gate(train_rows: list[dict[str, Any]], model_type: str):
    x = np.asarray([row["features"] for row in train_rows], dtype=np.float32)
    y = np.asarray([row["teacher_label"] for row in train_rows], dtype=np.int64)
    w = np.asarray([row["teacher_weight"] for row in train_rows], dtype=np.float32)
    if model_type == "logistic":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=2,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    try:
        clf.fit(x, y, **({"sample_weight": w} if model_type != "logistic" else {"logisticregression__sample_weight": w}))
    except TypeError:
        clf.fit(x, y)
    probs = clf.predict_proba(x)[:, 1]
    metrics = {
        "n_states": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else 0.0,
        "train_auroc": float(roc_auc_score(y, probs)) if len(set(y.tolist())) > 1 else None,
        "train_auprc": float(average_precision_score(y, probs)) if len(set(y.tolist())) > 1 else None,
    }
    return clf, metrics


def gate_prob(clf, features: list[float]) -> float:
    return float(clf.predict_proba(np.asarray([features], dtype=np.float32))[0, 1])


def choose_threshold(
    clf,
    val_summaries: list[dict[str, Any]],
    val_rows_by_id: dict[str, list[dict[str, Any]]],
    max_f1_drop: float,
    utility_lambda: float,
) -> dict[str, Any]:
    best = None
    for threshold in np.linspace(0.05, 0.95, 91):
        selected_f1 = []
        selected_steps = []
        baseline_f1 = []
        baseline_steps = []
        selected_probs = []
        for summary in val_summaries:
            state_rows = val_rows_by_id[summary["question_id"]]
            probs = [gate_prob(clf, row["features"]) for row in state_rows]
            stop = len(state_rows)
            for idx, prob in enumerate(probs, start=1):
                if prob >= threshold:
                    stop = idx
                    break
            selected_f1.append(summary["f1_by_step"][stop - 1])
            selected_steps.append(stop)
            baseline_f1.append(summary["f1_by_step"][-1])
            baseline_steps.append(len(summary["f1_by_step"]))
            selected_probs.append(probs[stop - 1])
        mean_selected_f1 = float(np.mean(selected_f1))
        mean_baseline_f1 = float(np.mean(baseline_f1))
        mean_selected_steps = float(np.mean(selected_steps))
        mean_baseline_steps = float(np.mean(baseline_steps))
        f1_delta = mean_selected_f1 - mean_baseline_f1
        effort = mean_baseline_steps - mean_selected_steps
        utility = mean_selected_f1 - utility_lambda * mean_selected_steps
        candidate = {
            "threshold": float(threshold),
            "val_baseline_f1": mean_baseline_f1,
            "val_gate_f1": mean_selected_f1,
            "val_f1_delta": f1_delta,
            "val_baseline_steps": mean_baseline_steps,
            "val_gate_steps": mean_selected_steps,
            "val_effort_reduction": effort,
            "val_relative_effort_reduction": effort / mean_baseline_steps if mean_baseline_steps else 0.0,
            "val_utility": utility,
            "mean_selected_stop_probability": float(np.mean(selected_probs)),
        }
        if effort <= 0 or f1_delta < -max_f1_drop:
            continue
        if best is None or candidate["val_utility"] > best["val_utility"]:
            best = candidate
    if best is None:
        best = {
            "threshold": 0.95,
            "val_baseline_f1": 0.0,
            "val_gate_f1": 0.0,
            "val_f1_delta": -1.0,
            "val_baseline_steps": 0.0,
            "val_gate_steps": 0.0,
            "val_effort_reduction": 0.0,
            "val_relative_effort_reduction": 0.0,
            "val_utility": -1.0,
            "mean_selected_stop_probability": 0.0,
        }
    return best


def run_gate_on_test(
    learned,
    flow,
    tokenizer,
    model,
    clf,
    threshold: float,
    examples: list[dict[str, Any]],
    fixed_k: int,
    max_new_tokens: int,
    max_length: int,
) -> list[dict[str, Any]]:
    rows = []
    for idx, ex in enumerate(examples, start=1):
        ranked = build_ranked_chunks(learned, ex, fixed_k)
        chunks = ranked["ranked_chunks"]
        retrieval_scores = ranked["retrieval_scores"]
        stop = len(chunks)
        stop_prob = 0.0
        state_records = []
        for step in range(1, len(chunks) + 1):
            seen = chunks[:step]
            attn = score_prefix_attention(flow, tokenizer, model, ex["question"], seen, max_length)
            attn_scores = [float(x) for x in attn["chunk_scores"]]
            vec, debug = feature_vector(ex["question"], seen, retrieval_scores[:step], attn_scores, fixed_k)
            prob = gate_prob(clf, vec)
            state_records.append(
                {
                    "step": step,
                    "stop_probability": prob,
                    "attention_scores_seen": attn_scores,
                    "feature_debug": debug,
                }
            )
            if prob >= threshold:
                stop = step
                stop_prob = prob
                break
        if stop_prob == 0.0 and state_records:
            stop_prob = state_records[-1]["stop_probability"]
        baseline_answer = generate(flow, tokenizer, model, ex["question"], chunks, max_new_tokens, max_length)
        gate_answer = generate(flow, tokenizer, model, ex["question"], chunks[:stop], max_new_tokens, max_length)
        row = {
            "question_id": ex["id"],
            "question": ex["question"],
            "answers": ex["answers"],
            "baseline_answer": baseline_answer,
            "evidence_use_gate_answer": gate_answer,
            "baseline_f1": learned.token_f1(baseline_answer, ex["answers"]),
            "evidence_use_gate_f1": learned.token_f1(gate_answer, ex["answers"]),
            "baseline_steps": len(chunks),
            "evidence_use_gate_steps": stop,
            "stop_probability": stop_prob,
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
                    "evidence_use_gate_f1": row["evidence_use_gate_f1"],
                    "baseline_steps": row["baseline_steps"],
                    "evidence_use_gate_steps": row["evidence_use_gate_steps"],
                }
            ),
            flush=True,
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    baseline_f1 = mean([r["baseline_f1"] for r in rows])
    gate_f1 = mean([r["evidence_use_gate_f1"] for r in rows])
    baseline_steps = mean([r["baseline_steps"] for r in rows])
    gate_steps = mean([r["evidence_use_gate_steps"] for r in rows])
    return {
        "n": len(rows),
        "baseline_mean_f1": baseline_f1,
        "evidence_use_gate_mean_f1": gate_f1,
        "f1_delta": gate_f1 - baseline_f1,
        "baseline_mean_steps": baseline_steps,
        "evidence_use_gate_mean_steps": gate_steps,
        "mean_effort_reduction": baseline_steps - gate_steps,
        "relative_effort_reduction": (baseline_steps - gate_steps) / baseline_steps if baseline_steps else 0.0,
        "answer_preservation_rate": mean([1.0 if r["evidence_use_gate_f1"] >= r["baseline_f1"] else 0.0 for r in rows]),
    }


def feature_importances(clf, feature_names: list[str]) -> list[dict[str, Any]]:
    model = clf
    if hasattr(clf, "named_steps"):
        model = list(clf.named_steps.values())[-1]
    values = getattr(model, "feature_importances_", None)
    if values is None:
        coef = getattr(model, "coef_", None)
        if coef is not None:
            values = np.abs(coef[0])
    if values is None:
        return []
    pairs = sorted(zip(feature_names, [float(v) for v in values]), key=lambda x: -x[1])
    return [{"feature": name, "importance": value} for name, value in pairs[:12]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--learn-limit", type=int, default=1536)
    parser.add_argument("--train-examples", type=int, default=80)
    parser.add_argument("--val-examples", type=int, default=32)
    parser.add_argument("--eval-examples", type=int, default=100)
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument("--max-f1-drop", type=float, default=0.01)
    parser.add_argument("--utility-lambda", type=float, default=0.015)
    parser.add_argument("--f1-epsilon", type=float, default=0.05)
    parser.add_argument("--min-good-f1", type=float, default=0.5)
    parser.add_argument("--min-counterfactual-drop", type=float, default=0.05)
    parser.add_argument("--gate-model", choices=["gradient_boosting", "random_forest", "logistic"], default="gradient_boosting")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--output-tag", default="evidence_use_gate_v0")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu = configure_gpu()
    learned = load_module("learned_trust", LEARNED_SCRIPT)
    flow = load_module("qwen_flow", FLOW_SCRIPT)
    examples = learned.load_answerable(args.learn_limit)
    rng = np.random.default_rng(20260609)
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
        rows, summary = teacher_rows_for_example(
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
        train_rows.extend(rows)
        train_summaries.append(summary)
        print(json.dumps({"phase": "teacher_train", "idx": idx, "question_id": ex["id"], "states": len(rows)}), flush=True)

    val_rows: list[dict[str, Any]] = []
    val_summaries = []
    for idx, ex in enumerate(val_examples, start=1):
        rows, summary = teacher_rows_for_example(
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
        val_rows.extend(rows)
        val_summaries.append(summary)
        print(json.dumps({"phase": "teacher_val", "idx": idx, "question_id": ex["id"], "states": len(rows)}), flush=True)

    clf, train_metrics = fit_gate(train_rows, args.gate_model)
    val_x = np.asarray([row["features"] for row in val_rows], dtype=np.float32)
    val_y = np.asarray([row["teacher_label"] for row in val_rows], dtype=np.int64)
    val_probs = clf.predict_proba(val_x)[:, 1] if len(val_rows) else np.asarray([])
    teacher_metrics = {
        **train_metrics,
        "val_states": int(len(val_rows)),
        "val_positive_rate": float(np.mean(val_y)) if len(val_y) else 0.0,
        "val_auroc": float(roc_auc_score(val_y, val_probs)) if len(set(val_y.tolist())) > 1 else None,
        "val_auprc": float(average_precision_score(val_y, val_probs)) if len(set(val_y.tolist())) > 1 else None,
    }
    val_rows_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in val_rows:
        val_rows_by_id.setdefault(row["question_id"], []).append(row)
    policy = choose_threshold(clf, val_summaries, val_rows_by_id, args.max_f1_drop, args.utility_lambda)

    test_rows = run_gate_on_test(
        learned,
        flow,
        tokenizer,
        model,
        clf,
        policy["threshold"],
        test_examples,
        args.fixed_k,
        args.max_new_tokens,
        args.max_length,
    )
    agg = aggregate(test_rows)
    verdict = "PASS" if agg["f1_delta"] >= -args.max_f1_drop and agg["mean_effort_reduction"] > 0 else "FAIL"
    _, debug = feature_vector("dummy question", ["dummy chunk"], [1.0], [1.0], args.fixed_k)
    feature_names = list(debug.keys())
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
            "sufficient_if": "prefix F1 is within f1_epsilon of best prefix/future F1 and >= min_good_f1",
            "counterfactual_if": "removing highest-attended chunk lowers F1 by at least min_counterfactual_drop",
            "runtime_features_only": "retrieval + Qwen attention behavior + lexical/position/length state",
            "no_runtime_contribution": True,
        },
        "teacher_metrics": teacher_metrics,
        "policy": policy,
        "feature_importances": feature_importances(clf, feature_names),
        "aggregate": agg,
        "verdict": verdict,
        "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 3),
        "test_rows": test_rows,
        "teacher_train_summaries": train_summaries,
        "teacher_val_summaries": val_summaries,
        "note": "EvidenceUseGate-v0 learns when Qwen attention/retrieval state predicts evidence sufficiency. Teacher labels use answer sufficiency and counterfactual drop probes; the local 71120 contribution teacher is model-specific and not used in this Qwen gate.",
    }
    json_path = OUT_DIR / f"{args.output_tag}.json"
    md_path = OUT_DIR / f"{args.output_tag}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# EvidenceUseGate-v0",
                "",
                f"Verdict: `{verdict}`",
                "",
                f"- Model: Qwen3-4B-Instruct",
                f"- Runtime gate: learned {args.gate_model} over cheap attention/retrieval state features",
                f"- Teacher: answer sufficiency + counterfactual drop probes",
                f"- Train/val/test examples: {len(train_examples)} / {len(val_examples)} / {len(test_examples)}",
                f"- Train/val states: {len(train_rows)} / {len(val_rows)}",
                f"- Teacher val AUROC: {teacher_metrics['val_auroc']:.4f}" if teacher_metrics["val_auroc"] is not None else "- Teacher val AUROC: n/a",
                f"- Teacher val AUPRC: {teacher_metrics['val_auprc']:.4f}" if teacher_metrics["val_auprc"] is not None else "- Teacher val AUPRC: n/a",
                f"- Selected stop threshold: {policy['threshold']:.3f}",
                f"- Baseline F1: {agg['baseline_mean_f1']:.4f}",
                f"- EvidenceUseGate F1: {agg['evidence_use_gate_mean_f1']:.4f}",
                f"- F1 delta: {agg['f1_delta']:.4f}",
                f"- Steps: {agg['baseline_mean_steps']:.4f} -> {agg['evidence_use_gate_mean_steps']:.4f}",
                f"- Effort reduction: {agg['relative_effort_reduction']:.2%}",
                f"- Answer preservation rate: {agg['answer_preservation_rate']:.2%}",
                f"- Peak CUDA allocation: {result['max_memory_allocated_mib']:.1f} MiB",
                "",
                "This is not a hand-written attention + relevance rule. The gate learns stop probability from runtime state features, with labels produced by expensive teacher probes. Contribution-flow remains a separate mechanistic teacher candidate because the available 71120 implementation is LLaMA/Gemma-specific.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path), "report": str(md_path), "verdict": verdict, "aggregate": agg, "policy": policy, "teacher_metrics": teacher_metrics, "max_memory_allocated_mib": result["max_memory_allocated_mib"]}, indent=2))


if __name__ == "__main__":
    main()
