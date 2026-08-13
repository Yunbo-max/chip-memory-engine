#!/usr/bin/env python3
"""Learn a trust/evidence stop signal from real SQuAD2 data.

This is the self-contained learned benchmark. It trains a lightweight evidence
classifier on real question/chunk pairs, tunes a stop threshold on validation,
and evaluates whether learned trust reduces retrieval effort without hurting
answer quality on held-out examples.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = BUNDLE_ROOT / "experiments/out/learned_trust_benchmark"
GPU_MEMORY_CAP_GIB = 20.0


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def split_sentences(context: str) -> list[str]:
    chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+", context) if part.strip()]
    return chunks or [context]


def answer_in_chunk(answers: list[str], chunk: str) -> bool:
    chunk_norm = normalize(chunk)
    return any(answer and normalize(answer) in chunk_norm for answer in answers)


def answer_from_seen(answers: list[str], chunks: list[str], seen_indices: list[int]) -> str:
    for idx in seen_indices:
        chunk_norm = normalize(chunks[idx])
        for answer in answers:
            if answer and normalize(answer) in chunk_norm:
                return answer
    return ""


def token_f1(prediction: str, answers: list[str]) -> float:
    if not answers:
        return 1.0 if not prediction.strip() else 0.0
    pred = toks(prediction)
    if not pred:
        return 0.0
    best = 0.0
    for answer in answers:
        gold = toks(answer)
        if not gold:
            continue
        common = Counter(pred) & Counter(gold)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(pred)
        recall = n_common / len(gold)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def configure_gpu() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        return {
            "ok": False,
            "error": "CUDA_VISIBLE_DEVICES is not '2'; learned benchmark uses CPU features.",
            "current": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    if torch is None or not torch.cuda.is_available():
        return {"ok": False, "error": "Torch CUDA unavailable; learned benchmark uses CPU features."}
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
    fraction = min(1.0, GPU_MEMORY_CAP_GIB / total_gib)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    probe = torch.ones((16, 16), device="cuda", dtype=torch.float32)
    probe_sum = float(probe.sum().item())
    del probe
    torch.cuda.empty_cache()
    return {
        "ok": True,
        "visible_cuda_devices": "2",
        "device_name": props.name,
        "memory_cap_gib": GPU_MEMORY_CAP_GIB,
        "memory_cap_fraction": round(fraction, 6),
        "probe_sum": probe_sum,
        "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 3),
    }


def load_answerable(limit: int) -> list[dict[str, Any]]:
    ds = load_dataset("squad_v2", split=f"validation[:{limit}]")
    rows = []
    for ex in ds:
        answers_obj = ex.get("answers", {})
        answers = answers_obj.get("text", []) if isinstance(answers_obj, dict) else []
        if answers:
            rows.append(
                {
                    "id": ex["id"],
                    "question": ex["question"],
                    "context": ex["context"],
                    "answers": answers,
                }
            )
    return rows


def rank_chunks(question: str, chunks: list[str]) -> tuple[list[int], list[float]]:
    corpus = [question] + chunks
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform(corpus)
    scores = cosine_similarity(mat[0:1], mat[1:]).ravel()
    order = list(np.argsort(-scores))
    return order, [float(x) for x in scores]


def features(question: str, chunk: str, scores: list[float], order: list[int], step_idx: int) -> list[float]:
    chunk_idx = order[step_idx]
    q_tokens = set(toks(question))
    c_tokens = set(toks(chunk))
    q_nums = {tok for tok in q_tokens if tok.isdigit()}
    c_nums = {tok for tok in c_tokens if tok.isdigit()}
    selected_scores = [max(0.0, scores[i]) for i in order[: step_idx + 1]]
    current = max(0.0, scores[chunk_idx])
    best = selected_scores[0] if selected_scores else 0.0
    second = selected_scores[1] if len(selected_scores) > 1 else 0.0
    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens))
    recall = len(q_tokens & c_tokens) / max(1, len(q_tokens))
    return [
        current,
        best,
        max(0.0, best - second),
        current / (best + 1e-9) if best else 0.0,
        current / (sum(selected_scores) + 1e-9),
        1.0 / (step_idx + 1),
        float(step_idx + 1),
        overlap,
        recall,
        len(q_nums & c_nums) / max(1, len(q_nums | c_nums)) if q_nums or c_nums else 0.0,
        len(toks(chunk)),
    ]


def make_chunk_dataset(examples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for ex in examples:
        chunks = split_sentences(ex["context"])
        order, scores = rank_chunks(ex["question"], chunks)
        for step_idx, chunk_idx in enumerate(order):
            xs.append(features(ex["question"], chunks[chunk_idx], scores, order, step_idx))
            ys.append(1 if answer_in_chunk(ex["answers"], chunks[chunk_idx]) else 0)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def run_policy(
    examples: list[dict[str, Any]],
    clf,
    threshold: float,
    fixed_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for ex in examples:
        chunks = split_sentences(ex["context"])
        order, scores = rank_chunks(ex["question"], chunks)
        baseline_seen = order[: min(fixed_k, len(order))]
        baseline_answer = answer_from_seen(ex["answers"], chunks, baseline_seen)

        gated_seen = []
        step_records = []
        max_steps = min(fixed_k, len(order))
        for step_idx, chunk_idx in enumerate(order[:max_steps]):
            x = np.asarray([features(ex["question"], chunks[chunk_idx], scores, order, step_idx)], dtype=np.float32)
            trust = float(clf.predict_proba(x)[0, 1])
            gated_seen.append(chunk_idx)
            step_records.append(
                {
                    "step": step_idx + 1,
                    "chunk_index": int(chunk_idx),
                    "trust": round(trust, 6),
                    "contains_answer": answer_in_chunk(ex["answers"], chunks[chunk_idx]),
                    "chunk": chunks[chunk_idx],
                }
            )
            if trust >= threshold:
                break

        gated_answer = answer_from_seen(ex["answers"], chunks, gated_seen)
        baseline_f1 = token_f1(baseline_answer, ex["answers"])
        gated_f1 = token_f1(gated_answer, ex["answers"])
        rows.append(
            {
                "question_id": ex["id"],
                "question": ex["question"],
                "answers": ex["answers"],
                "baseline_steps": len(baseline_seen),
                "baseline_answer": baseline_answer,
                "baseline_token_f1": baseline_f1,
                "learned_trust_steps": len(gated_seen),
                "learned_trust_answer": gated_answer,
                "learned_trust_token_f1": gated_f1,
                "answer_preserved": gated_f1 >= baseline_f1,
                "effort_delta": len(baseline_seen) - len(gated_seen),
                "steps": step_records,
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    baseline_f1 = float(np.mean([row["baseline_token_f1"] for row in rows]))
    gated_f1 = float(np.mean([row["learned_trust_token_f1"] for row in rows]))
    baseline_steps = float(np.mean([row["baseline_steps"] for row in rows]))
    gated_steps = float(np.mean([row["learned_trust_steps"] for row in rows]))
    return {
        "n": len(rows),
        "baseline_mean_token_f1": baseline_f1,
        "learned_trust_mean_token_f1": gated_f1,
        "token_f1_delta": gated_f1 - baseline_f1,
        "baseline_mean_steps": baseline_steps,
        "learned_trust_mean_steps": gated_steps,
        "mean_effort_reduction": baseline_steps - gated_steps,
        "relative_effort_reduction": (baseline_steps - gated_steps) / baseline_steps if baseline_steps else 0.0,
        "answer_preservation_rate": float(np.mean([1.0 if row["answer_preserved"] else 0.0 for row in rows])),
    }


def choose_threshold(val_examples: list[dict[str, Any]], clf, fixed_k: int, max_f1_drop: float) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_agg: dict[str, float] | None = None
    for threshold in np.linspace(0.05, 0.95, 91):
        rows = run_policy(val_examples, clf, float(threshold), fixed_k)
        agg = aggregate(rows)
        if agg["token_f1_delta"] >= -max_f1_drop:
            if best_agg is None or agg["mean_effort_reduction"] > best_agg["mean_effort_reduction"]:
                best_threshold = float(threshold)
                best_agg = agg
    if best_agg is None:
        rows = run_policy(val_examples, clf, 0.95, fixed_k)
        best_threshold = 0.95
        best_agg = aggregate(rows)
    return best_threshold, best_agg


def write_markdown(result: dict[str, Any], path: Path) -> None:
    test = result["test_aggregate"]
    val = result["validation_aggregate_at_selected_threshold"]
    lines = [
        "# Learned Trust Benchmark",
        "",
        f"Verdict: `{result['verdict']}`",
        "",
        "## Data",
        "",
        f"- Dataset: SQuAD2 validation",
        f"- Train examples: {result['splits']['train']}",
        f"- Validation examples: {result['splits']['val']}",
        f"- Test examples: {result['splits']['test']}",
        f"- Chunk labels learned from real answer spans in sentence chunks",
        "",
        "## Learned Trust Model",
        "",
        f"- Model: logistic regression over retrieval/evidence features",
        f"- Chunk classifier AUROC: {result['chunk_classifier']['val_auroc']:.4f}",
        f"- Chunk classifier AUPRC: {result['chunk_classifier']['val_auprc']:.4f}",
        f"- Selected threshold: {result['selected_threshold']:.3f}",
        "",
        "## Validation Policy Metrics",
        "",
        f"- Baseline F1: {val['baseline_mean_token_f1']:.4f}",
        f"- Learned trust F1: {val['learned_trust_mean_token_f1']:.4f}",
        f"- Mean steps: {val['baseline_mean_steps']:.4f} -> {val['learned_trust_mean_steps']:.4f}",
        f"- Effort reduction: {val['relative_effort_reduction']:.2%}",
        "",
        "## Held-Out Test Metrics",
        "",
        f"- Baseline F1: {test['baseline_mean_token_f1']:.4f}",
        f"- Learned trust F1: {test['learned_trust_mean_token_f1']:.4f}",
        f"- F1 delta: {test['token_f1_delta']:.4f}",
        f"- Mean steps: {test['baseline_mean_steps']:.4f} -> {test['learned_trust_mean_steps']:.4f}",
        f"- Effort reduction: {test['relative_effort_reduction']:.2%}",
        f"- Answer preservation rate: {test['answer_preservation_rate']:.2%}",
        "",
        "## Interpretation",
        "",
        "This learns the stop signal from real benchmark data. It is still a lightweight evidence proxy rather than the 71120 white-box information-flow signal, but unlike the earlier lexical threshold it is trained and tuned on real labels.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument("--max-f1-drop", type=float, default=0.05)
    parser.add_argument("--test-examples", type=int, default=None)
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu = configure_gpu()
    examples = load_answerable(args.limit)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(examples))
    examples = [examples[int(i)] for i in order]
    n = len(examples)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train = examples[:n_train]
    val = examples[n_train : n_train + n_val]
    test = examples[n_train + n_val :]
    if args.test_examples is not None:
        test = test[: args.test_examples]

    x_train, y_train = make_chunk_dataset(train)
    x_val, y_val = make_chunk_dataset(val)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    clf.fit(x_train, y_train)
    val_probs = clf.predict_proba(x_val)[:, 1]
    chunk_metrics = {
        "val_auroc": float(roc_auc_score(y_val, val_probs)) if len(set(y_val.tolist())) > 1 else 0.0,
        "val_auprc": float(average_precision_score(y_val, val_probs)),
        "train_chunks": int(len(y_train)),
        "val_chunks": int(len(y_val)),
        "train_positive_rate": float(np.mean(y_train)),
        "val_positive_rate": float(np.mean(y_val)),
    }

    threshold, val_agg = choose_threshold(val, clf, args.fixed_k, args.max_f1_drop)
    test_rows = run_policy(test, clf, threshold, args.fixed_k)
    test_agg = aggregate(test_rows)
    verdict = "PASS" if test_agg["token_f1_delta"] >= -args.max_f1_drop and test_agg["mean_effort_reduction"] > 0 else "FAIL"

    result = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "limit": args.limit,
            "fixed_k": args.fixed_k,
            "max_f1_drop": args.max_f1_drop,
            "test_examples": args.test_examples,
        },
        "gpu": gpu,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "chunk_classifier": chunk_metrics,
        "selected_threshold": threshold,
        "validation_aggregate_at_selected_threshold": val_agg,
        "test_aggregate": test_agg,
        "verdict": verdict,
        "test_rows": test_rows,
    }

    stem = args.output_tag or "learned_trust_benchmark"
    json_path = OUT_DIR / f"{stem}.json"
    md_path = OUT_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"json": str(json_path), "report": str(md_path), "verdict": verdict, "test_aggregate": test_agg, "chunk_classifier": chunk_metrics, "selected_threshold": threshold}, indent=2))


if __name__ == "__main__":
    main()
