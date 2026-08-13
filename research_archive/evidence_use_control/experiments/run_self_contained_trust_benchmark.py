#!/usr/bin/env python3
"""Self-contained benchmark test for the trust-gating idea itself.

This is not a replacement for the full 71120 attribution run. It isolates the
runtime policy question on real SQuAD2 examples:

Can a trust-gated stop policy inspect fewer retrieved chunks while preserving
answerability compared with a fixed-effort retrieval baseline?
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = BUNDLE_ROOT / "experiments/out/self_contained_benchmark"
GPU_MEMORY_CAP_GIB = 20.0


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def split_sentences(context: str) -> list[str]:
    chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+", context) if part.strip()]
    if len(chunks) <= 1:
        chunks = [part.strip() for part in context.split(";") if part.strip()]
    return chunks or [context]


def answer_in_chunk(answers: list[str], chunk: str) -> bool:
    chunk_norm = normalize(chunk)
    return any(answer and normalize(answer) in chunk_norm for answer in answers)


def best_answer_from_chunk(answers: list[str], chunk: str) -> str:
    for answer in answers:
        if answer and normalize(answer) in normalize(chunk):
            return answer
    return ""


def token_f1(prediction: str, answers: list[str]) -> float:
    if not answers:
        return 1.0 if not prediction.strip() else 0.0
    pred = tokens(prediction)
    if not pred:
        return 0.0
    best = 0.0
    for answer in answers:
        gold = tokens(answer)
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


def exact_match(prediction: str, answers: list[str]) -> float:
    pred = normalize(prediction)
    return 1.0 if any(pred == normalize(answer) for answer in answers if answer) else 0.0


def configure_gpu() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        return {
            "ok": False,
            "error": "CUDA_VISIBLE_DEVICES is not '2'; CPU path used for benchmark computation.",
            "current": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    if torch is None or not torch.cuda.is_available():
        return {"ok": False, "error": "Torch CUDA unavailable; CPU path used."}
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


def load_examples(limit: int, eval_examples: int | None = None) -> list[dict[str, Any]]:
    ds = load_dataset("squad_v2", split=f"validation[:{limit}]")
    examples = []
    for ex in ds:
        answers_obj = ex.get("answers", {})
        answers = answers_obj.get("text", []) if isinstance(answers_obj, dict) else []
        if not answers:
            continue
        examples.append(
            {
                "id": ex["id"],
                "question": ex["question"],
                "context": ex["context"],
                "answers": answers,
            }
        )
        if eval_examples is not None and len(examples) >= eval_examples:
            break
    return examples


def rank_chunks(question: str, chunks: list[str]) -> tuple[list[int], list[float]]:
    corpus = [question] + chunks
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(tfidf[0:1], tfidf[1:]).ravel()
    order = list(np.argsort(-scores))
    return order, [float(score) for score in scores]


def trust_features(scores: list[float], order: list[int], step_idx: int) -> dict[str, float]:
    selected = order[: step_idx + 1]
    selected_scores = [max(0.0, scores[i]) for i in selected]
    current = selected_scores[-1] if selected_scores else 0.0
    total_seen = sum(selected_scores) + 1e-9
    concentration = current / total_seen
    best = selected_scores[0] if selected_scores else 0.0
    margin = max(0.0, best - (selected_scores[1] if len(selected_scores) > 1 else 0.0))
    simulatability = current / (best + 1e-9) if best else 0.0
    trust = 0.55 * simulatability + 0.30 * concentration + 0.15 * min(1.0, margin * 4.0)
    return {
        "retrieval_score": current,
        "simulatability_proxy": simulatability,
        "concentration_proxy": concentration,
        "margin_proxy": margin,
        "trust_score": trust,
    }


def run_example(ex: dict[str, Any], fixed_k: int, threshold: float, policy: str) -> dict[str, Any]:
    chunks = split_sentences(ex["context"])
    order, scores = rank_chunks(ex["question"], chunks)
    fixed_seen = order[: min(fixed_k, len(order))]
    fixed_answer = ""
    for idx in fixed_seen:
        fixed_answer = best_answer_from_chunk(ex["answers"], chunks[idx])
        if fixed_answer:
            break

    gated_answer = ""
    stop_step = len(order)
    step_records = []
    for step_idx, chunk_idx in enumerate(order):
        feats = trust_features(scores, order, step_idx)
        contains_answer = answer_in_chunk(ex["answers"], chunks[chunk_idx])
        step_records.append(
            {
                "step": step_idx + 1,
                "chunk_index": int(chunk_idx),
                "chunk": chunks[chunk_idx],
                "contains_answer": contains_answer,
                **{key: round(float(value), 6) for key, value in feats.items()},
            }
        )
        if policy == "oracle_evidence":
            should_stop = contains_answer or step_idx == len(order) - 1
        elif policy == "lexical_trust":
            should_stop = feats["trust_score"] >= threshold or step_idx == len(order) - 1
        else:
            raise ValueError(f"unknown policy: {policy}")

        if should_stop:
            stop_step = step_idx + 1
            # Answer from the best evidence chunk seen so far. In oracle_evidence
            # mode this isolates the maximum possible gain from a perfect trust
            # detector; in lexical_trust mode it gives the cheap proxy a fair
            # retrieval-style readout.
            for seen_idx in order[: step_idx + 1]:
                gated_answer = best_answer_from_chunk(ex["answers"], chunks[seen_idx])
                if gated_answer:
                    break
            break

    fixed_f1 = token_f1(fixed_answer, ex["answers"])
    gated_f1 = token_f1(gated_answer, ex["answers"])
    return {
        "question_id": ex["id"],
        "question": ex["question"],
        "answers": ex["answers"],
        "n_chunks": len(chunks),
        "baseline_fixed_k": fixed_k,
        "baseline_steps": len(fixed_seen),
        "baseline_answer": fixed_answer,
        "baseline_exact_match": exact_match(fixed_answer, ex["answers"]),
        "baseline_token_f1": fixed_f1,
        "trust_threshold": threshold,
        "trust_policy": policy,
        "trust_gated_steps": stop_step,
        "trust_gated_answer": gated_answer,
        "trust_gated_exact_match": exact_match(gated_answer, ex["answers"]),
        "trust_gated_token_f1": gated_f1,
        "effort_delta": len(fixed_seen) - stop_step,
        "answer_preserved": gated_f1 >= fixed_f1,
        "steps": step_records,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [1 if row["baseline_token_f1"] > 0 else 0 for row in rows]
    y_score = [row["steps"][row["trust_gated_steps"] - 1]["trust_score"] for row in rows]
    auroc = None
    auprc = None
    if len(set(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_score))
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auprc = float(auc(recall, precision))
    baseline_f1 = float(np.mean([row["baseline_token_f1"] for row in rows])) if rows else 0.0
    gated_f1 = float(np.mean([row["trust_gated_token_f1"] for row in rows])) if rows else 0.0
    baseline_steps = float(np.mean([row["baseline_steps"] for row in rows])) if rows else 0.0
    gated_steps = float(np.mean([row["trust_gated_steps"] for row in rows])) if rows else 0.0
    preserved = float(np.mean([1.0 if row["answer_preserved"] else 0.0 for row in rows])) if rows else 0.0
    return {
        "n": len(rows),
        "baseline_mean_token_f1": baseline_f1,
        "trust_gated_mean_token_f1": gated_f1,
        "token_f1_delta": gated_f1 - baseline_f1,
        "baseline_mean_steps": baseline_steps,
        "trust_gated_mean_steps": gated_steps,
        "mean_effort_reduction": baseline_steps - gated_steps,
        "relative_effort_reduction": (baseline_steps - gated_steps) / baseline_steps if baseline_steps else 0.0,
        "answer_preservation_rate": preserved,
        "trust_auroc_for_baseline_answerability": auroc,
        "trust_auprc_for_baseline_answerability": auprc,
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    agg = result["aggregate"]
    lines = [
        "# Self-Contained Trust Benchmark",
        "",
        f"Verdict: `{result['verdict']}`",
        "",
        "## Setup",
        "",
        f"- Dataset: SQuAD2 validation",
        f"- Examples: {agg['n']}",
        f"- Baseline: inspect fixed top-{result['config']['fixed_k']} retrieved chunks",
        f"- Trust policy: {result['config']['policy']}",
        f"- Trust-gated: stop when trust >= {result['config']['threshold']} for lexical policy, or when evidence is found for oracle policy",
        "",
        "## Metrics",
        "",
        f"- Baseline mean token F1: {agg['baseline_mean_token_f1']:.4f}",
        f"- Trust-gated mean token F1: {agg['trust_gated_mean_token_f1']:.4f}",
        f"- Token F1 delta: {agg['token_f1_delta']:.4f}",
        f"- Baseline mean steps: {agg['baseline_mean_steps']:.4f}",
        f"- Trust-gated mean steps: {agg['trust_gated_mean_steps']:.4f}",
        f"- Mean effort reduction: {agg['mean_effort_reduction']:.4f}",
        f"- Relative effort reduction: {agg['relative_effort_reduction']:.2%}",
        f"- Answer preservation rate: {agg['answer_preservation_rate']:.2%}",
        "",
        "## Interpretation",
        "",
        "This evaluates the stop/continue policy itself on real benchmark examples. The oracle_evidence policy is an upper bound for a perfect evidence/trust detector; lexical_trust is the cheap deployable proxy. Neither claims to reproduce 71120 white-box transformer attribution.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=None)
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--policy", choices=["oracle_evidence", "lexical_trust"], default="oracle_evidence")
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu = configure_gpu()
    examples = load_examples(args.limit, args.eval_examples)
    rows = [run_example(ex, args.fixed_k, args.threshold, args.policy) for ex in examples]
    agg = aggregate(rows)
    verdict = "PASS" if agg["token_f1_delta"] >= -0.05 and agg["mean_effort_reduction"] > 0 else "FAIL"
    result = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "limit": args.limit,
            "eval_examples": args.eval_examples,
            "fixed_k": args.fixed_k,
            "threshold": args.threshold,
            "policy": args.policy,
        },
        "gpu": gpu,
        "aggregate": agg,
        "verdict": verdict,
        "rows": rows,
    }
    stem = args.output_tag or f"self_contained_trust_benchmark_{args.policy}"
    json_path = OUT_DIR / f"{stem}.json"
    md_path = OUT_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"json": str(json_path), "report": str(md_path), "verdict": verdict, "aggregate": agg}, indent=2))


if __name__ == "__main__":
    main()
