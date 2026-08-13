#!/usr/bin/env python3
"""Qwen3-4B calibrated attention-flow trust-gated retrieval.

This upgrades the lightweight learned proxy toward an internal trust signal:

- sparse retriever selects sentence chunks;
- Qwen3 internal attention flow scores each retrieved chunk;
- retrieval relevance and attention flow are calibrated on validation labels;
- held-out examples compare fixed top-k context vs calibrated flow-gated context.

It is still a bridge implementation, not the exact 71120 contribution/relevance
calibrator, but it tests whether a stronger model-internal signal can replace
the TF-IDF-only learned proxy.
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
OUT_DIR = BUNDLE_ROOT / "experiments/out/qwen_calibrated_flow_trust_case"
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


def minmax(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return []
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [float((x - lo) / (hi - lo)) for x in arr]


def build_ranked_chunks(learned, ex: dict[str, Any], fixed_k: int):
    chunks = learned.split_sentences(ex["context"])
    order, retrieval_scores = learned.rank_chunks(ex["question"], chunks)
    order = order[: min(fixed_k, len(order))]
    return {
        "chunks": chunks,
        "indices": order,
        "ranked_chunks": [chunks[i] for i in order],
        "retrieval_scores": [retrieval_scores[i] for i in order],
    }


def score_example(learned, flow, tokenizer, model, ex: dict[str, Any], fixed_k: int, max_length: int):
    ranked = build_ranked_chunks(learned, ex, fixed_k)
    flow_result = flow.attention_flow_scores(
        tokenizer,
        model,
        ex["question"],
        ranked["ranked_chunks"],
        max_length,
    )
    retrieval_norm = minmax(ranked["retrieval_scores"])
    flow_norm = minmax(flow_result["chunk_scores"])
    labels = [1 if learned.answer_in_chunk(ex["answers"], chunk) else 0 for chunk in ranked["ranked_chunks"]]
    return {
        **ranked,
        "flow_scores": flow_result["chunk_scores"],
        "flow_scores_norm": flow_norm,
        "retrieval_scores_norm": retrieval_norm,
        "labels": labels,
        "token_count": flow_result["token_count"],
    }


def combined_scores(scored: dict[str, Any], alpha: float) -> list[float]:
    return [
        alpha * f + (1.0 - alpha) * r
        for f, r in zip(scored["flow_scores_norm"], scored["retrieval_scores_norm"])
    ]


def stop_step(scores: list[float], threshold: float) -> int:
    for idx, score in enumerate(scores, start=1):
        if score >= threshold:
            return idx
    return len(scores)


def selected_has_answer(labels: list[int], step: int) -> bool:
    return any(labels[:step])


def calibrate(scored_val: list[dict[str, Any]], max_label_drop: float):
    best = None
    for alpha in np.linspace(0.0, 1.0, 11):
        for threshold in np.linspace(0.05, 0.95, 91):
            baseline_ok = [any(s["labels"]) for s in scored_val]
            gated_steps = []
            gated_ok = []
            for scored in scored_val:
                scores = combined_scores(scored, float(alpha))
                step = stop_step(scores, float(threshold))
                gated_steps.append(step)
                gated_ok.append(selected_has_answer(scored["labels"], step))
            baseline_rate = float(np.mean(baseline_ok))
            gated_rate = float(np.mean(gated_ok))
            baseline_steps = float(np.mean([len(s["labels"]) for s in scored_val]))
            mean_steps = float(np.mean(gated_steps))
            label_delta = gated_rate - baseline_rate
            effort_reduction = baseline_steps - mean_steps
            if label_delta >= -max_label_drop and effort_reduction > 0:
                candidate = {
                    "alpha": float(alpha),
                    "threshold": float(threshold),
                    "baseline_label_coverage": baseline_rate,
                    "gated_label_coverage": gated_rate,
                    "label_delta": label_delta,
                    "baseline_mean_steps": baseline_steps,
                    "gated_mean_steps": mean_steps,
                    "mean_effort_reduction": effort_reduction,
                    "relative_effort_reduction": effort_reduction / baseline_steps if baseline_steps else 0.0,
                }
                if best is None or candidate["mean_effort_reduction"] > best["mean_effort_reduction"]:
                    best = candidate
    if best is None:
        best = {
            "alpha": 1.0,
            "threshold": 0.95,
            "baseline_label_coverage": 0.0,
            "gated_label_coverage": 0.0,
            "label_delta": -1.0,
            "baseline_mean_steps": 0.0,
            "gated_mean_steps": 0.0,
            "mean_effort_reduction": 0.0,
            "relative_effort_reduction": 0.0,
        }
    return best


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    baseline_f1 = mean([r["baseline_f1"] for r in rows])
    gated_f1 = mean([r["calibrated_flow_f1"] for r in rows])
    baseline_steps = mean([r["baseline_steps"] for r in rows])
    gated_steps = mean([r["calibrated_flow_steps"] for r in rows])
    return {
        "n": len(rows),
        "baseline_mean_f1": baseline_f1,
        "calibrated_flow_mean_f1": gated_f1,
        "f1_delta": gated_f1 - baseline_f1,
        "baseline_mean_steps": baseline_steps,
        "calibrated_flow_mean_steps": gated_steps,
        "mean_effort_reduction": baseline_steps - gated_steps,
        "relative_effort_reduction": (baseline_steps - gated_steps) / baseline_steps if baseline_steps else 0.0,
        "answer_preservation_rate": mean([1.0 if r["calibrated_flow_f1"] >= r["baseline_f1"] else 0.0 for r in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--learn-limit", type=int, default=1024)
    parser.add_argument("--val-limit", type=int, default=24)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument("--max-label-drop", type=float, default=0.05)
    parser.add_argument("--max-f1-drop", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--output-tag", default="qwen_calibrated_flow_trust_case")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu = configure_gpu()
    learned = load_module("learned_trust", LEARNED_SCRIPT)
    flow = load_module("qwen_flow", FLOW_SCRIPT)

    examples = learned.load_answerable(args.learn_limit)
    rng = np.random.default_rng(11)
    examples = [examples[int(i)] for i in rng.permutation(len(examples))]
    val_examples = examples[int(len(examples) * 0.6) : int(len(examples) * 0.6) + args.val_limit]
    test_examples = examples[int(len(examples) * 0.8) : int(len(examples) * 0.8) + args.eval_limit]

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

    scored_val = []
    for idx, ex in enumerate(val_examples, start=1):
        scored_val.append(score_example(learned, flow, tokenizer, model, ex, args.fixed_k, args.max_length))
        print(json.dumps({"phase": "val_score", "idx": idx, "question_id": ex["id"]}), flush=True)
    calibration = calibrate(scored_val, args.max_label_drop)

    rows = []
    for idx, ex in enumerate(test_examples, start=1):
        scored = score_example(learned, flow, tokenizer, model, ex, args.fixed_k, args.max_length)
        scores = combined_scores(scored, calibration["alpha"])
        step = stop_step(scores, calibration["threshold"])
        baseline_chunks = scored["ranked_chunks"]
        gated_chunks = scored["ranked_chunks"][:step]
        baseline_prompt, _ = flow.make_prompt(ex["question"], baseline_chunks)
        gated_prompt, _ = flow.make_prompt(ex["question"], gated_chunks)
        baseline_answer = flow.generate_answer(tokenizer, model, baseline_prompt, args.max_new_tokens, args.max_length)
        gated_answer = flow.generate_answer(tokenizer, model, gated_prompt, args.max_new_tokens, args.max_length)
        row = {
            "question_id": ex["id"],
            "question": ex["question"],
            "answers": ex["answers"],
            "baseline_answer": baseline_answer,
            "calibrated_flow_answer": gated_answer,
            "baseline_f1": learned.token_f1(baseline_answer, ex["answers"]),
            "calibrated_flow_f1": learned.token_f1(gated_answer, ex["answers"]),
            "baseline_steps": len(baseline_chunks),
            "calibrated_flow_steps": step,
            "combined_scores_by_step": scores,
            "flow_scores_by_step": scored["flow_scores"],
            "retrieval_scores_by_step": scored["retrieval_scores"],
            "labels_by_step": scored["labels"],
        }
        rows.append(row)
        print(json.dumps({"phase": "test_gen", "idx": idx, "question_id": ex["id"], "baseline_f1": row["baseline_f1"], "calibrated_flow_f1": row["calibrated_flow_f1"], "baseline_steps": row["baseline_steps"], "calibrated_flow_steps": row["calibrated_flow_steps"]}), flush=True)

    agg = aggregate(rows)
    verdict = "PASS" if agg["f1_delta"] >= -args.max_f1_drop and agg["mean_effort_reduction"] > 0 else "FAIL"
    result = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "config": vars(args),
        "gpu": gpu,
        "calibration": calibration,
        "aggregate": agg,
        "verdict": verdict,
        "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 3),
        "rows": rows,
        "note": "Calibrated Qwen attention-flow + retrieval relevance bridge, not exact 71120 implementation.",
    }
    json_path = OUT_DIR / f"{args.output_tag}.json"
    md_path = OUT_DIR / f"{args.output_tag}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Qwen Calibrated Flow Trust Case",
                "",
                f"Verdict: `{verdict}`",
                "",
                f"- Validation examples for calibration: {len(scored_val)}",
                f"- Held-out generation examples: {agg['n']}",
                f"- Alpha internal-flow weight: {calibration['alpha']:.2f}",
                f"- Trust threshold: {calibration['threshold']:.3f}",
                f"- Baseline F1: {agg['baseline_mean_f1']:.4f}",
                f"- Calibrated-flow F1: {agg['calibrated_flow_mean_f1']:.4f}",
                f"- F1 delta: {agg['f1_delta']:.4f}",
                f"- Steps: {agg['baseline_mean_steps']:.4f} -> {agg['calibrated_flow_mean_steps']:.4f}",
                f"- Effort reduction: {agg['relative_effort_reduction']:.2%}",
                f"- Answer preservation rate: {agg['answer_preservation_rate']:.2%}",
                f"- Peak CUDA allocation: {result['max_memory_allocated_mib']:.1f} MiB",
                "",
                "This upgrades the gate from TF-IDF-only learned proxy to a calibrated blend of Qwen internal attention-flow and retrieval relevance. It is still a bridge toward, not an exact reproduction of, 71120 information-flow contribution/relevance calibration.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path), "report": str(md_path), "verdict": verdict, "aggregate": agg, "calibration": calibration, "max_memory_allocated_mib": result["max_memory_allocated_mib"]}, indent=2))


if __name__ == "__main__":
    main()
