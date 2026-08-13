#!/usr/bin/env python3
"""Run a real LLM baseline-vs-trust-gated SQuAD2 case on GPU 2."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = BUNDLE_ROOT / "experiments/out/real_llm_trust_case"
LEARNED_SCRIPT = BUNDLE_ROOT / "experiments/run_learned_trust_benchmark.py"
GPU_MEMORY_CAP_GIB = 20.0


def load_learned_module():
    spec = importlib.util.spec_from_file_location("learned_trust", LEARNED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {LEARNED_SCRIPT}")
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
    fraction = min(1.0, GPU_MEMORY_CAP_GIB / total_gib)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return {
        "visible_cuda_devices": visible,
        "device_name": props.name,
        "memory_cap_gib": GPU_MEMORY_CAP_GIB,
        "memory_cap_fraction": round(fraction, 6),
        "total_memory_gib": round(total_gib, 3),
    }


def clean_answer(prompt: str, decoded: str) -> str:
    answer = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded
    answer = answer.strip()
    answer = re.split(r"\n|Output:|Context:|Question:", answer)[0]
    answer = answer.strip().strip('"').strip()
    if "." in answer:
        answer = answer.split(".")[0].strip()
    return answer


def make_prompt(question: str, chunks: list[str]) -> str:
    context = " ".join(chunks)
    return (
        "Answer the question using only the context. "
        "Return a short phrase, not a sentence.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {key: value.to("cuda") for key, value in inputs.items() if key != "token_type_ids"}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return clean_answer(prompt, decoded)


def train_gate(learned, examples: list[dict[str, Any]], fixed_k: int, max_f1_drop: float):
    n = len(examples)
    train = examples[: int(n * 0.6)]
    val = examples[int(n * 0.6) : int(n * 0.8)]
    test = examples[int(n * 0.8) :]
    x_train, y_train = learned.make_chunk_dataset(train)
    x_val, y_val = learned.make_chunk_dataset(val)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score, roc_auc_score

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    clf.fit(x_train, y_train)
    val_probs = clf.predict_proba(x_val)[:, 1]
    threshold, val_agg = learned.choose_threshold(val, clf, fixed_k, max_f1_drop)
    return clf, threshold, test, {
        "train_examples": len(train),
        "val_examples": len(val),
        "test_examples": len(test),
        "train_chunks": int(len(y_train)),
        "val_chunks": int(len(y_val)),
        "val_auroc": float(roc_auc_score(y_val, val_probs)) if len(set(y_val.tolist())) > 1 else 0.0,
        "val_auprc": float(average_precision_score(y_val, val_probs)),
        "selected_threshold": threshold,
        "validation_policy": val_agg,
    }


def select_contexts(learned, ex: dict[str, Any], clf, threshold: float, fixed_k: int):
    chunks = learned.split_sentences(ex["context"])
    order, scores = learned.rank_chunks(ex["question"], chunks)
    baseline_indices = order[: min(fixed_k, len(order))]

    gated_indices = []
    step_records = []
    for step_idx, chunk_idx in enumerate(order[: min(fixed_k, len(order))]):
        x = np.asarray([learned.features(ex["question"], chunks[chunk_idx], scores, order, step_idx)], dtype=np.float32)
        trust = float(clf.predict_proba(x)[0, 1])
        gated_indices.append(chunk_idx)
        step_records.append(
            {
                "step": step_idx + 1,
                "chunk_index": int(chunk_idx),
                "trust": round(trust, 6),
                "contains_answer": learned.answer_in_chunk(ex["answers"], chunks[chunk_idx]),
            }
        )
        if trust >= threshold:
            break

    return {
        "chunks": chunks,
        "baseline_indices": baseline_indices,
        "gated_indices": gated_indices,
        "steps": step_records,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_f1 = float(np.mean([row["baseline_f1"] for row in rows])) if rows else 0.0
    gated_f1 = float(np.mean([row["gated_f1"] for row in rows])) if rows else 0.0
    baseline_steps = float(np.mean([row["baseline_steps"] for row in rows])) if rows else 0.0
    gated_steps = float(np.mean([row["gated_steps"] for row in rows])) if rows else 0.0
    return {
        "n": len(rows),
        "baseline_mean_f1": baseline_f1,
        "gated_mean_f1": gated_f1,
        "f1_delta": gated_f1 - baseline_f1,
        "baseline_mean_steps": baseline_steps,
        "gated_mean_steps": gated_steps,
        "mean_effort_reduction": baseline_steps - gated_steps,
        "relative_effort_reduction": (baseline_steps - gated_steps) / baseline_steps if baseline_steps else 0.0,
        "answer_preservation_rate": float(np.mean([1.0 if row["gated_f1"] >= row["baseline_f1"] else 0.0 for row in rows])) if rows else 0.0,
    }


def write_report(result: dict[str, Any], path: Path) -> None:
    agg = result["aggregate"]
    lines = [
        "# Real LLM Trust-Gated Case",
        "",
        f"Verdict: `{result['verdict']}`",
        "",
        "## Model",
        "",
        f"- Model path: `{result['model_path']}`",
        f"- GPU: {result['gpu']['device_name']} via CUDA_VISIBLE_DEVICES={result['gpu']['visible_cuda_devices']}",
        f"- Peak CUDA allocation: {result['max_memory_allocated_mib']:.1f} MiB",
        "",
        "## Learned Gate",
        "",
        f"- Validation AUROC: {result['gate']['val_auroc']:.4f}",
        f"- Validation AUPRC: {result['gate']['val_auprc']:.4f}",
        f"- Selected threshold: {result['gate']['selected_threshold']:.3f}",
        "",
        "## Held-Out LLM Generation Metrics",
        "",
        f"- Examples: {agg['n']}",
        f"- Baseline mean F1: {agg['baseline_mean_f1']:.4f}",
        f"- Gated mean F1: {agg['gated_mean_f1']:.4f}",
        f"- F1 delta: {agg['f1_delta']:.4f}",
        f"- Mean retrieval steps: {agg['baseline_mean_steps']:.4f} -> {agg['gated_mean_steps']:.4f}",
        f"- Relative effort reduction: {agg['relative_effort_reduction']:.2%}",
        f"- Answer preservation rate: {agg['answer_preservation_rate']:.2%}",
        "",
        "## Interpretation",
        "",
        "This is a real local LLM generation test. The gate is learned from real SQuAD2 answer-span labels, then used to shorten context before generation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/tf/notebooks/projects/latent-physics-reasoning/models/huginn-0125")
    parser.add_argument("--learn-limit", type=int, default=1024)
    parser.add_argument("--eval-limit", type=int, default=16)
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument("--max-f1-drop", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output-tag", default="real_llm_trust_case")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu = configure_gpu()
    learned = load_learned_module()
    examples = learned.load_answerable(args.learn_limit)
    rng = np.random.default_rng(7)
    examples = [examples[int(i)] for i in rng.permutation(len(examples))]
    clf, threshold, test_examples, gate_info = train_gate(learned, examples, args.fixed_k, args.max_f1_drop)
    eval_examples = test_examples[: args.eval_limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    rows = []
    for idx, ex in enumerate(eval_examples, start=1):
        selected = select_contexts(learned, ex, clf, threshold, args.fixed_k)
        chunks = selected["chunks"]
        baseline_chunks = [chunks[i] for i in selected["baseline_indices"]]
        gated_chunks = [chunks[i] for i in selected["gated_indices"]]
        baseline_prompt = make_prompt(ex["question"], baseline_chunks)
        gated_prompt = make_prompt(ex["question"], gated_chunks)
        baseline_answer = generate_answer(tokenizer, model, baseline_prompt, args.max_new_tokens)
        gated_answer = generate_answer(tokenizer, model, gated_prompt, args.max_new_tokens)
        baseline_f1 = learned.token_f1(baseline_answer, ex["answers"])
        gated_f1 = learned.token_f1(gated_answer, ex["answers"])
        row = {
            "question_id": ex["id"],
            "question": ex["question"],
            "answers": ex["answers"],
            "baseline_answer": baseline_answer,
            "gated_answer": gated_answer,
            "baseline_f1": baseline_f1,
            "gated_f1": gated_f1,
            "baseline_steps": len(baseline_chunks),
            "gated_steps": len(gated_chunks),
            "gate_steps": selected["steps"],
        }
        rows.append(row)
        print(json.dumps({"idx": idx, "question_id": ex["id"], "baseline_f1": baseline_f1, "gated_f1": gated_f1, "baseline_steps": len(baseline_chunks), "gated_steps": len(gated_chunks)}, indent=2), flush=True)

    agg = aggregate(rows)
    verdict = "PASS" if agg["f1_delta"] >= -args.max_f1_drop and agg["mean_effort_reduction"] > 0 else "FAIL"
    result = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "config": vars(args),
        "gpu": gpu,
        "gate": gate_info,
        "aggregate": agg,
        "verdict": verdict,
        "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 3),
        "rows": rows,
    }
    json_path = OUT_DIR / f"{args.output_tag}.json"
    md_path = OUT_DIR / f"{args.output_tag}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, md_path)
    print(json.dumps({"json": str(json_path), "report": str(md_path), "verdict": verdict, "aggregate": agg, "max_memory_allocated_mib": result["max_memory_allocated_mib"]}, indent=2))


if __name__ == "__main__":
    main()
