#!/usr/bin/env python3
"""Validate trust gating with a Qwen3 white-box attention-flow signal.

This is a pragmatic bridge toward the 71120 information-flow validation:
it uses the already-loaded real Qwen3-4B model and extracts internal
attention flow over retrieved sentence chunks. It is not identical to the
71120 contribution-matrix implementation, but it replaces the learned
logistic proxy with a real model-internal signal.
"""

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
OUT_DIR = BUNDLE_ROOT / "experiments/out/qwen_whitebox_flow_validation"
LEARNED_SCRIPT = BUNDLE_ROOT / "experiments/run_learned_trust_benchmark.py"
DEFAULT_MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
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


def make_prompt(question: str, chunks: list[str]) -> tuple[str, list[tuple[int, int]]]:
    prompt = "Answer the question using only the context. Return a short phrase.\nContext:\n"
    spans = []
    for idx, chunk in enumerate(chunks, start=1):
        prefix = f"[{idx}] "
        start = len(prompt) + len(prefix)
        prompt += prefix + chunk + "\n"
        end = start + len(chunk)
        spans.append((start, end))
    prompt += f"Question: {question}\nAnswer:"
    return prompt, spans


def char_to_token_spans(offsets, char_spans: list[tuple[int, int]]) -> list[list[int]]:
    token_spans = []
    for start, end in char_spans:
        ids = []
        for tok_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= start or tok_start >= end:
                continue
            ids.append(tok_idx)
        token_spans.append(ids)
    return token_spans


def attention_flow_scores(tokenizer, model, question: str, chunks: list[str], max_length: int) -> dict[str, Any]:
    prompt, char_spans = make_prompt(question, chunks)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {k: v.to("cuda") for k, v in encoded.items() if k != "token_type_ids"}
    token_spans = char_to_token_spans(offsets, char_spans)
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True, use_cache=False)
    # Last token's mean attention to source tokens, averaged over upper layers.
    attentions = outputs.attentions
    layer_start = max(0, len(attentions) - 8)
    score_vecs = []
    for attn in attentions[layer_start:]:
        # [batch, heads, target, source]
        score_vecs.append(attn[0, :, -1, :].float().mean(dim=0).detach().cpu().numpy())
    token_scores = np.mean(score_vecs, axis=0)
    chunk_scores = []
    for ids in token_spans:
        if not ids:
            chunk_scores.append(0.0)
        else:
            chunk_scores.append(float(np.sum(token_scores[ids])))
    total = sum(max(0.0, s) for s in chunk_scores) + 1e-9
    normalized = [max(0.0, s) / total for s in chunk_scores]
    return {
        "prompt": prompt,
        "chunk_scores": normalized,
        "raw_chunk_scores": chunk_scores,
        "token_count": int(encoded["input_ids"].shape[1]),
    }


def clean_answer(prompt: str, decoded: str) -> str:
    answer = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded
    answer = answer.strip()
    answer = re.split(r"\n|Output:|Context:|Question:", answer)[0].strip().strip('"')
    if "." in answer:
        answer = answer.split(".")[0].strip()
    return answer


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int, max_length: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {k: v.to("cuda") for k, v in encoded.items() if k != "token_type_ids"}
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return clean_answer(prompt, tokenizer.decode(out[0], skip_special_tokens=True))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    baseline_f1 = mean([r["baseline_f1"] for r in rows])
    flow_f1 = mean([r["flow_gated_f1"] for r in rows])
    baseline_steps = mean([r["baseline_steps"] for r in rows])
    flow_steps = mean([r["flow_gated_steps"] for r in rows])
    return {
        "n": len(rows),
        "baseline_mean_f1": baseline_f1,
        "flow_gated_mean_f1": flow_f1,
        "f1_delta": flow_f1 - baseline_f1,
        "baseline_mean_steps": baseline_steps,
        "flow_gated_mean_steps": flow_steps,
        "mean_effort_reduction": baseline_steps - flow_steps,
        "relative_effort_reduction": (baseline_steps - flow_steps) / baseline_steps if baseline_steps else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--learn-limit", type=int, default=1024)
    parser.add_argument("--eval-limit", type=int, default=8)
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument("--flow-threshold", type=float, default=0.30)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu = configure_gpu()
    learned = load_learned_module()
    examples = learned.load_answerable(args.learn_limit)
    rng = np.random.default_rng(7)
    examples = [examples[int(i)] for i in rng.permutation(len(examples))]
    test_examples = examples[int(len(examples) * 0.8) :][: args.eval_limit]

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

    rows = []
    for idx, ex in enumerate(test_examples, start=1):
        chunks = learned.split_sentences(ex["context"])
        order, _ = learned.rank_chunks(ex["question"], chunks)
        baseline_indices = order[: min(args.fixed_k, len(order))]
        baseline_chunks = [chunks[i] for i in baseline_indices]
        flow = attention_flow_scores(tokenizer, model, ex["question"], baseline_chunks, args.max_length)
        selected = []
        for step_idx, score in enumerate(flow["chunk_scores"]):
            selected.append(step_idx)
            if score >= args.flow_threshold:
                break
        flow_chunks = [baseline_chunks[i] for i in selected]
        baseline_prompt, _ = make_prompt(ex["question"], baseline_chunks)
        flow_prompt, _ = make_prompt(ex["question"], flow_chunks)
        baseline_answer = generate_answer(tokenizer, model, baseline_prompt, args.max_new_tokens, args.max_length)
        flow_answer = generate_answer(tokenizer, model, flow_prompt, args.max_new_tokens, args.max_length)
        row = {
            "question_id": ex["id"],
            "question": ex["question"],
            "answers": ex["answers"],
            "baseline_answer": baseline_answer,
            "flow_gated_answer": flow_answer,
            "baseline_f1": learned.token_f1(baseline_answer, ex["answers"]),
            "flow_gated_f1": learned.token_f1(flow_answer, ex["answers"]),
            "baseline_steps": len(baseline_chunks),
            "flow_gated_steps": len(flow_chunks),
            "flow_scores_by_step": flow["chunk_scores"],
            "raw_flow_scores_by_step": flow["raw_chunk_scores"],
            "flow_stop_step": len(flow_chunks),
            "token_count": flow["token_count"],
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in ["question_id", "baseline_f1", "flow_gated_f1", "baseline_steps", "flow_gated_steps"]}, indent=2), flush=True)

    agg = aggregate(rows)
    verdict = "PASS" if agg["f1_delta"] >= -0.05 and agg["mean_effort_reduction"] > 0 else "FAIL"
    result = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "config": vars(args),
        "gpu": gpu,
        "aggregate": agg,
        "verdict": verdict,
        "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 3),
        "rows": rows,
        "note": "White-box attention-flow bridge, not exact 71120 contribution-matrix implementation.",
    }
    stem = args.output_tag or "qwen_whitebox_flow_validation"
    json_path = OUT_DIR / f"{stem}.json"
    md_path = OUT_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Qwen White-Box Flow Validation",
                "",
                f"Verdict: `{verdict}`",
                "",
                f"- Examples: {agg['n']}",
                f"- Baseline F1: {agg['baseline_mean_f1']:.4f}",
                f"- Flow-gated F1: {agg['flow_gated_mean_f1']:.4f}",
                f"- F1 delta: {agg['f1_delta']:.4f}",
                f"- Steps: {agg['baseline_mean_steps']:.4f} -> {agg['flow_gated_mean_steps']:.4f}",
                f"- Effort reduction: {agg['relative_effort_reduction']:.2%}",
                f"- Peak CUDA allocation: {result['max_memory_allocated_mib']:.1f} MiB",
                "",
                "This uses real Qwen3-4B internal attentions as a white-box flow signal. It is a bridge validation, not the exact 71120 contribution/relevance/calibration pipeline.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path), "report": str(md_path), "verdict": verdict, "aggregate": agg, "max_memory_allocated_mib": result["max_memory_allocated_mib"]}, indent=2))


if __name__ == "__main__":
    main()
