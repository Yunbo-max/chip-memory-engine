#!/usr/bin/env python3
"""71120-style contribution-flow trust-gated retrieval pilot.

This is heavier than the Qwen attention-flow bridge. It imports the local
71120 LLaMA attributor and computes per-layer contribution matrices, then
aggregates token contribution from retrieved chunks to the next answer token.

The local cache does not contain the exact LLaMA-3.2/Gemma checkpoints used by
the paper, so the default model is the available LLaMA-family checkpoint. This
tests the real contribution-flow mechanism, but it is still a port/pilot rather
than a strict reproduction of the paper's full setup.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = BUNDLE_ROOT / "experiments/out/llama_contribution_flow_trust_case"
REPO_71120 = Path("/tf/notebooks/_orchestration/code_repos/icml2026/71120_RAG-information-flow")
LLAMA_ATTRIBUTOR = REPO_71120 / "proposed/Ours/llama/llama.py"
DEFAULT_MODEL = (
    "/root/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-chat-hf/"
    "snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee"
)
GPU_MEMORY_CAP_GIB = 20.0


def load_llama_attributor():
    spec = importlib.util.spec_from_file_location("llama_71120", LLAMA_ATTRIBUTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {LLAMA_ATTRIBUTOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["llama_71120"] = module
    spec.loader.exec_module(module)
    return module.AttrConfig, module.LlamaAttributor


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


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


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
        if not n_common:
            continue
        precision = n_common / len(pred)
        recall = n_common / len(gold)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def split_sentences(context: str) -> list[str]:
    chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+", context) if part.strip()]
    return chunks or [context]


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


def make_prompt(question: str, chunks: list[str]) -> tuple[str, list[tuple[int, int]]]:
    prompt = "Answer the question in no more than five words. Context:\n"
    spans = []
    for idx, chunk in enumerate(chunks, start=1):
        prefix = f"[{idx}] "
        start = len(prompt) + len(prefix)
        prompt += prefix + chunk + "\n"
        spans.append((start, start + len(chunk)))
    prompt += f"Question: {question} Answer:"
    return prompt, spans


def char_to_token_spans(offsets: list[tuple[int, int]], char_spans: list[tuple[int, int]]) -> list[list[int]]:
    token_spans = []
    for start, end in char_spans:
        ids = []
        for tok_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= start or tok_start >= end:
                continue
            ids.append(tok_idx)
        token_spans.append(ids)
    return token_spans


def contribution_flow_scores(tool, question: str, chunks: list[str], max_length: int) -> dict[str, Any]:
    prompt, char_spans = make_prompt(question, chunks)
    encoded_offsets = tool.tokenizer(
        prompt,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    offsets = [tuple(x) for x in encoded_offsets.pop("offset_mapping")[0].tolist()]
    encoded = {k: v.to(tool.cfg.device) for k, v in encoded_offsets.items()}
    input_ids = encoded["input_ids"]
    token_spans = char_to_token_spans(offsets, char_spans)

    with torch.no_grad():
        oproj_refs = tool._capture_o_proj_refs(encoded)
        contrib_mats = tool._manual_forward_once_and_calc_matrices(input_ids, oproj_refs)
        contrib = tool._accumulate_last_token(tool.cfg.device, contrib_mats, input_ids.shape[1])

    contrib = np.asarray(contrib, dtype=np.float64)
    contrib = np.maximum(contrib, 0.0)
    chunk_scores = []
    for ids in token_spans:
        valid = [idx for idx in ids if idx < len(contrib)]
        chunk_scores.append(float(contrib[valid].sum()) if valid else 0.0)
    total = sum(chunk_scores) + 1e-12
    normalized = [score / total for score in chunk_scores]

    del contrib_mats, oproj_refs, encoded
    torch.cuda.empty_cache()
    return {
        "prompt": prompt,
        "chunk_scores": normalized,
        "raw_chunk_scores": chunk_scores,
        "token_count": int(input_ids.shape[1]),
    }


def clean_answer(prompt: str, decoded: str) -> str:
    answer = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded
    answer = answer.strip()
    answer = re.split(r"\n|Context:|Question:|Answer:", answer)[0].strip().strip('"')
    if "." in answer:
        answer = answer.split(".")[0].strip()
    return answer


def generate_answer(tool, prompt: str, max_new_tokens: int, max_length: int) -> str:
    encoded = tool.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {k: v.to(tool.cfg.device) for k, v in encoded.items()}
    with torch.no_grad():
        out = tool.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tool.tokenizer.eos_token_id,
        )
    return clean_answer(prompt, tool.tokenizer.decode(out[0], skip_special_tokens=True))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    baseline_f1 = mean([r["baseline_f1"] for r in rows])
    gated_f1 = mean([r["contribution_flow_f1"] for r in rows])
    baseline_steps = mean([r["baseline_steps"] for r in rows])
    gated_steps = mean([r["contribution_flow_steps"] for r in rows])
    return {
        "n": len(rows),
        "baseline_mean_f1": baseline_f1,
        "contribution_flow_mean_f1": gated_f1,
        "f1_delta": gated_f1 - baseline_f1,
        "baseline_mean_steps": baseline_steps,
        "contribution_flow_mean_steps": gated_steps,
        "mean_effort_reduction": baseline_steps - gated_steps,
        "relative_effort_reduction": (baseline_steps - gated_steps) / baseline_steps if baseline_steps else 0.0,
        "answer_preservation_rate": mean([1.0 if r["contribution_flow_f1"] >= r["baseline_f1"] else 0.0 for r in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--data-limit", type=int, default=256)
    parser.add_argument("--eval-limit", type=int, default=8)
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--alpha", type=float, default=0.65, help="weight for contribution flow vs retrieval relevance")
    parser.add_argument("--i-block", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--output-tag", default="llama_contribution_flow_trust_case")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu = configure_gpu()
    AttrConfig, LlamaAttributor = load_llama_attributor()
    tool = LlamaAttributor(AttrConfig(model_path=args.model_path, device="cuda:0", i_block=args.i_block))
    examples = load_answerable(args.data_limit)
    rng = np.random.default_rng(71120)
    examples = [examples[int(i)] for i in rng.permutation(len(examples))][: args.eval_limit]

    rows = []
    for idx, ex in enumerate(examples, start=1):
        chunks = split_sentences(ex["context"])
        order, retrieval_scores_all = rank_chunks(ex["question"], chunks)
        baseline_indices = order[: min(args.fixed_k, len(order))]
        baseline_chunks = [chunks[i] for i in baseline_indices]
        retrieval_scores = [max(0.0, retrieval_scores_all[i]) for i in baseline_indices]
        rel_total = sum(retrieval_scores) + 1e-12
        relevance = [score / rel_total for score in retrieval_scores]

        flow = contribution_flow_scores(tool, ex["question"], baseline_chunks, args.max_length)
        trust_scores = [
            args.alpha * flow_score + (1.0 - args.alpha) * rel_score
            for flow_score, rel_score in zip(flow["chunk_scores"], relevance)
        ]
        selected = []
        for step_idx, trust in enumerate(trust_scores):
            selected.append(step_idx)
            if trust >= args.threshold:
                break
        if not selected:
            selected = list(range(len(baseline_chunks)))

        gated_chunks = [baseline_chunks[i] for i in selected]
        baseline_prompt, _ = make_prompt(ex["question"], baseline_chunks)
        gated_prompt, _ = make_prompt(ex["question"], gated_chunks)
        baseline_answer = generate_answer(tool, baseline_prompt, args.max_new_tokens, args.max_length)
        gated_answer = generate_answer(tool, gated_prompt, args.max_new_tokens, args.max_length)
        row = {
            "question_id": ex["id"],
            "question": ex["question"],
            "answers": ex["answers"],
            "baseline_answer": baseline_answer,
            "contribution_flow_answer": gated_answer,
            "baseline_f1": token_f1(baseline_answer, ex["answers"]),
            "contribution_flow_f1": token_f1(gated_answer, ex["answers"]),
            "baseline_steps": len(baseline_chunks),
            "contribution_flow_steps": len(gated_chunks),
            "contribution_scores_by_step": flow["chunk_scores"],
            "raw_contribution_scores_by_step": flow["raw_chunk_scores"],
            "relevance_scores_by_step": relevance,
            "trust_scores_by_step": trust_scores,
            "stop_step": len(gated_chunks),
            "token_count": flow["token_count"],
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "idx": idx,
                    "question_id": row["question_id"],
                    "baseline_f1": row["baseline_f1"],
                    "contribution_flow_f1": row["contribution_flow_f1"],
                    "baseline_steps": row["baseline_steps"],
                    "contribution_flow_steps": row["contribution_flow_steps"],
                }
            ),
            flush=True,
        )

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
        "note": "Uses the 71120 LLaMA contribution-matrix attributor. Default model is local LLaMA-2-7B chat, not the exact paper checkpoint.",
    }
    json_path = OUT_DIR / f"{args.output_tag}.json"
    md_path = OUT_DIR / f"{args.output_tag}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# LLaMA Contribution-Flow Trust Case",
                "",
                f"Verdict: `{verdict}`",
                "",
                f"- Examples: {agg['n']}",
                f"- Model: {args.model_path}",
                f"- Signal: 71120-style contribution-flow + retrieval relevance",
                f"- Alpha contribution-flow weight: {args.alpha:.2f}",
                f"- Trust threshold: {args.threshold:.3f}",
                f"- Baseline F1: {agg['baseline_mean_f1']:.4f}",
                f"- Contribution-flow F1: {agg['contribution_flow_mean_f1']:.4f}",
                f"- F1 delta: {agg['f1_delta']:.4f}",
                f"- Steps: {agg['baseline_mean_steps']:.4f} -> {agg['contribution_flow_mean_steps']:.4f}",
                f"- Effort reduction: {agg['relative_effort_reduction']:.2%}",
                f"- Answer preservation rate: {agg['answer_preservation_rate']:.2%}",
                f"- Peak CUDA allocation: {result['max_memory_allocated_mib']:.1f} MiB",
                "",
                "This uses the real 71120 LLaMA contribution-matrix code path, but with the locally available LLaMA-2-7B chat checkpoint rather than the exact LLaMA-3.2/Gemma paper checkpoints.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path), "report": str(md_path), "verdict": verdict, "aggregate": agg, "max_memory_allocated_mib": result["max_memory_allocated_mib"]}, indent=2))


if __name__ == "__main__":
    main()
