#!/usr/bin/env python3
"""Same-ID lexical distractor robustness for EvidenceUseGate selective fallback.

This runner keeps the clean validation discipline from the v5 safety-first run:

- train EvidenceUseGate heads on clean train examples;
- select v2/v3/selective-fallback policies on clean validation examples only;
- evaluate once on held-out test IDs after inserting one lexical distractor
  chunk before the clean retrieved chunks.

The goal is not to tune on noise, but to ask whether the selective fallback
reduces v3's wrong-stop risk when retrieval contains a high-overlap distractor.
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


ROOT = Path("/tf/notebooks")
BUNDLE_ROOT = ROOT / "llm_reasoning_agent_memory_pilot_bundle"
MISSION_ROOT = ROOT / "oral_research_memory_mission_2026_06_10"
OUT_DIR = MISSION_ROOT / "v0_experiments/evidence_use_gate_noise_robustness"
V4_SCRIPT = BUNDLE_ROOT / "experiments/run_evidence_use_gate_v4_clean_validation.py"
V5_SCRIPT = MISSION_ROOT / "scripts/run_evidence_use_gate_v5_safety_first.py"
DEFAULT_MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def build_distractor_pool(learned, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for ex in examples:
        for idx, chunk in enumerate(learned.split_sentences(ex["context"])):
            if len(chunk.split()) < 8:
                continue
            pool.append({"source_id": ex["id"], "chunk_index": idx, "chunk": chunk})
    return pool


def choose_lexical_distractor(
    v0,
    learned,
    ex: dict[str, Any],
    clean_chunks: list[str],
    pool: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    clean_set = set(clean_chunks)
    best: tuple[float, float, float, int] | None = None
    best_item: dict[str, Any] | None = None
    for item in pool:
        chunk = item["chunk"]
        if item["source_id"] == ex["id"] or chunk in clean_set:
            continue
        if learned.answer_in_chunk(ex["answers"], chunk):
            continue
        overlap, recall = v0.lexical_features(ex["question"], chunk)
        length_bonus = min(1.0, len(v0.toks(chunk)) / 80.0)
        key = (float(recall), float(overlap), float(length_bonus), -abs(len(v0.toks(chunk)) - 60))
        if best is None or key > best:
            best = key
            best_item = {**item, "lexical_overlap": float(overlap), "question_recall": float(recall)}

    if best_item is not None and (best_item["lexical_overlap"] > 0.0 or best_item["question_recall"] > 0.0):
        return best_item

    fallback = [item for item in pool if item["source_id"] != ex["id"] and item["chunk"] not in clean_set]
    if not fallback:
        return None
    item = dict(fallback[int(rng.integers(0, len(fallback)))])
    overlap, recall = v0.lexical_features(ex["question"], item["chunk"])
    item["lexical_overlap"] = float(overlap)
    item["question_recall"] = float(recall)
    return item


def build_noisy_ranked(
    v0,
    learned,
    ex: dict[str, Any],
    fixed_k: int,
    distractor_pool: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    clean = v0.build_ranked_chunks(learned, ex, fixed_k)
    clean_chunks = [str(x) for x in clean["ranked_chunks"]]
    clean_scores = [float(x) for x in clean["retrieval_scores"]]
    distractor = choose_lexical_distractor(v0, learned, ex, clean_chunks, distractor_pool, rng)
    if distractor is None:
        return {
            "clean_chunks": clean_chunks,
            "clean_retrieval_scores": clean_scores,
            "noisy_chunks": clean_chunks,
            "noisy_retrieval_scores": clean_scores,
            "distractor": None,
            "answer_in_clean": any(learned.answer_in_chunk(ex["answers"], c) for c in clean_chunks),
            "answer_in_noisy": any(learned.answer_in_chunk(ex["answers"], c) for c in clean_chunks),
        }

    top_score = max(clean_scores) if clean_scores else 1.0
    distractor_score = top_score + max(1e-3, abs(top_score) * 0.05)
    if len(clean_chunks) >= fixed_k:
        noisy_chunks = [distractor["chunk"], *clean_chunks[: fixed_k - 1]]
        noisy_scores = [float(distractor_score), *clean_scores[: fixed_k - 1]]
    else:
        noisy_chunks = [distractor["chunk"], *clean_chunks]
        noisy_scores = [float(distractor_score), *clean_scores]

    return {
        "clean_chunks": clean_chunks,
        "clean_retrieval_scores": clean_scores,
        "noisy_chunks": noisy_chunks,
        "noisy_retrieval_scores": noisy_scores,
        "distractor": distractor,
        "answer_in_clean": any(learned.answer_in_chunk(ex["answers"], c) for c in clean_chunks),
        "answer_in_noisy": any(learned.answer_in_chunk(ex["answers"], c) for c in noisy_chunks),
    }


def attention_stop_on_noisy(qcal, flow, tokenizer, model, ex: dict[str, Any], chunks: list[str], retrieval_scores: list[float], args):
    flow_result = flow.attention_flow_scores(tokenizer, model, ex["question"], chunks, args.max_length)
    scored = {
        "flow_scores_norm": qcal.minmax(flow_result["chunk_scores"]),
        "retrieval_scores_norm": qcal.minmax(retrieval_scores),
    }
    combined = qcal.combined_scores(scored, args.teacher_alpha)
    step = qcal.stop_step(combined, args.teacher_threshold)
    return step, {
        "combined_scores_by_step": [float(x) for x in combined],
        "flow_scores_by_step": [float(x) for x in flow_result["chunk_scores"]],
        "retrieval_scores_by_step": [float(x) for x in retrieval_scores],
        "token_count": int(flow_result.get("token_count", 0)),
    }


def metric_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    clean_f1 = mean([float(row["clean_full_f1"]) for row in rows])
    noisy_full_f1 = mean([float(row["noisy_full_f1"]) for row in rows])
    method_f1 = mean([float(row[f"{prefix}_f1"]) for row in rows])
    noisy_full_steps = mean([float(row["noisy_full_steps"]) for row in rows])
    method_steps = mean([float(row[f"{prefix}_steps"]) for row in rows])
    return {
        "n": len(rows),
        "clean_full_mean_f1": clean_f1,
        "noisy_full_mean_f1": noisy_full_f1,
        "method_mean_f1": method_f1,
        "f1_delta_vs_clean_full": method_f1 - clean_f1,
        "f1_delta_vs_noisy_full": method_f1 - noisy_full_f1,
        "noisy_full_mean_steps": noisy_full_steps,
        "method_mean_steps": method_steps,
        "mean_effort_reduction_vs_noisy_full": noisy_full_steps - method_steps,
        "relative_effort_reduction_vs_noisy_full": (
            (noisy_full_steps - method_steps) / noisy_full_steps if noisy_full_steps else 0.0
        ),
        "wrong_stop_rate_vs_clean_full": mean(
            [1.0 if float(row[f"{prefix}_f1"]) < float(row["clean_full_f1"]) - 0.05 else 0.0 for row in rows]
        ),
        "wrong_stop_rate_vs_noisy_full": mean(
            [1.0 if float(row[f"{prefix}_f1"]) < float(row["noisy_full_f1"]) - 0.05 else 0.0 for row in rows]
        ),
        "answer_preservation_vs_clean_full": mean(
            [1.0 if float(row[f"{prefix}_f1"]) >= float(row["clean_full_f1"]) else 0.0 for row in rows]
        ),
        "answer_preservation_vs_noisy_full": mean(
            [1.0 if float(row[f"{prefix}_f1"]) >= float(row["noisy_full_f1"]) else 0.0 for row in rows]
        ),
    }


def baseline_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    return {
        "n": len(rows),
        "mean_f1": mean([float(row[f"{prefix}_f1"]) for row in rows]),
        "mean_steps": mean([float(row[f"{prefix}_steps"]) for row in rows]),
    }


def run_noise_test(
    v0,
    v1,
    v2,
    v3,
    v4,
    qcal,
    learned,
    flow,
    tokenizer,
    model,
    heads: dict[str, Any],
    test_examples: list[dict[str, Any]],
    distractor_pool: list[dict[str, Any]],
    v2_policy: dict[str, Any],
    v3_policy: dict[str, Any],
    selected_policy: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.distractor_seed)
    condition = selected_policy["condition"]
    fallback_source = selected_policy["fallback_source"]
    for idx, ex in enumerate(test_examples, start=1):
        ranked = build_noisy_ranked(v0, learned, ex, args.fixed_k, distractor_pool, rng)
        chunks = ranked["noisy_chunks"]
        retrieval_scores = ranked["noisy_retrieval_scores"]
        total = len(chunks)
        state_records = []
        preds = []
        for step in range(1, total + 1):
            seen = chunks[:step]
            attn = v0.score_prefix_attention(flow, tokenizer, model, ex["question"], seen, args.max_length)
            attn_scores = [float(x) for x in attn["chunk_scores"]]
            features, debug = v0.feature_vector(ex["question"], seen, retrieval_scores[:step], attn_scores, args.fixed_k)
            pred = v2.predict_v2(v1, heads, features)
            preds.append(pred)
            state_records.append({"step": step, "attention_scores_seen": attn_scores, "feature_debug": debug, **pred})

        v2_s = v4.v2_stop(v2, preds, v2_policy, total)
        v3_s = v4.v3_stop(v3, preds, v3_policy, total)
        risk = v4.risk_features_from_state(state_records[v3_s - 1])
        fallback = bool(v4.condition_mask([{"risk_features": risk}], condition)[0])
        if fallback and fallback_source == "v2":
            selective_s = v2_s
            selective_source = "v2_fallback"
        elif fallback and fallback_source == "full_topk":
            selective_s = total
            selective_source = "noisy_full_topk"
        else:
            selective_s = v3_s
            selective_source = "v3_default"

        attention_s, attention_debug = attention_stop_on_noisy(
            qcal, flow, tokenizer, model, ex, chunks, retrieval_scores, args
        )
        attention_debug["labels_by_step"] = [1 if learned.answer_in_chunk(ex["answers"], c) else 0 for c in chunks]

        clean_chunks = ranked["clean_chunks"]
        noised_needed_steps = sorted(set([total, v2_s, v3_s, selective_s, attention_s]))
        noised_answers = {
            stop: v0.generate(flow, tokenizer, model, ex["question"], chunks[:stop], args.max_new_tokens, args.max_length)
            for stop in noised_needed_steps
        }
        clean_answer = v0.generate(
            flow, tokenizer, model, ex["question"], clean_chunks, args.max_new_tokens, args.max_length
        )

        row = {
            "id": ex["id"],
            "question": ex["question"],
            "gold_answers": ex["answers"],
            "distractor_source_id": ranked["distractor"]["source_id"] if ranked["distractor"] else None,
            "distractor_chunk": ranked["distractor"]["chunk"] if ranked["distractor"] else None,
            "distractor_lexical_overlap": ranked["distractor"]["lexical_overlap"] if ranked["distractor"] else 0.0,
            "distractor_question_recall": ranked["distractor"]["question_recall"] if ranked["distractor"] else 0.0,
            "answer_in_clean_chunks": bool(ranked["answer_in_clean"]),
            "answer_in_noisy_chunks": bool(ranked["answer_in_noisy"]),
            "clean_full_answer": clean_answer,
            "clean_full_f1": learned.token_f1(clean_answer, ex["answers"]),
            "clean_full_steps": len(clean_chunks),
            "noisy_full_answer": noised_answers[total],
            "noisy_full_f1": learned.token_f1(noised_answers[total], ex["answers"]),
            "noisy_full_steps": total,
            "calibrated_attention_answer": noised_answers[attention_s],
            "calibrated_attention_f1": learned.token_f1(noised_answers[attention_s], ex["answers"]),
            "calibrated_attention_steps": attention_s,
            "v2_answer": noised_answers[v2_s],
            "v2_f1": learned.token_f1(noised_answers[v2_s], ex["answers"]),
            "v2_steps": v2_s,
            "v3_answer": noised_answers[v3_s],
            "v3_f1": learned.token_f1(noised_answers[v3_s], ex["answers"]),
            "v3_steps": v3_s,
            "selective_answer": noised_answers[selective_s],
            "selective_f1": learned.token_f1(noised_answers[selective_s], ex["answers"]),
            "selective_steps": selective_s,
            "selective_source": selective_source,
            "fallback_triggered": fallback,
            "risk_features": risk,
            "state_records": state_records,
            "attention_debug": attention_debug,
        }
        row["v3_wrong_stop_vs_clean_full"] = bool(row["v3_f1"] < row["clean_full_f1"] - 0.05)
        row["selective_wrong_stop_vs_clean_full"] = bool(row["selective_f1"] < row["clean_full_f1"] - 0.05)
        rows.append(row)
        print(
            json.dumps(
                {
                    "phase": "noise_test",
                    "idx": idx,
                    "question_id": ex["id"],
                    "answer_in_noisy": row["answer_in_noisy_chunks"],
                    "clean_f1": row["clean_full_f1"],
                    "noisy_full_f1": row["noisy_full_f1"],
                    "v3_f1": row["v3_f1"],
                    "selective_f1": row["selective_f1"],
                    "v3_steps": v3_s,
                    "selective_steps": selective_s,
                    "selective_source": selective_source,
                }
            ),
            flush=True,
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_recompute_script(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np

def mean(xs):
    return float(np.mean(xs)) if xs else 0.0

def summary(rows, prefix):
    clean = mean([r["clean_full_f1"] for r in rows])
    noisy = mean([r["noisy_full_f1"] for r in rows])
    f1 = mean([r[f"{prefix}_f1"] for r in rows])
    base_steps = mean([r["noisy_full_steps"] for r in rows])
    steps = mean([r[f"{prefix}_steps"] for r in rows])
    return {
        "n": len(rows),
        "method_mean_f1": f1,
        "f1_delta_vs_clean_full": f1 - clean,
        "f1_delta_vs_noisy_full": f1 - noisy,
        "method_mean_steps": steps,
        "relative_effort_reduction_vs_noisy_full": (base_steps - steps) / base_steps if base_steps else 0.0,
        "wrong_stop_rate_vs_clean_full": mean([r[f"{prefix}_f1"] < r["clean_full_f1"] - 0.05 for r in rows]),
        "wrong_stop_rate_vs_noisy_full": mean([r[f"{prefix}_f1"] < r["noisy_full_f1"] - 0.05 for r in rows]),
    }

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
print(json.dumps({
    "clean_full": {"mean_f1": mean([r["clean_full_f1"] for r in rows]), "mean_steps": mean([r["clean_full_steps"] for r in rows])},
    "noisy_full": {"mean_f1": mean([r["noisy_full_f1"] for r in rows]), "mean_steps": mean([r["noisy_full_steps"] for r in rows])},
    "calibrated_attention": summary(rows, "calibrated_attention"),
    "v2": summary(rows, "v2"),
    "v3": summary(rows, "v3"),
    "selective": summary(rows, "selective"),
    "fallback_count": sum(1 for r in rows if r["fallback_triggered"]),
}, indent=2))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_report(result: dict[str, Any], md_path: Path) -> None:
    lines = [
        "# EvidenceUseGate Noise Robustness",
        "",
        f"Verdict: `{result['verdict']}`",
        "",
        "- Status: clean-validation-selected policy tested on same held-out IDs with lexical distractor insertion",
        "- Policy selection: train/validation are clean; no thresholds are selected on noisy test rows",
        "- Distractor: one high lexical-overlap chunk from another SQuAD2 example inserted before clean retrieved chunks",
        f"- Train/val/test examples: {result['splits']['train_examples']} / {result['splits']['val_examples']} / {result['splits']['test_examples']}",
        f"- GPU: {result['gpu']['device_name']} via CUDA_VISIBLE_DEVICES={result['gpu']['visible_cuda_devices']}",
        "",
        "## Selected Clean-Validation Policy",
        "",
        f"- Condition: `{result['selected_policy']['condition_text']}`",
        f"- Fallback source: `{result['selected_policy']['fallback_source']}`",
        f"- Noisy test fallback count: {result['noise_stats']['fallback_count']} / {result['noise_stats']['n']}",
        "",
        "## Noise Stats",
        "",
        f"- Answer retained in noised chunks: {result['noise_stats']['answer_retained_rate']:.2%}",
        f"- Mean distractor lexical overlap: {result['noise_stats']['mean_distractor_lexical_overlap']:.4f}",
        f"- Mean distractor question recall: {result['noise_stats']['mean_distractor_question_recall']:.4f}",
        "",
        "## Same-ID Noise Comparison",
        "",
        "| Method | F1 | Delta vs clean full | Delta vs noisy full | Steps | Effort vs noisy full | Wrong-stop vs clean | Wrong-stop vs noisy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Clean full top-k | {result['baselines']['clean_full']['mean_f1']:.4f} | +0.0000 | n/a | {result['baselines']['clean_full']['mean_steps']:.4f} | 0.00% | 0.00% | n/a |",
        f"| Noisy full top-k | {result['baselines']['noisy_full']['mean_f1']:.4f} | {result['baselines']['noisy_full']['mean_f1'] - result['baselines']['clean_full']['mean_f1']:+.4f} | +0.0000 | {result['baselines']['noisy_full']['mean_steps']:.4f} | 0.00% | {result['baselines']['noisy_full_wrong_stop_vs_clean']:.2%} | 0.00% |",
    ]
    for key, label in [
        ("calibrated_attention", "Calibrated attention-flow"),
        ("v2", "EvidenceUseGate-v2"),
        ("v3", "EvidenceUseGate-v3"),
        ("selective", "Selective fallback"),
    ]:
        metrics = result["test_metrics"][key]
        lines.append(
            f"| {label} | {metrics['method_mean_f1']:.4f} | {metrics['f1_delta_vs_clean_full']:+.4f} | "
            f"{metrics['f1_delta_vs_noisy_full']:+.4f} | {metrics['noisy_full_mean_steps']:.4f} -> {metrics['method_mean_steps']:.4f} | "
            f"{metrics['relative_effort_reduction_vs_noisy_full']:.2%} | {metrics['wrong_stop_rate_vs_clean_full']:.2%} | "
            f"{metrics['wrong_stop_rate_vs_noisy_full']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["interpretation"],
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--learn-limit", type=int, default=1536)
    parser.add_argument("--train-examples", type=int, default=120)
    parser.add_argument("--val-examples", type=int, default=60)
    parser.add_argument("--eval-examples", type=int, default=100)
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument("--v2-risk-budget", type=float, default=0.020)
    parser.add_argument("--v3-risk-budget", type=float, default=0.005)
    parser.add_argument("--utility-lambda", type=float, default=0.012)
    parser.add_argument("--max-f1-drop", type=float, default=0.005)
    parser.add_argument("--min-effort-reduction", type=float, default=0.35)
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
    parser.add_argument("--distractor-seed", type=int, default=20260611)
    parser.add_argument("--output-tag", default="evidence_use_gate_noise_robustness_shifted_100")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v4 = load_module("evidence_use_gate_v4_clean_validation", V4_SCRIPT)
    v5 = load_module("evidence_use_gate_v5_safety_first", V5_SCRIPT)
    v4.search_fallback_policy = v5.make_safety_first_search(v4)

    v2 = v4.load_module("evidence_use_gate_v2", v4.V2_SCRIPT)
    v3 = v4.load_module("evidence_use_gate_v3", v4.V3_SCRIPT)
    gpu = v2.configure_gpu()
    v0 = v2.load_module("evidence_use_gate_v0", v2.V0_SCRIPT)
    v1 = v2.load_module("evidence_use_gate_v1", v2.V1_SCRIPT)
    qcal = v2.load_module("qwen_calibrated_flow", v2.QCAL_SCRIPT)
    learned = v0.load_module("learned_trust", v0.LEARNED_SCRIPT)
    flow = v0.load_module("qwen_flow", v0.FLOW_SCRIPT)

    examples = learned.load_answerable(args.learn_limit)
    rng = np.random.default_rng(20260610)
    examples = [examples[int(i)] for i in rng.permutation(len(examples))]
    train_examples = examples[: args.train_examples]
    val_examples = examples[args.train_examples : args.train_examples + args.val_examples]
    test_start = args.train_examples + args.val_examples
    test_examples = examples[test_start : test_start + args.eval_examples]
    distractor_pool = build_distractor_pool(learned, examples)

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

    train_rows, train_summaries, train_attention_teacher = v2.build_teacher_rows(
        v0, v1, qcal, learned, flow, tokenizer, model, train_examples, args, "teacher_train"
    )
    val_rows, val_summaries, val_attention_teacher = v2.build_teacher_rows(
        v0, v1, qcal, learned, flow, tokenizer, model, val_examples, args, "teacher_val"
    )
    heads, train_metrics = v2.fit_v2_heads(v1, train_rows)
    validation_metrics = v2.validation_head_metrics(v1, heads, val_rows)
    v2_policy = v2.choose_policies(v1, heads, val_summaries, val_rows, [args.v2_risk_budget], args.utility_lambda)[0]
    v3_policy = v3.choose_policies(
        v2,
        v1,
        heads,
        val_summaries,
        val_rows,
        [args.v3_risk_budget],
        args.max_f1_drop,
        args.max_wrong_stop,
    )[0]
    val_decision_rows = v4.build_validation_decision_rows(v1, v2, v3, heads, val_summaries, val_rows, v2_policy, v3_policy)
    selected_policy, top_candidates = v4.search_fallback_policy(
        val_decision_rows,
        max_f1_drop=args.max_f1_drop,
        min_effort_reduction=args.min_effort_reduction,
        max_wrong_stop=args.max_wrong_stop,
    )

    test_rows = run_noise_test(
        v0,
        v1,
        v2,
        v3,
        v4,
        qcal,
        learned,
        flow,
        tokenizer,
        model,
        heads,
        test_examples,
        distractor_pool,
        v2_policy,
        v3_policy,
        selected_policy,
        args,
    )

    baselines = {
        "clean_full": baseline_summary(test_rows, "clean_full"),
        "noisy_full": baseline_summary(test_rows, "noisy_full"),
    }
    baselines["noisy_full_wrong_stop_vs_clean"] = mean(
        [1.0 if row["noisy_full_f1"] < row["clean_full_f1"] - 0.05 else 0.0 for row in test_rows]
    )
    test_metrics = {
        "calibrated_attention": metric_summary(test_rows, "calibrated_attention"),
        "v2": metric_summary(test_rows, "v2"),
        "v3": metric_summary(test_rows, "v3"),
        "selective": metric_summary(test_rows, "selective"),
    }
    fallback_count = int(sum(1 for row in test_rows if row["fallback_triggered"]))
    noise_stats = {
        "n": len(test_rows),
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(test_rows) if test_rows else 0.0,
        "answer_retained_rate": mean([1.0 if row["answer_in_noisy_chunks"] else 0.0 for row in test_rows]),
        "mean_distractor_lexical_overlap": mean([float(row["distractor_lexical_overlap"]) for row in test_rows]),
        "mean_distractor_question_recall": mean([float(row["distractor_question_recall"]) for row in test_rows]),
    }

    v3_metrics = test_metrics["v3"]
    selective_metrics = test_metrics["selective"]
    noise_target_pass = (
        selective_metrics["wrong_stop_rate_vs_clean_full"] < v3_metrics["wrong_stop_rate_vs_clean_full"]
        and selective_metrics["method_mean_f1"] >= v3_metrics["method_mean_f1"]
    )
    if noise_target_pass:
        interpretation = (
            "Selective fallback improved the noisy setting by lowering v3 wrong-stop rate while preserving or improving mean F1. "
            "Because policy thresholds were selected on clean validation only, this is a useful robustness signal."
        )
    else:
        interpretation = (
            "Selective fallback did not meet the proposed noise robustness criterion on this run. "
            "This means the current clean-selected fallback is not yet a reliable noise guardrail; the next version should train or validate with explicit hard distractor labels."
        )

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
            "test_question_ids": [ex["id"] for ex in test_examples],
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "v2_policy": v2_policy,
        "v3_policy": v3_policy,
        "selected_policy": {
            "condition": selected_policy["condition"],
            "condition_text": selected_policy["condition_text"],
            "fallback_source": selected_policy["fallback_source"],
            "validation_metrics": selected_policy["validation_metrics"],
            "validation_target_pass": selected_policy["validation_target_pass"],
        },
        "top_validation_candidates": [
            {
                "condition": candidate["condition"],
                "condition_text": candidate["condition_text"],
                "fallback_source": candidate["fallback_source"],
                "validation_metrics": candidate["validation_metrics"],
                "validation_target_pass": candidate["validation_target_pass"],
                "selection_score": candidate["selection_score"],
            }
            for candidate in top_candidates
        ],
        "noise_stats": noise_stats,
        "baselines": baselines,
        "test_metrics": test_metrics,
        "verdict": "PASS_NOISE_ROBUSTNESS_TARGET" if noise_target_pass else "NO_NOISE_ROBUSTNESS_TARGET_PASS",
        "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 3),
        "train_attention_teacher_records": train_attention_teacher,
        "val_attention_teacher_records": val_attention_teacher,
        "test_rows": test_rows,
        "interpretation": interpretation,
        "note": "Noise thresholds are not tuned on noisy test rows; distractors are lexical-bait chunks from other examples.",
    }

    json_path = OUT_DIR / f"{args.output_tag}.json"
    jsonl_path = OUT_DIR / f"{args.output_tag}.jsonl"
    md_path = OUT_DIR / f"{args.output_tag}.md"
    recompute_path = OUT_DIR / "recompute_noise_metrics.py"
    commands_path = OUT_DIR / "commands.txt"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(jsonl_path, test_rows)
    write_report(result, md_path)
    write_recompute_script(recompute_path)
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
                "jsonl": str(jsonl_path),
                "report": str(md_path),
                "recompute": str(recompute_path),
                "verdict": result["verdict"],
                "noise_stats": noise_stats,
                "test_metrics": test_metrics,
                "max_memory_allocated_mib": result["max_memory_allocated_mib"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
