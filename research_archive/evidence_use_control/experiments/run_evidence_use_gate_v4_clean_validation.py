#!/usr/bin/env python3
"""EvidenceUseGate-v4 clean validation.

This promotes the v4 selective-fallback diagnostic into a defensible validation:

- rebuild train/validation/test states on the same deterministic SQuAD2 split;
- fit the same EvidenceUseGate safety heads on train states;
- choose v2 fallback and v3 default policies on validation states;
- choose the v4 fallback threshold on validation states only;
- evaluate once on held-out test IDs.

The method under test:

    default = EvidenceUseGate-v3 guarded distill
    if learned risk selector says "unsafe":
        fallback to EvidenceUseGate-v2 or full top-k

The intended prototype stop criterion is:

- F1 delta >= -0.005
- effort reduction >= 35%
- wrong-stop rate <= 5%
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
OUT_DIR = BUNDLE_ROOT / "experiments/out/evidence_use_gate_v4_clean_validation"
V2_SCRIPT = BUNDLE_ROOT / "experiments/run_evidence_use_gate_v2_pareto.py"
V3_SCRIPT = BUNDLE_ROOT / "experiments/run_evidence_use_gate_v3_guarded_distill.py"
DEFAULT_MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
DEFAULT_REFERENCE_V3 = (
    BUNDLE_ROOT / "experiments/out/evidence_use_gate_v3_guarded_distill/evidence_use_gate_v3_guarded_distill_100.json"
)


RISK_FEATURE_DIRECTIONS = {
    "noise_risk": "hi",
    "predicted_f1_drop": "hi",
    "expected_f1_drop": "hi",
    "continue_advantage": "hi",
    "uncertainty": "lo",
    "stop_prob": "lo",
    "expected_f1_if_stop": "lo",
    "attention_teacher_stop_prob": "lo",
    "fd_attention_relevance_alignment": "lo",
    "fd_current_lexical_overlap": "lo",
    "fd_current_question_recall": "lo",
    "fd_max_lexical_overlap_seen": "lo",
    "fd_max_question_recall_seen": "lo",
    "fd_attention_concentration": "hi",
    "fd_retrieval_margin": "hi",
    "fd_current_retrieval": "lo",
    "fd_current_chunk_len_128": "lo",
    "fd_cumulative_chunk_len_512": "lo",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def metric_summary(rows: list[dict[str, Any]], method_prefix: str) -> dict[str, float]:
    baseline_f1 = mean([float(row["baseline_f1"]) for row in rows])
    method_f1 = mean([float(row[f"{method_prefix}_f1"]) for row in rows])
    baseline_steps = mean([float(row["baseline_steps"]) for row in rows])
    method_steps = mean([float(row[f"{method_prefix}_steps"]) for row in rows])
    return {
        "n": len(rows),
        "baseline_mean_f1": baseline_f1,
        "method_mean_f1": method_f1,
        "f1_delta": method_f1 - baseline_f1,
        "baseline_mean_steps": baseline_steps,
        "method_mean_steps": method_steps,
        "mean_effort_reduction": baseline_steps - method_steps,
        "relative_effort_reduction": (baseline_steps - method_steps) / baseline_steps if baseline_steps else 0.0,
        "answer_preservation_rate": mean(
            [1.0 if float(row[f"{method_prefix}_f1"]) >= float(row["baseline_f1"]) else 0.0 for row in rows]
        ),
        "wrong_stop_rate": mean(
            [1.0 if float(row[f"{method_prefix}_f1"]) < float(row["baseline_f1"]) - 0.05 else 0.0 for row in rows]
        ),
    }


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(row["question_id"], []).append(row)
    return by_id


def v2_stop(v2, preds: list[dict[str, float]], policy: dict[str, Any], total_steps: int) -> int:
    stop = total_steps
    for idx, pred in enumerate(preds, start=1):
        if v2.stop_condition(pred, idx, total_steps, policy):
            return idx
    return stop


def v3_stop(v3, preds: list[dict[str, float]], policy: dict[str, Any], total_steps: int) -> int:
    stop = total_steps
    for idx, pred in enumerate(preds, start=1):
        if v3.guarded_stop_condition(pred, idx, total_steps, policy):
            return idx
    return stop


def risk_features_from_state(state_record: dict[str, Any]) -> dict[str, float]:
    features = {
        "v3_step": float(state_record["step"]),
        "stop_prob": float(state_record["stop_prob"]),
        "expected_f1_if_stop": float(state_record["expected_f1_if_stop"]),
        "continue_value": float(state_record["continue_value"]),
        "noise_risk": float(state_record["noise_risk"]),
        "uncertainty": float(state_record["uncertainty"]),
        "attention_teacher_stop_prob": float(state_record["attention_teacher_stop_prob"]),
        "predicted_f1_drop": float(state_record["predicted_f1_drop"]),
        "expected_f1_drop": float(state_record["expected_f1_drop"]),
        "continue_advantage": float(state_record["continue_value"] - state_record["expected_f1_if_stop"]),
    }
    for key, value in state_record.get("feature_debug", {}).items():
        if isinstance(value, (int, float)):
            features[f"fd_{key}"] = float(value)
    return features


def risk_features_from_val_row(row: dict[str, Any], pred: dict[str, float]) -> dict[str, float]:
    state_record = {"step": row["step"], "feature_debug": row.get("feature_debug", {}), **pred}
    return risk_features_from_state(state_record)


def condition_mask(rows: list[dict[str, Any]], condition: dict[str, Any]) -> np.ndarray:
    if condition["kind"] == "none":
        return np.zeros(len(rows), dtype=bool)
    if condition["kind"] == "threshold":
        values = np.asarray([row["risk_features"][condition["feature"]] for row in rows], dtype=float)
        if condition["direction"] == "hi":
            return values >= float(condition["threshold"])
        return values <= float(condition["threshold"])
    if condition["kind"] == "compound":
        left = condition_mask(rows, condition["left"])
        right = condition_mask(rows, condition["right"])
        if condition["op"] == "or":
            return left | right
        if condition["op"] == "and":
            return left & right
        raise ValueError(f"unknown op {condition['op']}")
    raise ValueError(f"unknown condition kind {condition['kind']}")


def condition_text(condition: dict[str, Any]) -> str:
    if condition["kind"] == "none":
        return "never fallback"
    if condition["kind"] == "threshold":
        op = ">=" if condition["direction"] == "hi" else "<="
        return f"{condition['feature']} {op} {condition['threshold']:.6f}"
    if condition["kind"] == "compound":
        return f"({condition_text(condition['left'])}) {condition['op'].upper()} ({condition_text(condition['right'])})"
    return str(condition)


def apply_fallback_policy(rows: list[dict[str, Any]], condition: dict[str, Any], fallback_source: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    mask = condition_mask(rows, condition)
    out = []
    for fallback, row in zip(mask, rows):
        item = dict(row)
        item["fallback_triggered"] = bool(fallback)
        if fallback and fallback_source == "v2":
            item["v4_source"] = "v2_fallback"
            item["v4_answer"] = row["v2_answer"]
            item["v4_f1"] = row["v2_f1"]
            item["v4_steps"] = row["v2_steps"]
        elif fallback and fallback_source == "full_topk":
            item["v4_source"] = "full_topk"
            item["v4_answer"] = row["baseline_answer"]
            item["v4_f1"] = row["baseline_f1"]
            item["v4_steps"] = row["baseline_steps"]
        else:
            item["v4_source"] = "v3_default"
            item["v4_answer"] = row["v3_answer"]
            item["v4_f1"] = row["v3_f1"]
            item["v4_steps"] = row["v3_steps"]
        item["v4_wrong_stop"] = bool(float(item["v4_f1"]) < float(item["baseline_f1"]) - 0.05)
        out.append(item)
    metrics = metric_summary(out, "v4")
    metrics["fallback_count"] = int(mask.sum())
    metrics["fallback_rate"] = float(mask.mean()) if len(mask) else 0.0
    return out, metrics


def threshold_grid(rows: list[dict[str, Any]], feature: str) -> list[float]:
    values = np.asarray([row["risk_features"][feature] for row in rows], dtype=float)
    qs = np.quantile(values, np.linspace(0.05, 0.95, 19))
    return sorted(set(float(x) for x in qs))


def search_fallback_policy(
    val_rows: list[dict[str, Any]],
    max_f1_drop: float,
    min_effort_reduction: float,
    max_wrong_stop: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    simple_conditions = []
    for feature, direction in RISK_FEATURE_DIRECTIONS.items():
        if feature not in val_rows[0]["risk_features"]:
            continue
        for threshold in threshold_grid(val_rows, feature):
            simple_conditions.append(
                {
                    "kind": "threshold",
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                }
            )
    conditions: list[dict[str, Any]] = [{"kind": "none"}, *simple_conditions]
    for idx, left in enumerate(simple_conditions):
        for right in simple_conditions[idx + 1 :]:
            if left["feature"] == right["feature"]:
                continue
            for op in ("or", "and"):
                conditions.append({"kind": "compound", "op": op, "left": left, "right": right})

    candidates = []
    for condition in conditions:
        for fallback_source in ("v2", "full_topk"):
            _, metrics = apply_fallback_policy(val_rows, condition, fallback_source)
            target_pass = (
                metrics["f1_delta"] >= -max_f1_drop
                and metrics["relative_effort_reduction"] >= min_effort_reduction
                and metrics["wrong_stop_rate"] <= max_wrong_stop
            )
            score = (
                10.0 * float(target_pass)
                + metrics["relative_effort_reduction"]
                + 0.20 * metrics["f1_delta"]
                - 0.50 * metrics["wrong_stop_rate"]
                - 0.01 * metrics["fallback_rate"]
            )
            candidates.append(
                {
                    "condition": condition,
                    "condition_text": condition_text(condition),
                    "fallback_source": fallback_source,
                    "validation_metrics": metrics,
                    "validation_target_pass": bool(target_pass),
                    "selection_score": float(score),
                }
            )

    passing = [candidate for candidate in candidates if candidate["validation_target_pass"]]
    if passing:
        best = max(
            passing,
            key=lambda c: (
                c["validation_metrics"]["relative_effort_reduction"],
                c["validation_metrics"]["f1_delta"],
                -c["validation_metrics"]["fallback_count"],
            ),
        )
    else:
        best = max(
            candidates,
            key=lambda c: (
                c["validation_metrics"]["relative_effort_reduction"]
                - max(0.0, c["validation_metrics"]["wrong_stop_rate"] - max_wrong_stop)
                - max(0.0, -max_f1_drop - c["validation_metrics"]["f1_delta"]),
                c["validation_metrics"]["f1_delta"],
            ),
        )
    top = sorted(
        candidates,
        key=lambda c: (
            c["validation_target_pass"],
            c["validation_metrics"]["relative_effort_reduction"],
            c["validation_metrics"]["f1_delta"],
            -c["validation_metrics"]["wrong_stop_rate"],
        ),
        reverse=True,
    )[:25]
    return best, top


def build_validation_decision_rows(
    v1,
    v2,
    v3,
    heads: dict[str, Any],
    val_summaries: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    v2_policy: dict[str, Any],
    v3_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    rows_by_id = group_rows(val_rows)
    decision_rows = []
    for summary in val_summaries:
        qid = summary["question_id"]
        state_rows = rows_by_id[qid]
        preds = [v2.predict_v2(v1, heads, row["features"]) for row in state_rows]
        total = len(state_rows)
        v2_s = v2_stop(v2, preds, v2_policy, total)
        v3_s = v3_stop(v3, preds, v3_policy, total)
        baseline_f1 = float(summary["f1_by_step"][-1])
        risk = risk_features_from_val_row(state_rows[v3_s - 1], preds[v3_s - 1])
        decision_rows.append(
            {
                "id": qid,
                "question": summary["question"],
                "gold_answers": summary["answers"],
                "baseline_answer": summary["answers_by_step"][-1],
                "baseline_f1": baseline_f1,
                "baseline_steps": total,
                "v2_answer": summary["answers_by_step"][v2_s - 1],
                "v2_f1": float(summary["f1_by_step"][v2_s - 1]),
                "v2_steps": v2_s,
                "v3_answer": summary["answers_by_step"][v3_s - 1],
                "v3_f1": float(summary["f1_by_step"][v3_s - 1]),
                "v3_steps": v3_s,
                "risk_features": risk,
            }
        )
    return decision_rows


def run_test(
    v0,
    v1,
    v2,
    v3,
    qcal,
    learned,
    flow,
    tokenizer,
    model,
    heads: dict[str, Any],
    test_examples: list[dict[str, Any]],
    v2_policy: dict[str, Any],
    v3_policy: dict[str, Any],
    v4_policy: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    condition = v4_policy["condition"]
    fallback_source = v4_policy["fallback_source"]
    rows = []
    for idx, ex in enumerate(test_examples, start=1):
        ranked = v0.build_ranked_chunks(learned, ex, args.fixed_k)
        chunks = ranked["ranked_chunks"]
        retrieval_scores = ranked["retrieval_scores"]
        state_records = []
        preds = []
        for step in range(1, len(chunks) + 1):
            seen = chunks[:step]
            attn = v0.score_prefix_attention(flow, tokenizer, model, ex["question"], seen, args.max_length)
            attn_scores = [float(x) for x in attn["chunk_scores"]]
            features, debug = v0.feature_vector(ex["question"], seen, retrieval_scores[:step], attn_scores, args.fixed_k)
            pred = v2.predict_v2(v1, heads, features)
            preds.append(pred)
            state_records.append({"step": step, "attention_scores_seen": attn_scores, "feature_debug": debug, **pred})

        total = len(chunks)
        v2_s = v2_stop(v2, preds, v2_policy, total)
        v3_s = v3_stop(v3, preds, v3_policy, total)
        risk = risk_features_from_state(state_records[v3_s - 1])
        probe_row = {
            "risk_features": risk,
            "baseline_f1": 0.0,
            "baseline_steps": total,
            "v2_f1": 0.0,
            "v2_steps": v2_s,
            "v2_answer": "",
            "v3_f1": 0.0,
            "v3_steps": v3_s,
            "v3_answer": "",
            "baseline_answer": "",
        }
        fallback = bool(condition_mask([probe_row], condition)[0])
        if fallback and fallback_source == "v2":
            v4_s = v2_s
            v4_source = "v2_fallback"
        elif fallback and fallback_source == "full_topk":
            v4_s = total
            v4_source = "full_topk"
        else:
            v4_s = v3_s
            v4_source = "v3_default"

        attention_step, attention_scored = v2.attention_teacher_step(
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
        needed_steps = sorted(set([total, v2_s, v3_s, v4_s, attention_step]))
        answer_by_step = {}
        for stop in needed_steps:
            answer_by_step[stop] = v0.generate(
                flow,
                tokenizer,
                model,
                ex["question"],
                chunks[:stop],
                args.max_new_tokens,
                args.max_length,
            )
        baseline_answer = answer_by_step[total]
        v2_answer = answer_by_step[v2_s]
        v3_answer = answer_by_step[v3_s]
        v4_answer = answer_by_step[v4_s]
        attention_answer = answer_by_step[attention_step]
        row = {
            "id": ex["id"],
            "question": ex["question"],
            "gold_answers": ex["answers"],
            "baseline_answer": baseline_answer,
            "baseline_f1": learned.token_f1(baseline_answer, ex["answers"]),
            "baseline_steps": total,
            "calibrated_attention_answer": attention_answer,
            "calibrated_attention_f1": learned.token_f1(attention_answer, ex["answers"]),
            "calibrated_attention_steps": attention_step,
            "v2_answer": v2_answer,
            "v2_f1": learned.token_f1(v2_answer, ex["answers"]),
            "v2_steps": v2_s,
            "v3_answer": v3_answer,
            "v3_f1": learned.token_f1(v3_answer, ex["answers"]),
            "v3_steps": v3_s,
            "v4_answer": v4_answer,
            "v4_f1": learned.token_f1(v4_answer, ex["answers"]),
            "v4_steps": v4_s,
            "v4_source": v4_source,
            "fallback_triggered": fallback,
            "risk_features": risk,
            "state_records": state_records,
            "attention_teacher_combined_scores_by_step": [float(x) for x in attention_scored["combined_scores"]],
            "attention_teacher_labels_by_step": [int(x) for x in attention_scored["labels"]],
        }
        row["v4_wrong_stop"] = bool(row["v4_f1"] < row["baseline_f1"] - 0.05)
        rows.append(row)
        print(
            json.dumps(
                {
                    "phase": "test",
                    "idx": idx,
                    "question_id": ex["id"],
                    "baseline_f1": row["baseline_f1"],
                    "v2_f1": row["v2_f1"],
                    "v3_f1": row["v3_f1"],
                    "v4_f1": row["v4_f1"],
                    "v2_steps": v2_s,
                    "v3_steps": v3_s,
                    "v4_steps": v4_s,
                    "v4_source": v4_source,
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


def summary(rows, prefix):
    bf = np.mean([r["baseline_f1"] for r in rows])
    mf = np.mean([r[f"{prefix}_f1"] for r in rows])
    bs = np.mean([r["baseline_steps"] for r in rows])
    ms = np.mean([r[f"{prefix}_steps"] for r in rows])
    return {
        "n": len(rows),
        "baseline_mean_f1": float(bf),
        "method_mean_f1": float(mf),
        "f1_delta": float(mf - bf),
        "baseline_mean_steps": float(bs),
        "method_mean_steps": float(ms),
        "relative_effort_reduction": float((bs - ms) / bs),
        "wrong_stop_rate": float(np.mean([r[f"{prefix}_f1"] < r["baseline_f1"] - 0.05 for r in rows])),
        "answer_preservation_rate": float(np.mean([r[f"{prefix}_f1"] >= r["baseline_f1"] for r in rows])),
    }


rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
print(json.dumps({
    "calibrated_attention": summary(rows, "calibrated_attention"),
    "v2": summary(rows, "v2"),
    "v3": summary(rows, "v3"),
    "v4": summary(rows, "v4"),
}, indent=2))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_report(result: dict[str, Any], md_path: Path) -> None:
    lines = [
        "# EvidenceUseGate-v4 Clean Validation",
        "",
        f"Verdict: `{result['verdict']}`",
        "",
        "- Status: clean validation-selected fallback policy",
        "- Default policy: EvidenceUseGate-v3 guarded distill",
        "- Fallback choices searched on validation only: v2 or full top-k",
        "- Held-out test: one-shot evaluation on the same deterministic SQuAD2 IDs used by v1/v2/v3",
        f"- Train/val/test examples: {result['splits']['train_examples']} / {result['splits']['val_examples']} / {result['splits']['test_examples']}",
        f"- GPU: {result['gpu']['device_name']} via CUDA_VISIBLE_DEVICES={result['gpu']['visible_cuda_devices']}",
        "",
        "## Selected Fallback Policy",
        "",
        f"- Condition: `{result['selected_policy']['condition_text']}`",
        f"- Fallback source: `{result['selected_policy']['fallback_source']}`",
        f"- Validation fallback count: {result['selected_policy']['validation_metrics']['fallback_count']} / {result['selected_policy']['validation_metrics']['n']}",
        f"- Test fallback count: {result['test_metrics']['v4']['fallback_count']} / {result['test_metrics']['v4']['n']}",
        "",
        "## Same-Slice Test Comparison",
        "",
        "| Method | F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = [
        ("calibrated_attention", "Calibrated attention-flow"),
        ("v2", "EvidenceUseGate-v2"),
        ("v3", "EvidenceUseGate-v3"),
        ("v4", "EvidenceUseGate-v4 clean"),
    ]
    for key, label in names:
        metrics = result["test_metrics"][key]
        lines.append(
            f"| {label} | {metrics['method_mean_f1']:.4f} | {metrics['f1_delta']:+.4f} | "
            f"{metrics['baseline_mean_steps']:.4f} -> {metrics['method_mean_steps']:.4f} | "
            f"{metrics['relative_effort_reduction']:.2%} | {metrics['wrong_stop_rate']:.2%} | "
            f"{metrics['answer_preservation_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Validation-Selected Candidate Policies",
            "",
            "| Rank | Val pass | Fallback | Condition | Val F1 delta | Val effort | Val wrong-stop | Val fallbacks |",
            "|---:|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for idx, candidate in enumerate(result["top_validation_candidates"][:10], start=1):
        metrics = candidate["validation_metrics"]
        pass_text = "yes" if candidate["validation_target_pass"] else "no"
        lines.append(
            f"| {idx} | {pass_text} | {candidate['fallback_source']} | `{candidate['condition_text']}` | "
            f"{metrics['f1_delta']:+.4f} | {metrics['relative_effort_reduction']:.2%} | "
            f"{metrics['wrong_stop_rate']:.2%} | {metrics['fallback_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is the clean test of whether the v4 fallback selector generalizes from validation to held-out test. It should be used as the defensible method result, while the earlier v4 selective-fallback file remains a post-hoc diagnostic.",
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
    parser.add_argument("--reference-v3-json", type=Path, default=DEFAULT_REFERENCE_V3)
    parser.add_argument("--output-tag", default="evidence_use_gate_v4_clean_validation_100")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v2 = load_module("evidence_use_gate_v2", V2_SCRIPT)
    v3 = load_module("evidence_use_gate_v3", V3_SCRIPT)
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
    test_ids = [ex["id"] for ex in test_examples]
    reference_ids_match = None
    if args.reference_v3_json.exists():
        ref = json.loads(args.reference_v3_json.read_text(encoding="utf-8"))
        reference_ids = [row["question_id"] for row in ref.get("test_rows", [])]
        reference_ids_match = reference_ids == test_ids
        if not reference_ids_match:
            raise RuntimeError("Clean v4 test IDs do not match the saved v3 test IDs")

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
    val_decision_rows = build_validation_decision_rows(v1, v2, v3, heads, val_summaries, val_rows, v2_policy, v3_policy)
    selected_policy, top_candidates = search_fallback_policy(
        val_decision_rows,
        max_f1_drop=args.max_f1_drop,
        min_effort_reduction=args.min_effort_reduction,
        max_wrong_stop=args.max_wrong_stop,
    )

    test_rows = run_test(
        v0,
        v1,
        v2,
        v3,
        qcal,
        learned,
        flow,
        tokenizer,
        model,
        heads,
        test_examples,
        v2_policy,
        v3_policy,
        selected_policy,
        args,
    )
    test_metrics = {
        "calibrated_attention": metric_summary(test_rows, "calibrated_attention"),
        "v2": metric_summary(test_rows, "v2"),
        "v3": metric_summary(test_rows, "v3"),
        "v4": metric_summary(test_rows, "v4"),
    }
    test_metrics["v4"]["fallback_count"] = int(sum(1 for row in test_rows if row["fallback_triggered"]))
    test_metrics["v4"]["fallback_rate"] = test_metrics["v4"]["fallback_count"] / len(test_rows) if test_rows else 0.0

    v4 = test_metrics["v4"]
    target_pass = (
        v4["f1_delta"] >= -args.max_f1_drop
        and v4["relative_effort_reduction"] >= args.min_effort_reduction
        and v4["wrong_stop_rate"] <= args.max_wrong_stop
    )
    result = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "config": vars(args) | {"reference_v3_json": str(args.reference_v3_json)},
        "gpu": gpu,
        "splits": {
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "test_examples": len(test_examples),
            "train_states": len(train_rows),
            "val_states": len(val_rows),
            "reference_v3_ids_match": reference_ids_match,
            "test_question_ids": test_ids,
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
                "condition": c["condition"],
                "condition_text": c["condition_text"],
                "fallback_source": c["fallback_source"],
                "validation_metrics": c["validation_metrics"],
                "validation_target_pass": c["validation_target_pass"],
                "selection_score": c["selection_score"],
            }
            for c in top_candidates
        ],
        "test_metrics": test_metrics,
        "verdict": "PASS_CLEAN_TARGET" if target_pass else "NO_CLEAN_TARGET_PASS",
        "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 3),
        "train_attention_teacher_records": train_attention_teacher,
        "val_attention_teacher_records": val_attention_teacher,
        "test_rows": test_rows,
        "note": "Clean v4 selects fallback thresholds on validation only and evaluates once on held-out test IDs.",
    }

    json_path = OUT_DIR / f"{args.output_tag}.json"
    jsonl_path = OUT_DIR / f"{args.output_tag}.jsonl"
    md_path = OUT_DIR / f"{args.output_tag}.md"
    recompute_path = OUT_DIR / "recompute_v4_clean_metrics.py"
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
                f"--v2-risk-budget {args.v2_risk_budget}",
                f"--v3-risk-budget {args.v3_risk_budget}",
                f"--min-effort-reduction {args.min_effort_reduction}",
                f"--max-f1-drop {args.max_f1_drop}",
                f"--max-wrong-stop {args.max_wrong_stop}",
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
                "jsonl": str(jsonl_path),
                "report": str(md_path),
                "recompute": str(recompute_path),
                "verdict": result["verdict"],
                "selected_policy": result["selected_policy"],
                "test_metrics": test_metrics,
                "max_memory_allocated_mib": result["max_memory_allocated_mib"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
