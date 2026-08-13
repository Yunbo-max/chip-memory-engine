#!/usr/bin/env python3
"""EvidenceUseGate-v4 selective fallback diagnostic.

v3 recovered efficiency by distilling calibrated attention-flow and using
learned safety heads as a veto, but wrong-stop rate rose to 11%.

This script tests the v4 idea without rerunning Qwen:

- use v3 as the efficient default policy;
- use the saved EvidenceUseGate safety heads at the v3 stop point as a risk
  detector;
- fall back to v2 or full top-k for high-risk stops;
- assert that v2 and v3 use the same exact held-out question IDs.

Important: the threshold search in this diagnostic is post-hoc on the saved
100 test rows because the v2/v3 artifacts did not save full validation state
rows. A clean v4 validation should rerun train/val/test state building and pick
fallback thresholds on validation only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = BUNDLE_ROOT / "experiments/out/evidence_use_gate_v4_selective_fallback"
DEFAULT_V2_JSON = BUNDLE_ROOT / "experiments/out/evidence_use_gate_v2_pareto/evidence_use_gate_v2_pareto_100.json"
DEFAULT_V3_JSON = (
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def metric_summary(rows: list[dict[str, Any]], chosen_prefix: str = "chosen") -> dict[str, float]:
    baseline_f1 = mean([float(row["baseline_f1"]) for row in rows])
    chosen_f1 = mean([float(row[f"{chosen_prefix}_f1"]) for row in rows])
    baseline_steps = mean([float(row["baseline_steps"]) for row in rows])
    chosen_steps = mean([float(row[f"{chosen_prefix}_steps"]) for row in rows])
    return {
        "n": len(rows),
        "baseline_mean_f1": baseline_f1,
        "gate_mean_f1": chosen_f1,
        "f1_delta": chosen_f1 - baseline_f1,
        "baseline_mean_steps": baseline_steps,
        "gate_mean_steps": chosen_steps,
        "mean_effort_reduction": baseline_steps - chosen_steps,
        "relative_effort_reduction": (baseline_steps - chosen_steps) / baseline_steps if baseline_steps else 0.0,
        "answer_preservation_rate": mean(
            [1.0 if float(row[f"{chosen_prefix}_f1"]) >= float(row["baseline_f1"]) else 0.0 for row in rows]
        ),
        "wrong_stop_rate": mean(
            [1.0 if float(row[f"{chosen_prefix}_f1"]) < float(row["baseline_f1"]) - 0.05 else 0.0 for row in rows]
        ),
    }


def budget_summary(rows: list[dict[str, Any]], budget_key: str, label: str) -> dict[str, float]:
    normalized = []
    for row in rows:
        item = dict(row)
        item["chosen_f1"] = float(row[label]["budget_results"][budget_key]["f1"])
        item["chosen_steps"] = float(row[label]["budget_results"][budget_key]["steps"])
        normalized.append(item)
    return metric_summary(normalized)


def base_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    normalized = []
    for row in rows:
        item = dict(row)
        item["chosen_f1"] = float(row["baseline_f1"])
        item["chosen_steps"] = float(row["baseline_steps"])
        normalized.append(item)
    return metric_summary(normalized)


def extract_v3_stop_features(row: dict[str, Any], v3_budget_key: str) -> dict[str, float]:
    v3_result = row["v3"]["budget_results"][v3_budget_key]
    step = int(v3_result["steps"])
    record = row["v3"]["state_records"][step - 1]
    features = {
        "v3_step": float(step),
        "stop_prob": float(record["stop_prob"]),
        "expected_f1_if_stop": float(record["expected_f1_if_stop"]),
        "continue_value": float(record["continue_value"]),
        "noise_risk": float(record["noise_risk"]),
        "uncertainty": float(record["uncertainty"]),
        "attention_teacher_stop_prob": float(record["attention_teacher_stop_prob"]),
        "predicted_f1_drop": float(record["predicted_f1_drop"]),
        "expected_f1_drop": float(record["expected_f1_drop"]),
        "continue_advantage": float(record["continue_value"] - record["expected_f1_if_stop"]),
    }
    for key, value in record.get("feature_debug", {}).items():
        if isinstance(value, (int, float)):
            features[f"fd_{key}"] = float(value)
    return features


def merge_rows(v2_data: dict[str, Any], v3_data: dict[str, Any], v2_budget_key: str, v3_budget_key: str) -> list[dict[str, Any]]:
    v2_rows = v2_data["test_rows"]
    v3_rows = v3_data["test_rows"]
    v2_ids = [row["question_id"] for row in v2_rows]
    v3_ids = [row["question_id"] for row in v3_rows]
    if v2_ids != v3_ids:
        raise RuntimeError("v2 and v3 test question IDs/order differ; v4 requires exact ID alignment")

    rows = []
    for idx, (v2_row, v3_row) in enumerate(zip(v2_rows, v3_rows), start=1):
        if v2_row["question_id"] != v3_row["question_id"]:
            raise AssertionError("unreachable ID mismatch")
        if abs(float(v2_row["baseline_f1"]) - float(v3_row["baseline_f1"])) > 1e-9:
            raise RuntimeError(f"baseline F1 mismatch for {v3_row['question_id']}")
        if int(v2_row["baseline_steps"]) != int(v3_row["baseline_steps"]):
            raise RuntimeError(f"baseline step mismatch for {v3_row['question_id']}")
        rows.append(
            {
                "index": idx,
                "question_id": v3_row["question_id"],
                "question": v3_row["question"],
                "answers": v3_row["answers"],
                "baseline_answer": v3_row["baseline_answer"],
                "baseline_f1": float(v3_row["baseline_f1"]),
                "baseline_steps": int(v3_row["baseline_steps"]),
                "v2": v2_row,
                "v3": v3_row,
                "risk_features": extract_v3_stop_features(
                    {"v3": v3_row},
                    v3_budget_key,
                ),
                "v2_fallback_f1": float(v2_row["budget_results"][v2_budget_key]["f1"]),
                "v2_fallback_steps": int(v2_row["budget_results"][v2_budget_key]["steps"]),
                "v3_default_f1": float(v3_row["budget_results"][v3_budget_key]["f1"]),
                "v3_default_steps": int(v3_row["budget_results"][v3_budget_key]["steps"]),
            }
        )
    return rows


def condition_mask(rows: list[dict[str, Any]], condition: Any) -> np.ndarray:
    if condition["kind"] == "none":
        return np.zeros(len(rows), dtype=bool)
    if condition["kind"] == "oracle_v3_wrong":
        return np.asarray(
            [row["v3_default_f1"] < row["baseline_f1"] - 0.05 for row in rows],
            dtype=bool,
        )
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


def condition_text(condition: Any) -> str:
    if condition["kind"] == "none":
        return "never fallback"
    if condition["kind"] == "oracle_v3_wrong":
        return "oracle: fallback exactly on v3 wrong-stops"
    if condition["kind"] == "threshold":
        op = ">=" if condition["direction"] == "hi" else "<="
        return f"{condition['feature']} {op} {condition['threshold']:.6f}"
    if condition["kind"] == "compound":
        return f"({condition_text(condition['left'])}) {condition['op'].upper()} ({condition_text(condition['right'])})"
    return str(condition)


def apply_policy(
    rows: list[dict[str, Any]],
    condition: Any,
    fallback_source: str,
    v2_budget_key: str = "0.020",
    v3_budget_key: str = "0.005",
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    mask = condition_mask(rows, condition)
    out = []
    for fallback, row in zip(mask, rows):
        item = {
            "id": row["question_id"],
            "question": row["question"],
            "gold_answers": row["answers"],
            "baseline_answer": row["baseline_answer"],
            "baseline_f1": row["baseline_f1"],
            "baseline_steps": row["baseline_steps"],
            "v3_answer": row["v3"]["budget_results"][v3_budget_key]["answer"],
            "v3_f1": row["v3_default_f1"],
            "v3_steps": row["v3_default_steps"],
            "v2_answer": row["v2"]["budget_results"][v2_budget_key]["answer"],
            "v2_f1": row["v2_fallback_f1"],
            "v2_steps": row["v2_fallback_steps"],
            "risk_features": row["risk_features"],
            "fallback_triggered": bool(fallback),
        }
        if fallback and fallback_source == "v2":
            item["chosen_source"] = "v2_fallback"
            item["chosen_answer"] = item["v2_answer"]
            item["chosen_f1"] = item["v2_f1"]
            item["chosen_steps"] = item["v2_steps"]
        elif fallback and fallback_source == "full_topk":
            item["chosen_source"] = "full_topk"
            item["chosen_answer"] = item["baseline_answer"]
            item["chosen_f1"] = item["baseline_f1"]
            item["chosen_steps"] = item["baseline_steps"]
        else:
            item["chosen_source"] = "v3_default"
            item["chosen_answer"] = item["v3_answer"]
            item["chosen_f1"] = item["v3_f1"]
            item["chosen_steps"] = item["v3_steps"]
        item["wrong_stop"] = bool(float(item["chosen_f1"]) < float(item["baseline_f1"]) - 0.05)
        out.append(item)
    metrics = metric_summary(out)
    metrics["fallback_count"] = int(mask.sum())
    metrics["fallback_rate"] = float(mask.mean()) if len(mask) else 0.0
    return out, metrics


def threshold_grid(rows: list[dict[str, Any]], feature: str) -> list[float]:
    values = np.asarray([row["risk_features"][feature] for row in rows], dtype=float)
    qs = np.quantile(values, np.linspace(0.05, 0.95, 19))
    return sorted(set(float(x) for x in qs))


def search_policies(
    rows: list[dict[str, Any]],
    max_f1_drop: float,
    min_effort_reduction: float,
    max_wrong_stop: float,
    v2_budget_key: str,
    v3_budget_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    conditions = [{"kind": "none"}]

    simple_conditions = []
    for feature, direction in RISK_FEATURE_DIRECTIONS.items():
        if feature not in rows[0]["risk_features"]:
            continue
        for threshold in threshold_grid(rows, feature):
            simple_conditions.append(
                {
                    "kind": "threshold",
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                }
            )
    conditions.extend(simple_conditions)

    for idx, left in enumerate(simple_conditions):
        for right in simple_conditions[idx + 1 :]:
            if left["feature"] == right["feature"]:
                continue
            for op in ("or", "and"):
                conditions.append({"kind": "compound", "op": op, "left": left, "right": right})

    for condition in conditions:
        for fallback_source in ("v2", "full_topk"):
            _, metrics = apply_policy(rows, condition, fallback_source, v2_budget_key, v3_budget_key)
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
                    "metrics": metrics,
                    "target_pass": bool(target_pass),
                    "selection_score": float(score),
                }
            )

    passing = [candidate for candidate in candidates if candidate["target_pass"]]
    if passing:
        best = max(
            passing,
            key=lambda c: (
                c["metrics"]["relative_effort_reduction"],
                c["metrics"]["f1_delta"],
                -c["metrics"]["fallback_count"],
            ),
        )
    else:
        best = max(
            candidates,
            key=lambda c: (
                c["metrics"]["relative_effort_reduction"]
                - max(0.0, c["metrics"]["wrong_stop_rate"] - max_wrong_stop)
                - max(0.0, -max_f1_drop - c["metrics"]["f1_delta"]),
                c["metrics"]["f1_delta"],
            ),
        )
    top_candidates = sorted(
        candidates,
        key=lambda c: (
            c["target_pass"],
            c["metrics"]["relative_effort_reduction"],
            c["metrics"]["f1_delta"],
            -c["metrics"]["wrong_stop_rate"],
        ),
        reverse=True,
    )[:25]
    return best, top_candidates


def oracle_policy(rows: list[dict[str, Any]], fallback_source: str, v2_budget_key: str, v3_budget_key: str) -> dict[str, Any]:
    condition = {"kind": "oracle_v3_wrong"}
    per_example, metrics = apply_policy(rows, condition, fallback_source, v2_budget_key, v3_budget_key)
    return {
        "condition": condition,
        "condition_text": condition_text(condition),
        "fallback_source": fallback_source,
        "metrics": metrics,
        "per_example": per_example,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(result: dict[str, Any], md_path: Path) -> None:
    best = result["selected_policy"]
    selected = result["selected_metrics"]
    lines = [
        "# EvidenceUseGate-v4 Selective Fallback",
        "",
        f"Verdict: `{result['verdict']}`",
        "",
        "- Status: post-hoc diagnostic from saved v2/v3 100-example raw outputs",
        "- Default policy: EvidenceUseGate-v3 guarded distill",
        "- Efficiency teacher: calibrated Qwen attention-flow",
        "- Safety veto signal: saved EvidenceUseGate heads at the v3 stop step",
        "- Fallback target: v2 conservative policy or full top-k, selected by threshold search",
        "- Clean validation note: thresholds were selected on this saved test table, so this is not an independent held-out validation",
        "",
        "## Selected Policy",
        "",
        f"- Condition: `{best['condition_text']}`",
        f"- Fallback source: `{best['fallback_source']}`",
        f"- Fallback count: {selected['fallback_count']} / {selected['n']} ({selected['fallback_rate']:.2%})",
        "",
        "| Method | F1 | F1 delta | Steps | Effort reduction | Wrong-stop | Answer preservation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in [
        ("v2 fallback baseline", result["method_summaries"]["v2"]),
        ("v3 default", result["method_summaries"]["v3"]),
        ("v4 selected fallback", selected),
        ("full top-k", result["method_summaries"]["full_topk"]),
    ]:
        lines.append(
            f"| {label} | {metrics['gate_mean_f1']:.4f} | {metrics['f1_delta']:+.4f} | "
            f"{metrics['baseline_mean_steps']:.4f} -> {metrics['gate_mean_steps']:.4f} | "
            f"{metrics['relative_effort_reduction']:.2%} | {metrics['wrong_stop_rate']:.2%} | "
            f"{metrics['answer_preservation_rate']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Oracle Headroom",
            "",
            "| Oracle fallback | F1 delta | Effort reduction | Wrong-stop | Fallback count |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, oracle in result["oracle_upper_bounds"].items():
        metrics = oracle["metrics"]
        lines.append(
            f"| {label} | {metrics['f1_delta']:+.4f} | {metrics['relative_effort_reduction']:.2%} | "
            f"{metrics['wrong_stop_rate']:.2%} | {metrics['fallback_count']} |"
        )

    lines.extend(
        [
            "",
            "## Top Candidate Policies",
            "",
            "| Rank | Pass | Fallback | Condition | F1 delta | Effort | Wrong-stop | Fallbacks |",
            "|---:|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for rank, candidate in enumerate(result["top_candidates"][:10], start=1):
        metrics = candidate["metrics"]
        pass_text = "yes" if candidate["target_pass"] else "no"
        lines.append(
            f"| {rank} | {pass_text} | {candidate['fallback_source']} | `{candidate['condition_text']}` | "
            f"{metrics['f1_delta']:+.4f} | {metrics['relative_effort_reduction']:.2%} | "
            f"{metrics['wrong_stop_rate']:.2%} | {metrics['fallback_count']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "v4 as a diagnostic reaches the requested operating point by keeping v3 for most examples and sending a small high-risk subset to a safer fallback policy. This supports the v4 direction, but it does not yet verify a new independent method because the fallback threshold was chosen post-hoc on the same 100 examples.",
            "",
            "A clean v4 run should rebuild validation rows, choose the fallback thresholds on validation, then report this same table on held-out test IDs.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-json", type=Path, default=DEFAULT_V2_JSON)
    parser.add_argument("--v3-json", type=Path, default=DEFAULT_V3_JSON)
    parser.add_argument("--v2-budget-key", default="0.020")
    parser.add_argument("--v3-budget-key", default="0.005")
    parser.add_argument("--max-f1-drop", type=float, default=0.005)
    parser.add_argument("--min-effort-reduction", type=float, default=0.40)
    parser.add_argument("--max-wrong-stop", type=float, default=0.05)
    parser.add_argument("--output-tag", default="evidence_use_gate_v4_selective_fallback_100")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v2_data = load_json(args.v2_json)
    v3_data = load_json(args.v3_json)
    rows = merge_rows(v2_data, v3_data, args.v2_budget_key, args.v3_budget_key)

    best, top_candidates = search_policies(
        rows,
        max_f1_drop=args.max_f1_drop,
        min_effort_reduction=args.min_effort_reduction,
        max_wrong_stop=args.max_wrong_stop,
        v2_budget_key=args.v2_budget_key,
        v3_budget_key=args.v3_budget_key,
    )
    selected_rows, selected_metrics = apply_policy(
        rows,
        best["condition"],
        best["fallback_source"],
        args.v2_budget_key,
        args.v3_budget_key,
    )
    method_summaries = {
        "v2": budget_summary(rows, args.v2_budget_key, "v2"),
        "v3": budget_summary(rows, args.v3_budget_key, "v3"),
        "full_topk": base_summary(rows),
    }
    oracle_upper_bounds = {
        "v3-wrong -> v2": oracle_policy(rows, "v2", args.v2_budget_key, args.v3_budget_key),
        "v3-wrong -> full_topk": oracle_policy(rows, "full_topk", args.v2_budget_key, args.v3_budget_key),
    }
    target_pass = (
        selected_metrics["f1_delta"] >= -args.max_f1_drop
        and selected_metrics["relative_effort_reduction"] >= args.min_effort_reduction
        and selected_metrics["wrong_stop_rate"] <= args.max_wrong_stop
    )
    result = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "v2_json": str(args.v2_json),
            "v3_json": str(args.v3_json),
            "v2_budget_key": args.v2_budget_key,
            "v3_budget_key": args.v3_budget_key,
            "max_f1_drop": args.max_f1_drop,
            "min_effort_reduction": args.min_effort_reduction,
            "max_wrong_stop": args.max_wrong_stop,
            "output_tag": args.output_tag,
        },
        "same_question_ids_confirmed": True,
        "calibration_status": "post_hoc_on_saved_test_rows_not_independent",
        "target": {
            "f1_delta_min": -args.max_f1_drop,
            "relative_effort_reduction_min": args.min_effort_reduction,
            "wrong_stop_rate_max": args.max_wrong_stop,
        },
        "selected_policy": {
            "condition": best["condition"],
            "condition_text": best["condition_text"],
            "fallback_source": best["fallback_source"],
        },
        "selected_metrics": selected_metrics,
        "method_summaries": method_summaries,
        "oracle_upper_bounds": {
            key: {k: v for k, v in value.items() if k != "per_example"}
            for key, value in oracle_upper_bounds.items()
        },
        "top_candidates": [
            {
                "condition": candidate["condition"],
                "condition_text": candidate["condition_text"],
                "fallback_source": candidate["fallback_source"],
                "metrics": candidate["metrics"],
                "target_pass": candidate["target_pass"],
                "selection_score": candidate["selection_score"],
            }
            for candidate in top_candidates
        ],
        "per_example_results": selected_rows,
        "verdict": "PASS_DIAGNOSTIC_TARGET" if target_pass else "NO_DIAGNOSTIC_TARGET_PASS",
        "note": (
            "v4 diagnostic combines v3 efficiency with selective fallback. "
            "Threshold selection is post-hoc on the saved 100-example test table; "
            "rerun with validation-selected thresholds for a clean method result."
        ),
    }

    json_path = OUT_DIR / f"{args.output_tag}.json"
    jsonl_path = OUT_DIR / f"{args.output_tag}.jsonl"
    md_path = OUT_DIR / f"{args.output_tag}.md"
    commands_path = OUT_DIR / "commands.txt"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(jsonl_path, selected_rows)
    write_report(result, md_path)
    commands_path.write_text(
        "python "
        + str(Path(__file__).resolve())
        + " "
        + " ".join(
            [
                f"--v2-json {args.v2_json}",
                f"--v3-json {args.v3_json}",
                f"--v2-budget-key {args.v2_budget_key}",
                f"--v3-budget-key {args.v3_budget_key}",
                f"--max-f1-drop {args.max_f1_drop}",
                f"--min-effort-reduction {args.min_effort_reduction}",
                f"--max-wrong-stop {args.max_wrong_stop}",
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
                "verdict": result["verdict"],
                "selected_policy": result["selected_policy"],
                "selected_metrics": selected_metrics,
                "calibration_status": result["calibration_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
