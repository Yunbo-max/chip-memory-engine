#!/usr/bin/env python3
"""Verify preserved experience artifacts and recompute their key metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(relative_path: str) -> list[dict[str, Any]]:
    path = ROOT / relative_path
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def gated_summary(
    rows: list[dict[str, Any]],
    method: str,
    *,
    baseline: str = "baseline",
) -> dict[str, float | int]:
    baseline_f1 = mean(float(row[f"{baseline}_f1"]) for row in rows)
    method_f1 = mean(float(row[f"{method}_f1"]) for row in rows)
    baseline_steps = mean(float(row[f"{baseline}_steps"]) for row in rows)
    method_steps = mean(float(row[f"{method}_steps"]) for row in rows)
    return {
        "n": len(rows),
        "baseline_mean_f1": baseline_f1,
        "method_mean_f1": method_f1,
        "f1_delta": method_f1 - baseline_f1,
        "baseline_mean_steps": baseline_steps,
        "method_mean_steps": method_steps,
        "relative_effort_reduction": (
            (baseline_steps - method_steps) / baseline_steps if baseline_steps else 0.0
        ),
        "wrong_stop_rate": mean(
            1.0
            if float(row[f"{method}_f1"]) < float(row[f"{baseline}_f1"]) - 0.05
            else 0.0
            for row in rows
        ),
        "answer_preservation_rate": mean(
            1.0
            if float(row[f"{method}_f1"]) >= float(row[f"{baseline}_f1"])
            else 0.0
            for row in rows
        ),
    }


def noise_summary(rows: list[dict[str, Any]], method: str) -> dict[str, float | int]:
    clean_f1 = mean(float(row["clean_full_f1"]) for row in rows)
    noisy_f1 = mean(float(row["noisy_full_f1"]) for row in rows)
    method_f1 = mean(float(row[f"{method}_f1"]) for row in rows)
    baseline_steps = mean(float(row["noisy_full_steps"]) for row in rows)
    method_steps = mean(float(row[f"{method}_steps"]) for row in rows)
    return {
        "n": len(rows),
        "clean_full_mean_f1": clean_f1,
        "noisy_full_mean_f1": noisy_f1,
        "method_mean_f1": method_f1,
        "f1_delta_vs_clean_full": method_f1 - clean_f1,
        "f1_delta_vs_noisy_full": method_f1 - noisy_f1,
        "relative_effort_reduction_vs_noisy_full": (
            (baseline_steps - method_steps) / baseline_steps if baseline_steps else 0.0
        ),
        "wrong_stop_rate_vs_clean_full": mean(
            1.0
            if float(row[f"{method}_f1"]) < float(row["clean_full_f1"]) - 0.05
            else 0.0
            for row in rows
        ),
        "wrong_stop_rate_vs_noisy_full": mean(
            1.0
            if float(row[f"{method}_f1"]) < float(row["noisy_full_f1"]) - 0.05
            else 0.0
            for row in rows
        ),
    }


def verify_required_files() -> list[str]:
    required = [
        "docs/EXPERIMENTAL_EXPERIENCE.md",
        "docs/FAILURE_AND_DECISION_LOG.md",
        "research_archive/evidence_use_control/experiments/run_evidence_use_gate_v0.py",
        "research_archive/evidence_use_control/experiments/run_evidence_use_gate_v4_clean_validation.py",
        "research_archive/evidence_use_control/experiments/out/unified_100_example_comparison.md",
        "research_archive/evidence_use_control/experiments/out/verification_bundle/per_example_results.jsonl",
        "research_archive/evidence_use_control/experiments/out/evidence_use_gate_v4_clean_validation/evidence_use_gate_v4_clean_validation_100.jsonl",
        "research_archive/followups/v5_safety_first/evidence_use_gate_v5_safety_first_shifted_100.jsonl",
        "research_archive/followups/noise_robustness/smoke_noise_2.jsonl",
    ]
    return [path for path in required if not (ROOT / path).is_file()]


def recompute() -> dict[str, Any]:
    verification = read_jsonl(
        "research_archive/evidence_use_control/experiments/out/verification_bundle/"
        "per_example_results.jsonl"
    )
    v4 = read_jsonl(
        "research_archive/evidence_use_control/experiments/out/"
        "evidence_use_gate_v4_clean_validation/evidence_use_gate_v4_clean_validation_100.jsonl"
    )
    v5 = read_jsonl(
        "research_archive/followups/v5_safety_first/"
        "evidence_use_gate_v5_safety_first_shifted_100.jsonl"
    )
    noise = read_jsonl(
        "research_archive/followups/noise_robustness/smoke_noise_2.jsonl"
    )
    return {
        "verified_16": gated_summary(verification, "trust_gated"),
        "clean_v4": gated_summary(v4, "v4"),
        "shifted_v5": gated_summary(v5, "v4"),
        "noise_selective": noise_summary(noise, "selective"),
        "noise_v3": noise_summary(noise, "v3"),
    }


def assert_close(value: float, expected: float, tolerance: float = 1e-9) -> None:
    if abs(value - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, got {value}")


def verify_expected_metrics(metrics: dict[str, Any]) -> None:
    assert metrics["verified_16"]["n"] == 16
    assert_close(metrics["verified_16"]["relative_effort_reduction"], 0.3148148148148148)

    assert metrics["clean_v4"]["n"] == 100
    assert_close(metrics["clean_v4"]["relative_effort_reduction"], 0.44072164948453607)
    assert_close(metrics["clean_v4"]["wrong_stop_rate"], 0.07)

    assert metrics["shifted_v5"]["n"] == 100
    assert_close(metrics["shifted_v5"]["relative_effort_reduction"], 0.32653061224489793)
    assert_close(metrics["shifted_v5"]["wrong_stop_rate"], 0.02)

    assert metrics["noise_selective"]["n"] == 2
    assert_close(metrics["noise_selective"]["wrong_stop_rate_vs_clean_full"], 0.5)
    assert_close(metrics["noise_v3"]["wrong_stop_rate_vs_clean_full"], 0.0)


def main() -> int:
    missing = verify_required_files()
    if missing:
        print(json.dumps({"ok": False, "missing": missing}, indent=2))
        return 1
    metrics = recompute()
    verify_expected_metrics(metrics)
    print(json.dumps({"ok": True, "metrics": metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
