#!/usr/bin/env python3
"""Safety-first EvidenceUseGate v5 wrapper.

This is a prospective follow-up to the saved v4 clean-validation near miss.
It reuses the v4 clean-validation implementation but changes the validation
selection rule before running on a shifted held-out split:

- select thresholds/fallback source on validation only;
- among validation-passing candidates, prioritize lower wrong-stop first;
- use held-out test IDs once.

The wrapper writes into the current mission folder so it does not overwrite the
older retrieval bundle artifacts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path("/tf/notebooks")
V4_SCRIPT = ROOT / "llm_reasoning_agent_memory_pilot_bundle/experiments/run_evidence_use_gate_v4_clean_validation.py"
MISSION_OUT = ROOT / "oral_research_memory_mission_2026_06_10/v0_experiments/evidence_use_gate_v5_safety_first"


def load_v4():
    spec = importlib.util.spec_from_file_location("evidence_use_gate_v4_clean_validation", V4_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {V4_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_safety_first_search(v4):
    def search_fallback_policy(
        val_rows,
        max_f1_drop: float,
        min_effort_reduction: float,
        max_wrong_stop: float,
    ):
        simple_conditions = []
        for feature, direction in v4.RISK_FEATURE_DIRECTIONS.items():
            if feature not in val_rows[0]["risk_features"]:
                continue
            for threshold in v4.threshold_grid(val_rows, feature):
                simple_conditions.append(
                    {
                        "kind": "threshold",
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                    }
                )

        conditions = [{"kind": "none"}, *simple_conditions]
        for idx, left in enumerate(simple_conditions):
            for right in simple_conditions[idx + 1 :]:
                if left["feature"] == right["feature"]:
                    continue
                for op in ("or", "and"):
                    conditions.append({"kind": "compound", "op": op, "left": left, "right": right})

        candidates = []
        for condition in conditions:
            for fallback_source in ("v2", "full_topk"):
                _, metrics = v4.apply_fallback_policy(val_rows, condition, fallback_source)
                target_pass = (
                    metrics["f1_delta"] >= -max_f1_drop
                    and metrics["relative_effort_reduction"] >= min_effort_reduction
                    and metrics["wrong_stop_rate"] <= max_wrong_stop
                )
                f1_violation = max(0.0, -max_f1_drop - metrics["f1_delta"])
                effort_violation = max(0.0, min_effort_reduction - metrics["relative_effort_reduction"])
                wrong_violation = max(0.0, metrics["wrong_stop_rate"] - max_wrong_stop)
                score = (
                    10.0 * float(target_pass)
                    - 4.0 * wrong_violation
                    - 2.0 * f1_violation
                    - effort_violation
                    - 0.15 * metrics["wrong_stop_rate"]
                    + 0.10 * metrics["f1_delta"]
                    + 0.05 * metrics["relative_effort_reduction"]
                )
                candidates.append(
                    {
                        "condition": condition,
                        "condition_text": v4.condition_text(condition),
                        "fallback_source": fallback_source,
                        "validation_metrics": metrics,
                        "validation_target_pass": bool(target_pass),
                        "selection_score": float(score),
                        "selection_mode": "safety_first_wrong_stop_then_quality_then_effort",
                    }
                )

        passing = [candidate for candidate in candidates if candidate["validation_target_pass"]]
        if passing:
            best = max(
                passing,
                key=lambda c: (
                    -c["validation_metrics"]["wrong_stop_rate"],
                    c["validation_metrics"]["f1_delta"],
                    c["validation_metrics"]["answer_preservation_rate"],
                    c["validation_metrics"]["relative_effort_reduction"],
                    -c["validation_metrics"]["fallback_count"],
                ),
            )
        else:
            best = max(
                candidates,
                key=lambda c: (
                    -max(0.0, c["validation_metrics"]["wrong_stop_rate"] - max_wrong_stop),
                    -max(0.0, -max_f1_drop - c["validation_metrics"]["f1_delta"]),
                    -max(0.0, min_effort_reduction - c["validation_metrics"]["relative_effort_reduction"]),
                    c["validation_metrics"]["f1_delta"],
                    c["validation_metrics"]["relative_effort_reduction"],
                ),
            )

        top = sorted(
            candidates,
            key=lambda c: (
                c["validation_target_pass"],
                -c["validation_metrics"]["wrong_stop_rate"],
                c["validation_metrics"]["f1_delta"],
                c["validation_metrics"]["relative_effort_reduction"],
            ),
            reverse=True,
        )[:25]
        return best, top

    return search_fallback_policy


def main() -> None:
    v4 = load_v4()
    v4.OUT_DIR = MISSION_OUT
    v4.search_fallback_policy = make_safety_first_search(v4)
    v4.main()


if __name__ == "__main__":
    main()
