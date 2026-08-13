#!/usr/bin/env python3
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
