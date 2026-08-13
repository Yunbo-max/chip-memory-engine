#!/usr/bin/env python3
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
