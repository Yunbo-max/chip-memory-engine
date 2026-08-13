#!/usr/bin/env python3
import json
from pathlib import Path
p=Path(__file__).with_name('per_example_results.jsonl')
rows=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
mean=lambda xs: sum(xs)/len(xs)
baseline_f1=mean([r['baseline_f1'] for r in rows])
gated_f1=mean([r['trust_gated_f1'] for r in rows])
baseline_steps=mean([r['baseline_steps'] for r in rows])
gated_steps=mean([r['trust_gated_steps'] for r in rows])
summary={
  'n': len(rows),
  'baseline_mean_f1': baseline_f1,
  'trust_gated_mean_f1': gated_f1,
  'f1_delta': gated_f1-baseline_f1,
  'baseline_mean_steps': baseline_steps,
  'trust_gated_mean_steps': gated_steps,
  'mean_effort_reduction': baseline_steps-gated_steps,
  'relative_effort_reduction': (baseline_steps-gated_steps)/baseline_steps,
}
print(json.dumps(summary, indent=2))
