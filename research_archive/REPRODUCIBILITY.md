# Reproducibility guide for the historical experiments

## Recompute saved outputs without a model

The strongest verification path is to recompute metrics from the saved per-example JSONL files.

From the repository root:

```bash
python research_archive/evidence_use_control/experiments/out/verification_bundle/recompute_metrics.py

python research_archive/evidence_use_control/experiments/out/evidence_use_gate_v4_clean_validation/recompute_v4_clean_metrics.py \
  research_archive/evidence_use_control/experiments/out/evidence_use_gate_v4_clean_validation/evidence_use_gate_v4_clean_validation_100.jsonl

python research_archive/followups/v5_safety_first/recompute_v4_clean_metrics.py \
  research_archive/followups/v5_safety_first/evidence_use_gate_v5_safety_first_shifted_100.jsonl

python research_archive/followups/noise_robustness/recompute_noise_metrics.py \
  research_archive/followups/noise_robustness/smoke_noise_2.jsonl
```

The repository-level verifier checks archive presence and recomputes the saved 16-example, clean-v4, shifted-v5, and noise metrics using only the Python standard library:

```bash
python tools/verify_experience_archive.py
```

## Re-run model experiments

The historical experiment runners require optional dependencies not needed by the Chip Memory engine:

- NumPy;
- scikit-learn;
- Hugging Face Datasets;
- PyTorch with CUDA support;
- Transformers;
- locally available compatible model weights.

The LLaMA contribution-flow runner additionally requires the external information-flow repository used by the original workspace. Model weights and that repository are not included here.

Typical runners accept `--model-path`, `--train-examples`, `--val-examples`, and `--eval-examples`. Inspect each script's `--help` and the archived `commands.txt` before running it.

## Historical environment constraints

The archived scripts were developed in a workspace rooted at `/tf/notebooks`, with local model snapshots under `/root/.cache/huggingface`, and explicit `CUDA_VISIBLE_DEVICES` settings. Most v0-v4 runners resolve sibling scripts relative to their file location and accept a replacement model path. The v5 and noise wrappers preserve older absolute workspace paths and therefore require a portability edit or reconstruction of the historical layout before a full replay.

This is deliberate archival honesty: the saved outputs are reproducible from their raw rows, while full model execution is not claimed to be one-command portable in this repository.

## Clean evaluation protocol

For a new run:

1. freeze dataset version and deterministic IDs;
2. freeze train/validation/test split before inspecting test outcomes;
3. train teacher/gate heads only on train;
4. choose thresholds and fallback source only on validation;
5. evaluate the selected policy once on test;
6. retain per-example rows and recomputation code;
7. report F1/task quality, effort, wrong-stop/wrong-use, answer preservation, and fallback rate;
8. label any later test-table search as post-hoc diagnostic;
9. add noise and incompatibility cases to train/validation rather than tuning on noisy test rows.

## Known limits of the saved archive

- The two-example noise output is underpowered and should be treated only as a failed smoke diagnostic.
- Some “PASS” labels used a local maximum-F1-drop acceptance threshold; they do not imply unchanged quality.
- Several 100-example rows use different model checkpoints, calibration sets, or splits and are not direct head-to-head comparisons.
- The contribution-flow case is mechanistically closer to the external method but is not an exact reproduction.
- The archived SQuAD2 experiments are not direct evidence for multi-agent Chips.
