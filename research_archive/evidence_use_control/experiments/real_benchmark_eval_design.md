# Real Benchmark Evaluation Redesign

## Purpose

The previous pilot proved only structural composition. The real test must answer:

> Does a 71120-style trust signal improve stop/continue behavior on benchmark RAG examples without reducing answer quality?

## Benchmark Slice

Primary benchmark: `squad2`

Reason:

- It is one of the 71120 paper benchmarks.
- It has compact context/question/answer records.
- It can be prepared without MADQA's PDF/OCR/VLM infrastructure.
- It is small enough for a first GPU-bounded attribution run.

Secondary benchmarks after the first pass:

- `hotpot`: multi-hop RAG stress test.
- `msmarco`: retrieval-style QA stress test.
- `madqa`: agentic document benchmark after OCR/index/API or local VLM dependencies are available.

## Methods Compared

1. `baseline_final_answer`
   - Generate answer with the model.
   - Always continue until normal answer generation completes.

2. `trust_gated_stop`
   - Generate step/token-level 71120 contribution outputs.
   - Attach relevance layout from a ranker/SHAP stage.
   - Compute simulatability, concentration, and trust.
   - Stop once trust passes a threshold; continue otherwise.

3. `relevance_only_stop`
   - Same stopping interface, but uses external relevance only.
   - This controls for "ranker is enough".

## Required Real Artifacts

For every benchmark example:

- `benchmark_trace.jsonl`
  - `question_id`
  - `dataset`
  - `question`
  - `context`
  - `ground_truth`

- `baseline_predictions.jsonl`
  - model answer
  - answer tokens or generated text
  - prompt tokens if available

- `info_flow/loop_manhattan_contri_bf16_0.jsonl`
  - produced by `proposed/Ours/llama/main.py`

- `info_flow/loop_manhattan_rank_bf16_0.jsonl`
  - produced by `proposed/Ours/llama/main.py`

- `info_flow/loop_manhattan_path_bf16_0.jsonl`
  - produced by `proposed/Ours/llama/main.py`

- `relevance_layout.pt` or `relevance_layout.jsonl`
  - produced by `proposed/SHAP_Qwen3_8B.py`, `SHAP_MiniLM_L12.py`, or a documented lightweight replacement.

- `trust_gated_predictions.jsonl`
  - stop step/token
  - trust score
  - final answer under the stop policy

## Metrics

Answer quality:

- Exact match
- Token F1
- Unanswerable accuracy for SQuAD2
- Optional HHEM/semantic judge later

Effort quality:

- Generated tokens
- Attribution steps
- Stop step
- Continue-after-low-trust rate
- Wasted effort ratio: mean effort on incorrect answers divided by mean effort on correct answers

Trust quality:

- AUROC/AUPRC for correctness prediction
- ECE
- Coverage at target accuracy
- Trust threshold sweep

Composition criteria:

- Retrieved/context chunks align to contribution tokens.
- Relevance layout and contribution layout share a token/span mapping.
- Trust can be computed before final answer completion.
- Effort metrics are computed from real logs, not synthetic rows.

## GPU Constraint

All GPU stages must run with:

```bash
CUDA_VISIBLE_DEVICES=2
```

The runner sets:

```python
torch.cuda.set_per_process_memory_fraction(20 / total_gpu_memory)
```

71120 attribution should start with:

```bash
--i_block 1
```

because the code comments identify `i_block` as the peak-memory control.

## First Real Run

Prepare 8 SQuAD2 validation examples:

```bash
python experiments/run_real_benchmark_eval.py prepare --dataset squad2 --limit 8
```

Preflight the run:

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/run_real_benchmark_eval.py preflight --dataset squad2 --model-path <local_llama_or_compatible_model>
```

Run 71120 attribution:

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/run_real_benchmark_eval.py run-info-flow --dataset squad2 --model-path <local_llama_or_compatible_model> --i-block 1
```

Evaluate:

```bash
python experiments/run_real_benchmark_eval.py evaluate --dataset squad2
```

## Pass/Fail

Pass:

- At least 8 real benchmark examples are evaluated.
- Baseline and trust-gated outputs are both present.
- Trust fields are produced from real contribution/relevance layouts.
- Answer quality is not worse by more than 5 percentage points.
- Mean effort decreases or wasted effort ratio decreases.

Fail:

- Benchmark data or model outputs are synthetic.
- Contribution and relevance layouts cannot be aligned.
- Trust is only available after final answer completion.
- GPU memory exceeds 20 GiB on GPU 2.
