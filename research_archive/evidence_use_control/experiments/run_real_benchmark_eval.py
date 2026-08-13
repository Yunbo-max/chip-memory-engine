#!/usr/bin/env python3
"""Real benchmark evaluation harness for the trust-gated retrieval pilot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = BUNDLE_ROOT.parents[0]
RAG_REPO = WORKSPACE / "_orchestration/code_repos/icml2026/71120_RAG-information-flow"
OUT_ROOT = BUNDLE_ROOT / "experiments/out/real_benchmark"
GPU_MEMORY_CAP_GIB = 20.0


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dataset_dir(dataset: str) -> Path:
    return OUT_ROOT / dataset


def prepared_path(dataset: str) -> Path:
    return dataset_dir(dataset) / f"{dataset}_prepared.json"


def trace_path(dataset: str) -> Path:
    return dataset_dir(dataset) / "benchmark_trace.jsonl"


def load_squad2(limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("squad_v2", split=f"validation[:{limit}]")
    rows = []
    for idx, ex in enumerate(ds):
        answers = ex.get("answers", {})
        answer_text = answers.get("text", []) if isinstance(answers, dict) else []
        rows.append(
            {
                "id": ex.get("id") or f"squad2-{idx}",
                "dataset": "squad2",
                "context": ex["context"],
                "question": ex["question"],
                "answers": answer_text,
                "is_unanswerable": len(answer_text) == 0,
                "ground_truth": {
                    "answers": answer_text,
                    "answer_start": answers.get("answer_start", []) if isinstance(answers, dict) else [],
                },
            }
        )
    return rows


def prepare(args: argparse.Namespace) -> None:
    if args.dataset != "squad2":
        raise ValueError("Only squad2 preparation is implemented in the first real benchmark harness.")

    rows = load_squad2(args.limit)
    out_dir = dataset_dir(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 71120's loader expects a JSON dataset with at least id/context/question.
    prepared_rows = [
        {
            "id": row["id"],
            "context": row["context"],
            "question": row["question"],
            "answers": row["answers"],
            "is_unanswerable": row["is_unanswerable"],
        }
        for row in rows
    ]
    prepared_path(args.dataset).write_text(json.dumps(prepared_rows, ensure_ascii=False), encoding="utf-8")

    trace_rows = [
        {
            "question_id": row["id"],
            "dataset": row["dataset"],
            "question": row["question"],
            "context": row["context"],
            "ground_truth": row["ground_truth"],
            "is_unanswerable": row["is_unanswerable"],
        }
        for row in rows
    ]
    write_jsonl(trace_path(args.dataset), trace_rows)

    manifest = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "limit": args.limit,
        "prepared_data": str(prepared_path(args.dataset)),
        "benchmark_trace": str(trace_path(args.dataset)),
        "note": "Prepared from Hugging Face squad_v2 validation split for real benchmark eval.",
    }
    write_json(out_dir / "prepare_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def gpu_preflight() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        return {
            "ok": False,
            "error": "CUDA_VISIBLE_DEVICES must be exactly '2' for this pilot.",
            "current": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    import torch

    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA is not available."}
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
    fraction = min(1.0, GPU_MEMORY_CAP_GIB / total_gib)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    probe = torch.ones((16, 16), device="cuda", dtype=torch.float32)
    probe_sum = float(probe.sum().item())
    del probe
    torch.cuda.empty_cache()

    return {
        "ok": True,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_device": 0,
        "device_name": props.name,
        "total_memory_gib": round(total_gib, 3),
        "memory_cap_gib": GPU_MEMORY_CAP_GIB,
        "memory_cap_fraction": round(fraction, 6),
        "probe_sum": probe_sum,
        "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 3),
    }


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.startswith("version https://git-lfs.github.com/spec/")


def model_preflight(model_path: str) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        return {"ok": False, "error": "model path does not exist", "model_path": model_path}

    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model_type = getattr(cfg, "model_type", None)
        supported = model_type in {"llama", "gemma", "gemma2", "mistral"}
        return {
            "ok": supported,
            "model_path": model_path,
            "model_type": model_type,
            "architectures": getattr(cfg, "architectures", None),
            "error": None if supported else "model architecture may not match 71120 llama/gemma attribution code",
        }
    except Exception as exc:
        return {"ok": False, "model_path": model_path, "error": repr(exc)}


def preflight(args: argparse.Namespace) -> None:
    out_dir = dataset_dir(args.dataset)
    checks = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "prepared_data": {
            "path": str(prepared_path(args.dataset)),
            "exists": prepared_path(args.dataset).exists(),
            "is_lfs_pointer": is_lfs_pointer(prepared_path(args.dataset)),
        },
        "benchmark_trace": {
            "path": str(trace_path(args.dataset)),
            "exists": trace_path(args.dataset).exists(),
        },
        "rag_repo": {
            "path": str(RAG_REPO),
            "exists": RAG_REPO.exists(),
            "main_py": (RAG_REPO / "proposed/Ours/llama/main.py").exists(),
        },
        "gpu": gpu_preflight(),
        "model": model_preflight(args.model_path),
    }
    checks["ok"] = all(
        [
            checks["prepared_data"]["exists"],
            not checks["prepared_data"]["is_lfs_pointer"],
            checks["benchmark_trace"]["exists"],
            checks["rag_repo"]["main_py"],
            checks["gpu"]["ok"],
            checks["model"]["ok"],
        ]
    )
    write_json(out_dir / "preflight.json", checks)
    print(json.dumps(checks, indent=2))
    if not checks["ok"]:
        raise SystemExit(2)


def install_prepared_into_rag(dataset: str) -> None:
    src = prepared_path(dataset)
    dst = RAG_REPO / "preprocessed_data" / f"{dataset}_prepared.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def run_info_flow(args: argparse.Namespace) -> None:
    preflight(args)
    install_prepared_into_rag(args.dataset)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [
        sys.executable,
        str(RAG_REPO / "proposed/Ours/llama/main.py"),
        "--dataset",
        args.dataset,
        "--model_path",
        args.model_path,
        "--i_block",
        str(args.i_block),
    ]
    run_record = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cmd": cmd,
        "cwd": str(RAG_REPO / "proposed"),
        "env": {"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"], "PYTORCH_CUDA_ALLOC_CONF": env["PYTORCH_CUDA_ALLOC_CONF"]},
    }
    write_json(dataset_dir(args.dataset) / "info_flow_command.json", run_record)
    completed = subprocess.run(cmd, cwd=str(RAG_REPO / "proposed"), env=env, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def normalize_answer(text: str) -> list[str]:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()


def token_f1(prediction: str, answers: list[str]) -> float:
    if not answers:
        return 1.0 if not prediction.strip() else 0.0
    pred_tokens = normalize_answer(prediction)
    best = 0.0
    for answer in answers:
        gold_tokens = normalize_answer(answer)
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue
        common = 0
        remaining = gold_tokens.copy()
        for token in pred_tokens:
            if token in remaining:
                common += 1
                remaining.remove(token)
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def evaluate(args: argparse.Namespace) -> None:
    out_dir = dataset_dir(args.dataset)
    trace = read_jsonl(trace_path(args.dataset))
    results_dir = RAG_REPO / "results"
    expected = {
        "contri": list(results_dir.glob(f"*/loop_bf16/loop_manhattan_contri_bf16_*.jsonl")),
        "rank": list(results_dir.glob(f"*/loop_bf16/loop_manhattan_rank_bf16_*.jsonl")),
        "path": list(results_dir.glob(f"*/loop_bf16/loop_manhattan_path_bf16_*.jsonl")),
    }
    real_info_flow_present = all(expected[key] for key in expected)

    baseline_path = out_dir / "baseline_predictions.jsonl"
    trust_path = out_dir / "trust_gated_predictions.jsonl"
    baseline_present = baseline_path.exists()
    trust_present = trust_path.exists()

    eval_rows = []
    if baseline_present:
        by_id = {row.get("question_id") or row.get("id"): row for row in read_jsonl(baseline_path)}
        for row in trace:
            pred = by_id.get(row["question_id"], {}).get("answer", "")
            eval_rows.append(
                {
                    "question_id": row["question_id"],
                    "baseline_token_f1": token_f1(str(pred), row["ground_truth"]["answers"]),
                }
            )

    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "n_trace_examples": len(trace),
        "real_info_flow_present": real_info_flow_present,
        "info_flow_files": {key: [str(path) for path in paths] for key, paths in expected.items()},
        "baseline_predictions_present": baseline_present,
        "trust_gated_predictions_present": trust_present,
        "mean_baseline_token_f1": (
            sum(row["baseline_token_f1"] for row in eval_rows) / len(eval_rows) if eval_rows else None
        ),
        "verdict": "PASS" if real_info_flow_present and baseline_present and trust_present else "NOT_READY_FOR_METHOD_VERDICT",
        "missing": [
            name
            for name, ok in [
                ("real 71120 contribution/rank/path JSONL", real_info_flow_present),
                ("baseline_predictions.jsonl", baseline_present),
                ("trust_gated_predictions.jsonl", trust_present),
            ]
            if not ok
        ],
    }
    write_json(out_dir / "real_benchmark_eval.json", report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--dataset", default="squad2", choices=["squad2"])
    p_prepare.add_argument("--limit", type=int, default=8)
    p_prepare.set_defaults(func=prepare)

    p_preflight = sub.add_parser("preflight")
    p_preflight.add_argument("--dataset", default="squad2", choices=["squad2"])
    p_preflight.add_argument("--model-path", required=True)
    p_preflight.set_defaults(func=preflight)

    p_run = sub.add_parser("run-info-flow")
    p_run.add_argument("--dataset", default="squad2", choices=["squad2"])
    p_run.add_argument("--model-path", required=True)
    p_run.add_argument("--i-block", type=int, default=1)
    p_run.set_defaults(func=run_info_flow)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--dataset", default="squad2", choices=["squad2"])
    p_eval.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
