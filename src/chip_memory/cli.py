"""Command-line interface for validation, indexing, retrieval, and runtime audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from .engine import ChipMemoryEngine
from .loader import ChipLoader, ChipStore
from .projector import projection_to_markdown
from .retriever import ChipIndex
from .runtime import RuntimeEventStore


def _write_or_print(text: str, output: str | None) -> None:
    if output:
        path = Path(output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(str(path))
    else:
        print(text, end="" if text.endswith("\n") else "\n")


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _add_chip_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chips",
        action="append",
        required=True,
        metavar="PATH",
        help="Chip file or directory; repeat for multiple banks",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chip-memory",
        description="Grounded L1/L2/L3 retrieval over immutable paper Chips.",
    )
    parser.add_argument("--version", action="version", version="chip-memory 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate and summarize Chip files")
    _add_chip_paths(validate)
    validate.add_argument("--strict-warnings", action="store_true", help="Return failure when warnings exist")
    validate.add_argument("--output", help="Write JSON report to this path")

    build = commands.add_parser("build-index", help="Write a deterministic normalized index snapshot")
    _add_chip_paths(build)
    build.add_argument("--output", required=True, help="Output JSON path")

    query = commands.add_parser("query", help="Retrieve and project Chip knowledge")
    _add_chip_paths(query)
    query.add_argument("--query", required=True, help="Task or research question")
    query.add_argument("--role", default="researcher", help="planner, critic, executor, verifier, or researcher")
    query.add_argument("--layers", nargs="+", default=["L1", "L2", "L3"])
    query.add_argument("--candidate-limit", type=int, default=8)
    query.add_argument("--per-layer-limit", type=int, default=8)
    query.add_argument("--total-hit-limit", type=int, default=24)
    query.add_argument("--token-budget", type=int, default=4000)
    query.add_argument("--runtime", help="Optional runtime JSONL file/directory for feedback priors")
    query.add_argument("--format", choices=["json", "markdown"], default="markdown")
    query.add_argument("--include-payload", action="store_true", help="Include original graph payloads in JSON")
    query.add_argument("--output", help="Write result to this path")

    demo = commands.add_parser("demo-lifecycle", help="Run one retrieval and record a synthetic lifecycle")
    _add_chip_paths(demo)
    demo.add_argument("--query", required=True)
    demo.add_argument("--role", default="researcher")
    demo.add_argument("--runtime", required=True, help="Runtime JSONL file/directory")
    demo.add_argument("--success", action=argparse.BooleanOptionalAction, default=True)
    demo.add_argument("--output", help="Write result JSON to this path")

    runtime = commands.add_parser("runtime-summary", help="Summarize an append-only runtime store")
    runtime.add_argument("--runtime", required=True, help="Runtime JSONL file/directory")
    runtime.add_argument("--output", help="Write JSON report to this path")

    return parser


def _validate(paths: Iterable[str], strict_warnings: bool) -> tuple[dict[str, Any], int]:
    loader = ChipLoader()
    files = ChipStore.discover(paths)
    reports = [loader.validate(path) for path in files]
    valid_count = sum(report["valid"] for report in reports)
    warning_count = sum(len(report.get("warnings", [])) for report in reports)
    error_count = sum(len(report.get("errors", [])) for report in reports)
    result = {
        "summary": {
            "files": len(reports),
            "valid": valid_count,
            "invalid": len(reports) - valid_count,
            "warnings": warning_count,
            "errors": error_count,
        },
        "reports": reports,
    }
    return result, 2 if error_count or (strict_warnings and warning_count) else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "validate":
        report, code = _validate(args.chips, args.strict_warnings)
        _write_or_print(_json_text(report), args.output)
        return code

    if args.command == "build-index":
        store = ChipStore.from_paths(args.chips)
        snapshot = ChipIndex(store).snapshot()
        _write_or_print(_json_text(snapshot), args.output)
        return 0

    if args.command == "query":
        engine = ChipMemoryEngine.from_paths(args.chips, runtime_path=args.runtime)
        result = engine.retrieve_memory(
            args.query,
            role=args.role,
            layers=args.layers,
            candidate_limit=args.candidate_limit,
            per_layer_limit=args.per_layer_limit,
            total_hit_limit=args.total_hit_limit,
            token_budget=args.token_budget,
        )
        if args.format == "json":
            text = _json_text(result.to_dict(include_payload=args.include_payload))
        else:
            text = projection_to_markdown(result.projection)
        _write_or_print(text, args.output)
        return 0

    if args.command == "demo-lifecycle":
        engine = ChipMemoryEngine.from_paths(args.chips, runtime_path=args.runtime)
        engine.init_task_context(args.query, agent_roles=[args.role, "verifier"])
        result = engine.retrieve_memory(args.query, role=args.role, token_budget=2000)
        engine.add_agent_node(
            "demo-agent",
            "Used the projected Chip subgraph to propose a plan.",
            role=args.role,
        )
        engine.move_memory_state("propose grounded plan", "synthetic demonstration", reward=args.success)
        context = engine.save_task_context(args.success, "Synthetic CLI lifecycle demonstration")
        engine.backward(args.success)
        output = {"retrieval": result.to_dict(), "context": context.to_dict()}
        _write_or_print(_json_text(output), args.output)
        return 0

    if args.command == "runtime-summary":
        summary = RuntimeEventStore(args.runtime).summary()
        _write_or_print(_json_text(summary), args.output)
        return 0

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

