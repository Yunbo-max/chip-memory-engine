"""Append-only runtime observations kept separate from immutable paper Chips."""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .types import utc_now

try:  # Unix advisory locking; the thread lock remains the portable fallback.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FeedbackPriors:
    item_scores: Mapping[str, float]
    chip_scores: Mapping[str, float]
    completed_contexts: int


class RuntimeEventStore:
    """JSONL event store.

    The file is append-only.  Derived priors are recomputed from completed
    contexts, making provenance inspectable and avoiding silent mutations.
    """

    def __init__(self, path: str | Path):
        resolved = Path(path).resolve()
        if (resolved.exists() and resolved.is_dir()) or resolved.suffix.lower() != ".jsonl":
            self.path = resolved / "usage_events.jsonl"
        else:
            self.path = resolved
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.Lock()

    def append(self, event_type: str, context_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": utc_now(),
            "event_type": event_type,
            "context_id": context_id,
            "payload": dict(payload),
        }
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        with self._thread_lock:
            with self.path.open("a", encoding="utf-8") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return event

    def iter_events(self, *, tolerate_partial_tail: bool = True) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    if tolerate_partial_tail:
                        continue
                    raise ValueError(f"Invalid runtime JSONL at {self.path}:{line_number}")
                if isinstance(value, dict):
                    yield value

    def feedback_priors(self) -> FeedbackPriors:
        retrievals: dict[str, tuple[set[str], set[str]]] = {}
        outcomes: dict[str, float] = {}

        for event in self.iter_events():
            context_id = str(event.get("context_id", ""))
            payload = event.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            if event.get("event_type") == "retrieval":
                item_ids, chip_ids = retrievals.setdefault(context_id, (set(), set()))
                item_ids.update(str(value) for value in payload.get("item_ids", []) if value)
                chip_ids.update(str(value) for value in payload.get("chip_ids", []) if value)
            elif event.get("event_type") == "task_completed":
                success = payload.get("success")
                if isinstance(success, bool):
                    outcomes[context_id] = 1.0 if success else 0.0
            elif event.get("event_type") == "backward":
                reward = payload.get("reward")
                if isinstance(reward, (int, float)) and not isinstance(reward, bool):
                    outcomes[context_id] = 1.0 if reward > 0 else 0.0

        item_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        chip_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for context_id, outcome in outcomes.items():
            if context_id not in retrievals:
                continue
            success = 1 if outcome > 0 else 0
            item_ids, chip_ids = retrievals[context_id]
            for item_id in item_ids:
                item_counts[item_id][0] += success
                item_counts[item_id][1] += 1
            for chip_id in chip_ids:
                chip_counts[chip_id][0] += success
                chip_counts[chip_id][1] += 1

        def centered_prior(successes: int, trials: int) -> float:
            # Beta(1,1) smoothing; range approaches [-0.5, +0.5].
            return (successes + 1.0) / (trials + 2.0) - 0.5

        return FeedbackPriors(
            item_scores={key: centered_prior(*counts) for key, counts in item_counts.items()},
            chip_scores={key: centered_prior(*counts) for key, counts in chip_counts.items()},
            completed_contexts=sum(context_id in retrievals for context_id in outcomes),
        )

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = defaultdict(int)
        contexts: set[str] = set()
        for event in self.iter_events():
            counts[str(event.get("event_type", "unknown"))] += 1
            if event.get("context_id"):
                contexts.add(str(event["context_id"]))
        priors = self.feedback_priors()
        return {
            "path": str(self.path),
            "event_counts": dict(sorted(counts.items())),
            "contexts": len(contexts),
            "completed_contexts_with_retrieval": priors.completed_contexts,
            "scored_items": len(priors.item_scores),
            "scored_chips": len(priors.chip_scores),
        }
