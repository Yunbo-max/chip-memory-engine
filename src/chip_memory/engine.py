"""High-level Chip Memory engine with a multi-agent lifecycle API."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from .loader import ChipStore
from .memory_base import ChipMemoryBase
from .retriever import ChipRetriever
from .runtime import FeedbackPriors, RuntimeEventStore
from .types import RetrievalResult, TaskContext, utc_now


class ChipMemoryEngine(ChipMemoryBase):
    """Grounded paper retrieval plus separately recorded runtime experience."""

    def __init__(self, store: ChipStore, runtime_store: RuntimeEventStore | None = None):
        self.store = store
        self.runtime_store = runtime_store
        self.retriever = ChipRetriever(store)
        self.current_task_context: TaskContext | None = None
        self._lock = threading.RLock()

    @classmethod
    def from_paths(
        cls,
        chip_paths: Iterable[str | Path],
        *,
        runtime_path: str | Path | None = None,
        strict: bool = True,
    ) -> "ChipMemoryEngine":
        store = ChipStore.from_paths(chip_paths, strict=strict)
        runtime = RuntimeEventStore(runtime_path) if runtime_path is not None else None
        return cls(store, runtime)

    def init_task_context(
        self,
        task: str,
        task_description: str = "",
        agent_roles: Iterable[str] = (),
    ) -> TaskContext:
        if not task.strip():
            raise ValueError("task must not be empty")
        with self._lock:
            if self.current_task_context and not self.current_task_context.completed:
                raise RuntimeError("A task context is already active; complete it before starting another")
            context = TaskContext(
                context_id=str(uuid.uuid4()),
                task=task.strip(),
                task_description=(task_description or task).strip(),
                agent_roles=tuple(dict.fromkeys(str(role) for role in agent_roles if str(role).strip())),
            )
            self.current_task_context = context
            self._append_runtime("context_started", context.to_dict())
            return context

    def retrieve_memory(
        self,
        query_task: str | None = None,
        *,
        role: str = "researcher",
        layers: Iterable[str] | None = None,
        candidate_limit: int = 8,
        per_layer_limit: int = 8,
        total_hit_limit: int = 24,
        token_budget: int = 4000,
    ) -> RetrievalResult:
        query = (query_task or (self.current_task_context.task if self.current_task_context else "")).strip()
        if not query:
            raise ValueError("query_task is required when no task context is active")
        feedback = self.runtime_store.feedback_priors() if self.runtime_store else FeedbackPriors({}, {}, 0)
        result = self.retriever.retrieve(
            query,
            role=role,
            layers=layers,
            candidate_limit=candidate_limit,
            per_layer_limit=per_layer_limit,
            total_hit_limit=total_hit_limit,
            token_budget=token_budget,
            feedback=feedback,
        )
        with self._lock:
            if self.current_task_context:
                item_ids = [hit.item.item_id for hit in result.projection.hits]
                chip_ids = list(
                    dict.fromkeys(hit.item.chip_id for hit in result.projection.hits)
                )
                candidate_chip_ids = [chip_id for chip_id, _ in result.candidate_chips]
                self.current_task_context.retrieval_item_ids.extend(
                    item_id for item_id in item_ids if item_id not in self.current_task_context.retrieval_item_ids
                )
                self.current_task_context.retrieval_chip_ids.extend(
                    chip_id for chip_id in chip_ids if chip_id not in self.current_task_context.retrieval_chip_ids
                )
                self._append_runtime(
                    "retrieval",
                    {
                        "query": query,
                        "role": result.role,
                        "layers": [layer.value for layer in result.layers],
                        "item_ids": item_ids,
                        "chip_ids": chip_ids,
                        "candidate_chip_ids": candidate_chip_ids,
                        "diagnostics": dict(result.diagnostics),
                    },
                )
        return result

    def add_agent_node(
        self,
        agent_name: str | Any,
        message: str | None = None,
        *,
        role: str | None = None,
        upstream_agent_ids: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        context = self._require_context()
        # Accept lightweight objects with G-Memory-like attributes without
        # importing or depending on that implementation.
        if not isinstance(agent_name, str):
            obj = agent_name
            message = message or getattr(obj, "message", None)
            role = role or getattr(obj, "role", None) or getattr(obj, "agent_name", None)
            agent_name = str(getattr(obj, "agent_name", getattr(obj, "name", "agent")))
        event_id = str(uuid.uuid4())
        event = {
            "event_id": event_id,
            "event_type": "agent_message",
            "timestamp": utc_now(),
            "agent_name": agent_name,
            "role": role,
            "message": message or "",
            "upstream_agent_ids": list(upstream_agent_ids),
            "metadata": dict(metadata or {}),
        }
        with self._lock:
            context.agent_events.append(event)
            self._append_runtime("agent_event", event)
        return event_id

    def record_agent_event(
        self,
        *,
        agent_name: str,
        role: str,
        event_type: str,
        content: str,
        participants: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        return self.add_agent_node(
            agent_name,
            content,
            role=role,
            upstream_agent_ids=participants,
            metadata={"typed_event": event_type, **dict(metadata or {})},
        )

    def move_memory_state(
        self,
        action: str,
        observation: str,
        *,
        reward: float | bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        context = self._require_context()
        transition = {
            "transition_id": str(uuid.uuid4()),
            "timestamp": utc_now(),
            "action": action,
            "observation": observation,
            "reward": reward,
            "metadata": dict(metadata or {}),
        }
        with self._lock:
            context.state_transitions.append(transition)
            self._append_runtime("state_transition", transition)

    def save_task_context(self, label: bool, feedback: str | None = None) -> TaskContext:
        context = self._require_context()
        with self._lock:
            if context.completed:
                raise RuntimeError("Task context has already been completed")
            context.completed = True
            context.success = bool(label)
            context.feedback = feedback
            self._append_runtime(
                "task_completed",
                {
                    "success": context.success,
                    "feedback": feedback,
                    "retrieval_item_ids": list(context.retrieval_item_ids),
                    "retrieval_chip_ids": list(context.retrieval_chip_ids),
                    "agent_event_count": len(context.agent_events),
                    "state_transition_count": len(context.state_transitions),
                },
            )
            return context

    def backward(self, reward: bool | float) -> None:
        context = self._require_context(allow_completed=True)
        numeric_reward = (1.0 if reward else -1.0) if isinstance(reward, bool) else float(reward)
        self._append_runtime(
            "backward",
            {
                "reward": numeric_reward,
                "item_ids": list(context.retrieval_item_ids),
                "chip_ids": list(context.retrieval_chip_ids),
            },
        )

    def summarize(self) -> str:
        context = self._require_context(allow_completed=True)
        transition_text = "\n".join(
            f"> {transition['action']}\n{transition['observation']}"
            for transition in context.state_transitions
        )
        return "\n".join(filter(None, (context.task_description, transition_text)))

    @property
    def memory_size(self) -> int:
        return len(self.store)

    def _require_context(self, *, allow_completed: bool = False) -> TaskContext:
        if self.current_task_context is None:
            raise RuntimeError("No active task context; call init_task_context first")
        if self.current_task_context.completed and not allow_completed:
            raise RuntimeError("The current task context is already completed")
        return self.current_task_context

    def _append_runtime(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self.runtime_store is None or self.current_task_context is None:
            return
        self.runtime_store.append(event_type, self.current_task_context.context_id, payload)
