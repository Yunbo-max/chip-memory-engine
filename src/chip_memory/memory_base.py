"""Lifecycle interface inspired by common multi-agent memory integration points."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from .types import RetrievalResult, TaskContext


class ChipMemoryBase(ABC):
    """Minimal interface an agent framework needs from Chip Memory."""

    @abstractmethod
    def init_task_context(
        self,
        task: str,
        task_description: str = "",
        agent_roles: Iterable[str] = (),
    ) -> TaskContext:
        raise NotImplementedError

    @abstractmethod
    def retrieve_memory(self, query_task: str | None = None, **kwargs: Any) -> RetrievalResult:
        raise NotImplementedError

    @abstractmethod
    def add_agent_node(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def move_memory_state(self, action: str, observation: str, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_task_context(self, label: bool, feedback: str | None = None) -> TaskContext:
        raise NotImplementedError

    @abstractmethod
    def backward(self, reward: bool | float) -> None:
        raise NotImplementedError

