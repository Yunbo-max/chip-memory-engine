"""Chip Memory reference engine.

Paper Chips are immutable source records.  The package builds derived indexes
and keeps runtime observations in a separate append-only event log.
"""

from .engine import ChipMemoryEngine
from .loader import ChipLoader, ChipStore
from .runtime import RuntimeEventStore
from .types import (
    ChipRecord,
    KnowledgeItem,
    Layer,
    Projection,
    RetrievalResult,
    TaskContext,
    VerificationFinding,
)

__all__ = [
    "ChipLoader",
    "ChipMemoryEngine",
    "ChipRecord",
    "ChipStore",
    "KnowledgeItem",
    "Layer",
    "Projection",
    "RetrievalResult",
    "RuntimeEventStore",
    "TaskContext",
    "VerificationFinding",
]

__version__ = "0.1.0"

