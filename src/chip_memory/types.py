"""Typed values shared by loading, retrieval, projection, and runtime code."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


class Layer(str, Enum):
    """Chip representation layers.

    META is used for candidate ranking only.  It is not presented as one of the
    paper's three graph layers.
    """

    META = "META"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"

    @classmethod
    def parse_many(cls, values: Iterable[str] | None) -> tuple["Layer", ...]:
        if not values:
            return (cls.L1, cls.L2, cls.L3)
        parsed: list[Layer] = []
        for value in values:
            layer = cls(str(value).upper())
            if layer is cls.META:
                continue
            if layer not in parsed:
                parsed.append(layer)
        return tuple(parsed)


@dataclass(frozen=True)
class KnowledgeItem:
    """One retrievable unit derived from a Chip without changing its source."""

    item_id: str
    chip_id: str
    layer: Layer
    pin: str
    kind: str
    text: str
    local_id: str | None = None
    references: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    source_path: str = ""
    source_url: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def citation(self) -> str:
        target = self.source_url or self.source_path or self.chip_id
        return f"{target}#{self.item_id}"

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "item_id": self.item_id,
            "chip_id": self.chip_id,
            "layer": self.layer.value,
            "pin": self.pin,
            "kind": self.kind,
            "text": self.text,
            "local_id": self.local_id,
            "references": list(self.references),
            "evidence": list(self.evidence),
            "source_path": self.source_path,
            "source_url": self.source_url,
            "citation": self.citation(),
        }
        if include_payload:
            data["payload"] = dict(self.payload)
        return data


@dataclass(frozen=True)
class ChipRecord:
    """Normalized read-only view over one source paper Chip."""

    chip_id: str
    title: str
    chip_type: str
    schema_version: str
    source_path: Path
    source_url: str | None
    meta_text: str
    items: tuple[KnowledgeItem, ...]
    node_ids: frozenset[str]
    warnings: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def layer_counts(self) -> dict[str, int]:
        counts = {layer.value: 0 for layer in (Layer.L1, Layer.L2, Layer.L3)}
        for item in self.items:
            if item.layer.value in counts:
                counts[item.layer.value] += 1
        return counts

    def to_catalog_entry(self) -> dict[str, Any]:
        return {
            "chip_id": self.chip_id,
            "title": self.title,
            "chip_type": self.chip_type,
            "schema_version": self.schema_version,
            "source_path": str(self.source_path),
            "source_url": self.source_url,
            "layer_counts": self.layer_counts(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RetrievalHit:
    item: KnowledgeItem
    score: float
    lexical_score: float
    feedback_score: float = 0.0
    graph_bonus: float = 0.0
    role_bonus: float = 0.0
    reasons: tuple[str, ...] = ()

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        return {
            **self.item.to_dict(include_payload=include_payload),
            "score": round(self.score, 6),
            "score_components": {
                "lexical": round(self.lexical_score, 6),
                "feedback": round(self.feedback_score, 6),
                "graph": round(self.graph_bonus, 6),
                "role": round(self.role_bonus, 6),
            },
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class VerificationFinding:
    severity: str
    code: str
    message: str
    chip_id: str
    item_id: str | None = None
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "chip_id": self.chip_id,
            "item_id": self.item_id,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class Projection:
    role: str
    task: str
    hits: tuple[RetrievalHit, ...]
    findings: tuple[VerificationFinding, ...]
    estimated_tokens: int
    omitted_hits: int = 0

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        return {
            "role": self.role,
            "task": self.task,
            "estimated_tokens": self.estimated_tokens,
            "omitted_hits": self.omitted_hits,
            "hits": [hit.to_dict(include_payload=include_payload) for hit in self.hits],
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    role: str
    layers: tuple[Layer, ...]
    candidate_chips: tuple[tuple[str, float], ...]
    projection: Projection
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        return {
            "query": self.query,
            "role": self.role,
            "layers": [layer.value for layer in self.layers],
            "candidate_chips": [
                {"chip_id": chip_id, "score": round(score, 6)}
                for chip_id, score in self.candidate_chips
            ],
            "projection": self.projection.to_dict(include_payload=include_payload),
            "diagnostics": dict(self.diagnostics),
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskContext:
    context_id: str
    task: str
    task_description: str
    agent_roles: tuple[str, ...]
    created_at: str = field(default_factory=utc_now)
    retrieval_item_ids: list[str] = field(default_factory=list)
    retrieval_chip_ids: list[str] = field(default_factory=list)
    agent_events: list[dict[str, Any]] = field(default_factory=list)
    state_transitions: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False
    success: bool | None = None
    feedback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "task": self.task,
            "task_description": self.task_description,
            "agent_roles": list(self.agent_roles),
            "created_at": self.created_at,
            "retrieval_item_ids": list(self.retrieval_item_ids),
            "retrieval_chip_ids": list(self.retrieval_chip_ids),
            "agent_events": list(self.agent_events),
            "state_transitions": list(self.state_transitions),
            "completed": self.completed,
            "success": self.success,
            "feedback": self.feedback,
        }

