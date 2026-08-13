"""Load heterogeneous paper-Chip JSON into a stable runtime representation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .types import ChipRecord, KnowledgeItem, Layer


EXTENDED_PINS = (
    "problem_gap",
    "method_mechanism",
    "evaluation_validation",
    "experimental_setting",
    "result_outcome",
    "implementation",
    "reuse_transfer",
)
CLASSIC_PINS = ("gap", "method", "evaluation", "result")
META_KEYS = (
    "title",
    "authors",
    "abstract",
    "paper_metadata",
    "chip_metadata",
    "source_coverage",
    "target_domain_fit",
    "footprint",
    "recorded_gaps",
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key, inner in value.items():
            text = _clean_text(inner)
            if text:
                parts.append(f"{str(key).replace('_', ' ')}: {text}")
        return " ".join(parts)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(filter(None, (_clean_text(item) for item in value)))
    return _clean_text(str(value))


def _first_text(*values: Any, default: str = "") -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return default


def _stable_suffix(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _source_url(raw: Mapping[str, Any]) -> str | None:
    candidates = [
        raw.get("source_url"),
        raw.get("url"),
        (raw.get("chip_metadata") or {}).get("source_url") if isinstance(raw.get("chip_metadata"), Mapping) else None,
        (raw.get("paper_metadata") or {}).get("url") if isinstance(raw.get("paper_metadata"), Mapping) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
            return candidate
    return None


def _event_pin(kind: str) -> str:
    lowered = kind.lower()
    if "gap" in lowered or "failure" in lowered:
        return "problem_gap"
    if "eval" in lowered or "metric" in lowered:
        return "evaluation_validation"
    if "result" in lowered or "outcome" in lowered:
        return "result_outcome"
    if "ground" in lowered or "implement" in lowered or "code" in lowered:
        return "implementation"
    if "compat" in lowered or "transfer" in lowered or "temporal" in lowered:
        return "reuse_transfer"
    return "method_mechanism"


def _node_pin(kind: str) -> str:
    lowered = kind.lower()
    if any(term in lowered for term in ("gap", "failure", "limitation", "risk")):
        return "problem_gap"
    if any(term in lowered for term in ("dataset", "metric", "evaluation", "benchmark", "judge")):
        return "evaluation_validation"
    if any(term in lowered for term in ("result", "outcome", "significance")):
        return "result_outcome"
    if any(term in lowered for term in ("code", "repository", "implementation", "command", "artifact")):
        return "implementation"
    if any(term in lowered for term in ("scenario", "compatibility", "transfer")):
        return "reuse_transfer"
    return "method_mechanism"


def _extract_evidence(payload: Mapping[str, Any]) -> tuple[str, ...]:
    evidence: list[str] = []
    for key in (
        "evidence",
        "evidence_anchor",
        "evidence_anchors",
        "source",
        "sources",
        "citation",
        "citations",
        "section",
        "page",
        "table",
        "figure",
        "numerical",
    ):
        if key in payload:
            value = payload[key]
            if isinstance(value, list):
                evidence.extend(filter(None, (_clean_text(item) for item in value)))
            else:
                text = _clean_text(value)
                if text:
                    evidence.append(text)
    return tuple(dict.fromkeys(evidence))


def _known_references(payload: Mapping[str, Any], node_ids: set[str]) -> tuple[str, ...]:
    references: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            if value in node_ids:
                references.append(value)
        elif isinstance(value, Mapping):
            for inner in value.values():
                walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)

    walk(payload)
    return tuple(dict.fromkeys(references))


def _walk_graph_lists(value: Any, path: str = "root") -> Iterator[tuple[str, str, list[Any]]]:
    """Yield nested nodes/edges/events lists and their logical path."""

    if isinstance(value, Mapping):
        for key, inner in value.items():
            next_path = f"{path}.{key}"
            if key in {"nodes", "edges", "events"} and isinstance(inner, list):
                yield next_path, key, inner
            elif key in {"nodes", "edges", "events"} and isinstance(inner, Mapping):
                # A small number of historical Chips use one object directly;
                # others use an ID-keyed object.  Normalize both forms.
                if "id" in inner or "kind" in inner or "type" in inner:
                    yield next_path, key, [inner]
                else:
                    yield next_path, key, [item for item in inner.values() if isinstance(item, Mapping)]
            elif isinstance(inner, (Mapping, list)):
                yield from _walk_graph_lists(inner, next_path)
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            if isinstance(inner, (Mapping, list)):
                yield from _walk_graph_lists(inner, f"{path}[{index}]")


class ChipLoader:
    """Parser and structural validator for current Chip schema variants."""

    def load(self, path: str | Path) -> ChipRecord:
        source_path = Path(path).resolve()
        with source_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Chip root must be an object: {source_path}")

        chip_id = _first_text(raw.get("chip_id"), raw.get("id"))
        if not chip_id:
            raise ValueError(f"Chip is missing chip_id/id: {source_path}")
        title = _first_text(
            raw.get("title"),
            (raw.get("paper_metadata") or {}).get("title") if isinstance(raw.get("paper_metadata"), Mapping) else None,
            (raw.get("chip_metadata") or {}).get("title") if isinstance(raw.get("chip_metadata"), Mapping) else None,
            default=chip_id,
        )
        chip_type = _first_text(raw.get("chip_type"), default="static-paper")
        schema_version = _first_text(raw.get("schema_version"), default="unknown")
        url = _source_url(raw)

        pin_values: dict[str, Any] = {
            pin: raw[pin]
            for pin in EXTENDED_PINS
            if pin in raw and raw[pin] not in (None, "", [], {})
        }
        if not pin_values:
            pin_values = {
                pin: raw[pin]
                for pin in CLASSIC_PINS
                if pin in raw and raw[pin] not in (None, "", [], {})
            }

        meta_parts = [title]
        for key in META_KEYS:
            if key in raw:
                meta_parts.append(f"{key.replace('_', ' ')}: {_clean_text(raw[key])}")
        for pin, value in pin_values.items():
            meta_parts.append(f"{pin.replace('_', ' ')}: {_clean_text(value)}")
        meta_text = " ".join(filter(None, meta_parts))

        graph_lists = list(_walk_graph_lists(raw))
        node_payloads: list[tuple[str, Mapping[str, Any]]] = []
        for graph_path, graph_kind, entries in graph_lists:
            if graph_kind != "nodes":
                continue
            for entry in entries:
                if isinstance(entry, Mapping):
                    node_payloads.append((graph_path, entry))
        node_ids = {
            str(payload.get("id"))
            for _, payload in node_payloads
            if payload.get("id") not in (None, "")
        }

        items: list[KnowledgeItem] = []
        seen: set[tuple[str, str, str]] = set()

        def add_item(
            *,
            layer: Layer,
            pin: str,
            kind: str,
            payload: Mapping[str, Any],
            graph_path: str,
            local_id: str | None,
            text: str,
        ) -> None:
            normalized_text = " ".join(text.split())
            if not normalized_text:
                return
            identity = local_id or _stable_suffix(payload)
            dedupe_key = (layer.value, kind, identity)
            if dedupe_key in seen:
                return
            seen.add(dedupe_key)
            item_id = f"{chip_id}:{layer.value}:{kind}:{identity}"
            items.append(
                KnowledgeItem(
                    item_id=item_id,
                    chip_id=chip_id,
                    layer=layer,
                    pin=pin,
                    kind=kind,
                    text=normalized_text,
                    local_id=local_id,
                    references=_known_references(payload, node_ids),
                    evidence=_extract_evidence(payload),
                    source_path=str(source_path),
                    source_url=url,
                    payload={**dict(payload), "_graph_path": graph_path},
                )
            )

        for graph_path, payload in node_payloads:
            local_id = _first_text(payload.get("id")) or None
            kind = _first_text(payload.get("kind"), payload.get("type"), default="Node")
            label = _first_text(payload.get("label"), payload.get("text"), payload.get("name"), default=local_id or kind)
            props = _clean_text(payload.get("props"))
            add_item(
                layer=Layer.L1,
                pin=_node_pin(kind),
                kind=kind,
                payload=payload,
                graph_path=graph_path,
                local_id=local_id,
                text=f"{kind}: {label}. {props}" if props else f"{kind}: {label}",
            )

        for graph_path, graph_kind, entries in graph_lists:
            if graph_kind == "nodes":
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                local_id = _first_text(entry.get("id")) or None
                if graph_kind == "edges":
                    kind = _first_text(entry.get("rel"), entry.get("relation"), entry.get("kind"), entry.get("type"), default="related_to")
                    source = _first_text(entry.get("src"), entry.get("source"), entry.get("from"))
                    target = _first_text(entry.get("dst"), entry.get("target"), entry.get("to"))
                    pin = "cross_pin"
                    node_pin_by_id = {
                        str(payload.get("id")): _node_pin(_first_text(payload.get("kind"), payload.get("type"), default="Node"))
                        for _, payload in node_payloads
                        if payload.get("id") not in (None, "")
                    }
                    if source and node_pin_by_id.get(source) == node_pin_by_id.get(target):
                        pin = node_pin_by_id.get(source, "cross_pin")
                    text = f"{source or 'unknown'} {kind} {target or 'unknown'}. {_clean_text(entry)}"
                    add_item(
                        layer=Layer.L2,
                        pin=pin,
                        kind=kind,
                        payload=entry,
                        graph_path=graph_path,
                        local_id=local_id,
                        text=text,
                    )
                elif graph_kind == "events":
                    kind = _first_text(entry.get("kind"), entry.get("event_type"), entry.get("type"), default="Event")
                    layer = Layer.L1 if kind.lower() == "structure" else Layer.L3
                    label = _first_text(entry.get("label"), entry.get("name"), default=kind)
                    add_item(
                        layer=layer,
                        pin=_event_pin(kind),
                        kind=kind,
                        payload=entry,
                        graph_path=graph_path,
                        local_id=local_id,
                        text=f"{kind}: {label}. {_clean_text(entry)}",
                    )

        # Some classic Chips store result hyperedges outside an events list.
        for result_key in ("result", "result_outcome"):
            result_container = raw.get(result_key)
            if not isinstance(result_container, Mapping):
                continue
            results = result_container.get("results")
            if not isinstance(results, list):
                continue
            for result in results:
                if not isinstance(result, Mapping):
                    continue
                local_id = _first_text(result.get("id")) or None
                add_item(
                    layer=Layer.L3,
                    pin="result_outcome",
                    kind="Result",
                    payload=result,
                    graph_path=f"root.{result_key}.results",
                    local_id=local_id,
                    text=f"Result: {_clean_text(result)}",
                )

        warnings: list[str] = []
        layer_counts = Counter(item.layer for item in items)
        for layer in (Layer.L1, Layer.L2, Layer.L3):
            if layer_counts[layer] == 0:
                warnings.append(f"No {layer.value} items were found")
        if not pin_values:
            warnings.append("No recognized semantic pins were found")
        if not url:
            warnings.append("No HTTP(S) source URL was found")

        return ChipRecord(
            chip_id=chip_id,
            title=title,
            chip_type=chip_type,
            schema_version=schema_version,
            source_path=source_path,
            source_url=url,
            meta_text=meta_text,
            items=tuple(items),
            node_ids=frozenset(node_ids),
            warnings=tuple(warnings),
            raw=raw,
        )

    def validate(self, path: str | Path) -> dict[str, Any]:
        try:
            chip = self.load(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return {
                "path": str(Path(path).resolve()),
                "valid": False,
                "errors": [str(error)],
                "warnings": [],
            }
        counts = chip.layer_counts()
        return {
            "path": str(chip.source_path),
            "chip_id": chip.chip_id,
            "title": chip.title,
            "valid": True,
            "errors": [],
            "warnings": list(chip.warnings),
            "layer_counts": counts,
            "item_count": len(chip.items),
        }


class ChipStore:
    """Immutable collection of loaded paper Chips."""

    def __init__(self, chips: Iterable[ChipRecord]):
        by_id: dict[str, ChipRecord] = {}
        duplicates: dict[str, list[str]] = {}
        for chip in chips:
            if chip.chip_id in by_id:
                duplicates.setdefault(chip.chip_id, [str(by_id[chip.chip_id].source_path)]).append(str(chip.source_path))
                continue
            by_id[chip.chip_id] = chip
        if duplicates:
            details = "; ".join(f"{chip_id}: {paths}" for chip_id, paths in sorted(duplicates.items()))
            raise ValueError(f"Duplicate chip_id values: {details}")
        self._chips = by_id
        self._items = {
            item.item_id: item
            for chip in by_id.values()
            for item in chip.items
        }

    @staticmethod
    def discover(paths: Iterable[str | Path]) -> list[Path]:
        discovered: set[Path] = set()
        for value in paths:
            path = Path(value).resolve()
            if path.is_file():
                discovered.add(path)
            elif path.is_dir():
                matches = list(path.rglob("*.chip.json"))
                if not matches:
                    matches = [candidate for candidate in path.rglob("*.json") if candidate.is_file()]
                discovered.update(candidate.resolve() for candidate in matches)
            else:
                raise FileNotFoundError(path)
        return sorted(discovered)

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path], *, strict: bool = True) -> "ChipStore":
        loader = ChipLoader()
        chips: list[ChipRecord] = []
        failures: list[str] = []
        for path in cls.discover(paths):
            try:
                chips.append(loader.load(path))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                failures.append(f"{path}: {error}")
        if strict and failures:
            raise ValueError("Failed to load Chips:\n" + "\n".join(failures))
        return cls(chips)

    def __len__(self) -> int:
        return len(self._chips)

    def __iter__(self) -> Iterator[ChipRecord]:
        return iter(self._chips.values())

    def get_chip(self, chip_id: str) -> ChipRecord | None:
        return self._chips.get(chip_id)

    def get_item(self, item_id: str) -> KnowledgeItem | None:
        return self._items.get(item_id)

    @property
    def chips(self) -> Mapping[str, ChipRecord]:
        return self._chips

    @property
    def items(self) -> Mapping[str, KnowledgeItem]:
        return self._items

    def catalog(self) -> list[dict[str, Any]]:
        return [chip.to_catalog_entry() for chip in sorted(self, key=lambda value: value.chip_id)]
