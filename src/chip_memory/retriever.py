"""Staged candidate, layer, graph-expansion, and role-aware retrieval."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Mapping

from .loader import ChipStore
from .projector import RoleProjector, role_bonus
from .runtime import FeedbackPriors
from .text import BM25Index, unique_terms
from .types import Layer, RetrievalHit, RetrievalResult
from .verifier import CompatibilityVerifier


class ChipIndex:
    def __init__(self, store: ChipStore):
        self.store = store
        self.chip_index = BM25Index(
            {
                chip.chip_id: " ".join((chip.meta_text, *(item.text for item in chip.items)))
                for chip in store
            }
        )
        self.item_index = BM25Index({item_id: item.text for item_id, item in store.items.items()})
        self.items_by_chip: dict[str, set[str]] = defaultdict(set)
        self.items_by_reference: dict[tuple[str, str], set[str]] = defaultdict(set)
        for item_id, item in store.items.items():
            self.items_by_chip[item.chip_id].add(item_id)
            for reference in item.references:
                self.items_by_reference[(item.chip_id, reference)].add(item_id)

    def connected_item_ids(self, item_ids: Iterable[str]) -> dict[str, int]:
        connected: dict[str, int] = defaultdict(int)
        for item_id in item_ids:
            item = self.store.get_item(item_id)
            if item is None:
                continue
            for reference in item.references:
                for related_id in self.items_by_reference.get((item.chip_id, reference), ()):
                    if related_id != item_id:
                        connected[related_id] += 1
        return connected

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "chip-memory-runtime-index-v1",
            "chip_count": len(self.store),
            "item_count": len(self.store.items),
            "catalog": self.store.catalog(),
            "items": [
                item.to_dict(include_payload=False)
                for item in sorted(self.store.items.values(), key=lambda value: value.item_id)
            ],
        }


class ChipRetriever:
    def __init__(
        self,
        store: ChipStore,
        *,
        verifier: CompatibilityVerifier | None = None,
        projector: RoleProjector | None = None,
    ):
        self.store = store
        self.index = ChipIndex(store)
        self.verifier = verifier or CompatibilityVerifier()
        self.projector = projector or RoleProjector()

    def retrieve(
        self,
        query: str,
        *,
        role: str = "researcher",
        layers: Iterable[str] | None = None,
        candidate_limit: int = 8,
        per_layer_limit: int = 8,
        total_hit_limit: int = 24,
        token_budget: int = 4000,
        feedback: FeedbackPriors | None = None,
    ) -> RetrievalResult:
        selected_layers = Layer.parse_many(layers)
        feedback = feedback or FeedbackPriors({}, {}, 0)

        raw_candidates = self.index.chip_index.search(query, limit=candidate_limit)
        if not raw_candidates:
            # Empty or out-of-vocabulary queries still have deterministic behavior,
            # but the diagnostic clearly exposes that no lexical match was found.
            raw_candidates = [(chip.chip_id, 0.0) for chip in list(self.store)[:candidate_limit]]
        candidates = [
            (chip_id, score + feedback.chip_scores.get(chip_id, 0.0) * 0.5)
            for chip_id, score in raw_candidates
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        candidate_ids = {chip_id for chip_id, _ in candidates}

        allowed_item_ids = {
            item_id
            for chip_id in candidate_ids
            for item_id in self.index.items_by_chip.get(chip_id, ())
            if self.store.items[item_id].layer in selected_layers
        }
        query_terms = unique_terms(query)
        seed_hits: list[RetrievalHit] = []
        for item_id in allowed_item_ids:
            item = self.store.items[item_id]
            lexical = self.index.item_index.score(query, item_id)
            if lexical <= 0:
                continue
            item_feedback = feedback.item_scores.get(item_id, 0.0) * 0.75
            provisional = RetrievalHit(
                item=item,
                score=lexical + item_feedback,
                lexical_score=lexical,
                feedback_score=item_feedback,
                reasons=(
                    "lexical match: "
                    + ", ".join(sorted(query_terms & unique_terms(item.text))[:10]),
                ),
            )
            bonus, bonus_reasons = role_bonus(provisional, role)
            seed_hits.append(
                replace(
                    provisional,
                    score=provisional.score + bonus,
                    role_bonus=bonus,
                    reasons=tuple(dict.fromkeys((*provisional.reasons, *bonus_reasons))),
                )
            )

        seed_hits.sort(key=lambda hit: (-hit.score, hit.item.item_id))
        selected: list[RetrievalHit] = []
        for layer in selected_layers:
            selected.extend([hit for hit in seed_hits if hit.item.layer is layer][:per_layer_limit])
        selected.sort(key=lambda hit: (-hit.score, hit.item.item_id))
        selected = selected[:total_hit_limit]

        # One-hop expansion across shared participants adds structurally connected
        # context even when the neighboring item does not share query vocabulary.
        connected = self.index.connected_item_ids(hit.item.item_id for hit in selected)
        existing = {hit.item.item_id for hit in selected}
        expansion_hits: list[RetrievalHit] = []
        for item_id, shared_references in connected.items():
            if item_id in existing or item_id not in allowed_item_ids:
                continue
            item = self.store.items[item_id]
            graph_bonus = min(1.0, 0.20 * shared_references)
            lexical = self.index.item_index.score(query, item_id)
            item_feedback = feedback.item_scores.get(item_id, 0.0) * 0.75
            provisional = RetrievalHit(
                item=item,
                score=lexical + graph_bonus + item_feedback,
                lexical_score=lexical,
                feedback_score=item_feedback,
                graph_bonus=graph_bonus,
                reasons=(f"connected through {shared_references} shared graph reference(s)",),
            )
            bonus, bonus_reasons = role_bonus(provisional, role)
            expansion_hits.append(
                replace(
                    provisional,
                    score=provisional.score + bonus,
                    role_bonus=bonus,
                    reasons=tuple(dict.fromkeys((*provisional.reasons, *bonus_reasons))),
                )
            )

        expansion_hits.sort(key=lambda hit: (-hit.score, hit.item.item_id))
        # Preserve requested-layer coverage when a layer is connected to the
        # lexical seeds but its expansion score is lower than other layers.
        represented_layers = {hit.item.layer for hit in selected}
        for layer in selected_layers:
            if layer in represented_layers or len(selected) >= total_hit_limit:
                continue
            candidate = next((hit for hit in expansion_hits if hit.item.layer is layer), None)
            if candidate is not None:
                selected.append(candidate)
                existing.add(candidate.item.item_id)
                represented_layers.add(layer)
        for hit in expansion_hits:
            if len(selected) >= total_hit_limit:
                break
            if hit.item.item_id in existing:
                continue
            selected.append(hit)
            existing.add(hit.item.item_id)
        selected.sort(key=lambda hit: (-hit.score, hit.item.item_id))

        findings = self.verifier.verify(query, selected)
        projection = self.projector.project(
            query,
            role,
            selected,
            findings,
            token_budget=token_budget,
        )
        diagnostics = {
            "candidate_limit": candidate_limit,
            "per_layer_limit": per_layer_limit,
            "total_hit_limit": total_hit_limit,
            "candidate_count": len(candidates),
            "allowed_item_count": len(allowed_item_ids),
            "lexical_seed_count": len(seed_hits),
            "graph_expansion_count": len(expansion_hits),
            "projected_hit_count": len(projection.hits),
            "feedback_context_count": feedback.completed_contexts,
            "out_of_vocabulary_query": not any(score > 0 for _, score in raw_candidates),
        }
        return RetrievalResult(
            query=query,
            role=projection.role,
            layers=selected_layers,
            candidate_chips=tuple(candidates),
            projection=projection,
            diagnostics=diagnostics,
        )
