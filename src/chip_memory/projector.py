"""Deterministic role-conditioned projection under a context budget."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .text import estimate_tokens
from .types import Layer, Projection, RetrievalHit, VerificationFinding


ROLE_PIN_WEIGHTS: dict[str, dict[str, float]] = {
    "planner": {
        "problem_gap": 0.35,
        "method_mechanism": 0.45,
        "cross_pin": 0.25,
        "reuse_transfer": 0.30,
    },
    "critic": {
        "problem_gap": 0.55,
        "evaluation_validation": 0.30,
        "result_outcome": 0.30,
        "reuse_transfer": 0.45,
    },
    "executor": {
        "method_mechanism": 0.45,
        "experimental_setting": 0.45,
        "implementation": 0.60,
        "cross_pin": 0.20,
    },
    "verifier": {
        "problem_gap": 0.25,
        "evaluation_validation": 0.60,
        "result_outcome": 0.55,
        "implementation": 0.20,
        "reuse_transfer": 0.40,
    },
    "researcher": {
        "problem_gap": 0.25,
        "method_mechanism": 0.30,
        "evaluation_validation": 0.25,
        "result_outcome": 0.25,
        "implementation": 0.15,
        "reuse_transfer": 0.25,
    },
}

ROLE_LAYER_WEIGHTS: dict[str, dict[Layer, float]] = {
    "planner": {Layer.L1: 0.25, Layer.L2: 0.35, Layer.L3: 0.20},
    "critic": {Layer.L1: 0.10, Layer.L2: 0.30, Layer.L3: 0.45},
    "executor": {Layer.L1: 0.20, Layer.L2: 0.45, Layer.L3: 0.35},
    "verifier": {Layer.L1: 0.10, Layer.L2: 0.20, Layer.L3: 0.55},
    "researcher": {Layer.L1: 0.20, Layer.L2: 0.25, Layer.L3: 0.30},
}

ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "planner": ("gap", "structure", "depends", "requires", "transfer", "component"),
    "critic": ("failure", "limitation", "risk", "negative", "conflict", "worse", "cost"),
    "executor": ("implementation", "code", "forward", "training", "step", "pipeline", "command"),
    "verifier": ("evaluation", "result", "metric", "dataset", "evidence", "baseline", "ablation"),
    "researcher": ("method", "gap", "result", "evaluation", "transfer"),
}


def normalize_role(role: str) -> str:
    lowered = (role or "researcher").strip().lower()
    aliases = {
        "judge": "verifier",
        "reviewer": "critic",
        "review": "critic",
        "coder": "executor",
        "engineer": "executor",
        "method_agent": "researcher",
        "scientist": "researcher",
    }
    return aliases.get(lowered, lowered if lowered in ROLE_PIN_WEIGHTS else "researcher")


def role_bonus(hit: RetrievalHit, role: str) -> tuple[float, tuple[str, ...]]:
    normalized = normalize_role(role)
    pin_bonus = ROLE_PIN_WEIGHTS[normalized].get(hit.item.pin, 0.0)
    layer_bonus = ROLE_LAYER_WEIGHTS[normalized].get(hit.item.layer, 0.0)
    text = hit.item.text.lower()
    keyword_count = sum(keyword in text for keyword in ROLE_KEYWORDS[normalized])
    keyword_bonus = min(0.30, keyword_count * 0.06)
    reasons: list[str] = []
    if pin_bonus:
        reasons.append(f"{normalized} pin preference")
    if layer_bonus:
        reasons.append(f"{normalized} layer preference")
    if keyword_bonus:
        reasons.append(f"{normalized} keyword match")
    return pin_bonus + layer_bonus + keyword_bonus, tuple(reasons)


class RoleProjector:
    def project(
        self,
        task: str,
        role: str,
        hits: Iterable[RetrievalHit],
        findings: Iterable[VerificationFinding],
        *,
        token_budget: int = 4000,
    ) -> Projection:
        normalized_role = normalize_role(role)
        rescored: list[RetrievalHit] = []
        for hit in hits:
            bonus, reasons = role_bonus(hit, normalized_role)
            incremental_bonus = bonus - hit.role_bonus
            rescored.append(
                replace(
                    hit,
                    score=hit.score + incremental_bonus,
                    role_bonus=bonus,
                    reasons=tuple(dict.fromkeys((*hit.reasons, *reasons))),
                )
            )
        rescored.sort(key=lambda hit: (-hit.score, hit.item.item_id))

        selected: list[RetrievalHit] = []
        selected_ids: set[str] = set()
        # token_budget applies to injected memory context, not to the task text
        # that the caller already owns.
        used_tokens = 40

        # Preserve layer coverage before filling by score.
        for layer in (Layer.L1, Layer.L2, Layer.L3):
            candidate = next((hit for hit in rescored if hit.item.layer is layer), None)
            if candidate is None:
                continue
            cost = estimate_tokens(candidate.item.text) + 24
            if used_tokens + cost <= token_budget:
                selected.append(candidate)
                selected_ids.add(candidate.item.item_id)
                used_tokens += cost

        for hit in rescored:
            if hit.item.item_id in selected_ids:
                continue
            cost = estimate_tokens(hit.item.text) + 24
            if used_tokens + cost > token_budget:
                continue
            selected.append(hit)
            selected_ids.add(hit.item.item_id)
            used_tokens += cost

        selected.sort(key=lambda hit: (-hit.score, hit.item.item_id))
        selected_findings = tuple(
            finding
            for finding in findings
            if finding.item_id is None or finding.item_id in selected_ids
        )
        return Projection(
            role=normalized_role,
            task=task,
            hits=tuple(selected),
            findings=selected_findings,
            estimated_tokens=used_tokens,
            omitted_hits=max(0, len(rescored) - len(selected)),
        )


def projection_to_markdown(projection: Projection) -> str:
    lines = [
        f"# Chip Memory projection for `{projection.role}`",
        "",
        f"**Task:** {projection.task}",
        "",
        f"Estimated context tokens: {projection.estimated_tokens}",
        "",
    ]
    for layer in (Layer.L1, Layer.L2, Layer.L3):
        layer_hits = [hit for hit in projection.hits if hit.item.layer is layer]
        if not layer_hits:
            continue
        label = {Layer.L1: "Structure", Layer.L2: "Mechanisms", Layer.L3: "Events and evidence"}[layer]
        lines.extend([f"## {layer.value}: {label}", ""])
        for hit in layer_hits:
            lines.append(
                f"- **{hit.item.kind}** ({hit.item.chip_id}, score={hit.score:.3f}): "
                f"{hit.item.text}  \n  Source: `{hit.item.citation()}`"
            )
        lines.append("")
    if projection.findings:
        lines.extend(["## Compatibility and evidence checks", ""])
        for finding in projection.findings:
            lines.append(f"- **{finding.severity.upper()} — {finding.code}:** {finding.message}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
