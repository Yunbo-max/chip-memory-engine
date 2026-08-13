"""Evidence, compatibility, and negative-result checks for retrieved items."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .text import unique_terms
from .types import RetrievalHit, VerificationFinding


STRONG_CONFLICT_PATTERNS = (
    r"\bincompatible\b",
    r"\bconflicts? with\b",
    r"\bdoes not work\b",
    r"\bfails? under\b",
    r"\bcannot be (?:used|applied|transferred)\b",
)
NEGATIVE_PATTERNS = (
    r"\bfailure\b",
    r"\blimitation\b",
    r"\bnegative result\b",
    r"\bdegrad(?:e|es|ed|ation)\b",
    r"\bworse\b",
    r"\brisk\b",
    r"\bunsafe\b",
    r"\badds? cost\b",
    r"\bnot evaluated\b",
)
CONDITION_PATTERNS = (
    r"\brequires?\b",
    r"\bdepends? on\b",
    r"\bonly when\b",
    r"\bcondition(?:al|ed)?\b",
    r"\bassum(?:e|es|ed|ption)\b",
    r"\bbudget\b",
)


def _matches(patterns: Iterable[str], text: str) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


class CompatibilityVerifier:
    """Conservative verifier: flags evidence; it does not invent compatibility."""

    def verify(self, query: str, hits: Iterable[RetrievalHit]) -> tuple[VerificationFinding, ...]:
        findings: list[VerificationFinding] = []
        by_chip: dict[str, list[RetrievalHit]] = defaultdict(list)
        query_terms = unique_terms(query)

        for hit in hits:
            by_chip[hit.item.chip_id].append(hit)
            item = hit.item
            text = item.text.lower()
            strong_conflicts = _matches(STRONG_CONFLICT_PATTERNS, text)
            negatives = _matches(NEGATIVE_PATTERNS, text)
            conditions = _matches(CONDITION_PATTERNS, text)
            overlap = sorted(query_terms & unique_terms(item.text))

            if strong_conflicts:
                findings.append(
                    VerificationFinding(
                        severity="block",
                        code="explicit_incompatibility",
                        message=f"Retrieved evidence explicitly describes incompatibility or failure; matched query terms: {', '.join(overlap[:8]) or 'none'}.",
                        chip_id=item.chip_id,
                        item_id=item.item_id,
                        evidence=item.evidence or (item.text[:320],),
                    )
                )
            elif negatives:
                findings.append(
                    VerificationFinding(
                        severity="warning",
                        code="negative_or_limiting_evidence",
                        message="The retrieved item contains a limitation, risk, cost, or negative result that should be considered before transfer.",
                        chip_id=item.chip_id,
                        item_id=item.item_id,
                        evidence=item.evidence or (item.text[:320],),
                    )
                )
            if conditions:
                findings.append(
                    VerificationFinding(
                        severity="info",
                        code="conditional_transfer",
                        message="The item contains an explicit requirement, assumption, condition, or budget constraint.",
                        chip_id=item.chip_id,
                        item_id=item.item_id,
                        evidence=item.evidence or (item.text[:320],),
                    )
                )

        for chip_id, chip_hits in by_chip.items():
            l3_hits = [hit for hit in chip_hits if hit.item.layer.value == "L3"]
            if l3_hits and not any(hit.item.evidence for hit in l3_hits):
                findings.append(
                    VerificationFinding(
                        severity="info",
                        code="missing_item_evidence_anchor",
                        message="Retrieved L3 events have source-file provenance but no event-level evidence anchor.",
                        chip_id=chip_id,
                    )
                )

        severity_order = {"block": 0, "warning": 1, "info": 2}
        deduped: dict[tuple[str, str, str | None], VerificationFinding] = {}
        for finding in findings:
            key = (finding.code, finding.chip_id, finding.item_id)
            deduped.setdefault(key, finding)
        return tuple(
            sorted(
                deduped.values(),
                key=lambda finding: (
                    severity_order.get(finding.severity, 9),
                    finding.chip_id,
                    finding.item_id or "",
                ),
            )
        )

