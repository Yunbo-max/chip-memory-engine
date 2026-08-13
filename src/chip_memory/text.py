"""Dependency-free text normalization and BM25 scoring."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Mapping


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+.-]*|\d+(?:\.\d+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "has", "have", "how", "in", "into", "is", "it", "its", "of", "on", "or",
    "our", "that", "the", "their", "this", "to", "use", "using", "via", "was",
    "we", "were", "what", "when", "where", "which", "with",
}


def tokenize(text: str) -> list[str]:
    tokens = [token.lower().strip("._-") for token in TOKEN_RE.findall(text or "")]
    return [token for token in tokens if token and token not in STOPWORDS and len(token) > 1]


def unique_terms(text: str) -> set[str]:
    return set(tokenize(text))


def estimate_tokens(text: str) -> int:
    """Conservative tokenizer-independent estimate for context budgeting."""

    if not text:
        return 0
    return max(1, math.ceil(len(text) / 3.6))


class BM25Index:
    """Small in-memory BM25 index with deterministic behavior."""

    def __init__(self, documents: Mapping[str, str], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_frequencies: dict[str, Counter[str]] = {}
        self.document_lengths: dict[str, int] = {}
        self.document_frequency: Counter[str] = Counter()
        self.inverted: dict[str, set[str]] = defaultdict(set)

        for document_id, text in documents.items():
            frequencies = Counter(tokenize(text))
            self.term_frequencies[document_id] = frequencies
            self.document_lengths[document_id] = sum(frequencies.values())
            for term in frequencies:
                self.document_frequency[term] += 1
                self.inverted[term].add(document_id)

        self.document_count = len(self.term_frequencies)
        total_length = sum(self.document_lengths.values())
        self.average_length = total_length / self.document_count if self.document_count else 1.0

    def score(self, query: str, document_id: str) -> float:
        frequencies = self.term_frequencies.get(document_id)
        if not frequencies or self.document_count == 0:
            return 0.0
        document_length = self.document_lengths[document_id]
        score = 0.0
        for term, query_frequency in Counter(tokenize(query)).items():
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            df = self.document_frequency[term]
            inverse_document_frequency = math.log(1.0 + (self.document_count - df + 0.5) / (df + 0.5))
            denominator = term_frequency + self.k1 * (
                1.0 - self.b + self.b * document_length / max(self.average_length, 1.0)
            )
            score += query_frequency * inverse_document_frequency * (
                term_frequency * (self.k1 + 1.0) / denominator
            )
        return score

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        allowed_ids: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        if allowed_ids is None:
            candidates: set[str] = set()
            for term in query_terms:
                candidates.update(self.inverted.get(term, ()))
        else:
            candidates = set(allowed_ids)
        scored = [(document_id, self.score(query, document_id)) for document_id in candidates]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: max(0, limit)]

