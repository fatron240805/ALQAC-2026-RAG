from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


_TOKEN_RE = re.compile(r"\b[\wÀ-ỹ]+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def lexical_overlap_score(query: str, text: str) -> float:
    query_counts = Counter(tokenize(query))
    text_counts = Counter(tokenize(text))
    if not query_counts or not text_counts:
        return 0.0
    overlap = sum(min(count, text_counts[token]) for token, count in query_counts.items())
    precision = overlap / max(1, sum(text_counts.values()))
    recall = overlap / max(1, sum(query_counts.values()))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


class LexicalOverlapReranker:
    """Deterministic reranker used before adding a cross-encoder model."""

    def __init__(self, base_score_weight: float = 0.75, lexical_weight: float = 0.25) -> None:
        self.base_score_weight = base_score_weight
        self.lexical_weight = lexical_weight

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            item = dict(candidate)
            base = float(
                item.get("routed_score", item.get("fused_score", item.get("score", 0.0)))
            )
            lexical = lexical_overlap_score(query, str(item.get("content", "")))
            rerank_score = self.base_score_weight * base + self.lexical_weight * lexical
            if not math.isfinite(rerank_score):
                rerank_score = 0.0
            item["lexical_overlap_score"] = lexical
            item["rerank_score"] = rerank_score
            scored.append(item)
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored[:top_k]
