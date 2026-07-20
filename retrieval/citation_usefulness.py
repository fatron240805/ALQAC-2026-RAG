from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from retrieval.reranker import lexical_overlap_score


_LEGAL_SIGNAL_RE = re.compile(
    r"\b(điều|khoản|điểm|nghĩa vụ|trách nhiệm|bồi thường|hợp đồng|"
    r"thiệt hại|tài sản|quyền|phải|không được|được)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationJudgment:
    evidence_id: str
    judgment: str
    score: float
    reason: str


class HeuristicCitationUsefulnessFilter:
    """Phase-1 citation usefulness filter.

    It keeps passages that have both query overlap and legal-rule signals. This
    is intentionally transparent and deterministic so failures are inspectable
    before replacing it with a learned classifier or LLM judge.
    """

    def __init__(self, min_score: float = 0.03, max_results: int = 5) -> None:
        self.min_score = min_score
        self.max_results = max_results

    def judge(self, query: str, candidate: dict[str, Any]) -> CitationJudgment:
        text = str(candidate.get("content", ""))
        evidence_id = str(candidate.get("doc_id") or candidate.get("chunk_id") or "")
        overlap = lexical_overlap_score(query, text)
        legal_signal = 1.0 if _LEGAL_SIGNAL_RE.search(text) else 0.0
        retrieval_score = float(
            candidate.get("rerank_score", candidate.get("routed_score", candidate.get("fused_score", 0.0)))
        )
        score = (0.55 * overlap) + (0.25 * legal_signal) + (0.20 * max(0.0, retrieval_score))

        if score >= self.min_score and legal_signal:
            judgment = "useful"
            reason = "query overlap with legal-rule signal"
        elif score >= self.min_score:
            judgment = "uncertain"
            reason = "query overlap but weak legal-rule signal"
        else:
            judgment = "not_useful"
            reason = "low lexical/legal usefulness signal"

        return CitationJudgment(
            evidence_id=evidence_id,
            judgment=judgment,
            score=score,
            reason=reason,
        )

    def filter(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        preserve_graph_paths: bool = False,
    ) -> list[dict[str, Any]]:
        judged: list[dict[str, Any]] = []
        for candidate in candidates:
            item = dict(candidate)
            judgment = self.judge(query, item)
            item["citation_judgment"] = {
                "evidence_id": judgment.evidence_id,
                "judgment": judgment.judgment,
                "score": round(judgment.score, 6),
                "reason": judgment.reason,
            }
            graph_path = item.get("graph_path")
            is_graph_backed = isinstance(graph_path, list) and len(graph_path) >= 2
            if judgment.judgment in {"useful", "uncertain"} or (preserve_graph_paths and is_graph_backed):
                if preserve_graph_paths and is_graph_backed and judgment.judgment == "not_useful":
                    item["citation_judgment"]["judgment"] = "uncertain"
                    item["citation_judgment"]["reason"] = "graph traversal evidence retained for diversity"
                judged.append(item)

        # This component is a usefulness gate, not a second reranker. Keep
        # the caller's order so graph quota/interleaving and cluster ordering
        # survive the citation check.
        return judged[: self.max_results]
