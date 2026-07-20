from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from retrieval.citation_usefulness import HeuristicCitationUsefulnessFilter
from retrieval.contracts import EvidenceChain, RetrievalResult, RetrievalTrace
from retrieval.graph_retriever import LegalGraphRetriever
from retrieval.indexing import HybridIndexer
from retrieval.reranker import LexicalOverlapReranker
from retrieval.router import DocumentRouter


@dataclass(frozen=True)
class RetrievalServiceConfig:
    seed_top_k: int = 80
    final_top_k: int = 8
    hybrid_alpha: float = 0.50
    require_law_evidence: bool = True


class RetrievalService:
    """Public retrieval facade for the Researcher/Auditor stages.

    The service hides whether candidates came from flat hybrid retrieval or
    Neo4j traversal and always returns normalized evidence chains with source
    provenance, legal references, graph paths, and an execution trace.
    """

    def __init__(
        self,
        indexer: HybridIndexer,
        *,
        graph_retriever: LegalGraphRetriever | None = None,
        reranker: Any | None = None,
        citation_filter: Any | None = None,
        router: DocumentRouter | None = None,
        config: RetrievalServiceConfig | None = None,
    ) -> None:
        self.indexer = indexer
        self.graph_retriever = graph_retriever
        self.reranker = reranker or LexicalOverlapReranker()
        self.citation_filter = citation_filter or HeuristicCitationUsefulnessFilter()
        self.router = router or DocumentRouter()
        self.config = config or RetrievalServiceConfig()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        use_graph: bool | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        requested_top_k = max(0, int(top_k if top_k is not None else self.config.final_top_k))
        graph_enabled = self.graph_retriever is not None if use_graph is None else bool(use_graph)
        if graph_enabled and self.graph_retriever is not None:
            candidates = self.graph_retriever.retrieve(query)
            backend = "neo4j_graph"
            stats = dict(getattr(self.graph_retriever, "last_retrieval_stats", {}) or {})
            reranker = getattr(getattr(self.graph_retriever, "reranker", None), "__class__", type(None)).__name__
        else:
            candidates = self._retrieve_flat(query, requested_top_k)
            backend = "flat_hybrid"
            stats = {}
            reranker = self.reranker.__class__.__name__

        chains: list[EvidenceChain] = []
        for rank, candidate in enumerate(candidates[:requested_top_k], start=1):
            chain = EvidenceChain.from_candidate(candidate, rank)
            if self.config.require_law_evidence and not chain.law_evidence:
                continue
            chains.append(chain)

        latency_ms = (time.perf_counter() - started) * 1000.0
        trace = RetrievalTrace(
            backend=backend,
            reranker=reranker,
            query=query,
            seed_top_k=self.config.seed_top_k,
            requested_top_k=requested_top_k,
            returned_count=len(chains),
            latency_ms=round(latency_ms, 3),
            stats={
                **stats,
                "retrieval_backend": backend,
                "reranker_class": reranker,
                "law_backed_count": len(chains),
                "dropped_unmapped_candidates": max(0, min(requested_top_k, len(candidates)) - len(chains)),
            },
        )
        return RetrievalResult(query=query, chains=chains, trace=trace)

    def _retrieve_flat(self, query: str, top_k: int) -> list[dict[str, Any]]:
        candidates = self.indexer.search(
            query,
            top_k=max(0, self.config.seed_top_k),
            alpha=self.config.hybrid_alpha,
        )
        routed = self.router.apply(query, candidates)
        reranked = self.reranker.rerank(query, routed, top_k=max(0, self.config.seed_top_k))
        filtered = self.citation_filter.filter(query, reranked)
        return filtered[: max(0, min(top_k, self.config.final_top_k))]

    def retrieve_for_reasoning(
        self,
        case_id: str,
        query: str,
        *,
        case_evidence: list[Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        result = self.retrieve(query, top_k=top_k)
        return result.to_reasoning_payload(case_id=case_id, case_evidence=case_evidence)
