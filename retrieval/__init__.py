"""Public retrieval package for hybrid, graph, and evidence-chain RAG."""

from retrieval.contracts import EvidenceChain, RetrievalResult, RetrievalTrace
from retrieval.service import RetrievalService, RetrievalServiceConfig

__all__ = [
    "EvidenceChain",
    "RetrievalResult",
    "RetrievalTrace",
    "RetrievalService",
    "RetrievalServiceConfig",
]
