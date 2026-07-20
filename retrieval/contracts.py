from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as_float(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if score == score else 0.0


@dataclass(frozen=True)
class EvidenceChain:
    """Stable evidence contract consumed by the next RAG stage."""

    chain_id: str
    evidence_id: str
    doc_id: str
    content: str
    score: float
    rerank_score: float
    law_evidence: list[dict[str, Any]] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)
    graph_path: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    citation_judgment: dict[str, Any] | None = None
    rank: int = 0

    @classmethod
    def from_candidate(cls, candidate: dict[str, Any], rank: int) -> "EvidenceChain":
        metadata = dict(candidate.get("metadata") or {})
        doc_id = str(candidate.get("doc_id") or candidate.get("chunk_id") or "")
        evidence_id = str(candidate.get("evidence_id") or doc_id)
        graph_path_value = candidate.get("graph_path") or metadata.get("graph_path") or []
        graph_path = [str(node) for node in graph_path_value if str(node).strip()] if isinstance(graph_path_value, list) else []

        law_id = metadata.get("law_id")
        aid = metadata.get("aid")
        if aid in (None, ""):
            aid = metadata.get("article_number") or metadata.get("article_index")
        law_evidence = []
        if law_id not in (None, "") and aid not in (None, ""):
            law_evidence.append({"law_id": law_id, "aid": aid})

        source_chunks: list[str] = []
        for value in (
            metadata.get("source_chunk_id"),
            metadata.get("chunk_id"),
            candidate.get("source_chunk_id"),
            candidate.get("chunk_id"),
        ):
            chunk_id = str(value or "").strip()
            if chunk_id and chunk_id not in source_chunks:
                source_chunks.append(chunk_id)
        if not source_chunks and not graph_path and doc_id:
            source_chunks.append(doc_id)

        provenance_keys = (
            "retrieval_origin",
            "seed_chunk_id",
            "graph_distance",
            "cluster_id",
            "cluster_rank",
            "cluster_member_rank",
            "selected_ontology_community",
        )
        provenance = {
            key: metadata.get(key, candidate.get(key))
            for key in provenance_keys
            if metadata.get(key, candidate.get(key)) not in (None, "", [])
        }
        if graph_path:
            provenance["graph_path"] = graph_path

        return cls(
            chain_id=str(candidate.get("chain_id") or f"chain:{evidence_id}"),
            evidence_id=evidence_id,
            doc_id=doc_id,
            content=str(candidate.get("content") or ""),
            score=_as_float(candidate.get("score", candidate.get("rerank_score", candidate.get("fused_score", 0.0)))),
            rerank_score=_as_float(candidate.get("rerank_score", candidate.get("fused_score", 0.0))),
            law_evidence=law_evidence,
            source_chunks=source_chunks,
            graph_path=graph_path,
            provenance=provenance,
            metadata=metadata,
            citation_judgment=(
                dict(candidate["citation_judgment"])
                if isinstance(candidate.get("citation_judgment"), dict)
                else None
            ),
            rank=rank,
        )

    @property
    def is_graph_backed(self) -> bool:
        return len(self.graph_path) >= 2

    def as_candidate(self) -> dict[str, Any]:
        """Return a dict compatible with the existing reasoning interfaces."""
        candidate = dict(self.metadata)
        candidate.update(
            {
                "chain_id": self.chain_id,
                "evidence_id": self.evidence_id,
                "doc_id": self.doc_id,
                "content": self.content,
                "score": self.score,
                "rerank_score": self.rerank_score,
                "metadata": dict(self.metadata),
                "law_evidence": list(self.law_evidence),
                "source_chunks": list(self.source_chunks),
                "graph_path": list(self.graph_path),
                "provenance": dict(self.provenance),
                "rank": self.rank,
            }
        )
        if self.citation_judgment is not None:
            candidate["citation_judgment"] = dict(self.citation_judgment)
        return candidate

    def as_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "evidence_id": self.evidence_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "law_evidence": list(self.law_evidence),
            "source_chunks": list(self.source_chunks),
            "graph_path": list(self.graph_path),
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
            "citation_judgment": dict(self.citation_judgment) if self.citation_judgment else None,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class RetrievalTrace:
    backend: str
    reranker: str
    query: str
    seed_top_k: int
    requested_top_k: int
    returned_count: int
    latency_ms: float
    stats: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "reranker": self.reranker,
            "query": self.query,
            "seed_top_k": self.seed_top_k,
            "requested_top_k": self.requested_top_k,
            "returned_count": self.returned_count,
            "latency_ms": self.latency_ms,
            **dict(self.stats),
        }


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieval output that can be passed directly to reasoning/auditing."""

    query: str
    chains: list[EvidenceChain]
    trace: RetrievalTrace

    @property
    def evidence(self) -> list[dict[str, Any]]:
        return [chain.as_candidate() for chain in self.chains]

    @property
    def law_evidence(self) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for chain in self.chains:
            for item in chain.law_evidence:
                key = (str(item.get("law_id") or ""), str(item.get("aid") or ""))
                if not key[0] or not key[1] or key in seen:
                    continue
                evidence.append(dict(item))
                seen.add(key)
        return evidence

    def to_reasoning_payload(
        self,
        *,
        case_id: str | None = None,
        case_evidence: list[Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query,
            "law_evidence": self.evidence,
            "evidence_chains": [chain.as_dict() for chain in self.chains],
            "retrieval_trace": self.trace.as_dict(),
        }
        if case_id is not None:
            payload["case_id"] = str(case_id)
        if case_evidence is not None:
            payload["case_evidence"] = case_evidence
        return payload

    def as_dict(self) -> dict[str, Any]:
        return self.to_reasoning_payload()
