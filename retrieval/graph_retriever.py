from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from graph_construct.neo4j_store import Neo4jGraphStore
from retrieval.citation_usefulness import HeuristicCitationUsefulnessFilter
from retrieval.indexing import HybridIndexer
from retrieval.reranker import LexicalOverlapReranker, lexical_overlap_score
from retrieval.router import DocumentRouter, QueryAnalyzer, QueryAnalysis


@dataclass(frozen=True)
class GraphRetrieverConfig:
    seed_top_k: int = 80
    chain_top_k: int = 8
    expansion_depth_fast: int = 1
    expansion_depth_deep: int = 2
    bm25_weight: float = 0.25
    dense_weight: float = 0.25
    graph_weight: float = 0.20
    exact_citation_weight: float = 0.15
    legal_issue_weight: float = 0.10
    freshness_weight: float = 0.05
    hybrid_alpha: float = 0.50
    community_top_k: int = 1
    community_member_top_k: int = 20


class LegalGraphRetriever:
    """Graph-aware legal retriever backed by HybridIndexer + optional Neo4j."""

    def __init__(
        self,
        indexer: HybridIndexer,
        *,
        graph_store: Neo4jGraphStore | None = None,
        config: GraphRetrieverConfig | None = None,
        analyzer: QueryAnalyzer | None = None,
        reranker: LexicalOverlapReranker | None = None,
        citation_filter: HeuristicCitationUsefulnessFilter | None = None,
    ) -> None:
        self.indexer = indexer
        self.graph_store = graph_store
        self.config = config or GraphRetrieverConfig()
        self.analyzer = analyzer or QueryAnalyzer()
        self.router = DocumentRouter(self.analyzer)
        self.reranker = reranker or LexicalOverlapReranker()
        self.citation_filter = citation_filter or HeuristicCitationUsefulnessFilter(max_results=self.config.chain_top_k)

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        analysis = self.analyzer.analyze(query)
        seed_candidates = self.indexer.search(
            query,
            top_k=self.config.seed_top_k,
            alpha=self.config.hybrid_alpha,
        )
        routed = self.router.apply(query, seed_candidates)
        seeded = self.reranker.rerank(query, routed, top_k=self.config.seed_top_k)

        if self.graph_store is None:
            return self.citation_filter.filter(query, seeded[: self.config.chain_top_k])

        chain_candidates = self._expand_with_graph(query, analysis, seeded)
        if not chain_candidates:
            return self.citation_filter.filter(query, seeded[: self.config.chain_top_k])

        # Expansion improves recall for related provisions, but it must not turn
        # into a hard gate.  A useful seed can be outside a sparse/incomplete
        # graph neighbourhood, so let the final reranker compare both sources.
        candidate_pool = self._merge_seed_and_graph_candidates(seeded, chain_candidates)
        rerank_limit = min(len(candidate_pool), max(self.config.seed_top_k, self.config.chain_top_k * 10))
        reranked = self.reranker.rerank(query, candidate_pool, top_k=rerank_limit)
        return self.citation_filter.filter(query, reranked[: self.config.chain_top_k])

    @staticmethod
    def _candidate_identity(candidate: dict[str, Any]) -> str:
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        law_id = str(metadata.get("law_id") or "").strip()
        article_number = str(
            metadata.get("article_number")
            or metadata.get("article_index")
            or metadata.get("aid")
            or ""
        ).strip()
        if law_id and article_number:
            return f"{law_id}:{article_number.lower()}"
        return str(candidate.get("doc_id") or "").strip()

    def _merge_seed_and_graph_candidates(
        self,
        seeded: list[dict[str, Any]],
        chain_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fuse duplicate provisions while preserving the richer seed passage."""
        merged: dict[str, dict[str, Any]] = {}
        for source, candidates in (("seed", seeded), ("graph", chain_candidates)):
            for candidate in candidates:
                item = dict(candidate)
                metadata = dict(item.get("metadata") or {})
                item["metadata"] = metadata
                item["retrieval_origin"] = source
                metadata["retrieval_origin"] = source
                key = self._candidate_identity(item)
                if not key:
                    continue

                existing = merged.get(key)
                if existing is None:
                    merged[key] = item
                    continue

                existing_score = float(existing.get("fused_score", 0.0))
                candidate_score = float(item.get("fused_score", 0.0))
                primary, supporting = (existing, item) if existing_score >= candidate_score else (item, existing)
                primary_metadata = dict(primary.get("metadata") or {})
                supporting_metadata = supporting.get("metadata") if isinstance(supporting.get("metadata"), dict) else {}
                graph_path = primary.get("graph_path") or supporting.get("graph_path") or supporting_metadata.get("graph_path")
                if graph_path:
                    primary["graph_path"] = graph_path
                    primary_metadata["graph_path"] = graph_path
                primary_metadata["retrieval_origin"] = "seed+graph"
                primary_metadata["seed_score"] = max(
                    float(primary_metadata.get("seed_score", 0.0)),
                    float(supporting_metadata.get("seed_score", 0.0)),
                )
                primary["metadata"] = primary_metadata
                primary["retrieval_origin"] = "seed+graph"
                # The second retrieval path is corroboration, not a score that
                # can overwhelm semantic relevance before cross-encoder rerank.
                primary["fused_score"] = max(existing_score, candidate_score) + 0.08 * min(existing_score, candidate_score)
                primary["routed_score"] = primary["fused_score"]
                merged[key] = primary

        return sorted(merged.values(), key=lambda item: float(item.get("fused_score", 0.0)), reverse=True)

    def _expand_with_graph(
        self,
        query: str,
        analysis: QueryAnalysis,
        seeded: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not seeded:
            return []

        depth = (
            self.config.expansion_depth_deep
            if analysis.retrieval_depth == "deep"
            else self.config.expansion_depth_fast
        )
        target_layers = ["rule", "ontology", "fact"]
        seed_chunk_ids = [str(item.get("doc_id") or "").strip() for item in seeded[: min(12, len(seeded))]]
        seed_score_map = {
            str(item.get("doc_id") or ""): float(item.get("routed_score", item.get("fused_score", 0.0)))
            for item in seeded
        }

        expanded: list[dict[str, Any]] = []
        seen: set[str] = set()
        for seed_chunk_id in seed_chunk_ids:
            if not seed_chunk_id:
                continue
            seed_node_ids = self.graph_store.seed_nodes_for_chunk(seed_chunk_id)
            if not seed_node_ids:
                continue
            seed_score = float(seed_score_map.get(seed_chunk_id, 0.0))
            graph_rows = self.graph_store.expand_from_seeds(
                seed_node_ids,
                depth=depth,
                target_layers=target_layers,
                limit=max(self.config.chain_top_k * 5, 40),
            )
            for row in graph_rows:
                candidate = self._row_to_candidate(query, analysis, seed_chunk_id, seed_score, row)
                if candidate is None:
                    continue
                key = str(candidate["doc_id"])
                if key in seen:
                    existing = next((item for item in expanded if item["doc_id"] == key), None)
                    if existing is not None and float(candidate.get("fused_score", 0.0)) > float(existing.get("fused_score", 0.0)):
                        existing.update(candidate)
                    continue
                seen.add(key)
                expanded.append(candidate)

        expanded.sort(key=lambda item: float(item.get("fused_score", 0.0)), reverse=True)
        return self._community_guided_candidates(expanded)

    def _community_guided_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply LegalGraphRAG-style community expansion to ontology paths.

        LegalGraphRAG selects the query-aligned ontology community first, then
        retrieves the strongest members inside it.  This is deliberately done
        before the cross-encoder, so the cross-encoder still compares direct
        semantic/citation evidence against community-expanded evidence.
        """
        if not candidates:
            return []

        direct_candidates: list[dict[str, Any]] = []
        by_community: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
            graph_path = candidate.get("graph_path") or metadata.get("graph_path") or []
            communities = [
                str(node_id)
                for node_id in graph_path
                if str(node_id).startswith("ontology:")
            ]
            if not communities:
                direct_candidates.append(candidate)
                continue
            metadata = dict(metadata)
            metadata["ontology_communities"] = communities
            candidate["metadata"] = metadata
            for community_id in communities:
                by_community.setdefault(community_id, []).append(candidate)

        # A community is represented by its strongest query-aligned member.
        ranked_communities = sorted(
            by_community.items(),
            key=lambda item: max(float(candidate.get("fused_score", 0.0)) for candidate in item[1]),
            reverse=True,
        )[: max(0, self.config.community_top_k)]

        selected: list[dict[str, Any]] = list(direct_candidates)
        seen = {str(candidate.get("doc_id") or "") for candidate in selected}
        for community_id, members in ranked_communities:
            members.sort(key=lambda candidate: float(candidate.get("fused_score", 0.0)), reverse=True)
            for candidate in members[: max(0, self.config.community_member_top_k)]:
                doc_id = str(candidate.get("doc_id") or "")
                if not doc_id or doc_id in seen:
                    continue
                item = dict(candidate)
                metadata = dict(item.get("metadata") or {})
                metadata["selected_ontology_community"] = community_id
                item["metadata"] = metadata
                selected.append(item)
                seen.add(doc_id)

        selected.sort(key=lambda item: float(item.get("fused_score", 0.0)), reverse=True)
        return selected

    def _row_to_candidate(
        self,
        query: str,
        analysis: QueryAnalysis,
        seed_chunk_id: str,
        seed_score: float,
        row: dict[str, Any],
    ) -> dict[str, Any] | None:
        node_id = str(row.get("node_id") or "").strip()
        if not node_id:
            return None

        layer = str(row.get("layer") or "")
        node_type = str(row.get("node_type") or "")
        law_id = str(row.get("law_id") or "").strip()
        aid = row.get("aid")
        if not law_id or aid in (None, ""):
            return None
        graph_path = [str(node) for node in row.get("graph_path") or [] if str(node).strip()]
        distance = int(row.get("distance") or 0)
        text = str(row.get("text") or row.get("unit_path") or node_id)
        lexical = lexical_overlap_score(query, text)
        graph_score = 1.0 / (1.0 + max(0, distance))
        citation_aid = row.get("article_number") or row.get("article_index") or aid
        exact_citation = self._exact_citation_score(analysis, law_id, citation_aid, text)
        issue_match = lexical
        freshness = 0.0 if str(row.get("deprecated") or "").lower() in {"true", "1"} else 1.0
        fused_score = (
            self.config.bm25_weight * max(0.0, min(1.0, seed_score))
            + self.config.dense_weight * max(0.0, min(1.0, seed_score))
            + self.config.graph_weight * graph_score
            + self.config.exact_citation_weight * exact_citation
            + self.config.legal_issue_weight * issue_match
            + self.config.freshness_weight * freshness
        )
        if not math.isfinite(fused_score):
            fused_score = 0.0

        source_chunk_id = str(row.get("source_chunk_id") or row.get("chunk_id") or seed_chunk_id)
        candidate_doc_id = node_id if law_id else f"{seed_chunk_id}:{node_id}"
        metadata = {
            "node_id": node_id,
            "layer": layer,
            "node_type": node_type,
            "law_id": law_id,
            "aid": aid,
            "source_aid": row.get("source_aid"),
            "article_number": row.get("article_number"),
            "article_index": row.get("article_index"),
            "source_chunk_id": source_chunk_id,
            "chunk_id": row.get("chunk_id") or source_chunk_id,
            "graph_path": graph_path,
            "graph_distance": distance,
            "seed_chunk_id": seed_chunk_id,
            "seed_score": seed_score,
            "normalized_alias": row.get("normalized_alias"),
            "aliases": row.get("aliases"),
            "unit_path": row.get("unit_path"),
        }

        return {
            "rank": 0,
            "doc_id": candidate_doc_id,
            "content": text,
            "metadata": metadata,
            "bm25_score": float(seed_score),
            "dense_score": float(seed_score),
            "fused_score": float(fused_score),
            "routing_boost": 0.0,
            "routed_score": float(fused_score),
            "graph_path": graph_path,
            "graph_distance": distance,
        }

    @staticmethod
    def _exact_citation_score(analysis: QueryAnalysis, law_id: str, aid: Any, text: str) -> float:
        if not law_id:
            return 0.0
        lowered_text = text.lower()
        for ref in analysis.statute_references:
            if "/" in ref and ref == law_id:
                return 1.0
            if ref.isdigit() and aid is not None and str(ref) == str(aid):
                return 1.0
            if ref.lower() in lowered_text:
                return 0.8
        return 0.0
