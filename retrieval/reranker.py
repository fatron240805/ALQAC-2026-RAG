from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    from FlagEmbedding import FlagReranker  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    FlagReranker = None


_TOKEN_RE = re.compile(r"\b[\wÀ-ỹ]+\b", re.UNICODE)
_STATUTE_RE = re.compile(
    r"\b(?:điều|dieu)\s+([0-9]+[a-zA-Z]?)|"
    r"\b([0-9]{1,3}/[0-9]{4}/[A-ZĐ\-0-9]+)\b",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def _bigrams(tokens: list[str]) -> list[str]:
    return [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]


def _overlap_f1(query_items: list[str], text_items: list[str]) -> float:
    query_counts = Counter(query_items)
    text_counts = Counter(text_items)
    if not query_counts or not text_counts:
        return 0.0
    overlap = sum(min(count, text_counts[item]) for item, count in query_counts.items())
    precision = overlap / max(1, sum(text_counts.values()))
    recall = overlap / max(1, sum(query_counts.values()))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _extract_statute_refs(query: str) -> list[str]:
    refs: list[str] = []
    for match in _STATUTE_RE.finditer(query or ""):
        value = match.group(1) or match.group(2)
        if value:
            refs.append(value)
    return refs


def lexical_overlap_score(query: str, text: str) -> float:
    return _overlap_f1(tokenize(query), tokenize(text))


class LexicalOverlapReranker:
    """Deterministic reranker used before adding a cross-encoder model.

    The score blends base retrieval score, lexical overlap, phrase overlap,
    citation alignment, and legal-structure metadata signals.
    """

    def __init__(
        self,
        base_score_weight: float = 0.72,
        lexical_weight: float = 0.20,
        phrase_weight: float = 0.03,
        citation_weight: float = 0.04,
        structure_weight: float = 0.01,
        *,
        use_gpu_reranker: bool = True,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        self.base_score_weight = base_score_weight
        self.lexical_weight = lexical_weight
        self.phrase_weight = phrase_weight
        self.citation_weight = citation_weight
        self.structure_weight = structure_weight
        self.use_gpu_reranker = use_gpu_reranker
        self.reranker_model_name = reranker_model_name
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.gpu_reranker = self._load_gpu_reranker()

    @staticmethod
    def _resolve_device(device: str) -> str:
        value = (device or "auto").strip().lower()
        if value not in {"auto", "cpu", "cuda", "cuda:0"}:
            return value
        if value == "cpu":
            return "cpu"
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _load_gpu_reranker(self) -> Any:
        if not self.use_gpu_reranker or FlagReranker is None:
            return None
        try:
            kwargs: dict[str, Any] = {"use_fp16": self.device.startswith("cuda")}
            if self.device != "cpu":
                kwargs["devices"] = self.device
            return FlagReranker(self.reranker_model_name, **kwargs)
        except TypeError:
            try:
                return FlagReranker(self.reranker_model_name, use_fp16=self.device.startswith("cuda"))
            except Exception:
                return None
        except Exception:
            return None

    @staticmethod
    def _normalise_base_score(value: Any) -> float:
        score = float(value or 0.0)
        if not math.isfinite(score):
            return 0.0
        return max(0.0, min(1.0, score))

    def _citation_score(self, query: str, candidate: dict[str, Any]) -> float:
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        law_id = str(metadata.get("law_id") or "").strip().lower()
        aid = str(metadata.get("aid") or "").strip().lower()
        unit_path = str(metadata.get("unit_path") or candidate.get("content") or "").lower()
        text = str(candidate.get("content", "")).lower()

        if not query:
            return 0.0

        score = 0.0
        for ref in _extract_statute_refs(query):
            ref_text = str(ref).strip().lower()
            if not ref_text:
                continue
            if "/" in ref_text:
                if law_id and ref_text == law_id:
                    score = max(score, 1.0)
                elif ref_text in unit_path or ref_text in text:
                    score = max(score, 0.8)
            else:
                if aid and ref_text == aid:
                    score = max(score, 0.95)
                elif f"điều {ref_text}" in unit_path or f"dieu {ref_text}" in unit_path:
                    score = max(score, 0.85)
                elif ref_text in unit_path or ref_text in text:
                    score = max(score, 0.6)
        return score

    def _structure_score(self, candidate: dict[str, Any]) -> float:
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        score = 0.0
        if metadata.get("law_id"):
            score += 0.35
        if metadata.get("aid") is not None:
            score += 0.25
        if metadata.get("unit_path"):
            score += 0.15
        if metadata.get("article_number") or metadata.get("article_label"):
            score += 0.15
        if candidate.get("graph_path"):
            score += 0.10
        return min(1.0, score)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        if self.gpu_reranker is not None and candidates:
            try:
                pairs = [[query, str(candidate.get("content", ""))] for candidate in candidates]
                model_scores = self.gpu_reranker.compute_score(pairs, normalize=True)
                if isinstance(model_scores, float):
                    model_scores = [model_scores]
                scored: list[dict[str, Any]] = []
                for candidate, model_score in zip(candidates, model_scores):
                    item = dict(candidate)
                    base = self._normalise_base_score(
                        item.get("routed_score", item.get("fused_score", item.get("score", 0.0)))
                    )
                    lexical = lexical_overlap_score(query, str(item.get("content", "")))
                    phrase = _overlap_f1(_bigrams(tokenize(query)), _bigrams(tokenize(str(item.get("content", "")))))
                    citation = self._citation_score(query, item)
                    structure = self._structure_score(item)
                    rerank_score = (
                        0.82 * float(model_score)
                        + 0.10 * base
                        + 0.04 * citation
                        + 0.03 * structure
                        + 0.01 * lexical
                    )
                    if not math.isfinite(rerank_score):
                        rerank_score = 0.0
                    item["gpu_rerank_score"] = float(model_score)
                    item["lexical_overlap_score"] = lexical
                    item["phrase_overlap_score"] = phrase
                    item["citation_alignment_score"] = citation
                    item["structure_score"] = structure
                    item["rerank_score"] = rerank_score
                    scored.append(item)
                scored.sort(key=lambda item: item["rerank_score"], reverse=True)
                return scored[:top_k]
            except Exception:
                self.gpu_reranker = None

        scored: list[dict[str, Any]] = []
        query_tokens = tokenize(query)
        query_bigrams = _bigrams(query_tokens)
        for candidate in candidates:
            item = dict(candidate)
            base = self._normalise_base_score(
                item.get("routed_score", item.get("fused_score", item.get("score", 0.0)))
            )
            content = str(item.get("content", ""))
            lexical = lexical_overlap_score(query, content)
            phrase = _overlap_f1(query_bigrams, _bigrams(tokenize(content)))
            citation = self._citation_score(query, item)
            structure = self._structure_score(item)
            rerank_score = (
                self.base_score_weight * base
                + self.lexical_weight * lexical
                + self.phrase_weight * phrase
                + self.citation_weight * citation
                + self.structure_weight * structure
            )
            if not math.isfinite(rerank_score):
                rerank_score = 0.0
            item["lexical_overlap_score"] = lexical
            item["phrase_overlap_score"] = phrase
            item["citation_alignment_score"] = citation
            item["structure_score"] = structure
            item["rerank_score"] = rerank_score
            scored.append(item)
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored[:top_k]


class ClusterReranker(LexicalOverlapReranker):
    """Community-aware reranker for graph evidence.

    LegalGraphRAG does not define a separate trainable ``ClusterReranker``
    class. Its corresponding retrieval step scores the best ontology
    community first and then retrieves the strongest cases inside it. This
    adapter keeps the BGE cross-encoder score for each provision, aggregates
    candidates into explicit ontology communities, and orders the final
    evidence coarse-to-fine. Candidates without a graph community remain
    singleton clusters so direct semantic retrieval is not over-grouped by
    the statute they happen to cite.
    """

    def __init__(
        self,
        *args: Any,
        cluster_top_k: int = 1,
        cluster_member_top_k: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cluster_top_k = max(1, int(cluster_top_k))
        self.cluster_member_top_k = max(1, int(cluster_member_top_k))

    @staticmethod
    def _cluster_id(candidate: dict[str, Any]) -> str:
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        selected = str(metadata.get("selected_ontology_community") or "").strip()
        if selected:
            return f"community:{selected}"

        communities = metadata.get("ontology_communities")
        if isinstance(communities, (list, tuple)):
            values = sorted({str(value).strip() for value in communities if str(value).strip()})
            if values:
                return f"community:{values[0]}"

        graph_path = candidate.get("graph_path") or metadata.get("graph_path") or []
        if isinstance(graph_path, (list, tuple)):
            ontology_nodes = sorted(
                {str(node).strip() for node in graph_path if str(node).strip().startswith("ontology:")}
            )
            if ontology_nodes:
                return f"community:{ontology_nodes[0]}"

        source_chunk_id = str(metadata.get("source_chunk_id") or metadata.get("chunk_id") or "").strip()
        if source_chunk_id:
            return f"source:{source_chunk_id}"
        return f"document:{str(candidate.get('doc_id') or '').strip()}"

    @staticmethod
    def _cluster_summary(members: list[dict[str, Any]], max_members: int = 4) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for member in members[:max_members]:
            content = str(member.get("content") or "").strip()
            if not content:
                continue
            # Keep the community representation within the reranker context.
            content = content[:700]
            if content in seen:
                continue
            seen.add(content)
            parts.append(content)
        return "\n".join(parts)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        if not candidates:
            return []

        # Score provisions first. The cluster stage is deliberately a second
        # pass so a highly relevant article is not lost inside its cluster.
        member_scored = super().rerank(query, candidates, top_k=len(candidates))
        grouped: dict[str, list[dict[str, Any]]] = {}
        for member in member_scored:
            grouped.setdefault(self._cluster_id(member), []).append(member)

        clusters = list(grouped.items())
        cluster_rows: list[tuple[str, list[dict[str, Any]], float]] = []
        for cluster_id, members in clusters:
            members.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
            top_scores = [float(item.get("rerank_score", 0.0)) for item in members[:3]]
            max_member_score = top_scores[0] if top_scores else 0.0
            mean_top_score = sum(top_scores) / max(1, len(top_scores))
            is_community = cluster_id.startswith("community:")
            if is_community:
                summary_score = lexical_overlap_score(query, self._cluster_summary(members))
                cluster_score = 0.65 * max_member_score + 0.25 * mean_top_score + 0.10 * summary_score
            else:
                # Rsem/Rchg candidates do not have a community representation;
                # preserve their cross-encoder order instead of inventing one.
                cluster_score = max_member_score
            cluster_rows.append((cluster_id, members, cluster_score))
        cluster_rows.sort(key=lambda row: row[2], reverse=True)

        # Match the paper's coarse-to-fine behavior: expose the best community
        # first, then use other clusters as recall-preserving fallbacks.
        priority_count = min(self.cluster_top_k, len(cluster_rows))
        ordered_clusters = cluster_rows[:priority_count] + cluster_rows[priority_count:]
        ordered: list[dict[str, Any]] = []
        overflow: list[dict[str, Any]] = []
        for cluster_rank, (cluster_id, members, cluster_score) in enumerate(ordered_clusters, start=1):
            for member_rank, member in enumerate(members, start=1):
                item = dict(member)
                metadata = dict(item.get("metadata") or {})
                metadata.update(
                    {
                        "cluster_id": cluster_id,
                        "cluster_rank": cluster_rank,
                        "cluster_member_rank": member_rank,
                        "cluster_score": float(cluster_score),
                    }
                )
                item["metadata"] = metadata
                item["cluster_id"] = cluster_id
                item["cluster_rank"] = cluster_rank
                item["cluster_member_rank"] = member_rank
                item["cluster_score"] = float(cluster_score)
                item["rerank_method"] = "community_cluster"
                if member_rank <= self.cluster_member_top_k:
                    ordered.append(item)
                else:
                    overflow.append(item)

        ordered.extend(overflow)
        return ordered[:top_k]
