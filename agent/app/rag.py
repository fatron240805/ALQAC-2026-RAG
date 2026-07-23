"""Local Qdrant vector retrieval + JSONL one-hop graph expansion.

Owns only repository-local paths. Never imports sibling alqac-2026-rag.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.embeddings import create_embeddings
from app.schemas import ElementGraph, LawHit

logger = logging.getLogger("alqac.rag")

COLLECTION_NAME = "law_articles"


def _node_key(law_id: str, aid: str) -> str:
    return f"{law_id}::{aid}"


class LawGraph:
    """Adjacency from nodes.jsonl / edges.jsonl."""

    def __init__(self, nodes_path: Path, edges_path: Path) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.adj: dict[str, set[str]] = {}
        if nodes_path.exists():
            with nodes_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    node = json.loads(line)
                    key = node.get("id") or _node_key(
                        str(node.get("law_id", "")), str(node.get("aid", ""))
                    )
                    self.nodes[key] = node
        if edges_path.exists():
            with edges_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    edge = json.loads(line)
                    src = str(edge.get("source") or edge.get("src") or "")
                    dst = str(edge.get("target") or edge.get("dst") or "")
                    if not src or not dst:
                        continue
                    self.adj.setdefault(src, set()).add(dst)
                    self.adj.setdefault(dst, set()).add(src)

    def neighbors(self, key: str) -> list[str]:
        return sorted(self.adj.get(key, set()))

    def get_node(self, key: str) -> dict[str, Any] | None:
        return self.nodes.get(key)


class LawRAG:
    """Top-k Qdrant seeds + fixed graph hops; preserve law_id/aid."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.graph = LawGraph(settings.graph_nodes_path, settings.graph_edges_path)
        self._client = None
        self._embedder = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self.settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.settings.qdrant_path))
        return self._client

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed via OpenAI-compatible embeddings endpoint."""
        from openai import OpenAI

        client = OpenAI(
            base_url=self.settings.embedding_base_url,
            api_key=self.settings.embedding_api_key,
        )
        resp = create_embeddings(
            client,
            model=self.settings.embedding_model,
            inputs=texts,
        )
        # Preserve input order
        by_idx = {item.index: item.embedding for item in resp.data}
        return [by_idx[i] for i in range(len(texts))]

    def collection_ready(self) -> bool:
        try:
            client = self._get_client()
            client.get_collection(COLLECTION_NAME)
            return True
        except Exception:  # noqa: BLE001
            return False

    def search(
        self,
        query: str,
        element_graph: ElementGraph | None = None,
        top_k: int | None = None,
    ) -> list[LawHit]:
        k = top_k if top_k is not None else self.settings.law_rag_top_k
        max_hops = self.settings.graph_max_hops

        # Enrich query lightly with legal questions from element graph
        q = query.strip()
        if element_graph and element_graph.legal_questions:
            q = q + "\n" + " ".join(element_graph.legal_questions[:5])

        seeds: list[tuple[str, str, str, float, int]] = []
        # (law_id, aid, text, score, hops)

        if self.collection_ready() and q:
            try:
                client = self._get_client()
                vector = self._embed([q])[0]
                response = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vector,
                    limit=k,
                    with_payload=True,
                )
                for h in response.points:
                    payload = h.payload or {}
                    law_id = str(payload.get("law_id", ""))
                    aid = str(payload.get("aid", ""))
                    text = str(payload.get("text", ""))
                    if law_id and aid:
                        seeds.append((law_id, aid, text, float(h.score or 0.0), 0))
            except Exception as exc:  # noqa: BLE001
                logger.warning("qdrant_search_failed: %s", exc)

        # Fallback: scan graph nodes by naive token overlap if no vector hits
        if not seeds and self.graph.nodes:
            tokens = {t.lower() for t in q.split() if len(t) > 2}
            scored: list[tuple[float, str, dict[str, Any]]] = []
            for key, node in self.graph.nodes.items():
                text = str(node.get("text", ""))
                if not text:
                    continue
                tset = {t.lower() for t in text.split() if len(t) > 2}
                score = len(tokens & tset) / max(1, len(tokens))
                if score > 0:
                    scored.append((score, key, node))
            scored.sort(key=lambda x: (-x[0], x[1]))
            for score, _key, node in scored[:k]:
                seeds.append(
                    (
                        str(node.get("law_id", "")),
                        str(node.get("aid", "")),
                        str(node.get("text", "")),
                        float(score),
                        0,
                    )
                )

        # One-hop expansion
        seen: dict[tuple[str, str], LawHit] = {}
        for law_id, aid, text, score, hops in seeds:
            key = (law_id, aid)
            if key not in seen:
                seen[key] = LawHit(
                    law_id=law_id,
                    aid=aid,
                    text=text,
                    vector_score=score,
                    graph_hops=hops,
                )

        if max_hops >= 1:
            seed_keys = list(seen.keys())
            for law_id, aid in seed_keys:
                nkey = _node_key(law_id, aid)
                for neigh in self.graph.neighbors(nkey):
                    node = self.graph.get_node(neigh)
                    if not node:
                        # try parse neigh as law_id::aid
                        if "::" in neigh:
                            n_law, n_aid = neigh.split("::", 1)
                        else:
                            continue
                    else:
                        n_law = str(node.get("law_id", ""))
                        n_aid = str(node.get("aid", ""))
                        text = str(node.get("text", ""))
                    if not n_law or not n_aid:
                        continue
                    ntuple = (n_law, n_aid)
                    if ntuple in seen:
                        continue
                    if node:
                        text = str(node.get("text", ""))
                    else:
                        text = ""
                    parent_score = seen[(law_id, aid)].vector_score
                    seen[ntuple] = LawHit(
                        law_id=n_law,
                        aid=n_aid,
                        text=text,
                        vector_score=parent_score,
                        graph_hops=1,
                    )

        # Rank: seed score desc, then graph distance asc, stable by law_id/aid
        ranked = sorted(
            seen.values(),
            key=lambda h: (-h.vector_score, h.graph_hops, h.law_id, h.aid),
        )
        # Return top_k seeds expanded set but cap reasonably
        return ranked[: max(k * 3, k)]
