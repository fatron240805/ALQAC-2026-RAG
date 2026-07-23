import json
from pathlib import Path

from app.rag import LawGraph, LawRAG
from app.schemas import ElementGraph, LawHit
from scripts.build_legal_index import build_index


def test_graph_one_hop(tmp_path, settings, monkeypatch):
    nodes = tmp_path / "nodes.jsonl"
    edges = tmp_path / "edges.jsonl"
    nodes.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "L::1",
                        "law_id": "L",
                        "aid": "1",
                        "text": "article one damages dog",
                    }
                ),
                json.dumps(
                    {
                        "id": "L::2",
                        "law_id": "L",
                        "aid": "2",
                        "text": "article two liability",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    edges.write_text(
        json.dumps({"source": "L::1", "target": "L::2", "type": "adj"}) + "\n",
        encoding="utf-8",
    )
    g = LawGraph(nodes, edges)
    assert "L::2" in g.neighbors("L::1")

    # Point settings graph paths
    monkeypatch.setattr(settings, "graph_nodes_path", nodes)
    monkeypatch.setattr(settings, "graph_edges_path", edges)
    monkeypatch.setattr(settings, "law_rag_top_k", 1)
    monkeypatch.setattr(settings, "graph_max_hops", 1)
    monkeypatch.setattr(settings, "qdrant_path", tmp_path / "vector")

    rag = LawRAG(settings)
    hits = rag.search("damages dog")
    assert hits
    assert all(isinstance(h, LawHit) for h in hits)
    ids = {(h.law_id, h.aid) for h in hits}
    assert ("L", "1") in ids
    # one-hop should include neighbor
    assert ("L", "2") in ids


def test_search_returns_ranked_by_score(tmp_path, settings, monkeypatch):
    nodes = tmp_path / "nodes.jsonl"
    edges = tmp_path / "edges.jsonl"
    nodes.write_text(
        "\n".join(
            [
                json.dumps({"id": "X::1", "law_id": "X", "aid": "1", "text": "contract law obligation"}),
                json.dumps({"id": "X::2", "law_id": "X", "aid": "2", "text": "unrelated criminal law"}),
            ]
        ),
        encoding="utf-8",
    )
    edges.write_text("", encoding="utf-8")
    monkeypatch.setattr(settings, "graph_nodes_path", nodes)
    monkeypatch.setattr(settings, "graph_edges_path", edges)
    monkeypatch.setattr(settings, "law_rag_top_k", 2)
    monkeypatch.setattr(settings, "graph_max_hops", 0)
    monkeypatch.setattr(settings, "qdrant_path", tmp_path / "vector")

    rag = LawRAG(settings)
    hits = rag.search("contract obligation")
    assert len(hits) >= 1
    # First hit should be the more relevant one (higher token overlap)
    assert hits[0].law_id == "X"
    assert hits[0].aid == "1"


def test_search_with_element_graph_enrichment(tmp_path, settings, monkeypatch):
    nodes = tmp_path / "nodes.jsonl"
    edges = tmp_path / "edges.jsonl"
    nodes.write_text(
        json.dumps({"id": "Z::10", "law_id": "Z", "aid": "10", "text": "damages compensation"}) + "\n",
        encoding="utf-8",
    )
    edges.write_text("", encoding="utf-8")
    monkeypatch.setattr(settings, "graph_nodes_path", nodes)
    monkeypatch.setattr(settings, "graph_edges_path", edges)
    monkeypatch.setattr(settings, "law_rag_top_k", 1)
    monkeypatch.setattr(settings, "graph_max_hops", 0)
    monkeypatch.setattr(settings, "qdrant_path", tmp_path / "vector")

    rag = LawRAG(settings)
    graph = ElementGraph(legal_questions=["Who pays damages?"])
    hits = rag.search("compensation", element_graph=graph)
    assert len(hits) >= 1
    assert hits[0].law_id == "Z"
    assert hits[0].aid == "10"


def test_build_index_preserves_ids(tmp_path, settings, monkeypatch):
    corpus = tmp_path / "legal_corpus"
    corpus.mkdir()
    (corpus / "law.jsonl").write_text(
        json.dumps(
            {"law_id": "47/2010/QH12", "aid": "270", "text": "Quy định về hợp đồng."}
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "law_corpus_path", corpus)
    monkeypatch.setattr(settings, "graph_nodes_path", tmp_path / "graph" / "nodes.jsonl")
    monkeypatch.setattr(settings, "graph_edges_path", tmp_path / "graph" / "edges.jsonl")
    monkeypatch.setattr(settings, "qdrant_path", tmp_path / "vector")

    stats = build_index(settings, skip_embeddings=True)
    assert stats["records"] == 1
    node = json.loads(settings.graph_nodes_path.read_text(encoding="utf-8").strip())
    assert node["law_id"] == "47/2010/QH12"
    assert node["aid"] == "270"
