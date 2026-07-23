import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import APIConnectionError

from app.embeddings import is_retryable_embedding_error, is_retryable_openai_error
from app.rag import COLLECTION_NAME, LawGraph, LawRAG
from app.schemas import ElementGraph, LawHit
from scripts.build_legal_index import _build_collection_name, _swap_live_alias, build_index


def test_connection_error_is_retryable_for_embeddings():
    error = APIConnectionError(request=httpx.Request("POST", "https://embed.example/v1/embeddings"))

    assert is_retryable_embedding_error(error)
    assert is_retryable_openai_error(error)


def test_langchain_wrapped_server_error_is_retryable():
    error = ValueError({"message": "Internal Server Error", "code": 500})

    assert is_retryable_openai_error(error)


def test_build_collection_name_is_stable_and_input_specific():
    records = [{"law_id": "L", "aid": "1", "text": "Article one", "id": "L::1"}]

    name = _build_collection_name(records, "text-embedding-3-small", 1536)

    assert name == _build_collection_name(records, "text-embedding-3-small", 1536)
    assert name != _build_collection_name(records, "other-model", 1536)


def test_qdrant_alias_swap(tmp_path):
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm

    client = QdrantClient(path=str(tmp_path / "vector"))
    try:
        client.create_collection(
            collection_name="source",
            vectors_config=qm.VectorParams(size=2, distance=qm.Distance.COSINE),
        )
        client.create_collection(
            collection_name="target",
            vectors_config=qm.VectorParams(size=2, distance=qm.Distance.COSINE),
        )
        _swap_live_alias(client, qm, "source")
        _swap_live_alias(client, qm, "target")

        aliases = {
            alias.alias_name: alias.collection_name for alias in client.get_aliases().aliases
        }

        assert aliases["law_articles"] == "target"
    finally:
        client.close()


def test_collection_ready_resolves_alias(settings):
    class AliasClient:
        def get_collection(self, collection_name):
            assert collection_name == COLLECTION_NAME

    rag = LawRAG(settings)
    rag._get_client = lambda: AliasClient()

    assert rag.collection_ready()


def test_search_uses_qdrant_query_points(settings, monkeypatch, tmp_path):
    class Client:
        def get_collection(self, collection_name):
            assert collection_name == COLLECTION_NAME

        def query_points(self, **kwargs):
            assert kwargs["collection_name"] == COLLECTION_NAME
            assert kwargs["query"] == [0.25, 0.75]
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        payload={"law_id": "L", "aid": "1", "text": "law text"},
                        score=0.9,
                    )
                ]
            )

    nodes = tmp_path / "nodes.jsonl"
    edges = tmp_path / "edges.jsonl"
    nodes.write_text("", encoding="utf-8")
    edges.write_text("", encoding="utf-8")
    monkeypatch.setattr(settings, "graph_nodes_path", nodes)
    monkeypatch.setattr(settings, "graph_edges_path", edges)
    rag = LawRAG(settings)
    rag._get_client = lambda: Client()
    rag._embed = lambda texts: [[0.25, 0.75]]

    hits = rag.search("query", top_k=1)

    assert [(hit.law_id, hit.aid, hit.vector_score) for hit in hits] == [("L", "1", 0.9)]


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


def test_build_index_preserves_ids(tmp_path, settings, monkeypatch, capsys):
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
    stderr = capsys.readouterr().err

    assert stats["records"] == 1
    node = json.loads(settings.graph_nodes_path.read_text(encoding="utf-8").strip())
    assert node["law_id"] == "47/2010/QH12"
    assert node["aid"] == "270"
    assert "[load] records=1" in stderr
    assert "[graph] nodes=1 edges=0" in stderr
    assert "[embed] skipped" in stderr
    assert "[done] records=1 edges=0 embedded=0" in stderr


def test_build_index_ingests_nested_law_corpus(tmp_path, settings, monkeypatch):
    corpus = tmp_path / "legal_corpus"
    corpus.mkdir()
    (corpus / "private_law_corpus.json").write_text(
        json.dumps(
            [
                {
                    "law_id": "47/2010/QH12",
                    "content": [
                        {"aid": 270, "content_Article": "Điều khoản ngân hàng."},
                        {"aid": 271, "content_Article": "Điều khoản tín dụng."},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "law_corpus_path", corpus)
    monkeypatch.setattr(settings, "graph_nodes_path", tmp_path / "graph" / "nodes.jsonl")
    monkeypatch.setattr(settings, "graph_edges_path", tmp_path / "graph" / "edges.jsonl")
    monkeypatch.setattr(settings, "qdrant_path", tmp_path / "vector")

    stats = build_index(settings, skip_embeddings=True)
    nodes = [
        json.loads(line)
        for line in settings.graph_nodes_path.read_text(encoding="utf-8").splitlines()
    ]

    assert stats["records"] == 2
    assert [(node["law_id"], node["aid"]) for node in nodes] == [
        ("47/2010/QH12", "270"),
        ("47/2010/QH12", "271"),
    ]
