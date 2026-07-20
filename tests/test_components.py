from __future__ import annotations

import argparse
import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from evaluation.metrics import evaluate_alqac_system
from evaluation.retrieval_benchmark import (
    GoldProvision,
    ProvisionNormalizer,
    canonical_law_code,
    extract_article_numbers,
    is_graph_candidate,
    load_public_test_gold,
    per_case_metrics,
    provision_key,
    run_benchmark,
)
from graph_construct.builder import build_graph_records
from graph_construct.neo4j_store import Neo4jConfig, _redacted_uri
from orchestration.config import PipelineConfig
from orchestration.data_adapters import chunk_to_indexer_doc, stream_active_indexer_docs
from orchestration.interfaces import (
    LocalOllamaClient,
    PromptTemplateReasoningAgent,
    StatutoryConsistencyVerifier,
)
from orchestration.run_pipeline import (
    build_graph_retriever,
    build_submission_json,
    check_runtime,
    run_pipeline,
    validate_prediction_output,
)
from prepare_corpus import build_corpus_records, normalize_text
from retrieval.citation_usefulness import HeuristicCitationUsefulnessFilter
from retrieval.cleaning import build_clean_corpus
from retrieval.deprecated_filter import DeprecatedFilter
from retrieval.graph_retriever import GraphRetrieverConfig, LegalGraphRetriever
from retrieval.indexing import HybridIndexer
from retrieval.reranker import ClusterReranker, LexicalOverlapReranker
from retrieval.router import DocumentRouter, QueryAnalyzer
from retrieval.service import RetrievalService, RetrievalServiceConfig


class WorkspaceTempDir:
    def __enter__(self) -> str:
        tmp_root = Path.cwd() / "test_tmp"
        tmp_root.mkdir(exist_ok=True)
        self.path = tmp_root / f"case_{uuid.uuid4().hex}"
        self.path.mkdir()
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Managed sandbox chmod/rmtree can be flaky on Windows. Test artifacts
        # stay under test_tmp/ and are ignored by git.
        return False


def workspace_tempdir() -> WorkspaceTempDir:
    return WorkspaceTempDir()


class CorpusPreparationTests(unittest.TestCase):
    def test_normalize_text_standardizes_whitespace_and_punctuation(self) -> None:
        text = normalize_text("  Điều  1 \r\n  Nội dung   ,  quyền   ;nghĩa vụ  ")
        self.assertEqual(text, "Điều 1\nNội dung, quyền; nghĩa vụ")

    def test_build_corpus_records_outputs_required_schema(self) -> None:
        records = build_corpus_records(
            [
                {
                    "id": "law_1",
                    "law_id": "01/2024/QH15",
                    "content": [{"content_Article": "Điều 1. Nội dung luật"}],
                    "status": "active",
                }
            ]
        )
        self.assertEqual(records[0]["doc_id"], "law_1")
        self.assertIn("content", records[0])
        self.assertEqual(records[0]["metadata"]["title"], "01/2024/QH15")
        self.assertEqual(records[0]["metadata"]["status"], "active")


class CleaningTests(unittest.TestCase):
    def test_cleaning_preserves_aid_as_article_number_and_splits_points(self) -> None:
        with workspace_tempdir() as tmp:
            raw_path = Path(tmp) / "corpus.json"
            raw_path.write_text(
                json.dumps(
                    [
                        {
                            "id": 1,
                            "law_id": "99/2024/QH15",
                            "content": [
                                {
                                    "aid": 12,
                                    "content_Article": (
                                        "Trong luật này:\n"
                                        "1. Chủ sở hữu có nghĩa vụ sau đây: "
                                        "a) Bồi thường thiệt hại; b) Quản lý tài sản."
                                    ),
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            _, articles, chunks, audit = build_clean_corpus(raw_path, target_tokens=1)

        self.assertEqual(audit["schema_version"], 5)
        self.assertEqual(articles[0].article_number, "12")
        self.assertEqual(articles[0].article_number_source, "aid")
        self.assertTrue(any(chunk.unit_type.startswith("point") for chunk in chunks))
        self.assertIn("điểm a", {chunk.unit_path for chunk in chunks if chunk.point_label == "a"}.pop())
        self.assertTrue(any("obligation" in chunk.rule_signals for chunk in chunks))
        self.assertTrue(any("ontology:issue:tort_damage" in chunk.ontology_concepts for chunk in chunks))


class DeprecatedFilterTests(unittest.TestCase):
    def test_status_and_expiry_rules(self) -> None:
        deprecated_filter = DeprecatedFilter()
        self.assertTrue(deprecated_filter.is_deprecated({"metadata": {"status": "hết hiệu lực"}}))
        self.assertFalse(deprecated_filter.is_deprecated({"metadata": {"status": "còn hiệu lực"}}))
        self.assertTrue(
            deprecated_filter.is_deprecated({"metadata": {"status": "active", "valid_until": "2000-01-01"}})
        )

    def test_stream_active_indexer_docs_filters_flat_chunks(self) -> None:
        with workspace_tempdir() as tmp:
            path = Path(tmp) / "chunks.jsonl"
            rows = [
                {"chunk_id": "c1", "text": "Điều 1 quyền dân sự", "law_id": "a", "aid": 1},
                {"chunk_id": "c2", "text": "old", "law_id": "b", "aid": 2, "deprecated": True},
            ]
            path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8",
            )
            docs = list(stream_active_indexer_docs(path))
        self.assertEqual([doc["doc_id"] for doc in docs], ["c1"])
        self.assertEqual(docs[0]["metadata"]["law_id"], "a")


class RetrievalComponentTests(unittest.TestCase):
    def test_neo4j_error_log_redacts_credentials_from_uri(self) -> None:
        self.assertEqual(
            _redacted_uri("neo4j+s://user:secret@abc123.databases.neo4j.io:7687"),
            "neo4j+s://abc123.databases.neo4j.io:7687",
        )

    def test_neo4j_config_accepts_aura_uri_and_small_pool(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "NEO4J_URI": "neo4j+s://abc123.databases.neo4j.io",
                "NEO4J_USERNAME": "neo4j",
                "NEO4J_PASSWORD": "test-password",
                "NEO4J_DATABASE": "neo4j",
                "NEO4J_MAX_CONNECTION_POOL_SIZE": "8",
                "NEO4J_CONNECTION_ACQUISITION_TIMEOUT": "30",
            },
            clear=False,
        ):
            config = Neo4jConfig.from_env()

        self.assertTrue(config.is_aura)
        self.assertEqual(config.max_connection_pool_size, 8)
        self.assertEqual(config.connection_acquisition_timeout, 30.0)

    def test_hybrid_indexer_search_ranks_matching_document(self) -> None:
        corpus = [
            {
                "doc_id": "dog",
                "content": "Chủ sở hữu súc vật phải bồi thường thiệt hại do súc vật gây ra.",
                "metadata": {"law_id": "91/2015/QH13", "aid": 584, "unit_path": "91/2015/QH13 Điều 584"},
            },
            {
                "doc_id": "bank",
                "content": "Hoạt động ngân hàng và tổ chức tín dụng.",
                "metadata": {"law_id": "47/2010/QH12", "aid": 1, "unit_path": "47/2010/QH12 Điều 1"},
            },
        ]
        indexer = HybridIndexer(embedding_dim=64).build_index(corpus)
        results = indexer.search("Điều 584 bồi thường thiệt hại do súc vật gây ra", top_k=1)
        self.assertEqual(results[0]["doc_id"], "dog")
        self.assertGreater(results[0]["fused_score"], 0)

    def test_hybrid_indexer_semantic_text_includes_preprocessed_legal_features(self) -> None:
        indexer = HybridIndexer(embedding_dim=32)
        text = indexer._semantic_text(
            {
                "content": "Chu so huu phai boi thuong.",
                "metadata": {
                    "ontology_concepts": ["ontology:issue:tort_damage"],
                    "rule_signals": ["obligation"],
                    "article_references": ["585"],
                },
            }
        )

        self.assertIn("ontology:issue:tort_damage", text)
        self.assertIn("obligation", text)
        self.assertIn("585", text)

    def test_save_and_load_index_roundtrip(self) -> None:
        with workspace_tempdir() as tmp:
            indexer = HybridIndexer(embedding_dim=32).build_index(
                [{"doc_id": "x", "content": "quyền tài sản", "metadata": {"law_id": "L"}}]
            )
            indexer.save_index(tmp)
            loaded = HybridIndexer.load_index(tmp)
            self.assertEqual(loaded.doc_ids, ["x"])
            self.assertEqual(loaded.search("tài sản", top_k=1)[0]["doc_id"], "x")

    def test_router_reranker_and_citation_filter(self) -> None:
        query = "Điều 12 quy định bồi thường thiệt hại như thế nào?"
        candidates = [
            {
                "doc_id": "c1",
                "content": "Điều 12. Cá nhân phải bồi thường thiệt hại do lỗi của mình.",
                "metadata": {"law_id": "99/2024/QH15", "unit_path": "99/2024/QH15 Điều 12"},
                "fused_score": 0.5,
            },
            {"doc_id": "c2", "content": "Nội dung quản trị ngân hàng.", "metadata": {}, "fused_score": 0.4},
        ]
        analysis = QueryAnalyzer().analyze(query)
        self.assertIn("12", analysis.statute_references)

        routed = DocumentRouter().apply(query, candidates)
        reranked = LexicalOverlapReranker().rerank(query, routed, top_k=2)
        useful = HeuristicCitationUsefulnessFilter(min_score=0.01).filter(query, reranked)

        self.assertEqual(useful[0]["doc_id"], "c1")
        self.assertIn(useful[0]["citation_judgment"]["judgment"], {"useful", "uncertain"})

    def test_reranker_prefers_statute_aligned_candidate(self) -> None:
        query = "Điều 584 bồi thường thiệt hại do súc vật gây ra"
        candidates = [
            {
                "doc_id": "weak",
                "content": "Bài viết về trách nhiệm dân sự chung.",
                "metadata": {"law_id": "91/2015/QH13", "aid": 500, "unit_path": "91/2015/QH13 Điều 500"},
                "fused_score": 0.48,
            },
            {
                "doc_id": "strong",
                "content": "Chủ sở hữu súc vật phải bồi thường thiệt hại do súc vật gây ra.",
                "metadata": {"law_id": "91/2015/QH13", "aid": 584, "unit_path": "91/2015/QH13 Điều 584"},
                "fused_score": 0.4,
            },
        ]

        reranked = LexicalOverlapReranker().rerank(query, candidates, top_k=2)

        self.assertEqual(reranked[0]["doc_id"], "strong")
        self.assertGreater(reranked[0]["citation_alignment_score"], reranked[1]["citation_alignment_score"])

    def test_cluster_reranker_ranks_community_before_fallback_law(self) -> None:
        query = "fraud deception victim property"
        candidates = [
            {
                "doc_id": "fraud_1",
                "content": "Fraud by deception caused the victim to transfer property.",
                "metadata": {
                    "law_id": "law-fraud",
                    "selected_ontology_community": "ontology:issue:fraud",
                },
                "fused_score": 0.45,
            },
            {
                "doc_id": "contract_1",
                "content": "A contract dispute concerns payment obligations.",
                "metadata": {"law_id": "law-contract"},
                "fused_score": 0.50,
            },
        ]

        reranked = ClusterReranker(use_gpu_reranker=False).rerank(query, candidates, top_k=2)

        self.assertEqual(reranked[0]["doc_id"], "fraud_1")
        self.assertEqual(reranked[0]["rerank_method"], "community_cluster")
        self.assertEqual(reranked[0]["metadata"]["cluster_id"], "community:ontology:issue:fraud")
        self.assertIn("cluster_score", reranked[0]["metadata"])

    def test_citation_filter_preserves_cluster_selection_order(self) -> None:
        candidates = [
            {
                "doc_id": "graph_first",
                "content": "Relevant legal evidence.",
                "rerank_score": 0.20,
                "graph_path": ["chunk:seed", "rule:law:1"],
            },
            {
                "doc_id": "flat_second",
                "content": "Relevant legal evidence.",
                "rerank_score": 0.95,
            },
        ]

        filtered = HeuristicCitationUsefulnessFilter(max_results=2).filter(
            "legal evidence",
            candidates,
            preserve_graph_paths=True,
        )

        self.assertEqual([item["doc_id"] for item in filtered], ["graph_first", "flat_second"])

    def test_retrieval_service_returns_reasoning_ready_evidence_contract(self) -> None:
        candidate = {
            "doc_id": "rule:91/2015/QH13:584",
            "content": "The owner of an animal must compensate for damage caused by the animal.",
            "metadata": {
                "law_id": "91/2015/QH13",
                "aid": 584,
                "source_chunk_id": "chunk_584",
                "graph_distance": 1,
            },
            "fused_score": 0.65,
            "rerank_score": 0.82,
            "graph_path": ["ontology:liability:animal_damage", "rule:91/2015/QH13:584"],
        }

        class FakeIndexer:
            def search(self, query: str, top_k: int, alpha: float) -> list[dict[str, object]]:
                return [candidate]

        class FakeReranker:
            def rerank(self, query: str, candidates: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
                return candidates[:top_k]

        class FakeCitationFilter:
            def filter(self, query: str, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
                return candidates

        service = RetrievalService(
            FakeIndexer(),
            reranker=FakeReranker(),
            citation_filter=FakeCitationFilter(),
            config=RetrievalServiceConfig(seed_top_k=4, final_top_k=1),
        )

        result = service.retrieve("animal damage", top_k=1)
        payload = result.to_reasoning_payload(case_id="case_1")

        self.assertEqual(result.trace.backend, "flat_hybrid")
        self.assertEqual(result.trace.returned_count, 1)
        self.assertEqual(result.law_evidence, [{"law_id": "91/2015/QH13", "aid": 584}])
        self.assertEqual(result.evidence[0]["source_chunks"], ["chunk_584"])
        self.assertEqual(result.evidence[0]["graph_path"], candidate["graph_path"])
        self.assertEqual(payload["case_id"], "case_1")
        self.assertEqual(len(payload["evidence_chains"]), 1)

    def test_build_graph_records_creates_rule_and_concept_nodes(self) -> None:
        with workspace_tempdir() as tmp:
            chunks_path = Path(tmp) / "chunks.jsonl"
            chunks_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "chunk_id": "chunk_1",
                                "law_id": "91/2015/QH13",
                                "aid": 584,
                                "unit_type": "article",
                                "unit_path": "91/2015/QH13 Điều 584",
                                "text": "Chủ sở hữu súc vật phải bồi thường thiệt hại do súc vật gây ra.",
                            },
                            ensure_ascii=False,
                        )
                    ]
                ),
                encoding="utf-8",
            )
            nodes, edges, stats = build_graph_records(chunks_path)

        node_ids = {node["node_id"] for node in nodes}
        self.assertIn("law:91/2015/QH13", node_ids)
        self.assertIn("rule:91/2015/QH13:584", node_ids)
        self.assertIn("ontology:issue:tort_damage", node_ids)
        self.assertGreaterEqual(stats["node_count"], 3)
        self.assertGreaterEqual(len(edges), 2)

    def test_build_graph_records_uses_legal_article_number_not_source_aid(self) -> None:
        with workspace_tempdir() as tmp:
            chunks_path = Path(tmp) / "chunks.jsonl"
            chunks_path.write_text(
                json.dumps(
                    {
                        "chunk_id": "chunk_584",
                        "law_id": "91/2015/QH13",
                        "aid": 53354,
                        "article_number": "53354",
                        "article_index": 584,
                        "unit_type": "article",
                        "unit_path": "91/2015/QH13 Dieu 584",
                        "text": "Dieu 584. Boi thuong thiet hai.",
                    }
                ),
                encoding="utf-8",
            )
            nodes, _, _ = build_graph_records(chunks_path)

        article = next(node for node in nodes if node["node_id"] == "rule:91/2015/QH13:584")
        self.assertEqual(article["properties"]["aid"], 53354)
        self.assertEqual(article["properties"]["source_aid"], 53354)
        self.assertEqual(article["properties"]["article_number"], "584")
        self.assertEqual(article["properties"]["article_index"], 584)

    def test_build_graph_records_preserves_clause_hierarchy_and_preprocessed_features(self) -> None:
        with workspace_tempdir() as tmp:
            chunks_path = Path(tmp) / "chunks.jsonl"
            chunks_path.write_text(
                json.dumps(
                    {
                        "chunk_id": "point_a",
                        "law_id": "91/2015/QH13",
                        "aid": 584,
                        "article_index": 584,
                        "clause_number": "1",
                        "point_label": "a",
                        "unit_type": "point",
                        "unit_path": "91/2015/QH13 Dieu 584 khoan 1 diem a",
                        "text": "Chu so huu phai boi thuong thiet hai theo Dieu 585.",
                        "ontology_concepts": ["ontology:issue:tort_damage"],
                        "rule_signals": ["obligation"],
                        "article_references": ["585"],
                    }
                ),
                encoding="utf-8",
            )
            nodes, edges, _ = build_graph_records(chunks_path)

        node_ids = {node["node_id"] for node in nodes}
        edge_pairs = {(edge["src"], edge["dst"], edge["edge_type"]) for edge in edges}
        self.assertIn("rule:91/2015/QH13:584:clause:1", node_ids)
        self.assertIn("ontology:rule_signal:obligation", node_ids)
        self.assertIn(
            ("rule:91/2015/QH13:584", "rule:91/2015/QH13:584:clause:1", "CONTAINS"),
            edge_pairs,
        )
        self.assertIn(
            ("rule:91/2015/QH13:584:clause:1", "chunk:point_a", "CONTAINS"),
            edge_pairs,
        )
        self.assertIn(
            ("rule:91/2015/QH13:584", "rule:91/2015/QH13:585", "CITES"),
            edge_pairs,
        )

    def test_legal_graph_retriever_returns_graph_backed_law_nodes(self) -> None:
        class FakeGraphStore:
            def seed_nodes_for_chunk(self, chunk_id: str) -> list[str]:
                return ["rule:91/2015/QH13:584"] if chunk_id == "c1" else []

            def expand_from_seeds(
                self,
                seed_node_ids: list[str],
                *,
                depth: int = 1,
                target_layers: list[str] | None = None,
                limit: int = 20,
            ) -> list[dict[str, Any]]:
                return [
                    {
                        "node_id": "rule:91/2015/QH13:584",
                        "layer": "rule",
                        "node_type": "article",
                        "law_id": "91/2015/QH13",
                        "aid": 584,
                        "source_chunk_id": "c1",
                        "chunk_id": "c1",
                        "text": "Chủ sở hữu súc vật phải bồi thường thiệt hại do súc vật gây ra.",
                        "unit_path": "91/2015/QH13 Điều 584",
                        "distance": 1,
                        "graph_path": ["chunk:c1", "rule:91/2015/QH13:584"],
                    }
                ]

        indexer = HybridIndexer(embedding_dim=64).build_index(
            [{"doc_id": "c1", "content": "bồi thường thiệt hại do súc vật", "metadata": {}}]
        )
        retriever = LegalGraphRetriever(
            indexer,
            graph_store=FakeGraphStore(),
            config=GraphRetrieverConfig(seed_top_k=5, chain_top_k=3),
        )
        results = retriever.retrieve("Chủ sở hữu súc vật phải bồi thường thiệt hại như thế nào?")

        self.assertTrue(results)
        self.assertEqual(results[0]["metadata"]["law_id"], "91/2015/QH13")
        self.assertEqual(results[0]["metadata"]["aid"], 584)
        self.assertIn("rule:91/2015/QH13:584", results[0]["graph_path"])

    def test_graph_retriever_keeps_seed_candidates_when_graph_expands(self) -> None:
        retriever = LegalGraphRetriever(HybridIndexer(embedding_dim=32))
        seed = {
            "doc_id": "seed_chunk",
            "content": "seed evidence",
            "fused_score": 0.8,
            "metadata": {"law_id": "91/2015/QH13", "article_index": 584},
        }
        graph = {
            "doc_id": "rule:91/2015/QH13:585",
            "content": "neighbour evidence",
            "fused_score": 0.6,
            "metadata": {
                "law_id": "91/2015/QH13",
                "article_number": 585,
                "graph_path": ["chunk:seed_chunk", "rule:91/2015/QH13:585"],
            },
            "graph_path": ["chunk:seed_chunk", "rule:91/2015/QH13:585"],
        }

        merged = retriever._merge_seed_and_graph_candidates([seed], [graph])

        self.assertEqual(len(merged), 2)
        self.assertIn("seed_chunk", [item["doc_id"] for item in merged])
        self.assertIn("rule:91/2015/QH13:585", [item["doc_id"] for item in merged])

    def test_graph_retriever_reserves_graph_candidate_quota(self) -> None:
        retriever = LegalGraphRetriever(
            HybridIndexer(embedding_dim=32),
            config=GraphRetrieverConfig(graph_candidate_ratio=0.5),
        )
        flat = [
            {"doc_id": f"flat_{index}", "rerank_score": 1.0 - index * 0.01, "metadata": {}}
            for index in range(6)
        ]
        graph = [
            {
                "doc_id": f"graph_{index}",
                "rerank_score": 0.4 - index * 0.01,
                "metadata": {},
                "graph_path": ["chunk:seed", f"rule:law:{index}"],
            }
            for index in range(6)
        ]

        selected = retriever._select_graph_aware_candidates(flat + graph, 6)

        self.assertEqual(len(selected), 6)
        self.assertEqual(sum(1 for item in selected if retriever._is_graph_backed(item)), 3)

    def test_graph_retriever_selects_query_aligned_ontology_community(self) -> None:
        retriever = LegalGraphRetriever(
            HybridIndexer(embedding_dim=32),
            config=GraphRetrieverConfig(community_top_k=1, community_member_top_k=2),
        )
        candidates = [
            {
                "doc_id": "direct",
                "fused_score": 0.5,
                "metadata": {},
                "graph_path": ["chunk:seed", "rule:91/2015/QH13:584"],
            },
            {
                "doc_id": "community_a_best",
                "fused_score": 0.9,
                "metadata": {},
                "graph_path": ["chunk:seed", "ontology:issue:tort_damage", "rule:91/2015/QH13:585"],
            },
            {
                "doc_id": "community_a_second",
                "fused_score": 0.8,
                "metadata": {},
                "graph_path": ["chunk:seed", "ontology:issue:tort_damage", "rule:91/2015/QH13:586"],
            },
            {
                "doc_id": "community_b",
                "fused_score": 0.7,
                "metadata": {},
                "graph_path": ["chunk:seed", "ontology:issue:contract_dispute", "rule:91/2015/QH13:587"],
            },
        ]

        selected = retriever._community_guided_candidates(candidates)

        self.assertEqual(
            [item["doc_id"] for item in selected],
            ["community_a_best", "community_a_second", "direct"],
        )
        self.assertEqual(
            selected[0]["metadata"]["selected_ontology_community"],
            "ontology:issue:tort_damage",
        )

    def test_build_graph_retriever_can_require_graph(self) -> None:
        indexer = HybridIndexer(embedding_dim=32).build_index(
            [{"doc_id": "c1", "content": "bá»“i thÆ°á»ng thiá»‡t háº¡i", "metadata": {}}]
        )
        config = PipelineConfig(use_graph_retrieval=False)

        with self.assertRaisesRegex(RuntimeError, "Graph retrieval is disabled"):
            build_graph_retriever(config, indexer, require_graph=True)


class ReasoningAndRuntimeTests(unittest.TestCase):
    def test_local_ollama_client_uses_native_chat_endpoint(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "message": {
                        "role": "assistant",
                        "content": '{"label":"B_WIN","confidence":0.7,"evidence_ids":[],"justification":"ok"}',
                    }
                }

        client = LocalOllamaClient(
            base_url="http://localhost:11434",
            model_name="qwen2.5:7b-instruct",
            provider="ollama",
        )
        with patch("orchestration.interfaces.requests.post", return_value=FakeResponse()) as post:
            raw = client.generate("Return JSON", max_tokens=128, temperature=0.0)

        self.assertIn('"label":"B_WIN"', raw)
        _, kwargs = post.call_args
        self.assertEqual(post.call_args.args[0], "http://localhost:11434/api/chat")
        self.assertEqual(kwargs["json"]["format"], "json")
        self.assertFalse(kwargs["json"]["stream"])
        self.assertEqual(kwargs["json"]["options"]["num_predict"], 128)
        self.assertEqual(kwargs["json"]["model"], "qwen2.5:7b-instruct")

    def test_openai_compatible_client_uses_chat_completions_endpoint(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"label":"A_WIN","confidence":0.8,"evidence_ids":[],"justification":"ok"}'
                            }
                        }
                    ]
                }

        client = LocalOllamaClient(
            base_url="http://localhost:8000/v1",
            model_name="luanngo/Qwen3-4B-VietNamese-Legal-Chat",
            provider="openai-compatible",
            api_key="secret",
        )
        with patch("orchestration.interfaces.requests.post", return_value=FakeResponse()) as post:
            raw = client.generate("Return JSON", max_tokens=256, temperature=0.0)

        self.assertIn('"label":"A_WIN"', raw)
        _, kwargs = post.call_args
        self.assertEqual(post.call_args.args[0], "http://localhost:8000/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(kwargs["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["json"]["max_tokens"], 256)
        self.assertEqual(kwargs["json"]["model"], "luanngo/Qwen3-4B-VietNamese-Legal-Chat")

    def test_reasoning_parser_extracts_and_normalizes_json(self) -> None:
        parsed = PromptTemplateReasoningAgent._parse_json_output(
            "case_x",
            'prefix {"prediction":"A_WIN","confidence":"0.8","evidence_ids":["L1"],"justification":"ok"} suffix',
        )
        self.assertEqual(parsed["label"], "A_WIN")
        self.assertEqual(parsed["confidence"], 0.8)
        self.assertEqual(parsed["parser_status"], "ok")

    def test_reasoning_parser_falls_back_on_invalid_label(self) -> None:
        parsed = PromptTemplateReasoningAgent._parse_json_output(
            "case_x",
            '{"label":"UNKNOWN","confidence":2,"justification":"bad"}',
        )
        self.assertEqual(parsed["label"], "PARTIAL_B_WIN")
        self.assertEqual(parsed["confidence"], 1.0)
        self.assertEqual(parsed["parser_status"], "invalid_label")

    def test_verifier_filters_invalid_evidence_ids_and_lowers_confidence(self) -> None:
        verifier = StatutoryConsistencyVerifier()
        verified = verifier.verify(
            {
                "case_id": "case_x",
                "label": "A_WIN",
                "confidence": 0.9,
                "evidence_ids": ["L1", "bad"],
                "justification": "ok",
            },
            [{"doc_id": "chunk_1", "content": "Điều 1", "metadata": {}}],
        )
        self.assertEqual(verified["evidence_ids"], ["L1"])
        self.assertEqual(verified["verifier_status"], "ok")

    def test_check_runtime_reports_missing_env_without_network(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            chunks = root / "chunks.jsonl"
            public = root / "public.json"
            prompt = root / "prompt.md"
            for path in (chunks, public, prompt):
                path.write_text("[]", encoding="utf-8")
            config = PipelineConfig(
                chunks_path=chunks,
                public_test_path=public,
                prompt_path=prompt,
                index_path=root / "index",
            )
            with patch.dict("os.environ", {"ALQAC_TEAM_TOKEN": "replace_with_team_token"}, clear=False):
                report = check_runtime(config, ping_llm=False)
        self.assertFalse(report["ready_for_real_run"])
        self.assertFalse(report["env"]["ALQAC_TEAM_TOKEN"])

    def test_validate_prediction_output_detects_dry_run(self) -> None:
        with workspace_tempdir() as tmp:
            pred_path = Path(tmp) / "pred.json"
            pred_path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "case_1",
                            "prediction": "A_WIN",
                            "confidence": 0.5,
                            "justification": "[dry-run] placeholder",
                            "parser_status": "ok",
                            "law_evidence": [{"law_id": "L", "aid": 1}],
                            "case_evidence": ["c1"],
                            "api_calls": 1,
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            report = validate_prediction_output(pred_path)
        self.assertTrue(report["ready_for_submission_shape"])
        self.assertFalse(report["real_reasoning_ready"])
        self.assertEqual(report["dry_run_cases"], ["case_1"])

    def test_build_submission_json_cleans_schema_and_preserves_public_order(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            pred_path = root / "pred.json"
            public_path = root / "public.json"
            output_path = root / "submission.json"

            public_path.write_text(
                json.dumps(
                    [{"case_id": "case_2"}, {"case_id": "case_1"}],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            pred_path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "case_1",
                            "prediction": "A_WIN",
                            "case_evidence": [" chunk_a ", "chunk_a", ""],
                            "law_evidence": [
                                {"law_id": " 91/2015/QH13 ", "aid": "584"},
                                {"law_id": "91/2015/QH13", "aid": 584},
                                {"bad": "row"},
                            ],
                            "api_calls": 3,
                        },
                        {
                            "case_id": "case_2",
                            "prediction": "B_WIN",
                            "case_evidence": "chunk_b",
                            "law_evidence": [],
                            "confidence": 0.4,
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            build_submission_json(pred_path, output_path, public_test_path=public_path)
            submission = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual([item["case_id"] for item in submission], ["case_2", "case_1"])
        self.assertEqual(set(submission[0]), {"case_id", "prediction", "case_evidence", "law_evidence"})
        self.assertEqual(submission[0]["case_evidence"], ["chunk_b"])
        self.assertEqual(submission[1]["case_evidence"], ["chunk_a"])
        self.assertEqual(submission[1]["law_evidence"], [{"law_id": "91/2015/QH13", "aid": 584}])


class EvaluationTests(unittest.TestCase):
    def test_alqac_metrics_handles_exact_predictions(self) -> None:
        gold = [
            {
                "case_id": "case_1",
                "prediction": "A_WIN",
                "case_evidence": ["e1"],
                "law_evidence": [{"law_id": "L", "aid": 1}],
            }
        ]
        pred = [
            {
                "case_id": "case_1",
                "prediction": "A_WIN",
                "case_evidence": ["e1"],
                "law_evidence": [{"law_id": "L", "aid": 1}],
                "api_calls": 1,
            }
        ]
        report, cm = evaluate_alqac_system(gold, pred)
        self.assertEqual(report["ALQAC_Final_Score"], 1.0)
        self.assertIn("A_WIN", str(cm))

    def test_retrieval_benchmark_maps_public_law_labels_to_topk_metrics(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            public_path = root / "public.json"
            chunks_path = root / "chunks.jsonl"
            public_path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "case_1",
                            "case_query": "boi thuong thiet hai va an phi",
                            "related_law_provisions": (
                                "Bộ luật Dân sự năm 2015 | Điều 584\n"
                                "Bộ luật Tố tụng dân sự | Khoản 1 Điều 157\n"
                            ),
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            chunks_path.write_text(
                "\n".join(
                    json.dumps(row, ensure_ascii=False)
                    for row in [
                        {
                            "chunk_id": "chunk_civil",
                            "law_id": "91/2015/QH13",
                            "aid": 53353,
                            "article_number": "53353",
                            "article_index": 584,
                        },
                        {
                            "chunk_id": "chunk_proc",
                            "law_id": "92/2015/QH13",
                            "aid": 50822,
                            "article_number": "50822",
                            "article_index": 157,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            gold = load_public_test_gold(public_path)
            normalizer = ProvisionNormalizer(chunks_path)
            predicted = [
                {"metadata": {"chunk_id": "chunk_civil"}},
                {"metadata": {"chunk_id": "chunk_proc"}},
            ]

        gold_keys = {item.key for item in gold["case_1"] if item.key}
        predicted_keys = [normalizer.candidate_key(item) for item in predicted]
        self.assertEqual(
            gold_keys,
            {provision_key("91/2015/QH13", 584), provision_key("92/2015/QH13", 157)},
        )
        self.assertEqual(
            predicted_keys,
            [provision_key("91/2015/QH13", 584), provision_key("92/2015/QH13", 157)],
        )
        self.assertEqual(extract_article_numbers("Dieu 584, Dieu 585"), ["584", "585"])
        self.assertTrue(is_graph_candidate({"metadata": {"graph_path": ["seed", "rule:law:584"]}}))
        self.assertFalse(is_graph_candidate({"metadata": {"chunk_id": "chunk_civil"}}))
        self.assertEqual(per_case_metrics(gold_keys, predicted_keys, 1)["recall"], 0.5)
        self.assertEqual(per_case_metrics(gold_keys, predicted_keys, 2)["full_recall"], 1.0)

    def test_retrieval_benchmark_separates_legacy_laws_missing_from_index(self) -> None:
        with workspace_tempdir() as tmp:
            chunks_path = Path(tmp) / "chunks.jsonl"
            chunks_path.write_text(
                json.dumps(
                    {
                        "chunk_id": "chunk_current",
                        "law_id": "91/2015/QH13",
                        "aid": 53354,
                        "article_number": "53354",
                        "article_index": 584,
                    }
                ),
                encoding="utf-8",
            )
            normalizer = ProvisionNormalizer(chunks_path)

        self.assertEqual(canonical_law_code("Bo luat Dan su nam 2005"), "33/2005/QH11")
        legacy = GoldProvision(
            case_id="case_legacy",
            law_title="Bo luat Dan su nam 2005",
            article_number="604",
            law_code="33/2005/QH11",
            source="test",
        )
        self.assertIsNone(normalizer.gold_key(legacy))
        self.assertEqual(normalizer.gold_mapping_status(legacy), "legacy_law_not_in_index")

    def test_retrieval_benchmark_maps_resolution_code_alias(self) -> None:
        with workspace_tempdir() as tmp:
            chunks_path = Path(tmp) / "chunks.jsonl"
            chunks_path.write_text(
                json.dumps(
                    {
                        "chunk_id": "chunk_fee",
                        "law_id": "326/2016/UBTVQH14",
                        "aid": 1001,
                        "article_number": "1001",
                        "article_index": 26,
                    }
                ),
                encoding="utf-8",
            )
            normalizer = ProvisionNormalizer(chunks_path)

        self.assertEqual(
            canonical_law_code("Nghị quyết số 326/2016/QH14"),
            "326/2016/UBTVQH14",
        )
        gold = GoldProvision(
            case_id="case_fee",
            law_title="Nghị quyết số 326/2016/QH14",
            article_number="26",
            law_code="326/2016/QH14",
            source="test",
        )
        self.assertEqual(normalizer.gold_key(gold), "326/2016/UBTVQH14:26")
        self.assertEqual(normalizer.gold_mapping_status(gold), "mapped")

    def test_graph_benchmark_scores_only_neo4j_traversal_candidates(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            public_path = root / "public.json"
            chunks_path = root / "chunks.jsonl"
            public_path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "case_1",
                            "case_query": "boi thuong thiet hai",
                            "related_law_provisions": "Bo luat Dan su nam 2015 | Dieu 584",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            chunks_path.write_text(
                json.dumps(
                    {
                        "chunk_id": "graph_chunk",
                        "law_id": "91/2015/QH13",
                        "aid": 53354,
                        "article_number": "53354",
                        "article_index": 584,
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                public_test=public_path,
                gold=None,
                chunks=chunks_path,
                index=root / "index",
                retriever="graph",
                top_k=[1],
                seed_top_k=5,
                alpha=0.5,
                include_flat_fallback=False,
                score_scope="graph_traversal_only",
                limit=None,
                rebuild_index=False,
            )

            def retrieve(_: str) -> list[dict[str, object]]:
                return [
                    {"metadata": {"chunk_id": "flat_fallback"}},
                    {
                        "metadata": {"chunk_id": "graph_chunk"},
                        "graph_path": ["chunk:seed", "rule:91/2015/QH13:53354"],
                    },
                ]

            with patch(
                "evaluation.retrieval_benchmark.build_retriever",
                return_value=(retrieve, {"backend": "neo4j", "legal_node_count": 2}, lambda: None),
            ):
                report = run_benchmark(args)

        self.assertEqual(report["Retrieval_Benchmark"]["score_scope"], "neo4j_traversal_only")
        self.assertEqual(report["Graph_Execution"]["cases_with_neo4j_expansion"], 1)
        self.assertEqual(report["Metrics"]["@1"]["Case_Hit_Accuracy"], 1.0)
        self.assertEqual(report["Case_Rows"][0]["flat_fallback_candidate_count"], 1)

    def test_graph_benchmark_reports_end_to_end_and_traversal_metrics(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            public_path = root / "public.json"
            chunks_path = root / "chunks.jsonl"
            public_path.write_text(
                json.dumps([{"case_id": "case_1", "case_query": "boi thuong", "related_law_provisions": "Bo luat Dan su nam 2015 | Dieu 584"}]),
                encoding="utf-8",
            )
            chunks_path.write_text(json.dumps({"chunk_id": "seed_chunk", "law_id": "91/2015/QH13", "article_index": 584}), encoding="utf-8")
            args = argparse.Namespace(public_test=public_path, gold=None, chunks=chunks_path, index=root / "index", retriever="graph", top_k=[1], seed_top_k=5, alpha=0.5, include_flat_fallback=False, score_scope="end_to_end", limit=None, rebuild_index=False)
            with patch("evaluation.retrieval_benchmark.build_retriever", return_value=(lambda _: [{"metadata": {"chunk_id": "seed_chunk"}}], {"backend": "neo4j"}, lambda: None)):
                report = run_benchmark(args)

        self.assertEqual(report["Retrieval_Benchmark"]["score_scope"], "full_retrieval_pipeline")
        self.assertEqual(report["Scope_Metrics"]["end_to_end"]["@1"]["Case_Hit_Accuracy"], 1.0)
        self.assertEqual(report["Scope_Metrics"]["neo4j_traversal_only"]["@1"]["Case_Hit_Accuracy"], 0.0)


class PipelineDryRunTests(unittest.TestCase):
    def test_pipeline_dry_run_processes_one_case(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "index"
            public_test_path = root / "public_test.json"
            prompt_path = root / "prompt.md"
            gold_path = root / "gold.json"
            experiments_dir = root / "experiments"

            chunks_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "chunk_id": "law_1",
                                "text": "Điều 1. Chủ sở hữu phải bồi thường thiệt hại do tài sản gây ra.",
                                "law_id": "91/2015/QH13",
                                "aid": 584,
                                "unit_path": "91/2015/QH13 Điều 584",
                            },
                            ensure_ascii=False,
                        )
                    ]
                ),
                encoding="utf-8",
            )
            public_test_path.write_text(
                json.dumps(
                    [{"case_id": "case_1", "case_query": "Ai phải bồi thường thiệt hại tài sản?"}],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            prompt_path.write_text(
                "{{case_id}}\n{{case_query}}\n{{related_law_provisions}}\n{{evidence_blocks}}",
                encoding="utf-8",
            )
            gold_path.write_text("[]", encoding="utf-8")

            config = PipelineConfig(
                chunks_path=chunks_path,
                index_path=index_path,
                public_test_path=public_test_path,
                prompt_path=prompt_path,
                gold_path=gold_path,
                experiments_dir=experiments_dir,
                run_tag="test",
                top_k_before_rerank=5,
                top_k_after_rerank=3,
            )
            output_path = run_pipeline(config, dry_run=True, limit=1, force_rebuild_index=True)
            predictions = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["case_id"], "case_1")
        self.assertIn(predictions[0]["prediction"], {"A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"})
        self.assertEqual(predictions[0]["api_calls"], 1)
        self.assertEqual(len(predictions[0]["law_evidence"]), len({(e["law_id"], e["aid"]) for e in predictions[0]["law_evidence"]}))


if __name__ == "__main__":
    unittest.main()
