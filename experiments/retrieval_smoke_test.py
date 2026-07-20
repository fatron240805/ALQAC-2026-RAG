r"""Small retrieval smoke test using the first public test query.

Run from the project root:

    .venv\Scripts\python.exe experiments\retrieval_smoke_test.py

Use ``--flat`` to skip Neo4j and inspect only the local hybrid index.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.config import PipelineConfig
from orchestration.data_adapters import load_test_cases
from orchestration.run_pipeline import build_graph_retriever, load_or_build_index
from retrieval.citation_usefulness import HeuristicCitationUsefulnessFilter
from retrieval.reranker import LexicalOverlapReranker
from retrieval.service import RetrievalService, RetrievalServiceConfig


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _first_public_case(path: Path) -> dict[str, Any]:
    try:
        return next(load_test_cases(path))
    except StopIteration as exc:
        raise RuntimeError(f"Public test file has no cases: {path}") from exc


def _print_chain(rank: int, chain: Any, content_chars: int) -> None:
    print(f"\n--- DOCUMENT {rank} ---")
    print(f"chain_id:       {chain.chain_id}")
    print(f"evidence_id:    {chain.evidence_id}")
    print(f"doc_id:         {chain.doc_id}")
    print(f"score:          {chain.score:.6f}")
    print(f"rerank_score:   {chain.rerank_score:.6f}")
    print(f"law_evidence:   {json.dumps(chain.law_evidence, ensure_ascii=False)}")
    print(f"source_chunks:  {json.dumps(chain.source_chunks, ensure_ascii=False)}")
    print(f"graph_path:     {' -> '.join(chain.graph_path) or '(none)'}")
    print(f"provenance:     {json.dumps(chain.provenance, ensure_ascii=False, default=str)}")
    print(f"metadata:       {json.dumps(chain.metadata, ensure_ascii=False, default=str)}")
    if chain.citation_judgment:
        print(
            "citation:       "
            f"{json.dumps(chain.citation_judgment, ensure_ascii=False, default=str)}"
        )
    content = " ".join(chain.content.split())
    if len(content) > content_chars:
        content = f"{content[:content_chars]}..."
    print(f"content:        {content}")


def main() -> int:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Smoke test the ALQAC retrieval module")
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence documents to print")
    parser.add_argument(
        "--content-chars",
        type=int,
        default=900,
        help="Maximum content characters printed per document",
    )
    parser.add_argument("--flat", action="store_true", help="Skip Neo4j and use flat hybrid retrieval")
    parser.add_argument(
        "--require-graph",
        action="store_true",
        help="Fail if Neo4j is unavailable instead of falling back to flat retrieval",
    )
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the local retrieval index")
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.content_chars <= 0:
        parser.error("--content-chars must be positive")

    config = PipelineConfig()
    config = replace(
        config,
        top_k_before_rerank=max(config.top_k_before_rerank, args.top_k),
        top_k_after_rerank=args.top_k,
    )
    first_case = _first_public_case(config.public_test_path)
    query = str(first_case.get("case_query") or "").strip()
    if not query:
        raise RuntimeError(f"First public case has an empty query: {first_case}")

    setup_started = time.perf_counter()
    indexer = load_or_build_index(config, force_rebuild=args.rebuild_index)
    graph_retriever = None
    if not args.flat:
        graph_retriever = build_graph_retriever(
            config,
            indexer,
            require_graph=args.require_graph,
        )

    flat_reranker = LexicalOverlapReranker(
        use_gpu_reranker=config.use_gpu_reranker,
        reranker_model_name=config.reranker_model_name,
        device=config.reranker_device,
        batch_size=config.reranker_batch_size,
    )
    service = RetrievalService(
        indexer,
        graph_retriever=graph_retriever,
        reranker=flat_reranker,
        citation_filter=HeuristicCitationUsefulnessFilter(max_results=args.top_k),
        config=RetrievalServiceConfig(
            seed_top_k=config.top_k_before_rerank,
            final_top_k=args.top_k,
            hybrid_alpha=config.hybrid_alpha,
            require_law_evidence=False,
        ),
    )
    setup_ms = (time.perf_counter() - setup_started) * 1000.0

    result = service.retrieve(query, top_k=args.top_k)
    trace = result.trace.as_dict()

    print("=== RETRIEVAL SMOKE TEST ===")
    print(f"case_id:        {first_case.get('case_id')}")
    print(f"query:          {query}")
    print(f"backend:        {trace.get('backend')}")
    print(f"reranker:       {trace.get('reranker')}")
    print(f"index_docs:     {len(indexer.doc_ids)}")
    print(f"setup_ms:       {setup_ms:.3f}")
    print(f"retrieval_ms:   {trace.get('latency_ms', 0.0):.3f}")
    print(f"returned_count: {trace.get('returned_count')}")
    print(f"trace:          {json.dumps(trace, ensure_ascii=False, default=str)}")

    for rank, chain in enumerate(result.chains, start=1):
        _print_chain(rank, chain, args.content_chars)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
