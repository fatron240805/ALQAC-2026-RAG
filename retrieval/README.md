# Retrieval Package

`RetrievalService` is the public boundary between retrieval and the next RAG
stage. It hides whether the backend is flat hybrid retrieval or Neo4j graph
retrieval and returns the same evidence-chain contract in both modes.

```python
from retrieval import RetrievalService, RetrievalServiceConfig

service = RetrievalService(
    indexer,
    graph_retriever=graph_retriever,  # optional
    config=RetrievalServiceConfig(seed_top_k=80, final_top_k=8),
)

result = service.retrieve(case_query)
reasoning_payload = result.to_reasoning_payload(case_id=case_id)
```

The reasoning/auditing stage can consume:

- `result.evidence`: backward-compatible candidate dictionaries
- `result.chains`: normalized `EvidenceChain` objects
- `result.law_evidence`: deduplicated `(law_id, aid)` references
- `result.trace`: backend, reranker, graph expansion, latency, and drop counts
- `result.to_reasoning_payload(...)`: JSON-ready payload with `evidence_chains`

Every chain carries content, source chunk ids, legal references, graph path,
retrieval provenance, score, citation judgment, and rank. Candidates without a
mapped legal provision are dropped by default; set
`require_law_evidence=False` only for a non-legal retrieval use case.
