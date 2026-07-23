# Retrieval

Four retrieval sources. Two disabled by default.

## Sources

| Source | Tool | Default | Citable? | Output |
|---|---|---|---|---|
| Local vector RAG | `LawGraphSearchTool` | always on | Yes (law_evidence) | `list[LawHit]` |
| Local graph expansion | `LawGraph.expand()` | always on | Yes (law_evidence) | 1-hop neighbors |
| Official Case API | `OfficialCaseTop1Tool` | OFF | Yes (case_evidence) | `OfficialCaseHit` |
| Public Case Search | `PublicCaseSearchTool` | OFF | No (reasoning only) | `list[PublicCaseHit]` |

## Local RAG pipeline (`app/rag.py`)

```
query → Qdrant top-k (LAW_RAG_TOP_K) → 1-hop graph expansion (GRAPH_MAX_HOPS) → law_evidence
```

- `LawRAG.search(query, element_graph)` — vector similarity
- `LawGraph.expand(seed_ids)` — JSONL adjacency lookup
- Results filtered by element graph enrichment

## Official API (`app/tools.py`)

- Always top-1 per call
- Shared batch budget via `OfficialCallLedger`
- Returns `chunk_id` + text
- `chunk_id` added to `state.official_chunk_ids` allowlist

## Public search (`app/tools.py`)

- Returns `source_id` + text + score
- `source_id` never enters `case_evidence` — reasoning only
- Disabled by default (`PUBLIC_CASE_RETRIEVAL_ENABLED=false`)

## Index build (`scripts/build_legal_index.py`)

Atomic rebuild using Qdrant aliases:
1. Build into temp collection
2. Paginated copy into versioned collection
3. Validate point count
4. Alias swap (atomic)
5. Delete old collection

- Supports flat article records plus nested `articles` and `content` law documents.
- Preserves `law_id` and `aid`; `content_Article` becomes searchable article text.
- Embedding requests retry transient connection, timeout, rate-limit, and 5xx errors.
- A rerun resumes deterministic temporary collections and skips already uploaded points.
- Progress writes to `stderr`; final index statistics remain JSON on `stdout`.
- Alias readiness resolves `law_articles` with `get_collection`; vector retrieval uses Qdrant `query_points`.

Current private-test corpus: `data/legal_corpus/private_test_60_cases_extracted_corpus.json`, 2,820 articles across 14 laws.

```bash
python scripts/build_legal_index.py
```

## Related

- [[Provenance]] — how retrieved evidence is validated
- [[OfficialCallLedger]] — budget for official calls
- [[Configuration]] — all retrieval env vars
