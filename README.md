# alqac-2026-rag-hak
Heading to ALQAC 2026

## Phase 1 Components

This repo contains the baseline Agentic RAG pipeline described in `Plan.md`:

- corpus preparation: `prepare_corpus.py`
- legal corpus cleaning and legal-unit chunking: `retrieval/cleaning.py`
- deprecated-law filtering: `retrieval/deprecated_filter.py`
- hybrid BM25 + dense baseline indexing/search: `retrieval/indexing.py`
- query analysis and document routing: `retrieval/router.py`
- Neo4j-ready graph construction: `graph_construct/`
- graph-aware retrieval: `retrieval/graph_retriever.py`
- deterministic reranking: `retrieval/reranker.py`
- citation usefulness filtering: `retrieval/citation_usefulness.py`
- retrieval benchmark: `evaluation/retrieval_benchmark.py`
- orchestration CLI: `orchestration/run_pipeline.py`
- local metrics: `evaluation/metrics.py`

## Run Individual Steps

```powershell
python prepare_corpus.py --input data/raw/ALQAC/corpus_law_pub.json --output data/corpus.jsonl
python retrieval/cleaning.py
python -m orchestration.run_pipeline build-index
python -m orchestration.run_pipeline build-graph
python -m orchestration.run_pipeline run --dry-run --limit 3 --rebuild-index
python -m evaluation.retrieval_benchmark --limit 3 --top-k 1,3,5
python -m orchestration.run_pipeline check-runtime
python -m orchestration.run_pipeline check-runtime --ping-llm
python -m orchestration.run_pipeline export-submission
```

To use Neo4j AuraDB Free, copy the Aura Connection URI into `NEO4J_URI`
(`neo4j+s://...`), then fill its username and password in `.env`. Install the
Python driver, verify Aura connectivity, and import the graph in small batches:

```powershell
pip install neo4j
python -m orchestration.run_pipeline check-runtime --graph-only --ping-neo4j
python -m orchestration.run_pipeline build-graph --import-neo4j --clear-graph --batch-size 200
python -m evaluation.retrieval_benchmark --top-k 1,3,5,8
```

The retrieval benchmark defaults to Neo4j graph traversal and compares returned
`(law_id, article)` provisions against `related_law_provisions` in the public
test. It reports graph-expansion coverage separately; use
`--include-flat-fallback` only when evaluating the complete fallback-capable pipeline.

For real inference, create a local `.env` from `env.example`, set the ALQAC
token, configure one open-weight SLM under 10B parameters, and expose it through
an API:

```powershell
# Ollama native /api/chat
ollama pull qwen2.5:7b-instruct
$env:ALQAC_LLM_PROVIDER="ollama"
$env:ALQAC_LLM_BASE_URL="http://localhost:11434"

# or OpenAI-compatible /v1/chat/completions from vLLM, LM Studio, llama.cpp, etc.
$env:ALQAC_LLM_PROVIDER="openai-compatible"
$env:ALQAC_LLM_BASE_URL="http://localhost:8000"
```

Then run the runtime check and a small real batch:

```powershell
python -m orchestration.run_pipeline check-runtime --ping-llm
python -m orchestration.run_pipeline run --limit 1 --rebuild-index --require-graph
```

See `REAL_RUN.md` for the full preflight and validation sequence.

## Tests

The component tests are offline and do not download embedding or LLM models.

```powershell
python -m unittest discover -s tests -v
```

The tests cover corpus normalization, legal chunking, deprecated filtering,
hybrid retrieval, routing/reranking/citation filtering, metrics, and pipeline
dry-run wiring.
