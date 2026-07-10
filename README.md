# alqac-2026-rag-hak
Heading to ALQAC 2026

## Phase 1 Components

This repo contains the baseline Agentic RAG pipeline described in `Plan.md`:

- corpus preparation: `prepare_corpus.py`
- legal corpus cleaning and legal-unit chunking: `retrieval/cleaning.py`
- deprecated-law filtering: `retrieval/deprecated_filter.py`
- hybrid BM25 + dense baseline indexing/search: `retrieval/indexing.py`
- query analysis and document routing: `retrieval/router.py`
- deterministic reranking: `retrieval/reranker.py`
- citation usefulness filtering: `retrieval/citation_usefulness.py`
- orchestration CLI: `orchestration/run_pipeline.py`
- local metrics: `evaluation/metrics.py`

## Run Individual Steps

```powershell
python prepare_corpus.py --input data/raw/ALQAC/corpus_law_pub.json --output data/corpus.jsonl
python retrieval/cleaning.py
python -m orchestration.run_pipeline build-index
python -m orchestration.run_pipeline run --dry-run --limit 3 --rebuild-index
python -m orchestration.run_pipeline check-runtime
python -m orchestration.run_pipeline check-runtime --ping-llm
python -m orchestration.run_pipeline export-submission
```

For real inference, create a local `.env` from `env.example`, set the ALQAC
token, and run a local Ollama model:

```powershell
ollama pull qwen2.5:7b-instruct
python -m orchestration.run_pipeline check-runtime --ping-llm
```

Then run without `--dry-run`:

```powershell
python -m orchestration.run_pipeline run --limit 1 --rebuild-index
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
