# ALQAC 2026 Agent Skeleton

Multi-agent legal reasoning pipeline for ALQAC 2026, adapted from JurisMA (paper `docs/raw/2604.10470v1.pdf`) with ALQAC provenance, budget, and output gates.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# fill OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
```

Defaults:

- `PUBLIC_CASE_RETRIEVAL_ENABLED=false`
- `OFFICIAL_API_ENABLED=false`
- All LLM roles use single `OPENAI_MODEL`

## Build law index

Place permitted ALQAC law JSON files under `data/legal_corpus/`, then:

```bash
python scripts/build_legal_index.py
```

Writes Qdrant storage to `data/vector/` and graph JSONL to `data/graph/`.

## Public test input

Default path (data only; do not import sibling agents code):

`/home/nesfan/Desktop/HCMUS/Nam3/Legal_idk/alqac-2026-rag/data/ALQAC2026_public_test.json`

Fields used: `case_id`, `case_query` (50 cases).

## Run API

```bash
conda activate ml-env
uvicorn app.main:app --reload --port 8000
```

- `GET /health`
- `POST /v1/submission` — batch cases `{ "cases": [{"case_id","case_query"}] }`
- `POST /v1/submission/from_public_test` — load public test (`limit`, `case_ids` optional)
- `POST /v1/cases/{case_id}/debug` — single-case redacted trace

## Tests

```bash
pytest
```

## Architecture

See `docs/Architecture.md` and `docs/SystemDesign.md`.

## Note on official schema

Four-label ALQAC field names/values are provisional until the published competition schema is supplied. Serializer and validator enforce provenance allowlists regardless.
