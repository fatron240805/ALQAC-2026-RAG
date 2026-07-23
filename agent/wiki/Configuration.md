# Configuration

All via env vars (`.env`). Pydantic Settings with validation.

**Location**: `app/config.py`

## Required

| Var | Default | Purpose |
|---|---|---|
| `OPENAI_BASE_URL` | — | LLM endpoint |
| `OPENAI_API_KEY` | — | LLM key |
| `OPENAI_MODEL` | — | Single model for all roles |

## Security

| Var | Default | Purpose |
|---|---|---|
| `API_KEY` | (empty) | Service auth; empty = no auth |
| `MAX_BATCH_SIZE` | 50 | Max cases per request |
| `RATE_LIMIT_RPM` | 60 | Requests/min per principal |

## Retrieval

| Var | Default | Purpose |
|---|---|---|
| `QDRANT_PATH` | data/vector | Qdrant storage |
| `LAW_CORPUS_PATH` | data/legal_corpus/cleaned | Source law files |
| `GRAPH_NODES_PATH` | data/graph/graph/nodes.jsonl | Graph nodes |
| `GRAPH_EDGES_PATH` | data/graph/graph/edges.jsonl | Graph edges |
| `LAW_RAG_TOP_K` | 3 | Vector search top-k |
| `GRAPH_MAX_HOPS` | 1 | Graph expansion hops |

## External APIs

| Var | Default | Purpose |
|---|---|---|
| `PUBLIC_CASE_RETRIEVAL_ENABLED` | false | Enable public case search |
| `PUBLIC_CASE_RETRIEVAL_URL` | — | Public search endpoint |
| `PUBLIC_CASE_RETRIEVAL_API_KEY` | — | Public search key |
| `OFFICIAL_API_ENABLED` | false | Enable official ALQAC API |
| `OFFICIAL_API_URL` | — | Official endpoint |
| `OFFICIAL_API_KEY` | — | Official key |
| `OFFICIAL_CALL_BUDGET_MULTIPLIER` | 2.0 | Max calls = multiplier × n |
| `OFFICIAL_NO_GAIN_LIMIT` | 1 | Consecutive no-gain stop |

## Output

| Var | Default | Purpose |
|---|---|---|
| `SUBMISSION_OUTPUT_PATH` | artifacts/submission.json | Output file |
| `PUBLIC_TEST_PATH` | data/ALQAC2026_public_test.json | Public test input |

## Observability

| Var | Default | Purpose |
|---|---|---|
| `LANGFUSE_HOST` | — | Langfuse endpoint |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | — | Langfuse secret key |
| `LANGFUSE_ENVIRONMENT` | development | Environment tag |

## Debug Logging

Raw agent prompts are logged to `artifacts/logs/agent-prompts-YYYYMMDDTHHMMSS±ZZZZ.log`.

- One file per process, created on first invocation
- Contains DEBUG-level entries with timestamp, role, case_id, trace_id, model
- Exact system + user prompt text, no truncation or redaction
- `artifacts/` is gitignored

All observability paths (console logs, Langfuse metadata/input/output, debug-state endpoint) output raw unredacted values.

## Validators

Startup validation fails fast if:
- Missing chat config (base_url, api_key, model)
- Enabled retrieval agent lacks credentials
- `LAW_RAG_TOP_K` outside 1..50
- `GRAPH_MAX_HOPS` outside 0..3
- `OFFICIAL_CALL_BUDGET_MULTIPLIER` outside 0..2
