# System Design — ALQAC 2026 Agent Skeleton

## Component matrix

| Component | Responsibility | Model calls | Tools / data | Max |
|---|---|---|---|---|
| Element Agent | Paper Table 8 element graph | 1 | none | 1/case |
| Draft Agent | Four-label ALQAC draft / revise | 1 per draft/revision | state evidence only | loop-bounded |
| Manager Agent | Route: public / official / format / law / pass | 1 per iteration | no network | ≤5 iterations |
| Public Case Retrieval | Raw public judgment text | 0 LLM | HTTP once/action | Manager-gated; default OFF |
| Official Case API | Citable top-1 chunk | 0 LLM | HTTP top-1 | ≤`2n` batch; default OFF |
| Format Check | Editing suggestions only | 1 | none | Manager-gated |
| Law Search | Local law evidence | 0 LLM (retrieval) | Qdrant top-k + 1-hop graph | Manager-gated |
| Content Check | pass/fail support gate | 1 | state only | 1/case end |
| Deterministic validator | Provenance + serialize | 0 | allowlists | always |

All six LLM roles share one factory: `OPENAI_BASE_URL` + `OPENAI_API_KEY` + `OPENAI_MODEL`.

## Tool contracts

| Tool | Signature | Output may enter |
|---|---|---|
| `public_case_search` | `(query) -> [{source_id, text, score}]` | non-citable public context only |
| `official_case_top1` | `(query) -> {chunk_id, text, score?}` | official hits + case `chunk_id` allowlist |
| `law_graph_search` | `(query, element_graph) -> [{law_id, aid, text, vector_score, graph_hops}]` | law hits + `(law_id, aid)` allowlist |

No general-purpose tools. Agents never call each other.

## State (`AlqacState`)

- Input: `case_id`, `case_query`
- `element_graph`, `draft`, `manager_decision`
- `public_context` (non-citable), `official_hits`, `law_hits`, `format_suggestions`
- `content_result` (`pass`|`fail` + findings)
- `iteration` (0..5), shared `OfficialCallLedger`
- Per-case allowlists: `official_chunk_ids`, `law_pairs`
- `revision_history`, `trace_id`, `validation_errors`

## Manager routing rules

1. Disabled actions in Manager output → structured reject/retry; never silently enable.
2. When Format + Law both selected: Format Check → apply revision → Law Search → integrate.
3. Stop on `Pass` or after five iterations → Content Check.
4. Content Check `fail` → `reject_case` (no serializer).
5. Content Check `pass` → deterministic validator → optional `submission.json` write.

## Budgets

| Resource | Limit | Enforcement |
|---|---|---|
| Manager iterations | 5 | workflow loop |
| Official API calls | `OFFICIAL_CALL_BUDGET_MULTIPLIER * n` (default `2n`) | `OfficialCallLedger` |
| Official top-k | always 1 | tool adapter |
| Official no-gain | `OFFICIAL_NO_GAIN_LIMIT` (default 1) | ledger / tool |
| Law vector seeds | `LAW_RAG_TOP_K` (default 3) | RAG |
| Graph hops | `GRAPH_MAX_HOPS` (default 1) | RAG |

## Prompt sources

Paper `docs/raw/2604.10470v1.pdf` Table 8 (Appendix A.3). Role intent preserved; ALQAC machine contracts appended (JSON schema, no-ID-invention, four-label prediction). Content Check adapted from paper rewrite role to pass/fail support gate so deterministic serializer owns ALQAC output.

## Observability

Every agent/tool/validator step:

- structured log event (`agent`, `case_id`, `OPENAI_MODEL`, route, elapsed_ms, error)
- Langfuse span (prompt version, hashes, redacted previews)

Trace root: request ID, flags, model, batch size, official used/max, status. Never log secrets or full legal documents.

## API

| Endpoint | Behavior |
|---|---|
| `GET /health` | readiness without secrets |
| `POST /v1/submission` | ordered batch; shared `2n` ledger; per-case result/errors |
| `POST /v1/cases/{case_id}/debug` | single case; redacted state; no artifact write |

## Config validation (startup)

Fail if missing chat endpoint/key/model; if enabled retrieval agent lacks credentials; if `LAW_RAG_TOP_K`/`GRAPH_MAX_HOPS` invalid; if official budget multiplier outside `0..2`.
