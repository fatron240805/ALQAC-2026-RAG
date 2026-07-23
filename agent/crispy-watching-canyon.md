# Context

Repository contains only `docs/Architecture.md` and paper `docs/raw/2604.10470v1.pdf`. Build an ALQAC 2026 system skeleton in this repository, not from `../alqac-2026-rag`. It must retain ALQAC provenance, routing, output, and API-efficiency rules while making Public Case Retrieval and Official Case API agents runtime-toggleable and initially off. All LLM roles use one OpenAI-compatible chat model selected by `OPENAI_MODEL` in `.env`.

## Recommended architecture

Implement one request-scoped LangGraph workflow behind FastAPI:

```text
case_id + case_query
  -> Element Agent
  -> Draft Agent
  -> Manager Agent
       -> [optional Public Case Retrieval]
       -> [optional Official Case API, one top-1 result/call]
       -> [Format Check, then Law Search if both requested]
  -> Draft revision
  -> Manager loop (Pass or five iterations)
  -> Content Check pass/fail gate
       -> pass: deterministic validator -> submission.json
       -> fail: structured per-case rejection
```

The Manager is sole router and official-call budget controller. Public and official retrieval are only tools selected by this Manager, never supervisors. All tool calls validate strict Pydantic input/output schemas before state mutation. Content Check cannot add, alter, or invent any identifier. It returns a Boolean support decision; a failed check is fail-closed and can never reach serialization. Validator projects approved state into output and checks every evidence ID against a per-case official/law provenance ledger; no LLM writes `submission.json`.

### ALQAC constraints implemented

- Input: ordered batch of `{case_id, case_query}`.
- Draft output: official ALQAC four-label prediction schema, not consultation prose.
- `case_evidence`: only `chunk_id` values returned by Official Case Content API.
- Public raw-judgment hits: query-expansion/reasoning only; impossible to copy into `case_evidence`.
- Law RAG evidence: only retrieved `{law_id, aid}` pairs.
- Official API: always top-1; shared batch cap `2 * n` calls. Never exceed full-efficiency range. Stop early on budget exhaustion, duplicate chunk ID, or configured one-result diminishing-gain limit.
- Manager looping: stops on `Pass` or after five iterations. When both actions selected, executes Format Check before Law Search.
- Both togglable agents default off: `PUBLIC_CASE_RETRIEVAL_ENABLED=false`, `OFFICIAL_API_ENABLED=false`.
- Per-case error reporting: any Content Check or deterministic-validation failure returns a structured case error and is blocked from serialization. Preserve input ordering in API response; final artifact behavior will follow published ALQAC schema once locally available.

## Tech stack

- Python 3.12, FastAPI, Uvicorn, Pydantic Settings.
- LangGraph for bounded workflow/state; Deep Agents (`deepagents`) for each constrained role runtime; `langchain-openai` as one shared `ChatOpenAI` model factory.
- Qdrant local-persistent mode as self-owned vector DB inside this repository.
- GraphRAG as repository-local `nodes.jsonl` / `edges.jsonl` adjacency traversal, one hop from vector seeds. Avoid a graph server until query volume proves it necessary.
- Langfuse for request trace, every node span, generation trace, tool ledger, routing data, and validation error diagnostics.
- Pytest + HTTPX. No queue, Redis, worker, general-purpose Deep Agents delegation, tool factory, graph DB service, or database migrations.

## Files to add

```text
.env.example                    # Endpoint/model names, flags, budgets, Langfuse config
.gitignore                      # .env, vector artifacts, inputs, generated submissions
pyproject.toml                  # Minimal runtime/test dependencies
README.md                       # Setup, index build, service/test commands
docs/Architecture.md            # Keep implementation paths/flow aligned with source architecture
docs/SystemDesign.md            # Component, state, tool, prompt, budget, trace matrices
app/__init__.py
app/config.py                   # Typed env config and startup validation
app/schemas.py                  # API/state/tool/ALQAC schema contracts
app/prompts.py                  # Paper-derived roles + ALQAC machine contracts
app/agents.py                   # One OpenAI model factory; six constrained Deep Agent role calls
app/rag.py                      # Qdrant retrieval, JSONL graph expansion, ID preservation
app/tools.py                    # Public, official, law tool adapters and contracts
app/workflow.py                 # LangGraph nodes, conditional edges, call ledger, max loop
app/observability.py            # Logger/Langfuse trace/span helpers and redaction
app/validator.py                # Provenance checks, per-case errors, deterministic serializer
app/main.py                     # FastAPI health/submission/debug endpoints
scripts/build_legal_index.py    # Local legal corpus ingestion to Qdrant + graph artifacts
data/legal_corpus/cleaned/.gitkeep
data/graph/graph/.gitkeep
tests/test_config.py
tests/test_prompts.py
tests/test_workflow.py
tests/test_provenance_budget.py
tests/test_rag.py
tests/test_api.py
```

## Configuration

Add `.env.example` with no secrets:

```dotenv
# Every LLM role uses this exact model/client configuration.
OPENAI_BASE_URL=
OPENAI_API_KEY=
OPENAI_MODEL=
LLM_TEMPERATURE=0

# RAG infrastructure, not an agent. Defaults to OPENAI_BASE_URL/API_KEY.
OPENAI_EMBEDDING_MODEL=
OPENAI_EMBEDDING_BASE_URL=
OPENAI_EMBEDDING_API_KEY=

QDRANT_PATH=data/vector
LAW_CORPUS_PATH=data/legal_corpus/cleaned
GRAPH_NODES_PATH=data/graph/graph/nodes.jsonl
GRAPH_EDGES_PATH=data/graph/graph/edges.jsonl
LAW_RAG_TOP_K=3
GRAPH_MAX_HOPS=1

# Toggled agents. Both initially disabled.
PUBLIC_CASE_RETRIEVAL_ENABLED=false
PUBLIC_CASE_RETRIEVAL_URL=
PUBLIC_CASE_RETRIEVAL_API_KEY=
OFFICIAL_API_ENABLED=false
OFFICIAL_API_URL=
OFFICIAL_API_KEY=
OFFICIAL_CALL_BUDGET_MULTIPLIER=2
OFFICIAL_NO_GAIN_LIMIT=1

LANGFUSE_HOST=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_ENVIRONMENT=development
SUBMISSION_OUTPUT_PATH=artifacts/submission.json
```

`Settings` fails at startup for a missing chat endpoint/key/model, an enabled retrieval agent without endpoint credentials, invalid `LAW_RAG_TOP_K` or graph-hop limits, or an official budget multiplier outside `0..2`. Add model identity as both configuration and every agent/log trace field.

## Components and exact responsibilities

| Component | Responsibility | Permitted calls/data |
|---|---|---|
| Element Agent | Produce paper Table 8 legal element graph. | One model call. No retrieval/tools. |
| Draft Agent | Produce/revise strict four-label candidate using state evidence. | One model call per draft/revision. Cannot create IDs. |
| Manager Agent | Produce validated `revise/actions` or `pass`; choose only needed routes. | One model call. Cannot access network or mutate evidence. |
| Public Case Retrieval Agent | Retrieve permitted unannotated public judgment text. | One configured HTTP call per Manager action; output enters non-citable context only. |
| Official Case API Agent | Get one citable official hit. | One configured HTTP call/action with explicit top-1; only source that appends `chunk_id`. |
| Format Check Agent | Return editing/JSON/identifier correction instructions without changing legal result. | One model call. No retrieval. |
| Law Search Agent | Return authoritative local law evidence. | One Qdrant seed query plus fixed one-hop graph expansion; `{law_id, aid}` only. |
| Content Check Agent | Return `{decision: pass|fail, findings}` for claim/evidence support. | One model call. Cannot rewrite/add evidence; `fail` routes only to per-case rejection. |
| Deterministic validator | Validate schema, provenance ledger, order, budget, and ID formats; serialize only Content-Check-passed output. | No model/network call. |

Implement tool contracts:

- `public_case_search(query) -> [{source_id, text, score}]`
- `official_case_top1(query) -> {chunk_id, text, score?}`
- `law_graph_search(query, element_graph) -> [{law_id, aid, text, vector_score, graph_hops}]`

Instantiate all six LLM roles with Deep Agents and exact same `ChatOpenAI` instance from `OPENAI_MODEL`; do not set a per-subagent `model` override. Use `StateBackend`, no `LocalShellBackend`, `memory=None`, `FilesystemMiddleware(..., tools=[])`, and `SubAgentMiddleware(..., subagents=[])`; this removes filesystem, shell, memory, and open-ended delegation. Do not configure web tools. Todo state, if Deep Agents retains its internal middleware, gets no external permission and is not part of tool contract. Expose only typed wrappers listed above to owning role through deny-by-default backend. LangGraph invokes fixed roles/nodes; no role can discover, spawn, or call another role directly.

## Prompt derivation and contracts

`app/prompts.py` records paper source citations and keeps role intent from PDF Table 8, Vietnamese translated:

- Element: professional element extractor; JSON graph fields `entities`, `events`, `relationships`, `user_claims`, `key_facts`, `legal_questions`.
- Draft: legal draft generator, adapted only to required ALQAC four-label JSON prediction.
- Manager: chooses Format Check for clarity/logic/redundancy, Law Search for absent statutory support, both in that order, otherwise Pass.
- Format Check: concrete suggestions without changing legal meaning.
- Law Search: authoritative provisions only, adapted to local vector-plus-graph legal corpus and source IDs.
- Content Check: check support and preserve meaning, adapted to report findings only because deterministic ALQAC output code owns final serialization.

Append model-enforced JSON schema and ALQAC provenance rules to every prompt. Tests will assert source role clauses, output schemas, route-order instruction, and no-invention clauses remain present.

## LangGraph state and flow

`AlqacState` contains typed input, element graph, draft, Manager decision, non-citable public context, official hits, law hits, format suggestions, content result, iteration, request-shared official-call ledger, per-case `official_chunk_id` and `(law_id, aid)` allowlists, revision history, trace ID, and validation errors.

Nodes:

1. `extract_elements`
2. `create_initial_draft`
3. `manager_route`
4. conditional tool nodes, in this enforced order: `retrieve_public_context`, `retrieve_official_evidence`, `format_check`, `apply_format_revision`, `law_search`, `integrate_law_revision`
5. `content_check` on `pass` or at five-iteration limit
6. conditional `reject_case` on Content Check `fail`, otherwise `validate_and_serialize`

Manager output gets revalidated after every revision. A manager result containing disabled action is rejected/retried as structured output, not silently enabled. Public/offical disabled runs never instantiate HTTP transport calls. An Official API result appends its `chunk_id` to the case allowlist; public result types have no evidence-ID field. The validator rejects any `case_evidence` not in that allowlist and any law pair outside its law allowlist. A request-local `OfficialCallLedger` holds `max_calls = 2 * n`; it serializes official retrieval decisions across batch cases.

## RAG data path

`build_legal_index.py` ingests only user-supplied permitted ALQAC law files under `data/legal_corpus/cleaned/`. Each chunk retains canonical `law_id` and `aid`; script upserts chunks into current-branch Qdrant storage then writes `data/graph/graph/nodes.jsonl` / `edges.jsonl`.

At run time, Law Search gets Top-3 Qdrant seeds, expands exactly one graph hop, deduplicates `(law_id, aid)`, and ranks stable by seed score then graph distance. It never reads, imports, or points to `../alqac-2026-rag`.

## API and observability

- `GET /health`: configuration readiness without secrets.
- `POST /v1/submission`: full ordered case batch, shared `2n` ledger, per-case result/errors, deterministic artifact only for cases valid under final official schema.
- `POST /v1/cases/{case_id}/debug`: one case, redacted state/route/ledger/trace ID; never writes final artifact.

Create a structured log event and Langfuse span for each agent/tool/validator step. Trace root contains request ID, ALQAC flags, `OPENAI_MODEL`, batch size, configured/max/used official calls, and output status. Node spans capture prompt version, route action, elapsed time, document/identifier hashes, and errors. Never trace API keys; default to IDs, length, hash, and short redacted previews instead of full legal documents.

## Implementation order

1. Preserve and validate `docs/Architecture.md` current-branch Qdrant, `data/legal_corpus/cleaned`, and `data/graph/graph` paths against configuration; add project metadata, ignore rules, `.env.example`, typed settings, and system-design doc.
2. Define official schema adapter, Pydantic state/tool contracts, per-case provenance allowlists, Content Check pass/fail gate, deterministic validator, and stable serialization. Fetch ALQAC submission fields/labels from official published schema; do not invent names or values.
3. Add corpus ingestion plus Qdrant/graph RAG with canonical source-ID preservation.
4. Add shared OpenAI model factory, paper-aligned prompts, and six constrained Deep Agent role functions. Build each with disabled default tools/delegation and no model override.
5. Add typed tool adapters, ledger, LangGraph nodes/conditional edges, five-iteration and `2n` guards.
6. Add FastAPI endpoints, structured logs, Langfuse instrumentation, and debug response.
7. Add fake-client tests then optional live smoke checks once endpoint/corpus/API details are supplied.

## Verification

- Confirm Element, Draft, Manager, Format Check, Law Search, and Content Check are Deep Agent instances that share one `ChatOpenAI` client configured only by `.env` `OPENAI_MODEL`; no subagent model override or implicit provider exists.
- Confirm Deep Agents uses `StateBackend`, no memory, blank filesystem allowlist, no shell/web tools, and empty subagent delegation; each role sees only typed external tool wrapper/state slice. Todo state, if present, has no external tool permission.
- Confirm `PUBLIC_CASE_RETRIEVAL_ENABLED=false` and `OFFICIAL_API_ENABLED=false` defaults; enabled flags require configured endpoint credentials.
- Confirm paper flow: Element -> Draft -> Manager; both checks run Format then Law; Pass/five iterations -> Content Check -> validator.
- Confirm all calls fit role contracts: only Manager routes; only Official API tool appends a `chunk_id` to its case allowlist; only Law GraphRAG appends `{law_id, aid}` to its allowlist; public retrieval has no citable-ID field and cannot contaminate case evidence.
- Confirm Content Check `fail` takes `reject_case`, produces per-case error, and never invokes serializer; only `pass` reaches deterministic validation.
- For `n` cases, assert official transport calls `<= 2n`, each request has top-1, duplicate/no-gain result halts retrieval, and disabled flag causes zero network calls.
- Confirm Qdrant vector seed plus one-hop JSONL graph traversal returns deterministic Top-3 source-ID-preserved law evidence from current-branch data.
- Confirm malformed/unknown labels, identifier hallucinations, unsupported evidence, and invalid JSON produce structured per-case errors, not silent output.
- Confirm every node generates standard log event and Langfuse span, each trace includes `OPENAI_MODEL`, route history, budget usage, and redacted failures.
- Run `pytest`; run index build on a tiny permitted fixture corpus; run FastAPI smoke test with fake model/tools; run optional real OpenAI-compatible endpoint/Langfuse smoke test after user supplies credentials/contracts.

## Architecture validation

`docs/Architecture.md` matches this plan on source roles, order, paper five-iteration stop, Format-before-Law rule, two disabled-by-default retrieval flags, top-1 Official API contract, `2n` full-efficiency budget, fail-closed Content Check, per-case rejection, Qdrant plus one-hop JSONL GraphRAG, Deep Agents, LangGraph, FastAPI, Langfuse, and one `OPENAI_MODEL`.

Implementation configuration now matches architecture paths: `QDRANT_PATH=data/vector`, `LAW_CORPUS_PATH=data/legal_corpus/cleaned`, `GRAPH_NODES_PATH=data/graph/graph/nodes.jsonl`, and `GRAPH_EDGES_PATH=data/graph/graph/edges.jsonl`. Public test remains externally user-supplied input rather than a copied/reused RAG artifact. The only remaining unknowns are official schema/API/corpus/model values below; implementation will fail fast rather than invent them.

## Inputs still needed before live integrations

- Official ALQAC published schema path/content: exact four labels, allowed values, submission envelope/order, and invalid-case behavior. Web search did not expose a public schema, so implementation must source this from user-provided competition files or URL.
- Official and public retrieval endpoint/auth/request/response/top-1 details.
- Permitted legal corpus and canonical `law_id`/`aid` rules.
- OpenAI-compatible chat/embedding endpoint and model names.
