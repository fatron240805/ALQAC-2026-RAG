# AGENTS.md — Codebase schema for LLM agents

This file tells agents how to read, navigate, modify, and extend this codebase.
Co-evolve this file as the project changes.

---

## Project overview

ALQAC 2026 multi-agent legal reasoning skeleton. Adapts JurisMA paper (Table 8)
to Vietnamese legal competition with ALQAC provenance gates, budget control, and
deterministic output validation.

**Stack**: Python 3.12, FastAPI, Deep Agents, LangGraph, langchain-openai,
Qdrant (local-persistent), JSONL GraphRAG, Langfuse, Pydantic Settings, Pytest.

**Single model**: all 6 agent roles share one `OPENAI_MODEL` via `ChatOpenAI`.

---

## Directory routing

```
alqac2026_agent/
├── app/                      # Production code — read this first
│   ├── __init__.py
│   ├── agents.py             # 6 Deep Agent role instances + build_chat_model()
│   ├── config.py             # Pydantic Settings, env vars, validators
│   ├── main.py               # FastAPI endpoints: /health, /v1/submission, /v1/cases/{id}/debug
│   ├── observability.py      # Langfuse spans, structured logging, redaction
│   ├── prompts.py            # Vietnamese prompts derived from paper Table 8
│   ├── rag.py                # LawRAG (Qdrant) + LawGraph (JSONL adjacency)
│   ├── schemas.py            # All Pydantic models, enums, state, ledger
│   ├── tools.py              # PublicCaseSearchTool, OfficialCaseTop1Tool, LawGraphSearchTool
│   ├── validator.py          # Deterministic provenance checks + serializer
│   └── workflow.py           # CaseWorkflow (LangGraph-style), run_batch()
├── scripts/
│   └── build_legal_index.py  # Ingest corpus → Qdrant + graph JSONL (atomic alias swap)
├── tests/                    # 42 tests, all must pass
│   ├── conftest.py           # Fake LLM fixtures, env setup
│   ├── test_api.py           # Endpoint tests including auth + batch cap
│   ├── test_config.py        # Settings validation
│   ├── test_prompts.py       # Prompt structure checks
│   ├── test_provenance_budget.py  # Allowlist, budget, minimum-evidence
│   ├── test_rag.py           # Graph hops, search ranking, index build
│   └── test_workflow.py      # Flow integration, content fail, budget shared
├── docs/
│   ├── Architecture.md       # Mermaid flow, constraints, retrieval switches
│   ├── SystemDesign.md       # Component matrix, tool contracts, budgets
│   └── raw/                  # Paper PDF source
├── data/                     # Runtime data (symlinks OK)
│   ├── legal_corpus/cleaned/ # Source law files
│   ├── vector/               # Qdrant storage (built by build_legal_index.py)
│   └── graph/graph/          # nodes.jsonl + edges.jsonl
├── artifacts/                # Submission outputs (gitignored)
├── wiki/                     # LLM-maintained knowledge base
├── AGENTS.md                 # This file
├── README.md                 # Setup + run instructions
├── pyproject.toml            # Dependencies, pytest config
├── .env.example              # All config keys
└── .env                      # Live config (gitignored)
```

---

## Agent roles (6 roles, one model)

All roles live in `app/agents.py`. Each is a `create_deep_agent()` instance with
`StateBackend()`, `tools=[]`, Vietnamese system prompt, and shared `ChatOpenAI`.

| Role | File:Line | Purpose | Output model |
|---|---|---|---|
| Element | agents.py:~60 | Extract legal element graph | `ElementGraph` |
| Draft | agents.py:~100 | Initial/revision ALQAC prediction | `CaseDraft` |
| Manager | agents.py:~140 | Route: next action or pass | `ManagerDecision` |
| Format Check | agents.py:~180 | JSON/label validation, suggestions | `FormatSuggestions` |
| Content Check | agents.py:~220 | Evidence support gate (pass/fail) | `ContentCheckResult` |
| Law Search | N/A (tool) | Qdrant + graph retrieval | `list[LawHit]` |

**Routing** (Manager decides):
```
Manager → [Public Case Retrieval] → [Official Case API] → [Format Check → Law Search] → [Content Check]
```
- Disabled actions → reject, never silently enable
- Format + Law both selected: Format first, then Law
- Stop on `Pass` or 5 iterations → Content Check
- Content Check `fail` → reject (never serialized)

---

## Workflow flow (per case)

```
Element → Draft → Manager loop (max 5) → Content Check → Validator → submission.json
```

Defined in `app/workflow.py` `CaseWorkflow.run_case()`.

Key invariants:
- `case_evidence` only from official API `chunk_id` allowlist
- `law_evidence` only from local RAG `(law_id, aid)` allowlist
- Official API calls shared across batch via `OfficialCallLedger`
- Content Check fail → reject, never serialized
- Deterministic validator: no LLM, no network, fail-closed

---

## Data contracts

### Input
```json
{ "case_id": "string", "case_query": "string" }
```

### Output (submission.json)
```json
{
  "case_id": "string",
  "prediction": { "prediction": "A_WIN|B_WIN|PARTIAL_A_WIN|PARTIAL_B_WIN" },
  "case_evidence": ["chunk_id_from_official_api"],
  "law_evidence": [{ "law_id": "L1", "aid": "1" }]
}
```

### State (`AlqacState` — app/schemas.py:271)
- Input: `case_id`, `case_query`
- Accumulated: `element_graph`, `draft`, `official_chunk_ids`, `law_pairs`
- Control: `iteration` (0..5), `route_history`, `rejected`, `reject_reason`

---

## Configuration

All via env vars (`.env`). Key settings:

| Var | Default | Purpose |
|---|---|---|
| `OPENAI_BASE_URL` | — | LLM endpoint (required) |
| `OPENAI_API_KEY` | — | LLM key (required) |
| `OPENAI_MODEL` | — | Single model for all roles (required) |
| `API_KEY` | (empty) | Service auth; empty = no auth |
| `MAX_BATCH_SIZE` | 50 | Max cases per request |
| `RATE_LIMIT_RPM` | 60 | Requests/min per principal |
| `OFFICIAL_API_ENABLED` | false | Enable official ALQAC API |
| `PUBLIC_CASE_RETRIEVAL_ENABLED` | false | Enable public case search |
| `QDRANT_PATH` | data/vector | Qdrant storage |
| `LAW_RAG_TOP_K` | 3 | Vector search top-k |
| `GRAPH_MAX_HOPS` | 1 | Graph expansion hops |
| `OFFICIAL_CALL_BUDGET_MULTIPLIER` | 2.0 | Max official calls = multiplier × n |
| `SUBMISSION_OUTPUT_PATH` | artifacts/submission.json | Output file |

---

## Security rules

1. **Never log secrets** — API keys, full document text, or unredacted state.
2. **Never fabricate evidence** — `case_evidence` chunk IDs must come from official API. `law_evidence` must come from local RAG.
3. **Fail-closed** — Content Check fail → reject. Provenance error → reject. Unknown labels → reject.
4. **Auth required when `API_KEY` set** — `X-API-Key` header on `/v1/submission*` endpoints.
5. **Atomic writes** — submission artifacts use temp+rename. Index rebuild uses alias swap.
6. **No arbitrary file reads** — public test path from config only, never caller-controlled.

---

## Code conventions

- **Prompts**: Vietnamese, in `app/prompts.py`. Never hardcode in agent files.
- **Schemas**: All Pydantic models in `app/schemas.py`. Reuse across agents/validator/workflow.
- **No agent-to-agent calls** — agents read/write `AlqacState` only. Workflow orchestrates.
- **Deterministic validator** (`app/validator.py`) — no LLM, no network. Pure allowlist checks.
- **Tests must pass** — run `.venv/bin/pytest tests/ -x -v` before any commit.
- **No comments unless asked** — code should be self-documenting.
- **Imports**: `from __future__ import annotations` at top of every file.

---

## Common tasks

### Add a new agent role
1. Add system prompt to `app/prompts.py` (Vietnamese)
2. Add output Pydantic model to `app/schemas.py`
3. Add role instance in `app/agents.py` via `create_deep_agent()`
4. Wire into `app/workflow.py` `CaseWorkflow.run_case()`
5. Add tests in `tests/`

### Modify retrieval
- Vector search: `app/rag.py` `LawRAG.search()`
- Graph expansion: `app/rag.py` `LawGraph.expand()`
- Official API: `app/tools.py` `OfficialCaseTop1Tool`
- Public search: `app/tools.py` `PublicCaseSearchTool`

### Add config
1. Add field to `app/config.py` `Settings`
2. Add env var to `.env.example`
3. Add validator if needed
4. Update `README.md`

### Run tests
```bash
.venv/bin/pytest tests/ -x -v
```

### Build index
```bash
python scripts/build_legal_index.py
```

---

## Wiki

The `wiki/` directory is an LLM-maintained knowledge base following the
[LLM Wiki pattern](/home/nesfan/Desktop/HCMUS/Nam3/KhoaLuan/llm-wiki.md).

- `wiki/index.md` — content catalog, updated on every ingest
- `wiki/log.md` — chronological append-only record
- `wiki/*.md` — entity and concept pages

When adding new features or making significant changes, update the wiki:
1. Update `wiki/index.md` with new/changed pages
2. Update affected entity/concept pages
3. Append to `wiki/log.md`

---

## Verification checklist

Before marking any task complete:

- [ ] `.venv/bin/pytest tests/ -x -v` — all tests pass
- [ ] No secrets in output
- [ ] No fabricated evidence IDs
- [ ] Auth tested when `API_KEY` set
- [ ] Batch cap tested
- [ ] Rate limit tested
- [ ] Index build tested (alias swap, not delete-first)
- [ ] Minimum evidence enforced (law_evidence required; case_evidence when official API on)
- [ ] Submission writes atomic (temp+rename)
- [ ] Wiki updated if architecture changed
