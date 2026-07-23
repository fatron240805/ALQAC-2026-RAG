# AlqacState

Per-case workflow state. All agents read/write this; no agent-to-agent calls.

**Location**: `app/schemas.py:271`

## Fields

| Field | Type | Purpose |
|---|---|---|
| `case_id` | str | Input identifier |
| `case_query` | str | Input legal query text |
| `element_graph` | ElementGraph \| None | Extracted legal elements |
| `draft` | CaseDraft \| None | Current prediction + evidence |
| `official_chunk_ids` | list[str] | Allowlist: official API chunk IDs |
| `law_pairs` | list[LawEvidenceItem] | Allowlist: (law_id, aid) from RAG |
| `iteration` | int | 0..5, incremented per Manager loop |
| `route_history` | list[str] | Log of Manager actions taken |
| `rejected` | bool | True if case rejected |
| `reject_reason` | str \| None | Why rejected |

## Transitions

```
case_query → [Element] → element_graph
element_graph → [Draft] → draft
draft + tools → [Manager] → draft (revised), iteration++
draft → [Content Check] → content_result (pass/fail)
draft + content_result → [Validator] → CaseResult (ok/rejected/error)
```

## Invariants

- `official_chunk_ids` only populated by `OfficialCaseTop1Tool`
- `law_pairs` only populated by `LawGraphSearchTool`
- `rejected=True` → never serialized to submission
- `iteration > 5` → forced Content Check
