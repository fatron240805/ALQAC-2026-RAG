# CaseWorkflow

LangGraph-style pipeline orchestrating all agent roles per case.

**Location**: `app/workflow.py`

## Flow

```
Element → Draft → Manager loop (max 5) → Content Check → Validator
```

## Detailed routing

```
1. Element Agent → element_graph
2. Draft Agent → initial draft
3. Manager loop (max 5 iterations):
   a. Manager selects actions
   b. If PUBLIC_CASE_RETRIEVAL → tool (if enabled)
   c. If OFFICIAL_CASE_API → tool (if enabled, budgeted)
   d. If FORMAT_CHECK → tool, then revision applied
   e. If LAW_SEARCH → tool, then integration
   f. Draft Agent revises
4. Content Check → pass/fail
5. Validator → CaseResult
```

## Key files

- `CaseWorkflow.__init__()` — line ~52
- `CaseWorkflow.run_case()` — per-case pipeline
- `run_batch()` — batch orchestrator with shared ledger
- `redacted_debug_state()` — safe debug output

## Invariants

- Format + Law both selected → Format first, then Law
- Disabled actions → reject, never silently enable
- Content Check fail → reject, never serialized
- Deterministic validator: no LLM, no network
