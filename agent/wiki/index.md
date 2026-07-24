# ALQAC 2026 Agent Wiki

Content catalog. Updated on every ingest.

## Entities

| Page | Summary | Source |
|---|---|---|
| [[AlqacState]] | Per-case workflow state; holds inputs, accumulated evidence, control counters | app/schemas.py:271 |
| [[CaseWorkflow]] | Per-case pipeline plus atomic batch checkpoints and error sidecar artifacts | app/workflow.py |
| [[AgentRoles]] | 6 Deep Agent role instances sharing one ChatOpenAI model; transient API retry | app/agents.py |
| [[OfficialCallLedger]] | Shared batch budget tracker for official API calls | app/schemas.py:223 |
| [[ALQACPrediction]] | Four-label prediction: A_WIN, B_WIN, PARTIAL_A_WIN, PARTIAL_B_WIN | app/schemas.py:41 |

## Concepts

| Page | Summary |
|---|---|
| [[Provenance]] | Evidence rules: no fabrication, allowlist-only, fail-closed |
| [[Retrieval]] | Vector RAG + JSONL graph + official API + public search |
| [[ManagerRouting]] | Manager agent action selection and iteration control |
| [[SecurityRules]] | Auth, rate limiting, batch caps, atomic writes, debug logging security |
| [[Configuration]] | All env vars, defaults, validation |

## Sources

| Source | Description |
|---|---|
| docs/raw/2604.10470v1.pdf | JurisMA paper, Table 8 (Appendix A.3) — prompt origin |
| docs/Architecture.md | Mermaid flow, constraints, retrieval switches |
| docs/SystemDesign.md | Component matrix, tool contracts, budgets |
| ALQAC2026_public_test.json | 50 cases, case_id + case_query |
