# ManagerRouting

Manager agent action selection and iteration control.

## Actions

| Action | Enum | Effect |
|---|---|---|
| Public Case Retrieval | `PUBLIC_CASE_RETRIEVAL` | HTTP search (if enabled) |
| Official Case API | `OFFICIAL_CASE_API` | HTTP top-1 (if enabled, budgeted) |
| Format Check | `FORMAT_CHECK` | JSON/label validation |
| Law Search | `LAW_SEARCH` | Local Qdrant + graph |
| Pass | `PASS` | Stop loop, proceed to Content Check |

## Routing rules

1. **Disabled actions** → reject/retry. Never silently enable.
2. **Format + Law both selected** → Format first, then Law.
3. **Pass** or **5 iterations** → stop loop, proceed to Content Check.
4. **Content Check fail** → reject (never serialized).

## Iteration control

- Max 5 iterations per case
- `state.iteration` incremented each loop
- Manager `decision=pass` or `iteration >= 5` → force Content Check

## Action normalization

`ManagerDecision._normalize_actions()`:
- Deduplicates actions
- Preserves order
- Strips `PASS` from actions list (pass means "do nothing, stop")

## Related

- [[CaseWorkflow]] — loop implementation
- [[AgentRoles]] — Manager is one of 6 roles
- [[OfficialCallLedger]] — budget for official API calls
