# SecurityRules

Auth, rate limiting, batch caps, atomic writes.

## Auth

- `API_KEY` env var controls auth
- Empty = no auth (dev mode)
- Set = require `X-API-Key` header on `/v1/submission*` endpoints
- 401 on mismatch; `/health` and `/v1/cases/{id}/debug` ungated

## Rate limiting

- Sliding-window per principal (IP or API key)
- `RATE_LIMIT_RPM` (default 60) requests per 60 seconds
- 429 when exceeded
- In-memory only, not shared across workers

## Batch cap

- `MAX_BATCH_SIZE` (default 50)
- Rejects requests exceeding cap (400)
- Applies to both `/v1/submission` and `/v1/submission/from_public_test`

## Atomic writes

| What | Mechanism |
|---|---|
| Submission artifacts | Write to `.tmp`, then `rename()` |
| Request-scoped output | `submission_<trace_id>.json` (no collisions) |
| Index nodes/edges | Write to `.tmp`, then `rename()` |
| Qdrant index | Alias swap (never delete-first) |

## File access

- Public test path from config only (`PUBLIC_TEST_PATH`)
- Caller cannot specify arbitrary file paths
- No path traversal possible

## Debug logging

- Raw prompts logged to `artifacts/logs/agent-prompts-*.log` (gitignored)
- Contains unredacted API keys, legal content, and credential-bearing prompts
- Only accessible on the local filesystem
- Debug endpoint (`/v1/cases/{id}/debug`) returns unredacted state

## Related

- [[Provenance]] — evidence validation rules
- [[Configuration]] — all security env vars
