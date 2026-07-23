# OfficialCallLedger

Shared batch budget tracker for official API calls.

**Location**: `app/schemas.py:223`

## Budget formula

```
max_calls = OFFICIAL_CALL_BUDGET_MULTIPLIER × n_cases (default: 2n)
```

## Stop conditions

| Condition | `stopped_reason` |
|---|---|
| `used >= max_calls` | `budget_exhausted` |
| Duplicate chunk_id | `duplicate_chunk_id` |
| `consecutive_no_gain >= no_gain_limit` | `no_gain_limit` |

## API

- `can_call() → bool` — check before each official call
- `record(chunk_id, is_duplicate, is_no_gain)` — record call result
- `snapshot() → dict` — serialize for debug response

## Key invariant

Official API always returns top-1 chunk per call. Ledger shared across
entire batch (all cases in one request share the same ledger).

## Related

- [[ManagerRouting]] — Manager decides when to call official API
- [[Provenance]] — case_evidence only from official API chunk_ids
