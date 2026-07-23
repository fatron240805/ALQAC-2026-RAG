# Provenance

Evidence rules and enforcement. Fail-closed.

## Core rules

1. **No fabrication** — `case_evidence` chunk IDs only from official API. `law_evidence` only from local RAG.
2. **No public citation** — Public raw-judgment hits are reasoning-only; never copy `source_id` into `case_evidence`.
3. **Minimum evidence** — `law_evidence` always required. `case_evidence` required when official API enabled.
4. **Fail-closed** — Content Check fail → reject. Provenance error → reject. Unknown labels → reject.

## Enforcement layers

| Layer | Location | What it checks |
|---|---|---|
| Manager routing | workflow.py | Disabled actions → reject |
| Content Check | agents.py | Evidence supports prediction (LLM) |
| Validator | validator.py:29 | Allowlist checks, label validation |
| Serializer | validator.py:119 | Only serializes `status=ok` results |

## `validate_draft_provenance()`

```python
validate_draft_provenance(
    draft,
    official_allowlist=set[str],      # chunk_ids from official API
    law_allowlist=set[tuple[str,str]], # (law_id, aid) from RAG
    official_api_enabled=bool,
) → list[str]  # errors (empty = pass)
```

Checks:
1. `law_evidence` not empty (always)
2. `case_evidence` not empty (when official API enabled)
3. Every `case_evidence` chunk_id in official allowlist
4. Every `law_evidence` pair in law allowlist
5. Prediction label in `ALQAC_VALID_LABELS`

## Related

- [[OfficialCallLedger]] — official API budget
- [[Retrieval]] — how allowlists are populated
- [[AlqacState]] — where allowlists live
