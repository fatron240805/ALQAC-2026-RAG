# ALQACPrediction

Official ALQAC 2026 four-label prediction.

**Location**: `app/schemas.py:41`

## Labels

| Label | Meaning |
|---|---|
| `A_WIN` | Plaintiff (Nguyên đơn) wins fully |
| `B_WIN` | Defendant (Bị đơn) wins fully |
| `PARTIAL_A_WIN` | Plaintiff wins partially |
| `PARTIAL_B_WIN` | Defendant wins partially |

## Schema

```json
{ "prediction": "A_WIN|B_WIN|PARTIAL_A_WIN|PARTIAL_B_WIN" }
```

## Rules

- Single label only, no prose
- Always `case_type=Dân sự`, `court_level=Sơ thẩm`
- Validator rejects unknown labels (`ALQAC_VALID_LABELS` in schemas.py)
- Draft Agent produces; Validator enforces; Serializer copies

## Submission output

```json
{
  "case_id": "string",
  "prediction": { "prediction": "A_WIN" },
  "case_evidence": ["chunk_id"],
  "law_evidence": [{ "law_id": "L1", "aid": "1" }]
}
```

## Related

- [[Provenance]] — evidence must be from allowlists
- [[CaseWorkflow]] — Draft Agent produces, Validator checks
