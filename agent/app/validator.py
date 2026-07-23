"""Deterministic provenance checks + serializer. No LLM. No network."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.observability import Observability
from app.schemas import (
    ALQAC_VALID_LABELS,
    AlqacPrediction,
    CaseDraft,
    CaseError,
    CaseResult,
    LawEvidenceItem,
)

logger = logging.getLogger("alqac.validator")


def _law_key(item: LawEvidenceItem | dict[str, str]) -> tuple[str, str]:
    if isinstance(item, LawEvidenceItem):
        return item.law_id, item.aid
    return str(item["law_id"]), str(item["aid"])


def validate_draft_provenance(
    draft: CaseDraft,
    *,
    official_allowlist: set[str],
    law_allowlist: set[tuple[str, str]],
) -> list[str]:
    errors: list[str] = []
    for cid in draft.case_evidence:
        if cid not in official_allowlist:
            errors.append(f"case_evidence chunk_id not in allowlist: {cid}")
    for item in draft.law_evidence:
        key = _law_key(item)
        if key not in law_allowlist:
            errors.append(f"law_evidence pair not in allowlist: {key}")
    # Prediction label must be one of four valid ALQAC labels
    pred = draft.prediction
    label_val = pred.prediction.value if hasattr(pred.prediction, "value") else str(pred.prediction)
    if label_val not in ALQAC_VALID_LABELS:
        errors.append(f"prediction.label invalid: {label_val!r} not in {ALQAC_VALID_LABELS}")
    return errors


def validate_and_build_result(
    case_id: str,
    draft: CaseDraft | None,
    *,
    content_passed: bool,
    official_allowlist: list[str],
    law_pairs: list[LawEvidenceItem],
    content_findings: list[str] | None = None,
    obs: Observability | None = None,
) -> CaseResult:
    """Only Content-Check-passed cases may serialize. Fail-closed."""

    def _span(name: str):
        if obs:
            return obs.span(name, agent="validator", case_id=case_id)
        from contextlib import nullcontext

        return nullcontext()

    with _span("validate_and_serialize"):
        if not content_passed:
            return CaseResult(
                case_id=case_id,
                status="rejected",
                error=CaseError(
                    case_id=case_id,
                    stage="content_check",
                    message="Content Check failed; blocked from serialization",
                    details={"findings": content_findings or []},
                ),
            )
        if draft is None:
            return CaseResult(
                case_id=case_id,
                status="error",
                error=CaseError(
                    case_id=case_id,
                    stage="validator",
                    message="Missing draft",
                ),
            )

        errors = validate_draft_provenance(
            draft,
            official_allowlist=set(official_allowlist),
            law_allowlist={_law_key(p) for p in law_pairs},
        )
        if errors:
            return CaseResult(
                case_id=case_id,
                status="error",
                error=CaseError(
                    case_id=case_id,
                    stage="validator",
                    message="Provenance / schema validation failed",
                    details={"errors": errors},
                ),
            )

        return CaseResult(
            case_id=case_id,
            status="ok",
            prediction=draft.prediction,
            case_evidence=list(draft.case_evidence),
            law_evidence=list(draft.law_evidence),
        )


def serialize_submission(
    results: list[CaseResult],
    output_path: Path,
    *,
    write: bool = True,
) -> list[dict[str, Any]]:
    """Project OK cases into provisional ALQAC submission rows. Preserve order."""
    rows: list[dict[str, Any]] = []
    for r in results:
        if r.status != "ok" or r.prediction is None:
            continue
        rows.append(
            {
                "case_id": r.case_id,
                "prediction": r.prediction.model_dump(),
                "case_evidence": r.case_evidence,
                "law_evidence": [x.model_dump() for x in r.law_evidence],
            }
        )
    if write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("wrote submission n=%s path=%s", len(rows), output_path)
    return rows


def load_public_test(path: Path) -> list[dict[str, Any]]:
    """Load ALQAC public test JSON list. Expects case_id + case_query."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("public test must be a JSON list")
    return data
