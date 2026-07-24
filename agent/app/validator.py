"""Deterministic provenance checks + serializer. No LLM. No network."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from hashlib import sha256
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
    official_api_enabled: bool = False,
) -> list[str]:
    errors: list[str] = []

    # ---- Minimum evidence requirements ----
    # Law evidence is always required — a legal prediction without law support
    # is nonsensical regardless of official API availability.
    if not draft.law_evidence:
        errors.append("law_evidence is empty; at least one law pair required")
    # Case evidence: required when official API is enabled (means retrieval
    # was available). When disabled, warn but allow (public-only path).
    if official_api_enabled and not draft.case_evidence:
        errors.append(
            "case_evidence is empty; official API enabled but no retrieval results"
        )

    # ---- Allowlist checks ----
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
    official_api_enabled: bool = False,
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
            official_api_enabled=official_api_enabled,
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
    trace_id: str | None = None,
) -> list[dict[str, Any]]:
    """Project OK cases into provisional ALQAC submission rows. Preserve order.

    When write=True and trace_id is provided, writes to a request-scoped path
    (submission_<trace_id>.json) via atomic temp+rename to avoid concurrent
    overwrites. The canonical output_path is still returned for backwards
    compatibility.
    """
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
        if trace_id:
            dest = output_path.parent / f"submission_{trace_id}.json"
        else:
            dest = output_path
        _write_json_atomic(dest, rows)
        logger.info("wrote submission n=%s path=%s", len(rows), dest)
    return rows


def serialize_error_report(
    results: list[CaseResult],
    output_path: Path,
    *,
    write: bool = True,
    trace_id: str | None = None,
) -> list[dict[str, Any]]:
    """Write non-submittable case outcomes beside ALQAC submission rows."""
    rows = [
        result.model_dump(mode="json")
        for result in results
        if result.status != "ok"
    ]
    if write:
        dest = _error_report_path(output_path, trace_id)
        _write_json_atomic(dest, rows)
        logger.info("wrote error report n=%s path=%s", len(rows), dest)
    return rows


def serialize_case_artifact(
    result: CaseResult,
    output_path: Path,
    *,
    write: bool = True,
) -> tuple[Path | None, Path | None]:
    """Persist one completed case under a stable, case-specific filename."""
    case_id = _safe_case_id(result.case_id)
    if result.status == "ok" and result.prediction is not None:
        dest = output_path.parent / f"submission_{case_id}.json"
        row = {
            "case_id": result.case_id,
            "prediction": result.prediction.model_dump(),
            "case_evidence": result.case_evidence,
            "law_evidence": [item.model_dump() for item in result.law_evidence],
        }
        if write:
            _write_json_atomic(dest, [row])
            logger.info("wrote case submission case_id=%s path=%s", result.case_id, dest)
        return dest, None

    dest = output_path.parent / f"error_{case_id}.json"
    if write:
        _write_json_atomic(dest, [result.model_dump(mode="json")])
        logger.info("wrote case error case_id=%s path=%s", result.case_id, dest)
    return None, dest


def submission_artifact_path(output_path: Path, trace_id: str | None) -> Path:
    if trace_id:
        return output_path.parent / f"submission_{trace_id}.json"
    return output_path


def error_report_path(output_path: Path, trace_id: str | None) -> Path:
    return _error_report_path(output_path, trace_id)


def _error_report_path(output_path: Path, trace_id: str | None) -> Path:
    submission_path = submission_artifact_path(output_path, trace_id)
    return submission_path.with_name(f"{submission_path.stem}.errors.json")


def _write_json_atomic(dest: Path, payload: list[dict[str, Any]]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=dest.parent,
        prefix=f".{dest.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp_path = Path(tmp.name)
    try:
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)


def _safe_case_id(case_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("._") or "unknown"
    if normalized == case_id:
        return normalized
    return f"{normalized}_{sha256(case_id.encode()).hexdigest()[:10]}"


def load_public_test(path: Path) -> list[dict[str, Any]]:
    """Load ALQAC public test JSON list. Expects case_id + case_query."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("public test must be a JSON list")
    return data
