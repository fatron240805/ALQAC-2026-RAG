"""FastAPI: health, submission batch, single-case debug."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import clear_settings_cache, get_settings
from app.observability import Observability, configure_logging
from app.schemas import (
    CaseInput,
    DebugResponse,
    SubmissionRequest,
    SubmissionResponse,
)
from app.validator import load_public_test
from app.workflow import redacted_debug_state, run_batch

configure_logging()

app = FastAPI(title="ALQAC 2026 Agent", version="0.1.0")

# ---------------------------------------------------------------------------
# In-memory sliding-window rate limiter
# ---------------------------------------------------------------------------
_rate_buckets: dict[str, list[float]] = defaultdict(list)
_WINDOW_SECONDS = 60.0


def _check_rate_limit(key: str, rpm: int) -> None:
    now = time.monotonic()
    cutoff = now - _WINDOW_SECONDS
    bucket = _rate_buckets[key]
    # Prune old entries
    bucket[:] = [t for t in bucket if t > cutoff]
    if len(bucket) >= rpm:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    bucket.append(now)


# ---------------------------------------------------------------------------
# Auth dependency — enforced on execution endpoints
# ---------------------------------------------------------------------------
def _require_auth(x_api_key: str | None = Header(default=None)) -> str:
    """Validate API key when API_KEY is configured. Returns the principal id."""
    settings = get_settings()
    if not settings.api_key:
        # Auth disabled — fall back to IP-based principal
        return "anonymous"
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    return x_api_key


def _enforce_limits(cases: list[CaseInput], principal: str) -> None:
    """Batch-size cap + rate limit. Called before any LLM work."""
    settings = get_settings()
    if len(cases) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"batch size {len(cases)} exceeds limit {settings.max_batch_size}",
        )
    _check_rate_limit(principal, settings.rate_limit_rpm)


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------
class PublicTestSubmissionRequest(BaseModel):
    """Load cases from public test file (ordered). Optional slice.

    Path is always settings.public_test_path — caller cannot specify an
    arbitrary file.
    """

    case_ids: list[str] | None = None
    limit: int | None = None
    write_submission: bool = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    try:
        settings = get_settings()
        return {"status": "ok", "config": settings.readiness()}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": str(exc)}


@app.post("/v1/submission", response_model=SubmissionResponse)
def create_submission(
    body: SubmissionRequest,
    principal: str = Depends(_require_auth),
) -> SubmissionResponse:
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"config error: {exc}") from exc

    _enforce_limits(body.cases, principal)

    out = run_batch(body.cases, settings, write_submission=True)
    return SubmissionResponse(
        results=out["results"],
        official_calls_used=out["official_calls_used"],
        official_calls_max=out["official_calls_max"],
        openai_model=out["openai_model"],
        trace_id=out["trace_id"],
        submission_path=out["submission_path"],
    )


@app.post("/v1/submission/from_public_test", response_model=SubmissionResponse)
def create_submission_from_public_test(
    body: PublicTestSubmissionRequest,
    principal: str = Depends(_require_auth),
) -> SubmissionResponse:
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"config error: {exc}") from exc

    path = settings.public_test_path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"public test not found: {path}")
    try:
        raw = load_public_test(path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cases: list[CaseInput] = []
    for row in raw:
        cid = str(row.get("case_id", ""))
        query = str(row.get("case_query", ""))
        if not cid or not query:
            continue
        if body.case_ids is not None and cid not in body.case_ids:
            continue
        cases.append(CaseInput(case_id=cid, case_query=query))
        if body.limit is not None and len(cases) >= body.limit:
            break

    if not cases:
        raise HTTPException(status_code=400, detail="no cases selected")

    _enforce_limits(cases, principal)

    out = run_batch(cases, settings, write_submission=body.write_submission)
    return SubmissionResponse(
        results=out["results"],
        official_calls_used=out["official_calls_used"],
        official_calls_max=out["official_calls_max"],
        openai_model=out["openai_model"],
        trace_id=out["trace_id"],
        submission_path=out["submission_path"],
    )


@app.post("/v1/submission/from_private_test", response_model=SubmissionResponse)
def create_submission_from_private_test(
    body: PublicTestSubmissionRequest,
    principal: str = Depends(_require_auth),
) -> SubmissionResponse:
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"config error: {exc}") from exc

    path = settings.private_test_path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"private test not found: {path}")
    try:
        raw = load_public_test(path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cases: list[CaseInput] = []
    for row in raw:
        cid = str(row.get("case_id", ""))
        query = str(row.get("case_query", ""))
        if not cid or not query:
            continue
        if body.case_ids is not None and cid not in body.case_ids:
            continue
        cases.append(CaseInput(case_id=cid, case_query=query))
        if body.limit is not None and len(cases) >= body.limit:
            break

    if not cases:
        raise HTTPException(status_code=400, detail="no cases selected")

    _enforce_limits(cases, principal)

    out = run_batch(cases, settings, write_submission=body.write_submission)
    return SubmissionResponse(
        results=out["results"],
        official_calls_used=out["official_calls_used"],
        official_calls_max=out["official_calls_max"],
        openai_model=out["openai_model"],
        trace_id=out["trace_id"],
        submission_path=out["submission_path"],
    )


@app.post("/v1/cases/{case_id}/debug", response_model=DebugResponse)
def debug_case(case_id: str, body: CaseInput) -> DebugResponse:
    if body.case_id != case_id:
        raise HTTPException(status_code=400, detail="case_id path/body mismatch")
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"config error: {exc}") from exc

    obs = Observability(settings)
    out = run_batch(
        [body],
        settings,
        write_submission=False,
        obs=obs,
    )
    state = out["states"][0]
    return DebugResponse(
        case_id=case_id,
        trace_id=out["trace_id"],
        openai_model=out["openai_model"],
        redacted_state=redacted_debug_state(state),
        route_history=state.route_history,
        official_calls_used=out["official_calls_used"],
        official_calls_max=out["official_calls_max"],
    )


@app.post("/admin/reload-settings")
def reload_settings() -> dict[str, str]:
    clear_settings_cache()
    get_settings()
    return {"status": "reloaded"}
