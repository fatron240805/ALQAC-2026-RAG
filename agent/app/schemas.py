"""API / state / tool / ALQAC schema contracts.

Official ALQAC four-label prediction: A_WIN, B_WIN, PARTIAL_A_WIN, PARTIAL_B_WIN.
Source: alqac-2026-rag/reasoning/labels.md, orchestration/interfaces.py.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# ALQAC I/O (official four-label prediction)
# ---------------------------------------------------------------------------

ALQAC_VALID_LABELS = {"A_WIN", "B_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN"}
DEFAULT_FALLBACK_LABEL = "PARTIAL_B_WIN"


class AlqacLabel(str, Enum):
    """Official ALQAC 2026 prediction labels.

    A = Plaintiff (Nguyen don), B = Defendant (Bi don).
    Always case_type=Dân sự, court_level=Sơ thẩm.
    """

    A_WIN = "A_WIN"
    B_WIN = "B_WIN"
    PARTIAL_A_WIN = "PARTIAL_A_WIN"
    PARTIAL_B_WIN = "PARTIAL_B_WIN"


class LawEvidenceItem(BaseModel):
    law_id: str
    aid: str


class AlqacPrediction(BaseModel):
    """Official ALQAC four-label prediction block.

    Single prediction label chosen from four valid values.
    """

    prediction: AlqacLabel = Field(description="ALQAC prediction label")


class CaseDraft(BaseModel):
    """Structured draft held in workflow state (not consultation prose)."""

    prediction: AlqacPrediction
    case_evidence: list[str] = Field(
        default_factory=list,
        description="Official API chunk_id values only",
    )
    law_evidence: list[LawEvidenceItem] = Field(default_factory=list)
    reasoning: str = Field(
        default="",
        description="Internal reasoning; not serialized as consultation prose",
    )


class CaseInput(BaseModel):
    case_id: str
    case_query: str


class SubmissionRequest(BaseModel):
    cases: list[CaseInput]

    @field_validator("cases")
    @classmethod
    def _non_empty(cls, v: list[CaseInput]) -> list[CaseInput]:
        if not v:
            raise ValueError("cases must be non-empty")
        return v


class CaseError(BaseModel):
    case_id: str
    stage: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    case_id: str
    status: Literal["ok", "rejected", "error"]
    prediction: AlqacPrediction | None = None
    case_evidence: list[str] = Field(default_factory=list)
    law_evidence: list[LawEvidenceItem] = Field(default_factory=list)
    error: CaseError | None = None


class SubmissionResponse(BaseModel):
    results: list[CaseResult]
    official_calls_used: int
    official_calls_max: int
    openai_model: str
    trace_id: str | None = None
    submission_path: str | None = None


class DebugResponse(BaseModel):
    case_id: str
    trace_id: str | None = None
    openai_model: str
    redacted_state: dict[str, Any]
    route_history: list[str]
    official_calls_used: int
    official_calls_max: int


# ---------------------------------------------------------------------------
# Element graph (paper Table 8)
# ---------------------------------------------------------------------------


class EntityNode(BaseModel):
    name: str = ""
    type: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "type", mode="before")
    @classmethod
    def _none_to_str(cls, v: Any) -> str:
        return "" if v is None else str(v)


class EventNode(BaseModel):
    description: str = ""
    time: str | None = None

    @field_validator("description", mode="before")
    @classmethod
    def _none_to_str(cls, v: Any) -> str:
        return "" if v is None else str(v)


class RelationshipEdge(BaseModel):
    type: str = ""
    source: str | int | list[str | int] = ""
    target: str | int | list[str | int] = ""

    @field_validator("type", mode="before")
    @classmethod
    def _none_to_str(cls, v: Any) -> str:
        return "" if v is None else str(v)


class ElementGraph(BaseModel):
    entities: list[EntityNode] = Field(default_factory=list)
    events: list[EventNode] = Field(default_factory=list)
    relationships: list[RelationshipEdge] = Field(default_factory=list)
    user_claims: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    legal_questions: list[str] = Field(default_factory=list)

    @field_validator("user_claims", "key_facts", "legal_questions", mode="before")
    @classmethod
    def _coerce_to_strings(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return v
        out: list[str] = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(next(iter(item.values()), str(item)))
            else:
                out.append(str(item))
        return out


# ---------------------------------------------------------------------------
# Manager routing
# ---------------------------------------------------------------------------


class ManagerAction(str, Enum):
    PUBLIC_CASE_RETRIEVAL = "public_case_retrieval"
    OFFICIAL_CASE_API = "official_case_api"
    FORMAT_CHECK = "format_check"
    LAW_SEARCH = "law_search"
    PASS = "pass"


class ManagerDecision(BaseModel):
    """Validated Manager output. Disabled actions rejected by workflow."""

    decision: Literal["revise", "pass"]
    actions: list[ManagerAction] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("actions")
    @classmethod
    def _normalize_actions(cls, v: list[ManagerAction]) -> list[ManagerAction]:
        # Deduplicate, preserve order; Pass alone means empty revise set
        seen: set[ManagerAction] = set()
        out: list[ManagerAction] = []
        for a in v:
            if a == ManagerAction.PASS:
                continue
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out


# ---------------------------------------------------------------------------
# Tool I/O contracts
# ---------------------------------------------------------------------------


class PublicCaseHit(BaseModel):
    source_id: str
    text: str
    score: float = 0.0


class OfficialCaseHit(BaseModel):
    chunk_id: str
    text: str
    score: float | None = None


class LawHit(BaseModel):
    law_id: str
    aid: str
    text: str
    vector_score: float = 0.0
    graph_hops: int = 0


class FormatSuggestions(BaseModel):
    suggestions: list[str] = Field(default_factory=list)
    json_issues: list[str] = Field(default_factory=list)
    identifier_issues: list[str] = Field(default_factory=list)


class ContentCheckResult(BaseModel):
    decision: Literal["pass", "fail"]
    findings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Official call ledger (request-scoped, shared across batch)
# ---------------------------------------------------------------------------


class OfficialCallLedger:
    """Shared batch budget: max_calls = multiplier * n. Always top-1."""

    def __init__(self, max_calls: int, no_gain_limit: int = 1) -> None:
        self.max_calls = max(0, max_calls)
        self.no_gain_limit = max(0, no_gain_limit)
        self.used = 0
        self.seen_chunk_ids: set[str] = set()
        self.consecutive_no_gain = 0
        self.stopped_reason: str | None = None

    def can_call(self) -> bool:
        if self.stopped_reason:
            return False
        return self.used < self.max_calls

    def record(self, chunk_id: str | None, is_duplicate: bool, is_no_gain: bool) -> None:
        self.used += 1
        if chunk_id:
            if chunk_id in self.seen_chunk_ids:
                is_duplicate = True
            self.seen_chunk_ids.add(chunk_id)
        if is_duplicate or is_no_gain:
            self.consecutive_no_gain += 1
        else:
            self.consecutive_no_gain = 0
        if self.used >= self.max_calls:
            self.stopped_reason = "budget_exhausted"
        elif is_duplicate:
            self.stopped_reason = "duplicate_chunk_id"
        elif self.consecutive_no_gain >= self.no_gain_limit:
            self.stopped_reason = "no_gain_limit"

    def snapshot(self) -> dict[str, Any]:
        return {
            "used": self.used,
            "max_calls": self.max_calls,
            "seen_chunk_ids": sorted(self.seen_chunk_ids),
            "consecutive_no_gain": self.consecutive_no_gain,
            "stopped_reason": self.stopped_reason,
        }


# ---------------------------------------------------------------------------
# Workflow state (LangGraph TypedDict-friendly via dict casts)
# ---------------------------------------------------------------------------


class AlqacState(BaseModel):
    case_id: str
    case_query: str
    element_graph: ElementGraph | None = None
    draft: CaseDraft | None = None
    manager_decision: ManagerDecision | None = None
    public_context: list[PublicCaseHit] = Field(default_factory=list)
    official_hits: list[OfficialCaseHit] = Field(default_factory=list)
    law_hits: list[LawHit] = Field(default_factory=list)
    format_suggestions: FormatSuggestions | None = None
    content_result: ContentCheckResult | None = None
    iteration: int = 0
    official_chunk_ids: list[str] = Field(default_factory=list)
    law_pairs: list[LawEvidenceItem] = Field(default_factory=list)
    revision_history: list[str] = Field(default_factory=list)
    route_history: list[str] = Field(default_factory=list)
    trace_id: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    rejected: bool = False
    reject_reason: str | None = None
    openai_model: str = ""
    # Request-scoped bookkeeping (not serialized to clients raw)
    official_calls_used: int = 0
    official_calls_max: int = 0

    model_config = {"arbitrary_types_allowed": True}
