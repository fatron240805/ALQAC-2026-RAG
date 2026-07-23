"""LangGraph-style request-scoped workflow with Manager loop and budgets.

Flow: Element -> Draft -> Manager loop (max 5) -> Content Check -> validator.
Public + Official retrieval Manager-gated; default OFF = zero HTTP.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from app.agents import AgentRuntime
from app.config import Settings
from app.observability import Observability, redact
from app.schemas import (
    AlqacState,
    CaseDraft,
    CaseInput,
    CaseResult,
    LawEvidenceItem,
    ManagerAction,
    ManagerDecision,
    OfficialCallLedger,
)
from app.tools import LawGraphSearchTool, OfficialCaseTop1Tool, PublicCaseSearchTool
from app.validator import serialize_submission, validate_and_build_result

logger = logging.getLogger("alqac.workflow")


def _law_allowlist_dicts(pairs: list[LawEvidenceItem]) -> list[dict[str, str]]:
    return [p.model_dump() for p in pairs]


def _append_law_allowlist(state: AlqacState, hits: list[Any]) -> None:
    seen = {(p.law_id, p.aid) for p in state.law_pairs}
    for h in hits:
        key = (h.law_id, str(h.aid))
        if key not in seen:
            seen.add(key)
            state.law_pairs.append(LawEvidenceItem(law_id=h.law_id, aid=str(h.aid)))


def _append_official_allowlist(state: AlqacState, chunk_id: str) -> None:
    if chunk_id and chunk_id not in state.official_chunk_ids:
        state.official_chunk_ids.append(chunk_id)


class CaseWorkflow:
    """Per-case pipeline sharing batch OfficialCallLedger."""

    def __init__(
        self,
        settings: Settings,
        ledger: OfficialCallLedger,
        obs: Observability,
        agents: AgentRuntime | None = None,
        public_tool: PublicCaseSearchTool | None = None,
        official_tool: OfficialCaseTop1Tool | None = None,
        law_tool: LawGraphSearchTool | None = None,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.obs = obs
        self.agents = agents or AgentRuntime(settings, obs=obs)
        self.public_tool = public_tool or PublicCaseSearchTool(settings, obs=obs)
        self.official_tool = official_tool or OfficialCaseTop1Tool(
            settings, ledger, obs=obs
        )
        self.law_tool = law_tool or LawGraphSearchTool(settings)

    def run_case(self, case: CaseInput) -> tuple[AlqacState, CaseResult]:
        state = AlqacState(
            case_id=case.case_id,
            case_query=case.case_query,
            openai_model=self.settings.openai_model,
            trace_id=self.obs.trace_id,
            official_calls_max=self.ledger.max_calls,
        )

        try:
            # 1. Element
            with self.obs.span("extract_elements", agent="element", case_id=case.case_id):
                state.element_graph = self.agents.extract_elements(
                    case.case_id, case.case_query
                )
                state.route_history.append("extract_elements")

            # 2. Initial draft
            with self.obs.span("create_initial_draft", agent="draft", case_id=case.case_id):
                state.draft = self.agents.create_draft(
                    case.case_id,
                    case.case_query,
                    state.element_graph,
                    official_allowlist=state.official_chunk_ids,
                    law_allowlist=_law_allowlist_dicts(state.law_pairs),
                    official_hits=state.official_hits,
                    law_hits=state.law_hits,
                    public_context=state.public_context,
                )
                state.route_history.append("create_initial_draft")

            # 3. Manager loop
            max_iter = self.settings.manager_max_iterations
            while state.iteration < max_iter:
                with self.obs.span(
                    "manager_route",
                    agent="manager",
                    case_id=case.case_id,
                    iteration=state.iteration,
                ):
                    decision = self.agents.manager_route(
                        case.case_id,
                        state.draft,  # type: ignore[arg-type]
                        state.element_graph,
                        iteration=state.iteration,
                        official_remaining=max(0, self.ledger.max_calls - self.ledger.used),
                        official_max=self.ledger.max_calls,
                    )
                    state.manager_decision = decision
                    state.route_history.append(
                        f"manager:{decision.decision}:{[a.value for a in decision.actions]}"
                    )

                if decision.decision == "pass" or not decision.actions:
                    state.route_history.append("manager_pass")
                    break

                self._execute_actions(state, decision)
                state.iteration += 1
            else:
                state.route_history.append("manager_max_iterations")

            # 4. Content check
            with self.obs.span("content_check", agent="content_check", case_id=case.case_id):
                content = self.agents.content_check(
                    case.case_id,
                    case.case_query,
                    state.draft,  # type: ignore[arg-type]
                    official_hits=state.official_hits,
                    law_hits=state.law_hits,
                    official_allowlist=state.official_chunk_ids,
                    law_allowlist=_law_allowlist_dicts(state.law_pairs),
                )
                state.content_result = content
                state.route_history.append(f"content_check:{content.decision}")

            state.official_calls_used = self.ledger.used

            if content.decision == "fail":
                state.rejected = True
                state.reject_reason = "; ".join(content.findings) or "content_check_fail"
                result = validate_and_build_result(
                    case.case_id,
                    state.draft,
                    content_passed=False,
                    official_allowlist=state.official_chunk_ids,
                    law_pairs=state.law_pairs,
                    content_findings=content.findings,
                    obs=self.obs,
                )
                return state, result

            # 5. Deterministic validator (only on pass)
            result = validate_and_build_result(
                case.case_id,
                state.draft,
                content_passed=True,
                official_allowlist=state.official_chunk_ids,
                law_pairs=state.law_pairs,
                content_findings=content.findings,
                obs=self.obs,
            )
            state.route_history.append(f"validate:{result.status}")
            return state, result

        except Exception as exc:  # noqa: BLE001
            logger.exception("case_failed case_id=%s", case.case_id)
            state.validation_errors.append(str(exc))
            state.official_calls_used = self.ledger.used
            result = CaseResult(
                case_id=case.case_id,
                status="error",
                error={
                    "case_id": case.case_id,
                    "stage": "workflow",
                    "message": str(exc),
                    "details": {},
                },  # type: ignore[arg-type]
            )
            # Fix CaseError properly
            from app.schemas import CaseError

            result = CaseResult(
                case_id=case.case_id,
                status="error",
                error=CaseError(
                    case_id=case.case_id,
                    stage="workflow",
                    message=str(exc),
                ),
            )
            return state, result

    def _execute_actions(self, state: AlqacState, decision: ManagerDecision) -> None:
        actions = list(decision.actions)
        # Optional public first if present
        if ManagerAction.PUBLIC_CASE_RETRIEVAL in actions:
            with self.obs.span(
                "retrieve_public_context",
                agent="public_case_retrieval",
                case_id=state.case_id,
            ):
                hits = self.public_tool(state.case_query)
                state.public_context.extend(hits)
                state.route_history.append(f"public_hits:{len(hits)}")
                # Public hits never append to official_chunk_ids

        if ManagerAction.OFFICIAL_CASE_API in actions:
            with self.obs.span(
                "retrieve_official_evidence",
                agent="official_case_api",
                case_id=state.case_id,
            ):
                hit = self.official_tool(state.case_query)
                if hit is not None:
                    state.official_hits.append(hit)
                    _append_official_allowlist(state, hit.chunk_id)
                    state.route_history.append(f"official_chunk:{hit.chunk_id}")
                else:
                    state.route_history.append("official_empty")

        # Format then Law when both selected (enforced order)
        do_format = ManagerAction.FORMAT_CHECK in actions
        do_law = ManagerAction.LAW_SEARCH in actions

        if do_format:
            with self.obs.span("format_check", agent="format_check", case_id=state.case_id):
                suggestions = self.agents.format_check(
                    state.case_id,
                    state.draft,  # type: ignore[arg-type]
                    official_allowlist=state.official_chunk_ids,
                    law_allowlist=_law_allowlist_dicts(state.law_pairs),
                )
                state.format_suggestions = suggestions
                state.route_history.append("format_check")

            with self.obs.span(
                "apply_format_revision", agent="draft", case_id=state.case_id
            ):
                state.draft = self.agents.revise_format(
                    state.case_id,
                    state.draft,  # type: ignore[arg-type]
                    suggestions,
                    official_allowlist=state.official_chunk_ids,
                    law_allowlist=_law_allowlist_dicts(state.law_pairs),
                )
                state.revision_history.append("format_revision")
                state.route_history.append("apply_format_revision")

        if do_law:
            with self.obs.span("law_search", agent="law_search", case_id=state.case_id):
                hits = self.law_tool(state.case_query, state.element_graph)
                state.law_hits.extend(hits)
                _append_law_allowlist(state, hits)
                state.route_history.append(f"law_hits:{len(hits)}")

            with self.obs.span(
                "integrate_law_revision", agent="draft", case_id=state.case_id
            ):
                state.draft = self.agents.integrate_law(
                    state.case_id,
                    state.draft,  # type: ignore[arg-type]
                    hits,
                    official_allowlist=state.official_chunk_ids,
                    law_allowlist=_law_allowlist_dicts(state.law_pairs),
                )
                state.revision_history.append("law_integration")
                state.route_history.append("integrate_law_revision")


def run_batch(
    cases: list[CaseInput],
    settings: Settings,
    *,
    write_submission: bool = True,
    obs: Observability | None = None,
    agents: AgentRuntime | None = None,
    public_tool: PublicCaseSearchTool | None = None,
    official_tool: OfficialCaseTop1Tool | None = None,
    law_tool: LawGraphSearchTool | None = None,
) -> dict[str, Any]:
    """Ordered batch with shared 2n official ledger."""
    obs = obs or Observability(settings)
    n = len(cases)
    max_calls = settings.official_max_calls(n)
    ledger = OfficialCallLedger(
        max_calls=max_calls,
        no_gain_limit=settings.official_no_gain_limit,
    )
    obs.update_root(
        batch_size=n,
        official_calls_max=max_calls,
        public_case_retrieval_enabled=settings.public_case_retrieval_enabled,
        official_api_enabled=settings.official_api_enabled,
        openai_model=settings.openai_model,
    )

    # Official tool must share ledger
    official = official_tool or OfficialCaseTop1Tool(settings, ledger, obs=obs)
    # If caller passed tool built without this ledger, rebind
    if official_tool is not None:
        official.ledger = ledger

    wf = CaseWorkflow(
        settings,
        ledger,
        obs,
        agents=agents,
        public_tool=public_tool,
        official_tool=official,
        law_tool=law_tool,
    )

    results: list[CaseResult] = []
    states: list[AlqacState] = []
    for case in cases:
        state, result = wf.run_case(case)
        states.append(state)
        results.append(result)

    submission_path = None
    if write_submission:
        rows = serialize_submission(results, settings.submission_output_path, write=True)
        submission_path = str(settings.submission_output_path) if rows else None
    else:
        serialize_submission(results, settings.submission_output_path, write=False)

    obs.update_root(
        official_calls_used=ledger.used,
        output_status="done",
    )
    obs.flush()

    return {
        "results": results,
        "states": states,
        "official_calls_used": ledger.used,
        "official_calls_max": max_calls,
        "openai_model": settings.openai_model,
        "trace_id": obs.trace_id,
        "submission_path": submission_path,
        "ledger": ledger.snapshot(),
    }


def redacted_debug_state(state: AlqacState) -> dict[str, Any]:
    return redact(
        {
            "case_id": state.case_id,
            "iteration": state.iteration,
            "route_history": state.route_history,
            "revision_history": state.revision_history,
            "official_chunk_ids": state.official_chunk_ids,
            "law_pairs": [p.model_dump() for p in state.law_pairs],
            "content_result": state.content_result.model_dump()
            if state.content_result
            else None,
            "manager_decision": state.manager_decision.model_dump()
            if state.manager_decision
            else None,
            "draft_prediction": state.draft.prediction.model_dump()
            if state.draft
            else None,
            "public_context_count": len(state.public_context),
            "official_hits_count": len(state.official_hits),
            "law_hits_count": len(state.law_hits),
            "rejected": state.rejected,
            "reject_reason": state.reject_reason,
            "validation_errors": state.validation_errors,
            "openai_model": state.openai_model,
        }
    )
