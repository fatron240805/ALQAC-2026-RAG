"""Six constrained LLM roles. One shared OpenAI-compatible model."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Type, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from app import prompts
from app.config import Settings
from app.observability import Observability
from app.schemas import (
    CaseDraft,
    ContentCheckResult,
    ElementGraph,
    FormatSuggestions,
    ManagerAction,
    ManagerDecision,
)

logger = logging.getLogger("alqac.agents")

T = TypeVar("T", bound=BaseModel)


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """Single factory for every agent role."""
    return ChatOpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=settings.llm_temperature,
    )


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Best-effort first object/array
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise


def _invoke_json(
    model: ChatOpenAI,
    system: str,
    user: str,
    schema: Type[T],
    *,
    agent: str,
    settings: Settings,
    obs: Observability | None,
    case_id: str | None = None,
) -> T:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    span_cm = (
        obs.span(agent, agent=agent, case_id=case_id, prompt_version=prompts.PROMPT_VERSION)
        if obs
        else None
    )

    def _run() -> T:
        resp = model.invoke(messages)
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        data = _extract_json(content)
        return schema.model_validate(data)

    if span_cm is None:
        return _run()

    with span_cm as span:
        try:
            result = _run()
            span.set_output({"schema": schema.__name__, "ok": True})
            return result
        except Exception as exc:
            span.set_error(str(exc))
            raise


def _dump(obj: Any) -> str:
    if obj is None:
        return "null"
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=2)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


class AgentRuntime:
    """Thin wrappers: one model call per role, no tools inside agents."""

    def __init__(
        self,
        settings: Settings,
        obs: Observability | None = None,
        model: ChatOpenAI | None = None,
    ) -> None:
        self.settings = settings
        self.obs = obs
        self.model = model or build_chat_model(settings)
        self.openai_model = settings.openai_model

    def extract_elements(self, case_id: str, case_query: str) -> ElementGraph:
        user = prompts.ELEMENT_USER.format(case_id=case_id, case_query=case_query)
        return _invoke_json(
            self.model,
            prompts.ELEMENT_SYSTEM,
            user,
            ElementGraph,
            agent="element",
            settings=self.settings,
            obs=self.obs,
            case_id=case_id,
        )

    def create_draft(
        self,
        case_id: str,
        case_query: str,
        element_graph: ElementGraph,
        *,
        official_allowlist: list[str],
        law_allowlist: list[dict[str, str]],
        official_hits: list[Any],
        law_hits: list[Any],
        public_context: list[Any],
    ) -> CaseDraft:
        user = prompts.DRAFT_USER_INITIAL.format(
            case_id=case_id,
            case_query=case_query,
            element_graph=_dump(element_graph),
            official_allowlist=_dump(official_allowlist),
            official_hits=_dump(official_hits),
            law_allowlist=_dump(law_allowlist),
            law_hits=_dump(law_hits),
            public_context=_dump(public_context),
        )
        return _invoke_json(
            self.model,
            prompts.DRAFT_SYSTEM,
            user,
            CaseDraft,
            agent="draft",
            settings=self.settings,
            obs=self.obs,
            case_id=case_id,
        )

    def revise_format(
        self,
        case_id: str,
        draft: CaseDraft,
        suggestions: FormatSuggestions,
        *,
        official_allowlist: list[str],
        law_allowlist: list[dict[str, str]],
    ) -> CaseDraft:
        user = prompts.DRAFT_USER_REVISE_FORMAT.format(
            case_id=case_id,
            draft=_dump(draft),
            format_suggestions=_dump(suggestions),
            official_allowlist=_dump(official_allowlist),
            law_allowlist=_dump(law_allowlist),
        )
        return _invoke_json(
            self.model,
            prompts.DRAFT_SYSTEM,
            user,
            CaseDraft,
            agent="draft_format_revision",
            settings=self.settings,
            obs=self.obs,
            case_id=case_id,
        )

    def integrate_law(
        self,
        case_id: str,
        draft: CaseDraft,
        law_hits: list[Any],
        *,
        official_allowlist: list[str],
        law_allowlist: list[dict[str, str]],
    ) -> CaseDraft:
        user = prompts.DRAFT_USER_INTEGRATE_LAW.format(
            case_id=case_id,
            draft=_dump(draft),
            law_hits=_dump(law_hits),
            law_allowlist=_dump(law_allowlist),
            official_allowlist=_dump(official_allowlist),
        )
        return _invoke_json(
            self.model,
            prompts.DRAFT_SYSTEM,
            user,
            CaseDraft,
            agent="draft_law_integration",
            settings=self.settings,
            obs=self.obs,
            case_id=case_id,
        )

    def manager_route(
        self,
        case_id: str,
        draft: CaseDraft,
        element_graph: ElementGraph | None,
        *,
        iteration: int,
        official_remaining: int,
        official_max: int,
    ) -> ManagerDecision:
        user = prompts.MANAGER_USER.format(
            case_id=case_id,
            iteration=iteration,
            max_iterations=self.settings.manager_max_iterations,
            public_enabled=self.settings.public_case_retrieval_enabled,
            official_enabled=self.settings.official_api_enabled,
            official_remaining=official_remaining,
            official_max=official_max,
            draft=_dump(draft),
            element_graph=_dump(element_graph),
        )
        decision = _invoke_json(
            self.model,
            prompts.MANAGER_SYSTEM,
            user,
            ManagerDecision,
            agent="manager",
            settings=self.settings,
            obs=self.obs,
            case_id=case_id,
        )
        # Strip disabled actions (never silently enable)
        allowed: list[ManagerAction] = []
        for a in decision.actions:
            if a == ManagerAction.PUBLIC_CASE_RETRIEVAL and not self.settings.public_case_retrieval_enabled:
                logger.warning("manager requested disabled public_case_retrieval; stripped")
                continue
            if a == ManagerAction.OFFICIAL_CASE_API and not self.settings.official_api_enabled:
                logger.warning("manager requested disabled official_case_api; stripped")
                continue
            allowed.append(a)
        decision.actions = allowed
        if decision.decision == "pass":
            decision.actions = []
        # Enforce format before law in action list
        if (
            ManagerAction.FORMAT_CHECK in decision.actions
            and ManagerAction.LAW_SEARCH in decision.actions
        ):
            rest = [
                a
                for a in decision.actions
                if a not in (ManagerAction.FORMAT_CHECK, ManagerAction.LAW_SEARCH)
            ]
            decision.actions = rest + [ManagerAction.FORMAT_CHECK, ManagerAction.LAW_SEARCH]
        return decision

    def format_check(self, case_id: str, draft: CaseDraft, *, official_allowlist: list[str], law_allowlist: list[dict[str, str]]) -> FormatSuggestions:
        user = prompts.FORMAT_CHECK_USER.format(
            case_id=case_id,
            draft=_dump(draft),
            official_allowlist=_dump(official_allowlist),
            law_allowlist=_dump(law_allowlist),
        )
        return _invoke_json(
            self.model,
            prompts.FORMAT_CHECK_SYSTEM,
            user,
            FormatSuggestions,
            agent="format_check",
            settings=self.settings,
            obs=self.obs,
            case_id=case_id,
        )

    def content_check(
        self,
        case_id: str,
        case_query: str,
        draft: CaseDraft,
        *,
        official_hits: list[Any],
        law_hits: list[Any],
        official_allowlist: list[str],
        law_allowlist: list[dict[str, str]],
    ) -> ContentCheckResult:
        user = prompts.CONTENT_CHECK_USER.format(
            case_id=case_id,
            case_query=case_query,
            draft=_dump(draft),
            official_hits=_dump(official_hits),
            law_hits=_dump(law_hits),
            official_allowlist=_dump(official_allowlist),
            law_allowlist=_dump(law_allowlist),
        )
        return _invoke_json(
            self.model,
            prompts.CONTENT_CHECK_SYSTEM,
            user,
            ContentCheckResult,
            agent="content_check",
            settings=self.settings,
            obs=self.obs,
            case_id=case_id,
        )
