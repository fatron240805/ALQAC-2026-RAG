"""Six constrained Deep Agent role instances. One shared ChatOpenAI."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from app import prompts
from app.config import Settings
from app.embeddings import is_retryable_openai_error
from app.observability import Observability, log_raw_prompt
from app.schemas import (
    AlqacState,
    CaseDraft,
    ContentCheckResult,
    ElementGraph,
    FormatSuggestions,
    ManagerDecision,
)

logger = logging.getLogger("alqac.agents")


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """Single factory for every agent role."""
    kwargs: dict[str, Any] = {
        "base_url": settings.openai_base_url,
        "api_key": settings.openai_api_key,
        "model": settings.openai_model,
        "temperature": settings.llm_temperature,
        "max_retries": 0,
    }
    if settings.llm_max_tokens > 0:
        kwargs["max_tokens"] = settings.llm_max_tokens
    return ChatOpenAI(**kwargs)


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        idx = text.find(open_ch)
        if idx == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(idx, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[idx : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise json.JSONDecodeError("No valid JSON found", text, 0)


class EmptyResponseError(Exception):
    """Model returned empty or unparseable content."""


@retry(
    retry=retry_if_exception(is_retryable_openai_error),
    wait=wait_random_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _invoke_model(agent: Any, user: str) -> Any:
    return agent.invoke({"messages": [HumanMessage(content=user)]})


def _invoke_role(
    agent: Any,
    system: str,
    user: str,
    schema: type[BaseModel],
    *,
    role_name: str,
    obs: Observability | None,
    case_id: str | None = None,
    model: str | None = None,
) -> BaseModel:
    """Invoke a Deep Agent role and parse structured output."""
    log_raw_prompt(
        role=role_name,
        case_id=case_id,
        trace_id=obs.trace_id if obs else None,
        model=model,
        system_prompt=system,
        user_prompt=user,
    )

    span_cm = (
        obs.span(role_name, agent=role_name, case_id=case_id, prompt_version=prompts.PROMPT_VERSION)
        if obs
        else None
    )

    def _run() -> BaseModel:
        result = _invoke_model(agent, user)
        messages = result.get("messages", [])
        ai_content = ""
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "ai":
                ai_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
        data = _extract_json(ai_content)
        return schema.model_validate(data)

    max_json_retries = 3
    last_err: Exception | None = None

    def _run_with_retry() -> BaseModel:
        nonlocal last_err
        for attempt in range(max_json_retries):
            try:
                return _run()
            except (json.JSONDecodeError, ValidationError) as exc:
                last_err = exc
                if attempt < max_json_retries - 1:
                    logger.warning(
                        "structured_output_fail role=%s attempt=%d/%d error=%s",
                        role_name, attempt + 1, max_json_retries, exc,
                    )
                    continue
                raise
        raise last_err  # type: ignore[misc]

    if span_cm is None:
        return _run_with_retry()

    with span_cm as span:
        try:
            result = _run_with_retry()
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
    """Six Deep Agent roles. Each is a stripped-down Deep Agent (no filesystem, no shell,
    no memory, no sub-agent delegation). One shared ChatOpenAI instance."""

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

        backend = StateBackend()

        self._element = create_deep_agent(
            model=self.model,
            tools=[],
            system_prompt=prompts.ELEMENT_SYSTEM,
            backend=backend,
        )
        self._draft = create_deep_agent(
            model=self.model,
            tools=[],
            system_prompt=prompts.DRAFT_SYSTEM,
            backend=backend,
        )
        self._manager = create_deep_agent(
            model=self.model,
            tools=[],
            system_prompt=prompts.MANAGER_SYSTEM,
            backend=backend,
        )
        self._format_check = create_deep_agent(
            model=self.model,
            tools=[],
            system_prompt=prompts.FORMAT_CHECK_SYSTEM,
            backend=backend,
        )
        self._law_search = create_deep_agent(
            model=self.model,
            tools=[],
            system_prompt=prompts.LAW_SEARCH_SYSTEM,
            backend=backend,
        )
        self._content_check = create_deep_agent(
            model=self.model,
            tools=[],
            system_prompt=prompts.CONTENT_CHECK_SYSTEM,
            backend=backend,
        )

    def extract_elements(self, case_id: str, case_query: str) -> ElementGraph:
        user = prompts.ELEMENT_USER.format(case_id=case_id, case_query=case_query)
        return _invoke_role(
            self._element,
            prompts.ELEMENT_SYSTEM,
            user,
            ElementGraph,
            role_name="element",
            obs=self.obs,
            case_id=case_id,
            model=self.openai_model,
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
        return _invoke_role(
            self._draft,
            prompts.DRAFT_SYSTEM,
            user,
            CaseDraft,
            role_name="draft",
            obs=self.obs,
            case_id=case_id,
            model=self.openai_model,
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
        return _invoke_role(
            self._draft,
            prompts.DRAFT_SYSTEM,
            user,
            CaseDraft,
            role_name="draft_format_revision",
            obs=self.obs,
            case_id=case_id,
            model=self.openai_model,
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
        return _invoke_role(
            self._draft,
            prompts.DRAFT_SYSTEM,
            user,
            CaseDraft,
            role_name="draft_law_integration",
            obs=self.obs,
            case_id=case_id,
            model=self.openai_model,
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
        decision = _invoke_role(
            self._manager,
            prompts.MANAGER_SYSTEM,
            user,
            ManagerDecision,
            role_name="manager",
            obs=self.obs,
            case_id=case_id,
            model=self.openai_model,
        )
        from app.schemas import ManagerAction

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

    def format_check(
        self,
        case_id: str,
        draft: CaseDraft,
        *,
        official_allowlist: list[str],
        law_allowlist: list[dict[str, str]],
    ) -> FormatSuggestions:
        user = prompts.FORMAT_CHECK_USER.format(
            case_id=case_id,
            draft=_dump(draft),
            official_allowlist=_dump(official_allowlist),
            law_allowlist=_dump(law_allowlist),
        )
        return _invoke_role(
            self._format_check,
            prompts.FORMAT_CHECK_SYSTEM,
            user,
            FormatSuggestions,
            role_name="format_check",
            obs=self.obs,
            case_id=case_id,
            model=self.openai_model,
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
        return _invoke_role(
            self._content_check,
            prompts.CONTENT_CHECK_SYSTEM,
            user,
            ContentCheckResult,
            role_name="content_check",
            obs=self.obs,
            case_id=case_id,
            model=self.openai_model,
        )
