"""Agent invocation tests."""

from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError

from app.agents import _extract_json, _invoke_model, _invoke_role
from app.schemas import ContentCheckResult, ElementGraph, FormatSuggestions


class _Message:
    type = "ai"
    content = '{"key_facts": ["fact"]}'


class _RetryingAgent:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _: object) -> dict[str, list[_Message]]:
        self.calls += 1
        if self.calls == 1:
            raise APIConnectionError(request=httpx.Request("POST", "https://model.example/v1/chat"))
        return {"messages": [_Message()]}


class _WrappedErrorAgent:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _: object) -> dict[str, list[_Message]]:
        self.calls += 1
        if self.calls == 1:
            raise ValueError({"message": "Internal Server Error", "code": 500})
        return {"messages": [_Message()]}


def test_invoke_role_retries_transient_model_error(monkeypatch):
    monkeypatch.setattr(_invoke_model.retry, "sleep", lambda _: None)
    agent = _RetryingAgent()

    result = _invoke_role(
        agent,
        system="",
        user="case",
        schema=ElementGraph,
        role_name="element",
        obs=None,
    )

    assert result.key_facts == ["fact"]
    assert agent.calls == 2


def test_invoke_role_retries_langchain_wrapped_server_error(monkeypatch):
    monkeypatch.setattr(_invoke_model.retry, "sleep", lambda _: None)
    agent = _WrappedErrorAgent()

    result = _invoke_role(
        agent,
        system="",
        user="case",
        schema=ElementGraph,
        role_name="element",
        obs=None,
    )

    assert result.key_facts == ["fact"]
    assert agent.calls == 2


def test_extract_json_handles_trailing_text():
    raw = '{"key_facts": ["f1"]}\n\nSome trailing explanation text.'
    assert _extract_json(raw) == {"key_facts": ["f1"]}


def test_extract_json_handles_markdown_fences():
    raw = '```json\n{"key_facts": ["f1"]}\n```'
    assert _extract_json(raw) == {"key_facts": ["f1"]}


def test_extract_json_handles_nested_objects():
    raw = '{"prediction": {"prediction": "A_WIN"}, "reasoning": "ok"}'
    assert _extract_json(raw) == {"prediction": {"prediction": "A_WIN"}, "reasoning": "ok"}


def test_extract_json_handles_string_with_braces():
    raw = '{"text": "use {braces} carefully", "key_facts": []}'
    result = _extract_json(raw)
    assert result["text"] == "use {braces} carefully"
    assert result["key_facts"] == []


def test_extract_json_raises_on_no_json():
    with pytest.raises(Exception):
        _extract_json("no json here at all")


def test_format_suggestions_coerces_object_issues_to_lists():
    result = FormatSuggestions.model_validate(
        {
            "suggestions": "Giữ nguyên kết luận",
            "json_issues": {"case_evidence": "Danh sách không hợp lệ"},
            "identifier_issues": {"law_evidence": ["39/2015/QH13-1"]},
        }
    )

    assert result.suggestions == ["Giữ nguyên kết luận"]
    assert result.json_issues == ["case_evidence: Danh sách không hợp lệ"]
    assert result.identifier_issues == ["law_evidence: 39/2015/QH13-1"]


def test_invoke_role_retries_pydantic_validation_error():
    class Message:
        type = "ai"

        def __init__(self, content: str) -> None:
            self.content = content

    class Agent:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _: object) -> dict[str, list[Message]]:
            self.calls += 1
            content = '{"decision": "unknown"}' if self.calls == 1 else '{"decision": "pass"}'
            return {"messages": [Message(content)]}

    agent = Agent()
    result = _invoke_role(
        agent,
        system="",
        user="case",
        schema=ContentCheckResult,
        role_name="content_check",
        obs=None,
    )

    assert result.decision == "pass"
    assert agent.calls == 2
