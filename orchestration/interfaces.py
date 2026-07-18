from __future__ import annotations

import json
import re
import os
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable
import requests

logger = logging.getLogger(__name__)

RetrievedChunk = dict[str, Any]
ReasoningOutput = dict[str, Any]
ALQAC_VALID_LABELS = {"A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"}
DEFAULT_FALLBACK_LABEL = "PARTIAL_B_WIN"

# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------
class LLMClient(ABC):
    """Wrapper around an open-weight, <=10B-parameter model (competition rule)."""

    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        raise NotImplementedError


class NotConfiguredLLMClient(LLMClient):
    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        raise RuntimeError(
            "Chưa cấu hình LLMClient thật. Implement một subclass của LLMClient "
            "(HF transformers local / vLLM / Ollama HTTP) và truyền vào "
            "run_pipeline.py, hoặc chạy với --dry-run để test wiring."
        )


class DryRunLLMClient(LLMClient):
    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        return json.dumps(
            {
                "label": "PARTIAL_B_WIN",
                "confidence": 0.1,
                "evidence_ids": [],
                "citation_judgments": [],
                "justification": "[dry-run] no real LLM backend configured.",
            },
            ensure_ascii=False,
        )


class LocalOllamaClient(LLMClient):
    """Client for a local open-weight LLM server used in real inference."""

    OLLAMA_PROVIDERS = {"ollama", "ollama-native"}
    OPENAI_COMPATIBLE_PROVIDERS = {"openai-compatible", "openai", "vllm", "lmstudio", "llama.cpp"}

    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_retries: int = 5,
        provider: str = "ollama",
        timeout_seconds: float = 300.0,
        api_key: str | None = None,
    ):
        # Normalize common base URL variants before appending provider endpoints.
        clean_url = (
            base_url.split("/v1")[0]
            .split("/api/chat")[0]
            .split("/chat/completions")[0]
            .rstrip("/")
        )
        if not clean_url:
            raise ValueError("base_url must not be empty")

        self.base_url = clean_url
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.provider = provider.strip().lower() or "ollama"
        self.api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        if self.provider not in self.OLLAMA_PROVIDERS | self.OPENAI_COMPATIBLE_PROVIDERS:
            raise ValueError(
                "Unsupported ALQAC_LLM_PROVIDER="
                f"{provider!r}. Use 'ollama' or 'openai-compatible'."
            )

    @classmethod
    def from_env(cls) -> "LocalOllamaClient":
        """Nạp cấu hình tự động từ file môi trường .env"""
        base_url = os.environ.get("ALQAC_LLM_BASE_URL")
        model_name = os.environ.get("ALQAC_LLM_MODEL_NAME", "qwen2.5:7b-instruct")
        provider = os.environ.get("ALQAC_LLM_PROVIDER", "ollama")
        api_key = os.environ.get("ALQAC_LLM_API_KEY")
        try:
            timeout_seconds = float(os.environ.get("ALQAC_LLM_TIMEOUT_SECONDS", "300"))
        except ValueError:
            timeout_seconds = 300.0
        try:
            max_retries = int(os.environ.get("ALQAC_LLM_MAX_RETRIES", "2"))
        except ValueError:
            max_retries = 2
        if not base_url:
            raise RuntimeError("Thiếu cấu hình biến ALQAC_LLM_BASE_URL trong file .env!")
        return cls(
            base_url=base_url,
            model_name=model_name,
            provider=provider,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
        )

    def generate(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        headers = {"Content-Type": "application/json"}
        request_url, payload = self._build_request(prompt, max_tokens=max_tokens, temperature=temperature)
        if self.api_key and self.provider in self.OPENAI_COMPATIBLE_PROVIDERS:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Cơ chế hồi phục Exponential Backoff tự động ngủ bù tăng dần tại tầng LLM
        for attempt in range(self.max_retries):
            try:
                response = requests.post(request_url, headers=headers, json=payload, timeout=self.timeout_seconds)
                response.raise_for_status()
                return self._extract_content(response.json())
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
                # Nếu lỗi do tham số response_format ở các endpoint cũ, tiến hành loại bỏ và thử lại ngay
                if attempt == 0 and "response_format" in payload:
                    payload.pop("response_format")
                    continue
                
                # Tính toán thời gian ngủ lũy tiến lũy thừa: 2^(attempt) giây, chặn tối đa ở mức 30 giây
                try:
                    base_backoff = float(os.environ.get("FAILURE_BACKOFF_SECONDS", "2.0"))
                except ValueError:
                    base_backoff = 2.0
                backoff = min(base_backoff ** attempt, 30.0)
                logger.warning(f"Sự cố kết nối LLM Server ({e}). Đang thử lại lần {attempt + 1}/{self.max_retries} sau {backoff}s...")
                time.sleep(backoff)

        raise RuntimeError(f"LLM server unavailable at {request_url} after {self.max_retries} attempts.")


    def _build_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict[str, Any]]:
        if self.provider in self.OLLAMA_PROVIDERS:
            return (
                f"{self.base_url}/api/chat",
                {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )

        return (
            f"{self.base_url}/v1/chat/completions",
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            },
        )

    def _extract_content(self, payload: dict[str, Any]) -> str:
        if self.provider in self.OLLAMA_PROVIDERS:
            message = payload.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            response = payload.get("response")
            if isinstance(response, str):
                return response
            raise RuntimeError(f"Ollama response missing message.content: {payload}")

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"OpenAI-compatible response missing choices[0].message.content: {payload}") from exc
        if not isinstance(content, str):
            raise RuntimeError(f"LLM response content is not a string: {payload}")
        return content


# ---------------------------------------------------------------------------
# Reranker (T2-2)
# ---------------------------------------------------------------------------
class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError


class PassthroughReranker(Reranker):
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Citation usefulness filter (T2-3)
# ---------------------------------------------------------------------------
class CitationUsefulnessFilter(ABC):
    @abstractmethod
    def filter(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        raise NotImplementedError


class NoOpCitationFilter(CitationUsefulnessFilter):
    def filter(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return candidates


# ---------------------------------------------------------------------------
# Reasoning agent (T3-1)
# ---------------------------------------------------------------------------
class ReasoningAgent(ABC):
    @abstractmethod
    def answer(
        self,
        case_id: str,
        case_query: str,
        law_evidence: list[RetrievedChunk],
        case_evidence: list[Any],
    ) -> ReasoningOutput:
        raise NotImplementedError


class PromptTemplateReasoningAgent(ReasoningAgent):
    def __init__(self, llm_client: LLMClient, prompt_path: str | Path):
        self.llm_client = llm_client
        self.template = Path(prompt_path).read_text(encoding="utf-8")

    def _render(
        self,
        case_id: str,
        case_query: str,
        law_evidence: list[RetrievedChunk],
        case_evidence: list[Any],
    ) -> str:
        related_law_provisions = "\n".join(
            (
                f"L{i + 1} ({e.get('doc_id', '?')}, "
                f"{e['metadata'].get('law_id', '?')} - aid {e['metadata'].get('aid', '?')}, "
                f"score={float(e.get('rerank_score', e.get('fused_score', 0.0))):.3f}): "
                f"{e['content'][:700]}"
            )
            for i, e in enumerate(law_evidence)
        ) or "(không có)"
        evidence_blocks = "\n".join(
            f"C{i + 1} ({hit.chunk_id}, score={hit.score:.3f}): {hit.text[:700]}"
            for i, hit in enumerate(case_evidence)
            if hit.chunk_id
        ) or "(không có)"

        prompt = self.template
        for placeholder, value in (
            ("{{case_id}}", str(case_id)),
            ("{{case_query}}", case_query),
            ("{{related_law_provisions}}", related_law_provisions),
            ("{{evidence_blocks}}", evidence_blocks),
        ):
            prompt = prompt.replace(placeholder, value)
        return prompt

    def answer(
        self,
        case_id: str,
        case_query: str,
        law_evidence: list[RetrievedChunk],
        case_evidence: list[Any],
    ) -> ReasoningOutput:
        prompt = self._render(case_id, case_query, law_evidence, case_evidence)
        raw = self.llm_client.generate(prompt)
        return self._parse_json_output(case_id, raw)

    @staticmethod
    def _parse_json_output(case_id: str, raw: str) -> ReasoningOutput:
        text = extract_json_object(raw)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {
                "case_id": str(case_id),
                "label": DEFAULT_FALLBACK_LABEL,
                "confidence": 0.0,
                "evidence_ids": [],
                "citation_judgments": [],
                "justification": "invalid_json_output_from_llm",
                "parser_status": "invalid_json",
                "raw_output_preview": raw[:500],
            }
        return normalize_reasoning_output(parsed, case_id)


def extract_json_object(raw: str) -> str:
    """Extract the first balanced JSON object from a model response."""
    text = raw.strip()
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, confidence))


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def normalize_reasoning_output(payload: dict[str, Any], case_id: str) -> ReasoningOutput:
    """Normalize LLM output into the internal reasoning schema."""
    label = str(payload.get("label") or payload.get("prediction") or "").strip()
    parser_status = "ok"
    if label not in ALQAC_VALID_LABELS:
        parser_status = "invalid_label"
        label = DEFAULT_FALLBACK_LABEL

    citation_judgments = payload.get("citation_judgments")
    if not isinstance(citation_judgments, list):
        citation_judgments = []

    justification = str(payload.get("justification") or "").strip()
    if not justification:
        justification = "missing_model_justification"

    return {
        "case_id": str(payload.get("case_id") or case_id),
        "label": label,
        "confidence": _coerce_confidence(payload.get("confidence", 0.0)),
        "evidence_ids": _coerce_string_list(payload.get("evidence_ids", [])),
        "citation_judgments": citation_judgments,
        "justification": justification[:1000],
        "parser_status": parser_status,
    }


# ---------------------------------------------------------------------------
# Verifier (T3-2)
# ---------------------------------------------------------------------------
class Verifier(ABC):
    @abstractmethod
    def verify(self, answer: ReasoningOutput, evidence: list[RetrievedChunk]) -> ReasoningOutput:
        raise NotImplementedError


class PassthroughVerifier(Verifier):
    def verify(self, answer: ReasoningOutput, evidence: list[RetrievedChunk]) -> ReasoningOutput:
        return answer


class StatutoryConsistencyVerifier(Verifier):
    """Conservative verifier for real baseline runs.

    It does not try to re-reason the legal issue. It enforces the output
    contract, removes impossible evidence ids, lowers confidence when evidence
    is weak/missing, and guarantees that downstream submission export sees one
    of the four official labels.
    """

    def verify(self, answer: ReasoningOutput, evidence: list[RetrievedChunk]) -> ReasoningOutput:
        verified = normalize_reasoning_output(dict(answer), str(answer.get("case_id", "")))
        allowed_ids = self._allowed_evidence_ids(evidence)

        if allowed_ids:
            verified["evidence_ids"] = [
                evidence_id for evidence_id in verified.get("evidence_ids", []) if evidence_id in allowed_ids
            ]
        else:
            verified["evidence_ids"] = []

        statuses: list[str] = []
        if verified.get("parser_status") != "ok":
            statuses.append(str(verified["parser_status"]))
        if not evidence:
            statuses.append("no_law_evidence")
            verified["confidence"] = min(float(verified["confidence"]), 0.35)
        elif not verified["evidence_ids"]:
            statuses.append("no_valid_evidence_ids")
            verified["confidence"] = min(float(verified["confidence"]), 0.55)

        verified["verifier_status"] = "ok" if not statuses else ",".join(statuses)
        return verified

    @staticmethod
    def _allowed_evidence_ids(evidence: Iterable[RetrievedChunk]) -> set[str]:
        allowed: set[str] = set()
        for index, item in enumerate(evidence, start=1):
            doc_id = item.get("doc_id") or item.get("chunk_id")
            if doc_id:
                allowed.add(str(doc_id))
            allowed.add(f"L{index}")
        return allowed
