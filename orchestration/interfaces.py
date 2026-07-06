from __future__ import annotations

import json
import re
import os
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import requests

logger = logging.getLogger(__name__)

RetrievedChunk = dict[str, Any]
ReasoningOutput = dict[str, Any]

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
                "justification": "[dry-run] no real LLM backend configured.",
            },
            ensure_ascii=False,
        )


class LocalOllamaClient(LLMClient):
    """Client kết nối tới Server Emulator chạy mô hình < 10B (Gemma 3n E4B)."""

    def __init__(self, base_url: str, model_name: str, max_retries: int = 5):
        # Trích xuất loại bỏ /v1 hoặc chat/completions dư thừa từ ngrok URL
        clean_url = base_url.split("/v1")[0].rstrip("/")
        self.base_url = f"{clean_url}/v1/chat/completions"
        self.model_name = model_name
        self.max_retries = max_retries

    @classmethod
    def from_env(cls) -> "LocalOllamaClient":
        """Nạp cấu hình tự động từ file môi trường .env"""
        base_url = os.environ.get("ALQAC_LLM_BASE_URL")
        model_name = os.environ.get("ALQAC_LLM_MODEL_NAME", "google/gemma-3n-E4B-it")
        if not base_url:
            raise RuntimeError("Thiếu cấu hình biến ALQAC_LLM_BASE_URL trong file .env!")
        return cls(base_url=base_url, model_name=model_name)

    def generate(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }

        # Cơ chế hồi phục Exponential Backoff tự động ngủ bù tăng dần tại tầng LLM
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=90)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
                # Nếu lỗi do tham số response_format ở các endpoint cũ, tiến hành loại bỏ và thử lại ngay
                if attempt == 0 and "response_format" in payload:
                    payload.pop("response_format")
                    continue
                
                # Tính toán thời gian ngủ lũy tiến lũy thừa: 2^(attempt) giây, chặn tối đa ở mức 30 giây
                backoff = min(os.environ.get("FAILURE_BACKOFF_SECONDS", 2.0) ** attempt, 30.0)
                logger.warning(f"Sự cố kết nối LLM Server ({e}). Đang thử lại lần {attempt + 1}/{self.max_retries} sau {backoff}s...")
                time.sleep(backoff)

        raise RuntimeError(f"Hệ thống hoàn toàn mất kết nối đến LLM Server tại {self.base_url} sau {self.max_retries} lần thử.")


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
            f"- [{e['metadata'].get('law_id', '?')} - aid {e['metadata'].get('aid', '?')}] {e['content'][:500]}"
            for e in law_evidence
        ) or "(không có)"
        evidence_blocks = "\n".join(
            f"E{i + 1} ({hit.chunk_id}, score={hit.score:.3f}): {hit.text[:500]}"
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
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        text = match.group(0) if match else raw
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {
                "case_id": str(case_id),
                "label": "PARTIAL_B_WIN",
                "confidence": 0.0,
                "evidence_ids": [],
                "justification": "invalid_json_output_from_llm",
            }
        parsed.setdefault("case_id", str(case_id))
        return parsed


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