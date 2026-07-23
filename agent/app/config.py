"""Typed environment configuration and startup validation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Every LLM role shares OPENAI_* chat config."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Shared chat model — sole LLM for all agents
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="", alias="OPENAI_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")

    # Embeddings for RAG (not an agent)
    openai_embedding_model: str = Field(default="", alias="OPENAI_EMBEDDING_MODEL")
    openai_embedding_base_url: str = Field(default="", alias="OPENAI_EMBEDDING_BASE_URL")
    openai_embedding_api_key: str = Field(default="", alias="OPENAI_EMBEDDING_API_KEY")

    qdrant_path: Path = Field(default=Path("data/vector"), alias="QDRANT_PATH")
    law_corpus_path: Path = Field(default=Path("data/legal_corpus/cleaned"), alias="LAW_CORPUS_PATH")
    graph_nodes_path: Path = Field(
        default=Path("data/graph/graph/nodes.jsonl"), alias="GRAPH_NODES_PATH"
    )
    graph_edges_path: Path = Field(
        default=Path("data/graph/graph/edges.jsonl"), alias="GRAPH_EDGES_PATH"
    )
    law_rag_top_k: int = Field(default=3, alias="LAW_RAG_TOP_K")
    graph_max_hops: int = Field(default=1, alias="GRAPH_MAX_HOPS")

    # Toggled agents — initially off
    public_case_retrieval_enabled: bool = Field(
        default=False, alias="PUBLIC_CASE_RETRIEVAL_ENABLED"
    )
    public_case_retrieval_url: str = Field(default="", alias="PUBLIC_CASE_RETRIEVAL_URL")
    public_case_retrieval_api_key: str = Field(
        default="", alias="PUBLIC_CASE_RETRIEVAL_API_KEY"
    )
    official_api_enabled: bool = Field(default=False, alias="OFFICIAL_API_ENABLED")
    official_api_url: str = Field(default="", alias="OFFICIAL_API_URL")
    official_api_key: str = Field(default="", alias="OFFICIAL_API_KEY")
    official_call_budget_multiplier: float = Field(
        default=2.0, alias="OFFICIAL_CALL_BUDGET_MULTIPLIER"
    )
    official_no_gain_limit: int = Field(default=1, alias="OFFICIAL_NO_GAIN_LIMIT")

    langfuse_host: str = Field(default="", alias="LANGFUSE_HOST")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_environment: str = Field(default="development", alias="LANGFUSE_ENVIRONMENT")

    submission_output_path: Path = Field(
        default=Path("artifacts/submission.json"), alias="SUBMISSION_OUTPUT_PATH"
    )
    public_test_path: Path = Field(
        default=Path("data/ALQAC2026_public_test.json"),
        alias="PUBLIC_TEST_PATH",
    )

    manager_max_iterations: int = 5

    @field_validator("law_rag_top_k")
    @classmethod
    def _valid_top_k(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError("LAW_RAG_TOP_K must be in 1..50")
        return v

    @field_validator("graph_max_hops")
    @classmethod
    def _valid_hops(cls, v: int) -> int:
        if v < 0 or v > 3:
            raise ValueError("GRAPH_MAX_HOPS must be in 0..3")
        return v

    @field_validator("official_call_budget_multiplier")
    @classmethod
    def _valid_budget_mult(cls, v: float) -> float:
        if v < 0 or v > 2:
            raise ValueError("OFFICIAL_CALL_BUDGET_MULTIPLIER must be in 0..2")
        return v

    @model_validator(mode="after")
    def _validate_required(self) -> Self:
        missing = [
            name
            for name, val in (
                ("OPENAI_BASE_URL", self.openai_base_url),
                ("OPENAI_API_KEY", self.openai_api_key),
                ("OPENAI_MODEL", self.openai_model),
            )
            if not (val and str(val).strip())
        ]
        if missing:
            raise ValueError(f"Missing required chat config: {', '.join(missing)}")

        if self.public_case_retrieval_enabled:
            if not self.public_case_retrieval_url.strip():
                raise ValueError(
                    "PUBLIC_CASE_RETRIEVAL_ENABLED requires PUBLIC_CASE_RETRIEVAL_URL"
                )
            if not self.public_case_retrieval_api_key.strip():
                raise ValueError(
                    "PUBLIC_CASE_RETRIEVAL_ENABLED requires PUBLIC_CASE_RETRIEVAL_API_KEY"
                )

        if self.official_api_enabled:
            if not self.official_api_url.strip():
                raise ValueError("OFFICIAL_API_ENABLED requires OFFICIAL_API_URL")
            if not self.official_api_key.strip():
                raise ValueError("OFFICIAL_API_ENABLED requires OFFICIAL_API_KEY")

        return self

    @property
    def embedding_base_url(self) -> str:
        return self.openai_embedding_base_url.strip() or self.openai_base_url

    @property
    def embedding_api_key(self) -> str:
        return self.openai_embedding_api_key.strip() or self.openai_api_key

    @property
    def embedding_model(self) -> str:
        return self.openai_embedding_model.strip() or "text-embedding-3-small"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(
            self.langfuse_host.strip()
            and self.langfuse_public_key.strip()
            and self.langfuse_secret_key.strip()
        )

    def official_max_calls(self, n_cases: int) -> int:
        return int(self.official_call_budget_multiplier * n_cases)

    def readiness(self) -> dict:
        """Public health payload without secrets."""
        return {
            "openai_model": self.openai_model,
            "openai_base_url_set": bool(self.openai_base_url.strip()),
            "openai_api_key_set": bool(self.openai_api_key.strip()),
            "public_case_retrieval_enabled": self.public_case_retrieval_enabled,
            "official_api_enabled": self.official_api_enabled,
            "law_rag_top_k": self.law_rag_top_k,
            "graph_max_hops": self.graph_max_hops,
            "official_call_budget_multiplier": self.official_call_budget_multiplier,
            "manager_max_iterations": self.manager_max_iterations,
            "langfuse_enabled": self.langfuse_enabled,
            "qdrant_path": str(self.qdrant_path),
            "graph_nodes_path": str(self.graph_nodes_path),
            "graph_edges_path": str(self.graph_edges_path),
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
