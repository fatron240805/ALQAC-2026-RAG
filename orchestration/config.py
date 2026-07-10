"""Unified configuration for the ALQAC 2026 Agentic RAG pipeline.

Single source of truth for paths + hyperparameters referenced by
Plan.md's "Experiment matrix" (model, embedding, chunk size, top_k,
reranker, prompt version, score, latency, notes).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    # --- Data paths -----------------------------------------------------
    chunks_path: Path = PROJECT_ROOT / "data" / "chunks.jsonl"
    index_path: Path = PROJECT_ROOT / "data" / "index"
    public_test_path: Path = PROJECT_ROOT / "data" / "ALQAC2026_public_test.json"
    prompt_path: Path = PROJECT_ROOT / "reasoning" / "prompts" / "prompt_v0.md"
    gold_path: Path = PROJECT_ROOT / "data" / "local_validation_gold.json"

    # --- Retrieval (Plan.md defaults) ------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: str = "350-600 tokens (pre-chunked upstream by Hưng)"
    top_k_before_rerank: int = 50
    top_k_after_rerank: int = 5
    hybrid_alpha: float = 0.45  # score = alpha*BM25 + (1-alpha)*dense

    # --- Reasoning --------------------------------------------------------
    llm_model: str = "TBD"  # e.g. Qwen2.5-7B-Instruct / Mistral-7B-Instruct-v0.3
    prompt_version: str = "v0"

    # --- Case-content Retrieval API (CONFIRMED từ api-docs) ---------------
    # Domain KHÁC leaderboard: https://alqac-api.ngrok.pro, header X-API-Key.
    # Rate limit 1 req/5s áp dụng CHO ĐÚNG endpoint /retrieve này (KHÔNG
    # phải cho LLM reasoning nội bộ — LLM chạy local, không cần rate-limit).
    case_retrieval_base_url: str = "https://alqac-api.ngrok.pro"
    seconds_between_api_calls: float = 5.0
    max_case_retrieval_calls_per_case: int = 1  # TODO: tăng khi có query decomposition (T2-1, Hưng)

    # --- Submission guardrails ---------------------------------------------
    # NOTE: mâu thuẫn giữa email BTC (3/ngày) và api-docs (20/ngày) — đang
    # dùng mức AN TOÀN HƠN (3) theo quyết định của team. Cập nhật khi có
    # xác nhận chính thức từ BTC.
    max_submissions_per_day: int = 3
    submission_tracker_path: Path = PROJECT_ROOT / "leaderboard" / "submission_tracker.csv"

    # --- Experiment tracking ----------------------------------------------
    experiments_dir: Path = PROJECT_ROOT / "experiments"
    run_tag: str = "v0_baseline"

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        """Load overrides from a JSON config file, keeping unknown keys in `extra`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in payload.items() if k in known and k != "extra"}
        for key in (
            "chunks_path",
            "index_path",
            "public_test_path",
            "prompt_path",
            "gold_path",
            "submission_tracker_path",
            "experiments_dir",
        ):
            if key in kwargs:
                value = Path(kwargs[key])
                kwargs[key] = value if value.is_absolute() else PROJECT_ROOT / value
        extra = {k: v for k, v in payload.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def config_hash(self) -> str:
        """Short deterministic hash for experiment/submission tracking (Plan.md)."""
        payload = json.dumps(self.__dict__, default=str, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]

    def as_experiment_row(self) -> dict[str, Any]:
        """Fields matching Plan.md's experiment matrix schema."""
        return {
            "config_hash": self.config_hash(),
            "model": self.llm_model,
            "embedding": self.embedding_model,
            "chunk_size": self.chunk_size,
            "top_k": self.top_k_after_rerank,
            "reranker": "passthrough (stub)",
            "prompt_version": self.prompt_version,
            "hybrid_alpha": self.hybrid_alpha,
        }
