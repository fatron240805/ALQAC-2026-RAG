"""Shared fixtures. Fake LLM/tools; no sibling rag imports."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Ensure chat config present before Settings() import paths
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:20128/v1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")
os.environ.setdefault("LLM_TEMPERATURE", "0")
os.environ.setdefault("PUBLIC_CASE_RETRIEVAL_ENABLED", "false")
os.environ.setdefault("OFFICIAL_API_ENABLED", "false")


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path, monkeypatch):
    from app.config import Settings, clear_settings_cache

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("PUBLIC_CASE_RETRIEVAL_ENABLED", "false")
    monkeypatch.setenv("OFFICIAL_API_ENABLED", "false")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "vector"))
    monkeypatch.setenv("LAW_CORPUS_PATH", str(tmp_path / "legal_corpus"))
    monkeypatch.setenv("GRAPH_NODES_PATH", str(tmp_path / "graph" / "nodes.jsonl"))
    monkeypatch.setenv("GRAPH_EDGES_PATH", str(tmp_path / "graph" / "edges.jsonl"))
    monkeypatch.setenv("SUBMISSION_OUTPUT_PATH", str(tmp_path / "submission.json"))
    clear_settings_cache()
    s = Settings()
    yield s
    clear_settings_cache()
