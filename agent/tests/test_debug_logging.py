"""Debug logging: raw prompt files, unredacted observability, debug state."""

from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.observability import (
    _DEBUG_FILE_LOGGER,
    _DEBUG_HANDLER_INSTALLED,
    configure_debug_file_handler,
    log_event,
    log_raw_prompt,
)
from app.schemas import (
    AlqacLabel,
    AlqacPrediction,
    AlqacState,
    CaseDraft,
    LawEvidenceItem,
)
from app.workflow import redacted_debug_state


@pytest.fixture(autouse=True)
def _reset_debug_handler():
    """Clear the debug file handler state between tests."""
    import app.observability as obs_mod

    obs_mod._DEBUG_HANDLER_INSTALLED = False
    pl = logging.getLogger(_DEBUG_FILE_LOGGER)
    pl.handlers.clear()
    yield
    pl.handlers.clear()
    obs_mod._DEBUG_HANDLER_INSTALLED = False


# ---------------------------------------------------------------------------
# Prompt file creation
# ---------------------------------------------------------------------------

class TestPromptFileHandler:
    def test_file_created_with_timestamp_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pl = configure_debug_file_handler()

        log_dir = tmp_path / "artifacts" / "logs"
        assert log_dir.exists()
        files = list(log_dir.glob("agent-prompts-*.log"))
        assert len(files) == 1
        fname = files[0].name
        # Timestamp pattern: agent-prompts-YYYYMMDDTHHMMSS±HHMM.log
        assert re.match(r"agent-prompts-\d{8}T\d{6}[+-]\d{4}\.log$", fname)
        assert pl.name == _DEBUG_FILE_LOGGER

    def test_duplicate_calls_do_not_add_handlers(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        configure_debug_file_handler()
        pl = configure_debug_file_handler()
        file_handlers = [h for h in pl.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_raw_prompt_writes_exact_text(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        system = "You are a legal expert. System prompt with secret_api_key=abc123."
        user = "Case ID: case_99\nQuery: long text " + "x" * 500

        log_raw_prompt(
            role="element",
            case_id="case_99",
            trace_id="trace_abc",
            model="gpt-4o",
            system_prompt=system,
            user_prompt=user,
        )

        log_dir = tmp_path / "artifacts" / "logs"
        files = list(log_dir.glob("agent-prompts-*.log"))
        content = files[0].read_text(encoding="utf-8")

        assert "role=element" in content
        assert "case_id=case_99" in content
        assert "trace_id=trace_abc" in content
        assert "model=gpt-4o" in content
        assert "--- SYSTEM ---" in content
        assert system in content
        assert "--- USER ---" in content
        assert user in content
        assert "--- END ---" in content
        # Secret-like string preserved verbatim
        assert "secret_api_key=abc123" in content

    def test_long_user_prompt_not_truncated(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        long_text = "A" * 10_000
        log_raw_prompt(
            role="draft",
            case_id="c1",
            trace_id=None,
            model=None,
            system_prompt="sys",
            user_prompt=long_text,
        )

        log_dir = tmp_path / "artifacts" / "logs"
        files = list(log_dir.glob("agent-prompts-*.log"))
        content = files[0].read_text(encoding="utf-8")
        assert long_text in content


# ---------------------------------------------------------------------------
# Unredacted log_event
# ---------------------------------------------------------------------------

class TestLogEventRaw:
    def test_sensitive_fields_not_redacted(self, caplog):
        with caplog.at_level(logging.INFO, logger="alqac.agent"):
            log_event(
                "test_event",
                api_key="super_secret_123",
                openai_api_key="sk-FAKE",
                some_data="normal",
            )
        records = [r for r in caplog.records if "test_event" in r.getMessage()]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "super_secret_123" in msg
        assert "sk-FAKE" in msg
        assert "normal" in msg

    def test_long_string_not_previewed(self, caplog):
        long = "x" * 500
        with caplog.at_level(logging.INFO, logger="alqac.agent"):
            log_event("long_event", payload=long)
        records = [r for r in caplog.records if "long_event" in r.getMessage()]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert long in msg
        assert "preview" not in msg


# ---------------------------------------------------------------------------
# Unredacted Langfuse spans
# ---------------------------------------------------------------------------

class TestLangfuseSpansRaw:
    def test_span_input_not_redacted(self, settings, monkeypatch):
        from app.observability import Observability

        monkeypatch.setenv("LANGFUSE_HOST", "http://fake")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        settings.langfuse_host = "http://fake"
        settings.langfuse_public_key = "pk"
        settings.langfuse_secret_key = "sk"

        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_langfuse.start_observation.return_value = mock_trace
        mock_span = MagicMock()
        mock_trace.start_observation.return_value = mock_span

        with patch("langfuse.Langfuse", return_value=mock_langfuse):
            obs = Observability(settings)

        secret_input = {"api_key": "real_secret", "content": "a" * 300}
        with obs.span("test_span", agent="test", case_id="c1", input_data=secret_input):
            pass

        call_kwargs = mock_trace.start_observation.call_args
        assert call_kwargs.kwargs["input"] == secret_input
        meta = call_kwargs.kwargs["metadata"]
        assert meta["agent"] == "test"
        assert meta["case_id"] == "c1"

    def test_span_output_not_redacted(self, settings, monkeypatch):
        from app.observability import Observability

        monkeypatch.setenv("LANGFUSE_HOST", "http://fake")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        settings.langfuse_host = "http://fake"
        settings.langfuse_public_key = "pk"
        settings.langfuse_secret_key = "sk"

        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_langfuse.start_observation.return_value = mock_trace
        mock_span = MagicMock()
        mock_trace.start_observation.return_value = mock_span

        with patch("langfuse.Langfuse", return_value=mock_langfuse):
            obs = Observability(settings)

        secret_output = {"result": "sk-secret-output", "detail": "x" * 300}
        with obs.span("test_span", agent="test", case_id="c1") as span:
            span.set_output(secret_output)

        mock_span.update.assert_called_once()
        call_args = mock_span.update.call_args
        assert call_args.kwargs["output"] == secret_output

    def test_update_root_metadata_not_redacted(self, settings, monkeypatch):
        from app.observability import Observability

        monkeypatch.setenv("LANGFUSE_HOST", "http://fake")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        settings.langfuse_host = "http://fake"
        settings.langfuse_public_key = "pk"
        settings.langfuse_secret_key = "sk"

        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_langfuse.start_observation.return_value = mock_trace

        with patch("langfuse.Langfuse", return_value=mock_langfuse):
            obs = Observability(settings)

        obs.update_root(api_key="real_key", batch_size=5)
        mock_trace.update.assert_called_once()
        meta = mock_trace.update.call_args.kwargs["metadata"]
        assert meta["api_key"] == "real_key"


# ---------------------------------------------------------------------------
# Unredacted debug state
# ---------------------------------------------------------------------------

class TestDebugStateRaw:
    def test_full_values_preserved(self):
        state = AlqacState(
            case_id="case_1",
            case_query="long query " + "y" * 500,
        )
        state.official_chunk_ids = ["chunk_secret_123"]
        state.law_pairs = [LawEvidenceItem(law_id="L1", aid="1")]
        state.draft = CaseDraft(
            prediction=AlqacPrediction(prediction=AlqacLabel.A_WIN),
            case_evidence=["chunk_secret_123"],
            law_evidence=[LawEvidenceItem(law_id="L1", aid="1")],
            reasoning="a" * 300,
        )

        result = redacted_debug_state(state)

        assert result["case_id"] == "case_1"
        assert result["official_chunk_ids"] == ["chunk_secret_123"]
        assert "preview" not in str(result)
        assert "length" not in str(result)
        # Draft reasoning full text preserved
        assert result["draft_prediction"] is not None
