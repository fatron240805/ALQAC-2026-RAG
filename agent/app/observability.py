"""Structured logging + optional Langfuse spans. Never log secrets."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Generator, Iterator

from app.config import Settings

logger = logging.getLogger("alqac.agent")

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "openai_api_key",
        "authorization",
        "secret",
        "password",
        "token",
        "langfuse_secret_key",
        "langfuse_public_key",
        "official_api_key",
        "public_case_retrieval_api_key",
    }
)


def configure_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    logger.setLevel(level)


def new_trace_id() -> str:
    return uuid.uuid4().hex


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def redact(value: Any, max_len: int = 120) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if str(k).lower() in _SENSITIVE_KEYS or any(
                s in str(k).lower() for s in ("api_key", "secret", "password", "token")
            ):
                out[k] = "***"
            else:
                out[k] = redact(v, max_len=max_len)
        return out
    if isinstance(value, list):
        return [redact(v, max_len=max_len) for v in value[:20]]
    if isinstance(value, str):
        if len(value) > max_len:
            return {
                "preview": value[:max_len],
                "length": len(value),
                "hash": short_hash(value),
            }
        return value
    return value


def log_event(
    event: str,
    *,
    agent: str | None = None,
    case_id: str | None = None,
    openai_model: str | None = None,
    trace_id: str | None = None,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        "agent": agent,
        "case_id": case_id,
        "openai_model": openai_model,
        "trace_id": trace_id,
        **redact(fields),
    }
    # Drop Nones for cleaner logs
    payload = {k: v for k, v in payload.items() if v is not None}
    logger.info("%s", payload)


class NullSpan:
    def __enter__(self) -> NullSpan:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def update(self, **kwargs: Any) -> None:
        return None

    def end(self, **kwargs: Any) -> None:
        return None


class Observability:
    """Per-request trace helper. Langfuse optional."""

    def __init__(self, settings: Settings, trace_id: str | None = None) -> None:
        self.settings = settings
        self.trace_id = trace_id or new_trace_id()
        self.openai_model = settings.openai_model
        self._langfuse = None
        self._trace = None
        if settings.langfuse_enabled:
            try:
                from langfuse import Langfuse

                self._langfuse = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                self._trace = self._langfuse.trace(
                    id=self.trace_id,
                    name="alqac_submission",
                    metadata={
                        "openai_model": settings.openai_model,
                        "public_case_retrieval_enabled": settings.public_case_retrieval_enabled,
                        "official_api_enabled": settings.official_api_enabled,
                        "environment": settings.langfuse_environment,
                    },
                )
            except Exception as exc:  # noqa: BLE001 — observability must not break pipeline
                log_event("langfuse_init_failed", error=str(exc), trace_id=self.trace_id)
                self._langfuse = None
                self._trace = None

    def update_root(self, **metadata: Any) -> None:
        meta = {"openai_model": self.openai_model, **metadata}
        log_event("trace_root", trace_id=self.trace_id, **meta)
        if self._trace is not None:
            try:
                self._trace.update(metadata=redact(meta))
            except Exception as exc:  # noqa: BLE001
                log_event("langfuse_update_failed", error=str(exc), trace_id=self.trace_id)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        agent: str | None = None,
        case_id: str | None = None,
        input_data: Any = None,
        **meta: Any,
    ) -> Generator[Any, None, None]:
        start = time.perf_counter()
        log_event(
            "span_start",
            agent=agent or name,
            case_id=case_id,
            openai_model=self.openai_model,
            trace_id=self.trace_id,
            span=name,
            **meta,
        )
        lf_span = None
        if self._trace is not None:
            try:
                lf_span = self._trace.span(
                    name=name,
                    input=redact(input_data) if input_data is not None else None,
                    metadata={
                        "agent": agent or name,
                        "case_id": case_id,
                        "openai_model": self.openai_model,
                        **redact(meta),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                log_event("langfuse_span_failed", error=str(exc), trace_id=self.trace_id)

        class _SpanProxy:
            def __init__(self, outer: Observability, span_obj: Any) -> None:
                self._outer = outer
                self._span = span_obj
                self.error: str | None = None
                self.output: Any = None

            def set_output(self, output: Any) -> None:
                self.output = output

            def set_error(self, error: str) -> None:
                self.error = error

        proxy = _SpanProxy(self, lf_span)
        try:
            yield proxy
        except Exception as exc:
            proxy.set_error(str(exc))
            raise
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            log_event(
                "span_end",
                agent=agent or name,
                case_id=case_id,
                openai_model=self.openai_model,
                trace_id=self.trace_id,
                span=name,
                elapsed_ms=elapsed_ms,
                error=proxy.error,
                **meta,
            )
            if lf_span is not None:
                try:
                    if proxy.error:
                        lf_span.end(output=redact({"error": proxy.error}), level="ERROR")
                    else:
                        lf_span.end(output=redact(proxy.output) if proxy.output is not None else None)
                except Exception as exc:  # noqa: BLE001
                    log_event("langfuse_span_end_failed", error=str(exc), trace_id=self.trace_id)

    def flush(self) -> None:
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception as exc:  # noqa: BLE001
                log_event("langfuse_flush_failed", error=str(exc), trace_id=self.trace_id)
