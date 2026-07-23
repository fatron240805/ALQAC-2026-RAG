"""Retryable OpenAI-compatible embedding requests."""

from __future__ import annotations

import logging
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

logger = logging.getLogger("alqac.embeddings")


def is_retryable_openai_error(error: BaseException) -> bool:
    if isinstance(
        error,
        (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError),
    ):
        return True

    if not isinstance(error, ValueError) or len(error.args) != 1:
        return False

    detail = error.args[0]
    if isinstance(detail, dict):
        code = detail.get("code")
        message = str(detail.get("message", ""))
        try:
            if int(code) in {408, 429, 500, 502, 503, 504}:
                return True
        except (TypeError, ValueError):
            pass
    else:
        message = str(detail)

    return any(
        marker in message.lower()
        for marker in (
            "rate limit",
            "too many requests",
            "internal server error",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        )
    )


is_retryable_embedding_error = is_retryable_openai_error


@retry(
    retry=retry_if_exception(is_retryable_embedding_error),
    wait=wait_random_exponential(multiplier=1, min=30, max=600),
    stop=stop_after_attempt(5),
    reraise=True,
)
def create_embeddings(client: Any, *, model: str, inputs: list[str]) -> Any:
    try:
        return client.embeddings.create(model=model, input=inputs)
    except Exception as error:
        logger.error("embedding_request_failed model=%s error=%s", model, error)
        raise
