from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


class DeprecatedFilter:
    """Flag deprecated or superseded documents using metadata-driven rules."""

    _ACTIVE_STATUSES = {
        "active",
        "in_force",
        "effective",
        "current",
        "valid",
        "in-force",
        "live",
        "còn hiệu lực",
        "con hieu luc",
        "đang có hiệu lực",
        "dang co hieu luc",
    }
    _DEPRECATED_STATUSES = {
        "deprecated",
        "superseded",
        "repealed",
        "retired",
        "expired",
        "inactive",
        "obsolete",
        "removed",
        "replaced",
        "void",
        "revoked",
        "hết hiệu lực",
        "het hieu luc",
        "không còn hiệu lực",
        "khong con hieu luc",
        "bị bãi bỏ",
        "bi bai bo",
        "được thay thế",
        "duoc thay the",
    }

    def __init__(self, current_date: date | None = None) -> None:
        self.current_date = current_date or date.today()

    def _coerce_date(self, value: Any) -> date | None:
        if value in (None, ""):
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"):
                try:
                    return datetime.strptime(text, fmt).date()
                except ValueError:
                    continue
        return None

    def _metadata_for(self, document: Mapping[str, Any]) -> dict[str, Any]:
        if "metadata" in document and isinstance(document["metadata"], Mapping):
            metadata = dict(document["metadata"])
        else:
            metadata = dict(document)
        return metadata

    def _status_for(self, metadata: Mapping[str, Any]) -> str | None:
        for key in ("status", "state", "doc_status"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return None

    def _effective_date_for(self, metadata: Mapping[str, Any]) -> date | None:
        for key in ("effective_date", "effectiveDate", "effective_from", "effectiveFrom", "date", "issued_at", "issuedAt"):
            value = metadata.get(key)
            parsed = self._coerce_date(value)
            if parsed is not None:
                return parsed
        return None

    def _expiry_date_for(self, metadata: Mapping[str, Any]) -> date | None:
        for key in ("expiry_date", "expiryDate", "valid_until", "validUntil", "until", "end_date", "endDate"):
            value = metadata.get(key)
            parsed = self._coerce_date(value)
            if parsed is not None:
                return parsed
        return None

    def is_deprecated(self, document: Mapping[str, Any]) -> bool:
        """Return True when a document should be treated as deprecated."""
        metadata = self._metadata_for(document)
        if document.get("deprecated") is True or metadata.get("deprecated") is True:
            return True

        status = self._status_for(metadata)
        if status is not None:
            if status in self._DEPRECATED_STATUSES:
                return True

        expiry_date = self._expiry_date_for(metadata)
        if expiry_date is not None and expiry_date < self.current_date:
            return True

        if status is not None and status in self._ACTIVE_STATUSES:
            return False

        effective_date = self._effective_date_for(metadata)
        if effective_date is not None and effective_date > self.current_date:
            return False

        return False

    def filter_documents(self, documents: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Return only active documents from an iterable of document dictionaries."""
        kept: list[dict[str, Any]] = []
        for document in documents:
            if not isinstance(document, Mapping):
                raise TypeError(f"Expected mapping document, got {type(document).__name__}")
            if not self.is_deprecated(document):
                kept.append(dict(document))
        return kept

    def filter_corpus_file(self, corpus_path: str | Path) -> list[dict[str, Any]]:
        """Load a JSONL corpus file and return only active documents."""
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        documents: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
                if not isinstance(payload, dict):
                    raise TypeError(f"Expected JSON object on line {line_number}, got {type(payload).__name__}")
                documents.append(payload)

        return self.filter_documents(documents)
