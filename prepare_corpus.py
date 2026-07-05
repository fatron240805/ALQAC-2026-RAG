from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
PUNCTUATION_RE = re.compile(r"\s+([,.;:!?])")


def normalize_text(value: Any) -> str:
    """Normalize text for downstream retrieval while preserving meaning."""
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\ufeff", "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-").replace("—", "-")
    text = PUNCTUATION_RE.sub(r"\1", text)
    text = re.sub(r"\s+([\.,;:!?])", r"\1", text)
    text = re.sub(r"([\.,;:!?])(?=\S)", r"\1 ", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _coerce_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _extract_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        for key in ("content", "text", "body", "content_Article", "article_text", "value"):
            if key in candidate and candidate[key] is not None:
                return _extract_text(candidate[key])
        return ""
    if isinstance(candidate, list):
        pieces = [_extract_text(item) for item in candidate if _extract_text(item)]
        return "\n\n".join(piece for piece in pieces if piece)
    return ""


def _extract_content(record: dict[str, Any]) -> str:
    if isinstance(record.get("content"), str):
        return normalize_text(record["content"])

    content_parts: list[str] = []
    for candidate in [record.get("content"), record.get("text"), record.get("body")]:
        text = _extract_text(candidate)
        if text:
            content_parts.append(normalize_text(text))

    if content_parts:
        return "\n\n".join(content_parts)

    # Fall back to all string values in the record except metadata-like fields.
    for key, value in record.items():
        if key in {"id", "law_id", "metadata", "date", "status", "title", "updated_at", "created_at"}:
            continue
        text = _extract_text(value)
        if text:
            content_parts.append(normalize_text(text))

    return "\n\n".join(content_parts)


def _extract_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    title = _coerce_string(record.get("title") or record.get("name") or record.get("law_name") or record.get("law_id") or record.get("id"), "Untitled document")
    date_value = None
    for key in ("date", "effective_date", "effectiveDate", "issued_at", "issuedAt", "updated_at", "updatedAt"):
        if key in record and record[key] not in (None, ""):
            date_value = record[key]
            break
    if date_value is None and isinstance(metadata, dict):
        for key in ("date", "effective_date", "effectiveDate", "issued_at", "issuedAt", "updated_at", "updatedAt"):
            if key in metadata and metadata[key] not in (None, ""):
                date_value = metadata[key]
                break

    status = _coerce_string(record.get("status") or metadata.get("status") or metadata.get("state"), "active")
    normalized_metadata = dict(metadata)
    normalized_metadata.update(
        {
            "title": title,
            "date": date_value,
            "status": status.lower(),
            "source_id": _coerce_string(record.get("id") or record.get("doc_id")),
            "law_id": _coerce_string(record.get("law_id")),
        }
    )
    return normalized_metadata


def build_corpus_records(payload: Any) -> list[dict[str, Any]]:
    """Normalize a JSON payload into the target corpus schema."""
    if isinstance(payload, dict):
        items = payload.get("documents") or payload.get("items") or [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        raise TypeError(f"Unsupported corpus payload type: {type(payload).__name__}")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"Expected document object at index {index}, got {type(item).__name__}")

        doc_id = str(item.get("doc_id") or item.get("id") or f"doc_{index + 1}")
        content = _extract_content(item)
        metadata = _extract_metadata(item)
        records.append(
            {
                "doc_id": doc_id,
                "content": normalize_text(content),
                "metadata": metadata,
            }
        )
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def prepare_corpus(input_path: Path, output_path: Path) -> list[dict[str, Any]]:
    """Read the raw corpus, normalize it, and write the cleaned JSONL output."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = build_corpus_records(payload)
    write_jsonl(output_path, records)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a cleaned JSONL corpus for retrieval.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/ALQAC/corpus_law_pub.json"), help="Path to the raw JSON corpus")
    parser.add_argument("--output", type=Path, default=Path("data/corpus.jsonl"), help="Path to the cleaned JSONL output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = prepare_corpus(args.input, args.output)
    print(f"Prepared {len(records)} documents into {args.output}")


if __name__ == "__main__":
    main()
