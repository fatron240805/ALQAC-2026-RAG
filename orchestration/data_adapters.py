"""Schema adapters between raw data files and internal component contracts.

Gap this closes: retrieval/indexing.py's HybridIndexer expects each corpus
item to look like {"doc_id": ..., "content": ..., "metadata": {...}}. The
real data/chunks.jsonl (chunker output) instead has *flat* fields (aid,
law_id, chunk_id, doc_id, text, deprecated, unit_path, ...) with no nested
"metadata" key. Without this adapter, HybridIndexer._document_metadata()
silently returns {} for every chunk and law_id/aid disappear — breaking the
law_evidence field required by evaluation/metrics.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from retrieval.deprecated_filter import DeprecatedFilter


def chunk_to_indexer_doc(chunk: dict[str, Any]) -> dict[str, Any]:
    """Map one data/chunks.jsonl record into HybridIndexer's expected shape.

    The retrieval unit is the *chunk* (article/clause/point), not the whole
    law document, so chunk_id becomes the indexer's doc_id.
    """
    metadata_keys = (
        "aid",
        "article_id",
        "article_index",
        "article_label",
        "article_number",
        "clause_number",
        "point_label",
        "law_id",
        "doc_id",
        "source_type",
        "unit_type",
        "unit_path",
        "deprecated",
    )
    return {
        "doc_id": chunk.get("chunk_id") or chunk.get("id"),
        "content": chunk.get("text") or chunk.get("content") or "",
        "metadata": {key: chunk.get(key) for key in metadata_keys if key in chunk},
    }


def stream_active_indexer_docs(
    chunks_path: str | Path, deprecated_filter: DeprecatedFilter | None = None
) -> Iterator[dict[str, Any]]:
    """Read chunks.jsonl line-by-line, drop deprecated chunks, adapt schema.

    Streams one JSON object at a time instead of loading the full corpus
    into memory (chunks.jsonl can be large), per team convention of
    preferring line-by-line processing.
    """
    deprecated_filter = deprecated_filter or DeprecatedFilter()
    path = Path(chunks_path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                chunk = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc

            # chunks.jsonl already carries a per-chunk `deprecated` flag from
            # the chunker; DeprecatedFilter also honours status/date rules
            # if/when those get added upstream (T1-2, Hưng).
            if deprecated_filter.is_deprecated(chunk):
                continue

            yield chunk_to_indexer_doc(chunk)


def load_case_id_filter(path: str | Path) -> set[str]:
    """Đọc danh sách case_id cần chạy từ 1 file JSON (gold thật hoặc template
    do scaffold_gold_labels.py sinh ra) — dùng để giới hạn `run` chỉ xử lý
    đúng tập case đang cần đối chiếu, tránh đốt quota rate-limit (1 req/5s)
    vào toàn bộ public test khi chỉ cần validate trên 20-50 case đã gán nhãn.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("cases", payload.get("data", []))
    return {str(item["case_id"]) for item in items}


def load_test_cases(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one case at a time from ALQAC2026_public_test.json.

    NOTE: exact schema of this file wasn't in the files reviewed yet — this
    normalizes a couple of plausible key names. Confirm against the real
    file and adjust if needed (case_id/id, case_query/query/query_text).
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("cases") or payload.get("data") or [payload]
    else:
        raise TypeError(f"Unsupported public_test payload type: {type(payload).__name__}")

    for item in items:
        yield {
            "case_id": str(item.get("case_id") or item.get("id")),
            "case_query": item.get("case_query") or item.get("query") or item.get("query_text") or "",
            "n_segments": item.get("n_segments"),
        }
