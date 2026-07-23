#!/usr/bin/env python3
"""Ingest permitted ALQAC law files under data/legal_corpus/ into Qdrant + graph JSONL.

Canonical fields preserved: law_id, aid.
Does not read or import ../alqac-2026-rag.

Atomic rebuild using Qdrant aliases:
  1. Build into a temporary collection with full validation.
  2. Paginated copy into a versioned permanent collection.
  3. Validate exact point count.
  4. Atomically swap the live alias to point at the new collection.
  5. Delete the old live collection and the temporary build collection.

The live alias (COLLECTION_NAME) is never left empty or pointing at a partial build.
Exits non-zero on any failure; prior index remains live.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

# Allow running as script from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.embeddings import create_embeddings  # noqa: E402
from app.rag import COLLECTION_NAME  # noqa: E402

SCROLL_PAGE = 1000


def _progress(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", file=sys.stderr, flush=True)


def _load_records(corpus_dir: Path) -> list[dict]:
    records: list[dict] = []
    if not corpus_dir.exists():
        return records

    def append_data(data: object, path: Path) -> None:
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            articles = item.get("articles")
            if not isinstance(articles, list):
                content = item.get("content")
                articles = content if isinstance(content, list) else None
            if articles is None:
                records.append(item)
                continue
            for article in articles:
                if not isinstance(article, dict):
                    continue
                rec = dict(article)
                rec.setdefault("law_id", item.get("law_id", path.stem))
                records.append(rec)

    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    append_data(json.loads(line), path)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            append_data(data, path)
    return records


def _normalize(rec: dict) -> dict | None:
    law_id = rec.get("law_id") or rec.get("doc_id") or rec.get("id")
    aid = rec.get("aid")
    if aid is None:
        aid = rec.get("article_id") or rec.get("article_number") or rec.get("article_label")
    text = (
        rec.get("text")
        or rec.get("content")
        or rec.get("article_text")
        or rec.get("content_Article")
        or ""
    )
    if not law_id or aid is None or not str(text).strip():
        return None
    return {
        "law_id": str(law_id),
        "aid": str(aid),
        "text": str(text),
        "id": f"{law_id}::{aid}",
    }


def _scroll_all(client, collection: str) -> list:
    """Paginate through every point in a collection."""
    all_points = []
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_PAGE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        all_points.extend(result)
        if next_offset is None:
            break
        offset = next_offset
    return all_points


def _build_collection_name(norms: list[dict], model: str, dimension: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"{model}:{dimension}\n".encode())
    for record in norms:
        digest.update(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return f"{COLLECTION_NAME}_build_{digest.hexdigest()[:12]}"


def _point_id(record: dict) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, record["id"]))


def _existing_point_ids(client, collection: str) -> set[str]:
    point_ids: set[str] = set()
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_PAGE,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        point_ids.update(str(point.id) for point in points)
        if next_offset is None:
            return point_ids
        offset = next_offset


def _delete_collection_safe(client, name: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        client.delete_collection(name)


def _swap_live_alias(
    client,
    qm,
    collection_name: str,
    alias_name: str = COLLECTION_NAME,
) -> None:
    client.update_collection_aliases(
        change_aliases_operations=[
            qm.DeleteAliasOperation(
                delete_alias=qm.DeleteAlias(alias_name=alias_name)
            ),
            qm.CreateAliasOperation(
                create_alias=qm.CreateAlias(
                    alias_name=alias_name,
                    collection_name=collection_name,
                )
            ),
        ]
    )


def build_index(
    settings: Settings | None = None,
    *,
    skip_embeddings: bool = False,
) -> dict:
    settings = settings or Settings()
    corpus_dir = settings.law_corpus_path
    nodes_path = settings.graph_nodes_path
    edges_path = settings.graph_edges_path
    qdrant_path = settings.qdrant_path

    _progress("load", f"reading corpus={corpus_dir}")
    raw = _load_records(corpus_dir)
    norms = []
    for rec in raw:
        n = _normalize(rec)
        if n:
            norms.append(n)

    if not norms:
        print("ERROR: no valid records in corpus", file=sys.stderr)
        sys.exit(1)

    _progress("load", f"records={len(norms)}")

    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    qdrant_path.mkdir(parents=True, exist_ok=True)

    # Write nodes (atomic: write to temp then rename)
    tmp_nodes = nodes_path.with_suffix(".tmp")
    tmp_nodes.parent.mkdir(parents=True, exist_ok=True)
    with tmp_nodes.open("w", encoding="utf-8") as f:
        for n in norms:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    tmp_nodes.replace(nodes_path)

    # Simple co-law edges: consecutive articles same law_id
    edges: list[dict] = []
    by_law: dict[str, list[dict]] = {}
    for n in norms:
        by_law.setdefault(n["law_id"], []).append(n)
    for law_id, arts in by_law.items():
        arts_sorted = sorted(arts, key=lambda x: x["aid"])
        for a, b in zip(arts_sorted, arts_sorted[1:]):
            edges.append(
                {
                    "source": a["id"],
                    "target": b["id"],
                    "type": "same_law_adjacent",
                }
            )
    tmp_edges = edges_path.with_suffix(".tmp")
    tmp_edges.parent.mkdir(parents=True, exist_ok=True)
    with tmp_edges.open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    tmp_edges.replace(edges_path)
    _progress("graph", f"nodes={len(norms)} edges={len(edges)}")

    embedded = 0
    if not skip_embeddings and norms:
        try:
            from openai import OpenAI
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qm

            client = QdrantClient(path=str(qdrant_path))
            oai = OpenAI(
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,
            )
            _progress("embed", "probing embedding dimension")
            probe = create_embeddings(
                oai,
                model=settings.embedding_model,
                inputs=[norms[0]["text"][:2000]],
            )
            dim = len(probe.data[0].embedding)

            # ---- Phase 1: build into a disposable temp collection ----
            tmp_collection = _build_collection_name(norms, settings.embedding_model, dim)
            existing_names = {c.name for c in client.get_collections().collections}
            if tmp_collection in existing_names:
                completed_point_ids = _existing_point_ids(client, tmp_collection)
            else:
                client.create_collection(
                    collection_name=tmp_collection,
                    vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                )
                completed_point_ids = set()

            batch_size = 32
            batch_count = (len(norms) + batch_size - 1) // batch_size
            _progress(
                "resume",
                f"collection={tmp_collection} completed={len(completed_point_ids)}/{len(norms)}",
            )
            for i in range(0, len(norms), batch_size):
                batch_number = i // batch_size + 1
                batch = [
                    record
                    for record in norms[i : i + batch_size]
                    if _point_id(record) not in completed_point_ids
                ]
                if not batch:
                    continue
                _progress(
                    "embed",
                    f"batch={batch_number}/{batch_count} start points={len(completed_point_ids)}/{len(norms)}",
                )
                texts = [b["text"][:8000] for b in batch]
                emb = create_embeddings(
                    oai,
                    model=settings.embedding_model,
                    inputs=texts,
                )
                vectors = [d.embedding for d in sorted(emb.data, key=lambda x: x.index)]
                points = []
                for j, (rec, vec) in enumerate(zip(batch, vectors)):
                    points.append(
                        qm.PointStruct(
                            id=_point_id(rec),
                            vector=vec,
                            payload={
                                "law_id": rec["law_id"],
                                "aid": rec["aid"],
                                "text": rec["text"],
                            },
                        )
                    )
                client.upsert(collection_name=tmp_collection, points=points)
                completed_point_ids.update(_point_id(record) for record in batch)
                embedded += len(batch)
                _progress(
                    "embed",
                    f"batch={batch_number}/{batch_count} done points={len(completed_point_ids)}/{len(norms)}",
                )

            # Validate temp collection count
            tmp_info = client.get_collection(tmp_collection)
            tmp_count = tmp_info.points_count or 0
            if tmp_count != len(norms):
                _delete_collection_safe(client, tmp_collection)
                print(
                    f"ERROR: temp collection has {tmp_count}/{len(norms)} points; "
                    f"build discarded",
                    file=sys.stderr,
                )
                sys.exit(1)

            # ---- Phase 2: paginated copy into a versioned permanent collection ----
            version_tag = uuid.uuid4().hex[:8]
            permanent_collection = f"{COLLECTION_NAME}_v{version_tag}"
            _progress("copy", f"reading temp collection={tmp_collection}")
            client.create_collection(
                collection_name=permanent_collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )

            all_points = _scroll_all(client, tmp_collection)
            if len(all_points) != len(norms):
                _delete_collection_safe(client, tmp_collection)
                _delete_collection_safe(client, permanent_collection)
                print(
                    f"ERROR: scroll returned {len(all_points)}/{len(norms)} points; "
                    f"build discarded",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Upsert in batches
            for i in range(0, len(all_points), batch_size):
                client.upsert(
                    collection_name=permanent_collection,
                    points=all_points[i : i + batch_size],
                )
                _progress(
                    "copy",
                    f"points={min(i + batch_size, len(all_points))}/{len(all_points)}",
                )

            # Validate permanent collection count
            perm_info = client.get_collection(permanent_collection)
            perm_count = perm_info.points_count or 0
            if perm_count != len(norms):
                _delete_collection_safe(client, tmp_collection)
                _delete_collection_safe(client, permanent_collection)
                print(
                    f"ERROR: permanent collection has {perm_count}/{len(norms)} points; "
                    f"build discarded",
                    file=sys.stderr,
                )
                sys.exit(1)

            # ---- Phase 3: atomic alias swap ----
            # Resolve current live target (if alias exists, point to its collection)
            old_live_target = None
            existing_aliases = client.get_collections().collections
            existing_names = {c.name for c in existing_aliases}
            try:
                aliases = client.get_aliases().aliases
                for alias in aliases:
                    if alias.alias_name == COLLECTION_NAME:
                        old_live_target = alias.collection_name
                        break
            except Exception:  # noqa: BLE001
                pass

            # Create or update alias to point at new collection
            _swap_live_alias(client, qm, permanent_collection)

            # Cleanup: delete temp collection and old live collection
            _delete_collection_safe(client, tmp_collection)
            if old_live_target and old_live_target in existing_names:
                _delete_collection_safe(client, old_live_target)
            _progress(
                "swap",
                f"alias={COLLECTION_NAME} collection={permanent_collection}",
            )

        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR: embedding/qdrant failed; rerun to resume: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        _progress("embed", "skipped")

    _progress(
        "done",
        f"records={len(norms)} edges={len(edges)} embedded={embedded}",
    )

    return {
        "records": len(norms),
        "edges": len(edges),
        "embedded": embedded,
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "qdrant_path": str(qdrant_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local law vector+graph index")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Only write graph JSONL (no Qdrant)",
    )
    args = parser.parse_args()
    stats = build_index(skip_embeddings=args.skip_embeddings)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
