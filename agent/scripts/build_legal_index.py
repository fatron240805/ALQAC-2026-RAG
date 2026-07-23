#!/usr/bin/env python3
"""Ingest permitted ALQAC law files under data/legal_corpus/ into Qdrant + graph JSONL.

Canonical fields preserved: law_id, aid.
Does not read or import ../alqac-2026-rag.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

# Allow running as script from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.rag import COLLECTION_NAME  # noqa: E402


def _load_records(corpus_dir: Path) -> list[dict]:
    records: list[dict] = []
    if not corpus_dir.exists():
        return records
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                if "articles" in data:
                    for art in data["articles"]:
                        rec = dict(art)
                        rec.setdefault("law_id", data.get("law_id", path.stem))
                        records.append(rec)
                else:
                    records.append(data)
    return records


def _normalize(rec: dict) -> dict | None:
    law_id = rec.get("law_id") or rec.get("doc_id") or rec.get("id")
    aid = rec.get("aid")
    if aid is None:
        aid = rec.get("article_id") or rec.get("article_number") or rec.get("article_label")
    text = rec.get("text") or rec.get("content") or rec.get("article_text") or ""
    if not law_id or aid is None or not str(text).strip():
        return None
    return {
        "law_id": str(law_id),
        "aid": str(aid),
        "text": str(text),
        "id": f"{law_id}::{aid}",
    }


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

    raw = _load_records(corpus_dir)
    norms = []
    for rec in raw:
        n = _normalize(rec)
        if n:
            norms.append(n)

    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    qdrant_path.mkdir(parents=True, exist_ok=True)

    # Write nodes
    with nodes_path.open("w", encoding="utf-8") as f:
        for n in norms:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

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
    with edges_path.open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

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
            # Probe dim
            probe = oai.embeddings.create(
                model=settings.embedding_model, input=[norms[0]["text"][:2000]]
            )
            dim = len(probe.data[0].embedding)

            if COLLECTION_NAME in {c.name for c in client.get_collections().collections}:
                client.delete_collection(COLLECTION_NAME)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )

            batch_size = 32
            for i in range(0, len(norms), batch_size):
                batch = norms[i : i + batch_size]
                texts = [b["text"][:8000] for b in batch]
                emb = oai.embeddings.create(model=settings.embedding_model, input=texts)
                vectors = [d.embedding for d in sorted(emb.data, key=lambda x: x.index)]
                points = []
                for j, (rec, vec) in enumerate(zip(batch, vectors)):
                    points.append(
                        qm.PointStruct(
                            id=str(uuid.uuid5(uuid.NAMESPACE_URL, rec["id"])),
                            vector=vec,
                            payload={
                                "law_id": rec["law_id"],
                                "aid": rec["aid"],
                                "text": rec["text"],
                            },
                        )
                    )
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                embedded += len(batch)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: embedding/qdrant upsert skipped: {exc}", file=sys.stderr)

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
