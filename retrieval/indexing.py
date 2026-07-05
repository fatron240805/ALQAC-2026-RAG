from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    BM25Okapi = None


TOKEN_RE = re.compile(r"\b[\wÀ-ỹ]+\b", re.UNICODE)


class _SimpleBM25:
    """Minimal BM25 implementation used as a fallback when rank_bm25 is unavailable."""

    def __init__(self, corpus: Sequence[Sequence[str]]) -> None:
        self.corpus = list(corpus)
        self.doc_count = len(self.corpus)
        self.avgdl = sum(len(doc) for doc in self.corpus) / max(1, self.doc_count)
        self.df = Counter(token for doc in self.corpus for token in set(doc))
        self.idf = {
            token: np.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
            for token, freq in self.df.items()
        }

    def get_scores(self, documents: Sequence[Sequence[str]]) -> np.ndarray:
        scores = np.zeros((len(documents), self.doc_count), dtype=np.float32)
        for row_index, document in enumerate(documents):
            frequencies = Counter(document)
            for token, freq in frequencies.items():
                if token not in self.idf:
                    continue
                doc_freq = self.df[token]
                numerator = freq * (1.0 + 1.0)
                denominator = freq + 1.5 * (1.0 - 0.75 + 0.75 * (len(document) / self.avgdl))
                scores[row_index] += self.idf[token] * (numerator / denominator)
        return scores


class HybridIndexer:
    """Build sparse BM25 and dense embeddings for document retrieval baselines."""

    def __init__(self, model_name: str | None = None, embedding_dim: int = 384) -> None:
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dim = embedding_dim
        self.documents: list[dict[str, Any]] = []
        self.doc_ids: list[str] = []
        self.contents: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.bm25_model: Any = None
        self.embeddings: np.ndarray | None = None
        self.embedding_model: Any = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]

    def _document_text(self, document: Mapping[str, Any]) -> str:
        if isinstance(document.get("content"), str):
            return document["content"]
        if isinstance(document.get("text"), str):
            return document["text"]
        if isinstance(document.get("body"), str):
            return document["body"]
        return ""

    def _document_metadata(self, document: Mapping[str, Any]) -> dict[str, Any]:
        metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        return dict(metadata)

    def _prepare_documents(self, corpus: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for index, document in enumerate(corpus):
            if not isinstance(document, Mapping):
                raise TypeError(f"Expected mapping document at index {index}, got {type(document).__name__}")
            doc_id = str(document.get("doc_id") or document.get("id") or f"doc_{index + 1}")
            content = self._document_text(document)
            prepared.append(
                {
                    "doc_id": doc_id,
                    "content": content,
                    "metadata": self._document_metadata(document),
                }
            )
        return prepared

    def _build_bm25(self, texts: Sequence[str]) -> Any:
        tokenized = [self._tokenize(text) for text in texts]
        if BM25Okapi is not None:
            return BM25Okapi(tokenized)
        return _SimpleBM25(tokenized)

    def _build_dense_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        if SentenceTransformer is not None and torch is not None:
            self.embedding_model = SentenceTransformer(self.model_name)
            embeddings = self.embedding_model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(embeddings, dtype=np.float32)

        # Deterministic fallback for environments without sentence-transformers.
        vectors: list[np.ndarray] = []
        for text in texts:
            tokens = self._tokenize(text)
            vector = np.zeros(self.embedding_dim, dtype=np.float32)
            for token in tokens:
                index = abs(hash(token)) % self.embedding_dim
                vector[index] += 1.0
            if np.linalg.norm(vector) > 0:
                vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.vstack(vectors) if vectors else np.zeros((0, self.embedding_dim), dtype=np.float32)

    def build_index(self, corpus: Iterable[Mapping[str, Any]]) -> "HybridIndexer":
        self.documents = self._prepare_documents(corpus)
        self.doc_ids = [doc["doc_id"] for doc in self.documents]
        self.contents = [doc["content"] for doc in self.documents]
        self.metadata = [doc["metadata"] for doc in self.documents]

        if not self.contents:
            self.bm25_model = None
            self.embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)
            return self

        self.bm25_model = self._build_bm25(self.contents)
        self.embeddings = self._build_dense_embeddings(self.contents)
        return self

    def save_index(self, path: str | Path) -> Path:
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        documents_path = save_path / "documents.jsonl"
        with documents_path.open("w", encoding="utf-8") as handle:
            for document in self.documents:
                handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True))
                handle.write("\n")

        with (save_path / "doc_ids.json").open("w", encoding="utf-8") as handle:
            json.dump(self.doc_ids, handle, ensure_ascii=False, indent=2)

        with (save_path / "bm25_model.pkl").open("wb") as handle:
            pickle.dump(self.bm25_model, handle)

        if self.embeddings is not None:
            np.save(save_path / "dense_embeddings.npy", self.embeddings)

        return save_path
