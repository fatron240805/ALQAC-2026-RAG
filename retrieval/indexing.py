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

    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.corpus = list(corpus)
        self.k1 = k1
        self.b = b
        self.doc_count = len(self.corpus)
        self.avgdl = sum(len(doc) for doc in self.corpus) / max(1, self.doc_count)
        self.df = Counter(token for doc in self.corpus for token in set(doc))
        self.doc_term_frequencies = [Counter(doc) for doc in self.corpus]
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.idf = {
            token: np.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
            for token, freq in self.df.items()
        }

    def get_scores(self, queries: Sequence[Sequence[str]]) -> np.ndarray:
        """Return a batch matrix shaped (n_queries, n_indexed_docs).

        This intentionally mirrors rank_bm25's scoring semantics: query tokens
        are matched against each indexed document's term frequencies. The
        previous fallback accidentally used query frequencies as if they were
        document frequencies, which made scores nearly constant across docs.
        """
        scores = np.zeros((len(queries), self.doc_count), dtype=np.float32)
        for row_index, query_tokens in enumerate(queries):
            for doc_index, term_frequencies in enumerate(self.doc_term_frequencies):
                doc_len = self.doc_lengths[doc_index] or 1
                score = 0.0
                for token in query_tokens:
                    if token not in self.idf:
                        continue
                    freq = term_frequencies.get(token, 0)
                    if freq == 0:
                        continue
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (
                        1.0 - self.b + self.b * (doc_len / max(self.avgdl, 1e-9))
                    )
                    score += float(self.idf[token]) * (numerator / denominator)
                scores[row_index, doc_index] = score
        return scores


class HybridIndexer:
    """Build sparse BM25 and dense embeddings for document retrieval baselines."""

    def __init__(
        self,
        model_name: str | None = None,
        embedding_dim: int = 384,
        *,
        use_sentence_transformer: bool = False,
    ) -> None:
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dim = embedding_dim
        self.use_sentence_transformer = use_sentence_transformer
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
        if self.use_sentence_transformer and SentenceTransformer is not None and torch is not None:
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

    # ------------------------------------------------------------------
    # Query-time API (added for orchestration/run_pipeline.py, T0-1/T1-3).
    # HybridIndexer previously only supported build_index()/save_index();
    # these methods close the gap so the pipeline can actually retrieve.
    # Hưng: feel free to move/refactor this into retrieval/router.py (T2-1)
    # once query decomposition / routing lands on top of it.
    # ------------------------------------------------------------------

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into the same embedding space as the corpus."""
        text = query or ""
        if self.embedding_model is not None:
            vector = self.embedding_model.encode(
                [text], convert_to_numpy=True, normalize_embeddings=True
            )
            return np.asarray(vector[0], dtype=np.float32)

        # Must mirror _build_dense_embeddings' fallback exactly, or query and
        # corpus vectors would live in inconsistent spaces.
        tokens = self._tokenize(text)
        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        for token in tokens:
            index = abs(hash(token)) % self.embedding_dim
            vector[index] += 1.0
        if np.linalg.norm(vector) > 0:
            vector /= np.linalg.norm(vector)
        return vector

    def bm25_scores(self, query: str) -> np.ndarray:
        """Score every indexed document against a raw query string."""
        if self.bm25_model is None or not self.doc_ids:
            return np.zeros(0, dtype=np.float32)

        tokens = self._tokenize(query)
        if isinstance(self.bm25_model, _SimpleBM25):
            # _SimpleBM25.get_scores expects a batch of tokenized "documents";
            # wrap the single query and unwrap the single resulting row.
            return np.asarray(self.bm25_model.get_scores([tokens])[0], dtype=np.float32)
        return np.asarray(self.bm25_model.get_scores(tokens), dtype=np.float32)

    @staticmethod
    def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_v, max_v = float(scores.min()), float(scores.max())
        if max_v - min_v < 1e-9:
            return np.zeros_like(scores)
        return (scores - min_v) / (max_v - min_v)

    def search(self, query: str, top_k: int = 10, alpha: float = 0.45) -> list[dict[str, Any]]:
        """Hybrid BM25 + dense retrieval: score = alpha*BM25_norm + (1-alpha)*cosine.

        Matches the fusion formula and default alpha=0.45 from Plan.md.
        Returns a list of result dicts ordered by descending fused_score.
        """
        if not self.doc_ids:
            return []

        bm25_raw = self.bm25_scores(query)
        alpha = min(1.0, max(0.0, float(alpha)))
        bm25_norm = self._min_max_normalize(bm25_raw)

        query_vector = self.encode_query(query)
        if self.embeddings is not None and len(self.embeddings) > 0:
            dense_scores = self.embeddings @ query_vector
        else:
            dense_scores = np.zeros(len(self.doc_ids), dtype=np.float32)

        fused = alpha * bm25_norm + (1.0 - alpha) * dense_scores
        top_k = max(0, min(top_k, len(self.doc_ids)))
        ranked_indices = np.argsort(-fused)[:top_k]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices):
            results.append(
                {
                    "rank": rank + 1,
                    "doc_id": self.doc_ids[idx],
                    "content": self.contents[idx],
                    "metadata": self.metadata[idx],
                    "bm25_score": float(bm25_raw[idx]) if len(bm25_raw) else 0.0,
                    "dense_score": float(dense_scores[idx]),
                    "fused_score": float(fused[idx]),
                }
            )
        return results

    @classmethod
    def load_index(
        cls,
        path: str | Path,
        model_name: str | None = None,
        *,
        use_sentence_transformer: bool = False,
    ) -> "HybridIndexer":
        """Reconstruct an indexer instance from artifacts written by save_index()."""
        load_path = Path(path)
        indexer = cls(
            model_name=model_name,
            use_sentence_transformer=use_sentence_transformer,
        )

        with (load_path / "documents.jsonl").open("r", encoding="utf-8") as handle:
            indexer.documents = [json.loads(line) for line in handle if line.strip()]
        with (load_path / "doc_ids.json").open("r", encoding="utf-8") as handle:
            indexer.doc_ids = json.load(handle)
        with (load_path / "bm25_model.pkl").open("rb") as handle:
            indexer.bm25_model = pickle.load(handle)

        embeddings_path = load_path / "dense_embeddings.npy"
        indexer.embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
        if indexer.embeddings is not None and indexer.embeddings.ndim == 2:
            indexer.embedding_dim = int(indexer.embeddings.shape[1])

        indexer.contents = [doc.get("content", "") for doc in indexer.documents]
        indexer.metadata = [doc.get("metadata", {}) for doc in indexer.documents]

        if indexer.use_sentence_transformer and SentenceTransformer is not None and torch is not None:
            indexer.embedding_model = SentenceTransformer(indexer.model_name)

        return indexer

