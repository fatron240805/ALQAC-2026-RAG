from __future__ import annotations

import json
import logging
import pickle
import re
import unicodedata
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
    from FlagEmbedding import BGEM3FlagModel  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    BGEM3FlagModel = None

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    BM25Okapi = None


TOKEN_RE = re.compile(r"\b[\wÀ-ỹ]+\b", re.UNICODE)


logger = logging.getLogger(__name__)


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
        embedding_dim: int = 1024,
        *,
        use_bge_m3: bool = True,
        use_sentence_transformer: bool = False,
        device: str = "auto",
        batch_size: int = 12,
        max_length: int = 1024,
    ) -> None:
        self.use_bge_m3 = use_bge_m3
        self.embedding_dim = embedding_dim
        self.use_sentence_transformer = use_sentence_transformer
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = self.device.startswith("cuda")
        self.model_name = model_name or (
            "BAAI/bge-m3" if self.use_bge_m3 else "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.documents: list[dict[str, Any]] = []
        self.doc_ids: list[str] = []
        self.contents: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.bm25_model: Any = None
        self.embeddings: np.ndarray | None = None
        self.embedding_model: Any = None
        self.sparse_weights: list[dict[str, float]] = []
        self.retrieval_backend: str = "fallback_hash"

    @staticmethod
    def _resolve_device(device: str) -> str:
        value = (device or "auto").strip().lower()
        if value not in {"auto", "cpu", "cuda", "cuda:0"}:
            return value
        if value == "cpu":
            return "cpu"
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]

    @staticmethod
    def _strip_diacritics(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        return "".join(char for char in normalized if not unicodedata.combining(char))

    def _semantic_text(self, document: Mapping[str, Any]) -> str:
        """Compose a richer text view for dense retrieval.

        Legal queries often mention article numbers, law ids, or unit paths that
        do not appear verbatim in the paragraph body. Folding those signals into
        the dense side helps the semantic retriever connect references to the
        right provision even when the wording differs.
        """
        pieces: list[str] = []
        for key in (
            "content",
            "text",
            "body",
            "unit_path",
            "article_label",
            "article_number",
            "law_id",
            "aid",
            "source_type",
        ):
            value = document.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                pieces.append(text)

        metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        if isinstance(metadata, dict):
            for key in ("law_id", "aid", "unit_path", "article_label", "article_number", "point_label", "clause_number"):
                value = metadata.get(key)
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    pieces.append(text)

        return " \n ".join(pieces)

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
        merged = dict(metadata)
        for key in ("law_id", "aid", "unit_path", "article_label", "article_number", "point_label", "clause_number", "source_type"):
            value = document.get(key)
            if value is not None and key not in merged:
                merged[key] = value
        return merged

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

    def _load_bge_model(self) -> bool:
        if not self.use_bge_m3 or BGEM3FlagModel is None:
            return False
        try:
            kwargs: dict[str, Any] = {"use_fp16": self.use_fp16}
            if self.device != "cpu":
                kwargs["device"] = self.device
            self.embedding_model = BGEM3FlagModel(self.model_name, **kwargs)
            self.retrieval_backend = "bge_m3"
            return True
        except TypeError:
            try:
                self.embedding_model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
                self.retrieval_backend = "bge_m3"
                return True
            except Exception as exc:
                logger.warning("BGE-M3 init failed, falling back to classic retrieval: %s", exc)
        except Exception as exc:
            logger.warning("BGE-M3 init failed, falling back to classic retrieval: %s", exc)
        self.embedding_model = None
        return False

    def _build_dense_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        self.sparse_weights = []
        if self._load_bge_model():
            try:
                output = self.embedding_model.encode(  # type: ignore[union-attr]
                    list(texts),
                    batch_size=self.batch_size,
                    max_length=self.max_length,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
                embeddings = np.asarray(output.get("dense_vecs"), dtype=np.float32)
                sparse_weights = output.get("lexical_weights") or output.get("sparse_vecs") or []
                self.sparse_weights = [
                    {str(token): float(weight) for token, weight in dict(weights).items() if float(weight) > 0.0}
                    for weights in sparse_weights
                ]
                if embeddings.ndim == 2 and embeddings.shape[1] > 0:
                    self.embedding_dim = int(embeddings.shape[1])
                return embeddings
            except Exception as exc:
                logger.warning("BGE-M3 encoding failed, falling back to classic retrieval: %s", exc)
                self.embedding_model = None
                self.retrieval_backend = "fallback_hash"

        if self.use_sentence_transformer and SentenceTransformer is not None and torch is not None:
            try:
                self.embedding_model = SentenceTransformer(self.model_name)
                embeddings = self.embedding_model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
                self.retrieval_backend = "sentence_transformer"
                return np.asarray(embeddings, dtype=np.float32)
            except Exception as exc:
                logger.warning("SentenceTransformer encoding failed, falling back to hash vectors: %s", exc)
                self.embedding_model = None

        # Deterministic fallback for environments without GPU retrieval models.
        vectors: list[np.ndarray] = []
        for text in texts:
            tokens = self._tokenize(text)
            stripped_tokens = self._tokenize(self._strip_diacritics(text))
            vector = np.zeros(self.embedding_dim, dtype=np.float32)
            for token in tokens:
                index = abs(hash(token)) % self.embedding_dim
                vector[index] += 1.0
            for token in stripped_tokens:
                index = abs(hash(f"noaccent:{token}")) % self.embedding_dim
                vector[index] += 0.75
            for left, right in zip(tokens, tokens[1:]):
                index = abs(hash(f"bi:{left} {right}")) % self.embedding_dim
                vector[index] += 1.5
            if np.linalg.norm(vector) > 0:
                vector /= np.linalg.norm(vector)
            vectors.append(vector)
        self.retrieval_backend = "fallback_hash"
        return np.vstack(vectors) if vectors else np.zeros((0, self.embedding_dim), dtype=np.float32)

    def build_index(self, corpus: Iterable[Mapping[str, Any]]) -> "HybridIndexer":
        self.documents = self._prepare_documents(corpus)
        self.doc_ids = [doc["doc_id"] for doc in self.documents]
        self.contents = [doc["content"] for doc in self.documents]
        self.metadata = [doc["metadata"] for doc in self.documents]
        semantic_texts = [self._semantic_text(doc) for doc in self.documents]

        if not self.contents:
            self.bm25_model = None
            self.embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)
            return self

        self.embeddings = self._build_dense_embeddings(semantic_texts)
        if self.use_bge_m3 and self.sparse_weights:
            self.bm25_model = self.sparse_weights
        else:
            self.bm25_model = self._build_bm25(self.contents)
            self.retrieval_backend = "bm25"
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

        with (save_path / "index_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "embedding_dim": self.embedding_dim,
                    "model_name": self.model_name,
                    "use_bge_m3": self.use_bge_m3,
                    "use_sentence_transformer": self.use_sentence_transformer,
                    "retrieval_backend": self.retrieval_backend,
                    "device": self.device,
                    "batch_size": self.batch_size,
                    "max_length": self.max_length,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        with (save_path / "bm25_model.pkl").open("wb") as handle:
            pickle.dump(self.bm25_model, handle)

        if self.embeddings is not None:
            np.save(save_path / "dense_embeddings.npy", self.embeddings)

        return save_path

    def _encode_query_dense(self, query: str) -> np.ndarray:
        text = query or ""
        if self.use_bge_m3 and self.embedding_model is not None and hasattr(self.embedding_model, "encode"):
            try:
                output = self.embedding_model.encode(  # type: ignore[union-attr]
                    [text],
                    batch_size=1,
                    max_length=self.max_length,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
                dense_vecs = output.get("dense_vecs")
                if dense_vecs is not None and len(dense_vecs) > 0:
                    return np.asarray(dense_vecs[0], dtype=np.float32)
            except Exception as exc:
                logger.warning("BGE-M3 query dense encoding failed: %s", exc)

        if self.embedding_model is not None and not self.use_bge_m3:
            vector = self.embedding_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(vector[0], dtype=np.float32)

        tokens = self._tokenize(text)
        stripped_tokens = self._tokenize(self._strip_diacritics(text))
        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        for token in tokens:
            index = abs(hash(token)) % self.embedding_dim
            vector[index] += 1.0
        for token in stripped_tokens:
            index = abs(hash(f"noaccent:{token}")) % self.embedding_dim
            vector[index] += 0.75
        for left, right in zip(tokens, tokens[1:]):
            index = abs(hash(f"bi:{left} {right}")) % self.embedding_dim
            vector[index] += 1.5
        if np.linalg.norm(vector) > 0:
            vector /= np.linalg.norm(vector)
        return vector

    def _encode_query_sparse(self, query: str) -> dict[str, float]:
        text = query or ""
        if self.use_bge_m3 and self.embedding_model is not None and hasattr(self.embedding_model, "encode"):
            try:
                output = self.embedding_model.encode(  # type: ignore[union-attr]
                    [text],
                    batch_size=1,
                    max_length=self.max_length,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
                sparse_weights = output.get("lexical_weights") or output.get("sparse_vecs") or []
                if sparse_weights:
                    return {str(token): float(weight) for token, weight in dict(sparse_weights[0]).items() if float(weight) > 0.0}
            except Exception as exc:
                logger.warning("BGE-M3 query sparse encoding failed: %s", exc)

        return {token: float(count) for token, count in Counter(self._tokenize(text)).items() if count > 0}

    # ------------------------------------------------------------------
    # Query-time API (added for orchestration/run_pipeline.py, T0-1/T1-3).
    # HybridIndexer previously only supported build_index()/save_index();
    # these methods close the gap so the pipeline can actually retrieve.
    # Hưng: feel free to move/refactor this into retrieval/router.py (T2-1)
    # once query decomposition / routing lands on top of it.
    # ------------------------------------------------------------------

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into the same embedding space as the corpus."""
        return self._encode_query_dense(query)

    def bm25_scores(self, query: str) -> np.ndarray:
        """Score every indexed document against a raw query string."""
        if not self.doc_ids:
            return np.zeros(0, dtype=np.float32)

        if self.use_bge_m3 and isinstance(self.bm25_model, list):
            query_weights = self._encode_query_sparse(query)
            scores = [self._sparse_score(query_weights, doc_weights) for doc_weights in self.bm25_model]
            return np.asarray(scores, dtype=np.float32)

        if self.bm25_model is None:
            return np.zeros(len(self.doc_ids), dtype=np.float32)

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

    def _sparse_score(self, query_weights: dict[str, float], doc_weights: dict[str, float]) -> float:
        if self.use_bge_m3 and self.embedding_model is not None and hasattr(self.embedding_model, "compute_lexical_matching_score"):
            try:
                return float(self.embedding_model.compute_lexical_matching_score(query_weights, doc_weights))
            except Exception:
                pass
        if not query_weights or not doc_weights:
            return 0.0
        overlap = set(query_weights) & set(doc_weights)
        return float(sum(query_weights[token] * doc_weights[token] for token in overlap))

    def search(self, query: str, top_k: int = 10, alpha: float = 0.45) -> list[dict[str, Any]]:
        """Hybrid BM25 + dense retrieval: score = alpha*BM25_norm + (1-alpha)*cosine.

        The dense side uses metadata-enriched semantic text and n-gram hashing
        (or sentence-transformer embeddings when enabled), so article numbers,
        law ids, and unit paths can influence the semantic neighborhood.
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
        use_bge_m3: bool = True,
        use_sentence_transformer: bool = False,
        device: str = "auto",
        batch_size: int = 12,
        max_length: int = 1024,
    ) -> "HybridIndexer":
        """Reconstruct an indexer instance from artifacts written by save_index()."""
        load_path = Path(path)
        indexer = cls(
            model_name=model_name,
            use_bge_m3=use_bge_m3,
            use_sentence_transformer=use_sentence_transformer,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

        with (load_path / "documents.jsonl").open("r", encoding="utf-8") as handle:
            indexer.documents = [json.loads(line) for line in handle if line.strip()]
        with (load_path / "doc_ids.json").open("r", encoding="utf-8") as handle:
            indexer.doc_ids = json.load(handle)
        with (load_path / "bm25_model.pkl").open("rb") as handle:
            indexer.bm25_model = pickle.load(handle)

        meta_path = load_path / "index_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    if "embedding_dim" in meta:
                        indexer.embedding_dim = int(meta["embedding_dim"])
                    if model_name is None and isinstance(meta.get("model_name"), str):
                        indexer.model_name = meta["model_name"]
                    if "use_bge_m3" in meta:
                        indexer.use_bge_m3 = bool(meta["use_bge_m3"])
                    if "use_sentence_transformer" in meta:
                        indexer.use_sentence_transformer = bool(meta["use_sentence_transformer"])
                    if isinstance(meta.get("device"), str) and device == "auto":
                        indexer.device = meta["device"]
                    if "batch_size" in meta:
                        indexer.batch_size = int(meta["batch_size"])
                    if "max_length" in meta:
                        indexer.max_length = int(meta["max_length"])
                    if isinstance(meta.get("retrieval_backend"), str):
                        indexer.retrieval_backend = meta["retrieval_backend"]
            except Exception:
                pass

        embeddings_path = load_path / "dense_embeddings.npy"
        indexer.embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
        if indexer.embeddings is not None and indexer.embeddings.ndim == 2:
            indexer.embedding_dim = int(indexer.embeddings.shape[1])

        indexer.contents = [doc.get("content", "") for doc in indexer.documents]
        indexer.metadata = [doc.get("metadata", {}) for doc in indexer.documents]

        if indexer.use_bge_m3:
            indexer._load_bge_model()
        elif indexer.use_sentence_transformer and SentenceTransformer is not None and torch is not None:
            indexer.embedding_model = SentenceTransformer(indexer.model_name)

        if isinstance(indexer.bm25_model, list):
            indexer.sparse_weights = indexer.bm25_model
            indexer.retrieval_backend = "bge_m3"
        elif indexer.bm25_model is not None:
            indexer.retrieval_backend = "bm25"

        return indexer

