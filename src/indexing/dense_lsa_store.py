import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.datasets.schema import DocumentChunk, RetrievedChunk


class DenseLSAStore:
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self._svd: TruncatedSVD | None = None
        self._doc_embeddings: np.ndarray | None = None
        self._tfidf_matrix = None
        self._chunks: list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build dense LSA index from empty chunks.")
        self._chunks = chunks
        texts = [chunk.text for chunk in chunks]
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

        max_components = min(
            128,
            max(1, self._tfidf_matrix.shape[0] - 1),
            max(1, self._tfidf_matrix.shape[1] - 1),
        )
        if max_components <= 1:
            dense = self._tfidf_matrix.toarray()
            self._svd = None
        else:
            self._svd = TruncatedSVD(n_components=max_components, random_state=42)
            dense = self._svd.fit_transform(self._tfidf_matrix)
        self._doc_embeddings = normalize(dense)

    def query(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self._doc_embeddings is None or self._tfidf_matrix is None:
            raise RuntimeError("Dense LSA index is not built or loaded.")
        query_tfidf = self._vectorizer.transform([query])
        if self._svd is None:
            query_dense = query_tfidf.toarray()
        else:
            query_dense = self._svd.transform(query_tfidf)
        query_dense = normalize(query_dense)

        scores = (self._doc_embeddings @ query_dense.T).reshape(-1)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                chunk_id=self._chunks[idx].chunk_id,
                text=self._chunks[idx].text,
                score=float(scores[idx]),
                metadata=self._chunks[idx].metadata,
            )
            for idx in order
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vectorizer": self._vectorizer,
            "svd": self._svd,
            "doc_embeddings": self._doc_embeddings,
            "tfidf_matrix": self._tfidf_matrix,
            "chunks": self._chunks,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path: Path) -> None:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        self._vectorizer = payload["vectorizer"]
        self._svd = payload["svd"]
        self._doc_embeddings = payload["doc_embeddings"]
        self._tfidf_matrix = payload["tfidf_matrix"]
        self._chunks = payload["chunks"]
