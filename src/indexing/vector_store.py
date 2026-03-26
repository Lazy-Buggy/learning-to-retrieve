import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.datasets.schema import DocumentChunk, RetrievedChunk


class TfidfVectorStore:
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer()
        self._matrix = None
        self._chunks: list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build index from empty chunks.")
        self._chunks = chunks
        texts = [chunk.text for chunk in chunks]
        self._matrix = self._vectorizer.fit_transform(texts)

    def query(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self._matrix is None:
            raise RuntimeError("Index is not built or loaded.")
        query_vec = self._vectorizer.transform([query])
        scores = (self._matrix @ query_vec.T).toarray().reshape(-1)
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
            "matrix": self._matrix,
            "chunks": self._chunks,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path: Path) -> None:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        self._vectorizer = payload["vectorizer"]
        self._matrix = payload["matrix"]
        self._chunks = payload["chunks"]
