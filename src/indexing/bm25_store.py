import math
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.datasets.schema import DocumentChunk, RetrievedChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


class BM25Store:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: list[DocumentChunk] = []
        self._doc_term_freqs: list[Counter[str]] = []
        self._doc_lengths: np.ndarray | None = None
        self._avg_doc_len: float = 0.0
        self._idf: dict[str, float] = {}

    def build(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunks.")
        self._chunks = chunks
        tokenized_docs = [_tokenize(chunk.text) for chunk in chunks]
        self._doc_term_freqs = [Counter(tokens) for tokens in tokenized_docs]

        doc_freq: Counter[str] = Counter()
        for tokens in tokenized_docs:
            doc_freq.update(set(tokens))

        num_docs = len(chunks)
        self._doc_lengths = np.array([len(tokens) for tokens in tokenized_docs], dtype=float)
        self._avg_doc_len = float(self._doc_lengths.mean()) if num_docs else 0.0

        self._idf = {
            term: math.log(1.0 + (num_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def query(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self._doc_lengths is None:
            raise RuntimeError("BM25 index is not built or loaded.")
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = np.zeros(len(self._chunks), dtype=float)
        for idx, tf in enumerate(self._doc_term_freqs):
            doc_len = self._doc_lengths[idx]
            denom_adjust = self._k1 * (1.0 - self._b + self._b * (doc_len / max(self._avg_doc_len, 1e-6)))
            score = 0.0
            for term in tokens:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                score += idf * ((f * (self._k1 + 1.0)) / (f + denom_adjust))
            scores[idx] = score

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
            "k1": self._k1,
            "b": self._b,
            "chunks": self._chunks,
            "doc_term_freqs": self._doc_term_freqs,
            "doc_lengths": self._doc_lengths,
            "avg_doc_len": self._avg_doc_len,
            "idf": self._idf,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path: Path) -> None:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        self._k1 = payload["k1"]
        self._b = payload["b"]
        self._chunks = payload["chunks"]
        self._doc_term_freqs = payload["doc_term_freqs"]
        self._doc_lengths = payload["doc_lengths"]
        self._avg_doc_len = payload["avg_doc_len"]
        self._idf = payload["idf"]
