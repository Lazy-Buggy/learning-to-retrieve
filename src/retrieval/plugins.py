from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Protocol

from src.datasets.schema import DocumentChunk, RetrievedChunk
from src.indexing.bm25_store import BM25Store
from src.indexing.dense_lsa_store import DenseLSAStore
from src.indexing.vector_store import TfidfVectorStore


class RetrieverPlugin(Protocol):
    def build(self, chunks: list[DocumentChunk]) -> None: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


class TfidfRetrieverPlugin:
    def __init__(self) -> None:
        self._store = TfidfVectorStore()

    def build(self, chunks: list[DocumentChunk]) -> None:
        self._store.build(chunks)

    def save(self, path: Path) -> None:
        self._store.save(path)

    def load(self, path: Path) -> None:
        self._store.load(path)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._store.query(query=query, top_k=top_k)


class BM25RetrieverPlugin:
    def __init__(self) -> None:
        self._store = BM25Store()

    def build(self, chunks: list[DocumentChunk]) -> None:
        self._store.build(chunks)

    def save(self, path: Path) -> None:
        self._store.save(path)

    def load(self, path: Path) -> None:
        self._store.load(path)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._store.query(query=query, top_k=top_k)


class DenseLSARetrieverPlugin:
    def __init__(self) -> None:
        self._store = DenseLSAStore()

    def build(self, chunks: list[DocumentChunk]) -> None:
        self._store.build(chunks)

    def save(self, path: Path) -> None:
        self._store.save(path)

    def load(self, path: Path) -> None:
        self._store.load(path)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._store.query(query=query, top_k=top_k)


class HybridRRFPlugin:
    def __init__(self, rank_constant: int = 60, fanout_multiplier: int = 4) -> None:
        self._rank_constant = rank_constant
        self._fanout_multiplier = fanout_multiplier
        self._tfidf = TfidfVectorStore()
        self._bm25 = BM25Store()

    def build(self, chunks: list[DocumentChunk]) -> None:
        self._tfidf.build(chunks)
        self._bm25.build(chunks)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rank_constant": self._rank_constant,
            "fanout_multiplier": self._fanout_multiplier,
            "tfidf_store": self._tfidf,
            "bm25_store": self._bm25,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path: Path) -> None:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        self._rank_constant = payload["rank_constant"]
        self._fanout_multiplier = payload["fanout_multiplier"]
        self._tfidf = payload["tfidf_store"]
        self._bm25 = payload["bm25_store"]

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        candidate_k = max(top_k * self._fanout_multiplier, top_k)
        tfidf_hits = self._tfidf.query(query=query, top_k=candidate_k)
        bm25_hits = self._bm25.query(query=query, top_k=candidate_k)

        fused_scores: dict[str, float] = {}
        chosen_hits: dict[str, RetrievedChunk] = {}

        for rank, hit in enumerate(tfidf_hits, start=1):
            fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + 1.0 / (
                self._rank_constant + rank
            )
            chosen_hits[hit.chunk_id] = hit

        for rank, hit in enumerate(bm25_hits, start=1):
            fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + 1.0 / (
                self._rank_constant + rank
            )
            chosen_hits[hit.chunk_id] = hit

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=chosen_hits[chunk_id].text,
                score=float(score),
                metadata=chosen_hits[chunk_id].metadata,
            )
            for chunk_id, score in ranked
        ]


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_PATTERN.findall(text)}


class TfidfRerankPlugin:
    def __init__(self, fanout_multiplier: int = 5) -> None:
        self._fanout_multiplier = fanout_multiplier
        self._tfidf = TfidfVectorStore()

    def build(self, chunks: list[DocumentChunk]) -> None:
        self._tfidf.build(chunks)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fanout_multiplier": self._fanout_multiplier, "tfidf_store": self._tfidf}
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path: Path) -> None:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        self._fanout_multiplier = payload["fanout_multiplier"]
        self._tfidf = payload["tfidf_store"]

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        initial_k = max(top_k * self._fanout_multiplier, top_k)
        candidates = self._tfidf.query(query=query, top_k=initial_k)
        query_terms = _tokenize(query)
        scored: list[tuple[float, RetrievedChunk]] = []

        for hit in candidates:
            doc_terms = _tokenize(hit.text)
            if not query_terms:
                overlap = 0.0
            else:
                overlap = len(query_terms.intersection(doc_terms)) / len(query_terms)
            combined = (0.85 * overlap) + (0.15 * hit.score)
            scored.append((combined, hit))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=hit.chunk_id,
                text=hit.text,
                score=float(score),
                metadata=hit.metadata,
            )
            for score, hit in scored[:top_k]
        ]


class IterativeHybridPlugin:
    def __init__(self) -> None:
        self._hybrid = HybridRRFPlugin()

    def build(self, chunks: list[DocumentChunk]) -> None:
        self._hybrid.build(chunks)

    def save(self, path: Path) -> None:
        self._hybrid.save(path)

    def load(self, path: Path) -> None:
        self._hybrid.load(path)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        first_hop = self._hybrid.retrieve(query=query, top_k=top_k)
        if not first_hop:
            return []

        expansion_terms = " ".join(first_hop[0].text.split()[:12])
        expanded_query = f"{query} {expansion_terms}".strip()
        second_hop = self._hybrid.retrieve(query=expanded_query, top_k=top_k)

        fused: dict[str, RetrievedChunk] = {}
        for hit in first_hop:
            fused[hit.chunk_id] = RetrievedChunk(
                chunk_id=hit.chunk_id,
                text=hit.text,
                score=hit.score,
                metadata=hit.metadata,
            )
        for hit in second_hop:
            existing = fused.get(hit.chunk_id)
            if existing is None:
                fused[hit.chunk_id] = RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    score=hit.score,
                    metadata=hit.metadata,
                )
            else:
                existing.score = max(existing.score, hit.score) + 0.05

        ranked = sorted(fused.values(), key=lambda x: x.score, reverse=True)
        return ranked[:top_k]


_RETRIEVER_REGISTRY: dict[str, type[RetrieverPlugin]] = {
    "tfidf": TfidfRetrieverPlugin,
    "bm25": BM25RetrieverPlugin,
    "dense_lsa": DenseLSARetrieverPlugin,
    "hybrid_rrf": HybridRRFPlugin,
    "tfidf_rerank": TfidfRerankPlugin,
    "iterative_hybrid": IterativeHybridPlugin,
}


def available_retrievers() -> list[str]:
    return sorted(_RETRIEVER_REGISTRY.keys())


def create_retriever_plugin(name: str) -> RetrieverPlugin:
    plugin_cls = _RETRIEVER_REGISTRY.get(name)
    if plugin_cls is None:
        available = ", ".join(available_retrievers())
        raise ValueError(f"Unknown retriever plugin '{name}'. Available: {available}")
    return plugin_cls()
