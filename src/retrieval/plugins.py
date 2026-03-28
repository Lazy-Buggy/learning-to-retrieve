from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.datasets.schema import DocumentChunk, RetrievedChunk
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


_RETRIEVER_REGISTRY: dict[str, type[RetrieverPlugin]] = {
    "tfidf": TfidfRetrieverPlugin,
}


def available_retrievers() -> list[str]:
    return sorted(_RETRIEVER_REGISTRY.keys())


def create_retriever_plugin(name: str) -> RetrieverPlugin:
    plugin_cls = _RETRIEVER_REGISTRY.get(name)
    if plugin_cls is None:
        available = ", ".join(available_retrievers())
        raise ValueError(f"Unknown retriever plugin '{name}'. Available: {available}")
    return plugin_cls()
