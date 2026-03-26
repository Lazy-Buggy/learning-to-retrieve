from pathlib import Path

from src.datasets.schema import DocumentChunk, RetrievedChunk
from src.indexing.vector_store import TfidfVectorStore


class Retriever:
    def __init__(self, index_path: Path) -> None:
        self._index_path = index_path
        self._store = TfidfVectorStore()

    def build(self, chunks: list[DocumentChunk]) -> None:
        self._store.build(chunks)

    def save(self) -> None:
        self._store.save(self._index_path)

    def load(self) -> None:
        self._store.load(self._index_path)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._store.query(query=query, top_k=top_k)
