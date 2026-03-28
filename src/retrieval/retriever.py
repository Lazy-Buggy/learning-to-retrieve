from pathlib import Path
import json

from src.datasets.schema import DocumentChunk, RetrievedChunk
from src.retrieval.plugins import available_retrievers, create_retriever_plugin, RetrieverPlugin


class Retriever:
    def __init__(self, index_path: Path, plugin: str | None = None) -> None:
        self._index_path = index_path
        self._plugin_name = plugin
        self._backend: RetrieverPlugin | None = None
        if plugin is not None:
            self._backend = create_retriever_plugin(plugin)

    @property
    def plugin_name(self) -> str | None:
        return self._plugin_name

    def _metadata_path(self) -> Path:
        return self._index_path.with_name(f"{self._index_path.name}.meta.json")

    def _ensure_backend(self) -> RetrieverPlugin:
        if self._backend is None:
            # Default to tfidf for backward compatibility when plugin is omitted.
            self._plugin_name = "tfidf"
            self._backend = create_retriever_plugin(self._plugin_name)
        return self._backend

    def _write_metadata(self) -> None:
        if self._plugin_name is None:
            raise RuntimeError("Cannot save metadata without a configured retriever plugin.")
        meta_path = self._metadata_path()
        payload = {"plugin": self._plugin_name}
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

    def _configure_backend_from_metadata(self) -> None:
        meta_path = self._metadata_path()
        plugin_name = "tfidf"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            plugin_name = str(payload.get("plugin", "tfidf"))
        self._plugin_name = plugin_name
        self._backend = create_retriever_plugin(plugin_name)

    def build(self, chunks: list[DocumentChunk]) -> None:
        backend = self._ensure_backend()
        backend.build(chunks)

    def save(self) -> None:
        backend = self._ensure_backend()
        backend.save(self._index_path)
        self._write_metadata()

    def load(self) -> None:
        if self._backend is None:
            self._configure_backend_from_metadata()
        backend = self._ensure_backend()
        backend.load(self._index_path)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        backend = self._ensure_backend()
        return backend.retrieve(query=query, top_k=top_k)


__all__ = ["Retriever", "available_retrievers"]
