import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    data_root: Path
    index_path: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int


def load_settings() -> Settings:
    data_root = Path(os.getenv("RAG_DATA_ROOT", "./data"))
    index_path = Path(os.getenv("RAG_INDEX_PATH", str(data_root / "index" / "index.pkl")))
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    top_k = int(os.getenv("RAG_TOP_K", "5"))
    return Settings(
        data_root=data_root,
        index_path=index_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
    )
