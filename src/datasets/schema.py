from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NormalizedQARecord:
    dataset: str
    split: str
    example_id: str
    question: str
    answer: str
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)
    supporting_facts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
