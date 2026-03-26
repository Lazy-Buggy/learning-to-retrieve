from src.datasets.schema import NormalizedQARecord
from src.processing.chunker import chunk_records


def test_chunker_preserves_metadata_and_overlap():
    record = NormalizedQARecord(
        dataset="quality",
        split="train",
        example_id="ex1",
        question="q",
        answer="a",
        document=" ".join(["token"] * 200),
        metadata={"source": "fixture"},
        supporting_facts=[],
    )

    chunks = chunk_records([record], chunk_size=120, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(chunk.metadata["dataset"] == "quality" for chunk in chunks)
    assert all(chunk.metadata["example_id"] == "ex1" for chunk in chunks)
