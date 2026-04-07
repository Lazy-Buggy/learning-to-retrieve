from src.datasets.schema import NornalizedContextSentence
from src.processing.chunker import chunk_contexts


def test_chunker_preserves_metadata_and_overlap():
    context = NornalizedContextSentence(
        dataset="hotpot_qa",
        split="train",
        example_id="ex1",
        sent_id="0",
        title="T",
        text=" ".join(["token"] * 200),
    )

    chunks = chunk_contexts([context], chunk_size=120, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(chunk.metadata["dataset"] == "hotpot_qa" for chunk in chunks)
    assert all(chunk.metadata["example_id"] == "ex1" for chunk in chunks)
