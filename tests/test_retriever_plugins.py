from src.datasets.schema import DocumentChunk
from src.retrieval.retriever import Retriever, available_retrievers


def test_available_retrievers_includes_tfidf():
    assert "tfidf" in available_retrievers()


def test_retriever_rejects_unknown_plugin(tmp_path):
    try:
        Retriever(index_path=tmp_path / "index.pkl", plugin="does_not_exist")
        assert False, "Expected ValueError for unknown retriever plugin."
    except ValueError as exc:
        assert "Unknown retriever plugin" in str(exc)


def test_retriever_autoloads_plugin_from_index_metadata(tmp_path):
    index_path = tmp_path / "index.pkl"
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            text="HotpotQA is a multi-hop question answering dataset.",
            metadata={"dataset": "hotpot_qa", "example_id": "hp1"},
        )
    ]

    builder = Retriever(index_path=index_path, plugin="tfidf")
    builder.build(chunks)
    builder.save()

    evaluator = Retriever(index_path=index_path)
    evaluator.load()
    retrieved = evaluator.retrieve("What dataset is multi-hop?", top_k=1)

    assert evaluator.plugin_name == "tfidf"
    assert len(retrieved) == 1
    assert retrieved[0].chunk_id == "c1"
