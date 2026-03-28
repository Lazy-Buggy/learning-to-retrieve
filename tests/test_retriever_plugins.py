from src.datasets.schema import DocumentChunk
from src.retrieval.retriever import Retriever, available_retrievers


def test_available_retrievers_includes_tfidf():
    assert "tfidf" in available_retrievers()
    assert "bm25" in available_retrievers()
    assert "dense_lsa" in available_retrievers()
    assert "hybrid_rrf" in available_retrievers()
    assert "tfidf_rerank" in available_retrievers()
    assert "iterative_hybrid" in available_retrievers()


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


def test_bm25_retrieves_expected_chunk(tmp_path):
    index_path = tmp_path / "bm25_index.pkl"
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            text="The Eiffel Tower is located in Paris, France.",
            metadata={"dataset": "hotpot_qa", "example_id": "hp1"},
        ),
        DocumentChunk(
            chunk_id="c2",
            text="Mount Everest is the tallest mountain above sea level.",
            metadata={"dataset": "hotpot_qa", "example_id": "hp2"},
        ),
    ]
    retriever = Retriever(index_path=index_path, plugin="bm25")
    retriever.build(chunks)
    retriever.save()
    retriever.load()
    retrieved = retriever.retrieve("Where is the Eiffel Tower?", top_k=1)
    assert retrieved[0].chunk_id == "c1"


def test_hybrid_and_iterative_plugins_return_results(tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            text="Arthur's Magazine was an American literary periodical published in Philadelphia.",
            metadata={"dataset": "hotpot_qa", "example_id": "hp1"},
        ),
        DocumentChunk(
            chunk_id="c2",
            text="First for Women is a woman's magazine published by Bauer Media Group in the USA.",
            metadata={"dataset": "hotpot_qa", "example_id": "hp2"},
        ),
    ]

    hybrid = Retriever(index_path=tmp_path / "hybrid.pkl", plugin="hybrid_rrf")
    hybrid.build(chunks)
    hybrid.save()
    hybrid.load()
    hybrid_results = hybrid.retrieve("Which magazine is published in the USA?", top_k=2)
    assert len(hybrid_results) == 2

    iterative = Retriever(index_path=tmp_path / "iterative.pkl", plugin="iterative_hybrid")
    iterative.build(chunks)
    iterative.save()
    iterative.load()
    iterative_results = iterative.retrieve("Which magazine is published in the USA?", top_k=2)
    assert len(iterative_results) == 2
