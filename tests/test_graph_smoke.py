from src.datasets.schema import DocumentChunk
from src.graph.rag_graph import build_rag_graph
from src.retrieval.retriever import Retriever


def test_graph_smoke_returns_answer_and_metrics(tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            text="Google was founded by Larry Page and Sergey Brin.",
            metadata={"dataset": "natural_questions", "example_id": "nq1"},
        ),
        DocumentChunk(
            chunk_id="c2",
            text="HotpotQA contains multi-hop questions.",
            metadata={"dataset": "hotpot_qa", "example_id": "hp1"},
        ),
    ]

    retriever = Retriever(index_path=tmp_path / "index.pkl")
    retriever.build(chunks)
    retriever.save()

    graph = build_rag_graph(retriever)
    result = graph.invoke({"query": "Who founded Google?", "top_k": 1})

    assert "answer" in result
    assert "metrics" in result
    assert result["metrics"]["retrieved_count"] == 1
    assert "Google" in result["answer"]
