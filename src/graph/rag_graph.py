from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.eval.metrics import evaluate_result
from src.retrieval.retriever import Retriever


class RAGState(TypedDict):
    query: str
    top_k: NotRequired[int]
    retrieved_docs: NotRequired[list[dict]]
    context: NotRequired[str]
    answer: NotRequired[str]
    metrics: NotRequired[dict]


def build_rag_graph(retriever: Retriever):
    def retrieve_node(state: RAGState) -> RAGState:
        top_k = int(state.get("top_k", 5))
        retrieved = retriever.retrieve(state["query"], top_k=top_k)
        return {
            "retrieved_docs": [
                {"chunk_id": item.chunk_id, "text": item.text, "score": item.score, "metadata": item.metadata}
                for item in retrieved
            ]
        }

    def context_node(state: RAGState) -> RAGState:
        docs = state.get("retrieved_docs", [])
        context = "\n".join(doc["text"] for doc in docs)
        return {"context": context}

    def answer_node(state: RAGState) -> RAGState:
        context = state.get("context", "")
        if context.strip():
            answer = f"Based on retrieved context: {context.splitlines()[0]}"
        else:
            answer = "No relevant context found."
        return {"answer": answer}

    def score_node(state: RAGState) -> RAGState:
        docs = state.get("retrieved_docs", [])
        answer = state.get("answer", "")
        return {"metrics": evaluate_result(state["query"], len(docs), answer)}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("assemble_context", context_node)
    graph.add_node("generate", answer_node)
    graph.add_node("score", score_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "assemble_context")
    graph.add_edge("assemble_context", "generate")
    graph.add_edge("generate", "score")
    graph.add_edge("score", END)
    return graph.compile()
