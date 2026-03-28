import argparse
import json
from pathlib import Path

from src.config import load_settings
from src.eval.metrics import reciprocal_rank_from_docs
from src.graph.rag_graph import build_rag_graph
from src.retrieval.retriever import Retriever, available_retrievers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LangGraph RAG evaluation.")
    parser.add_argument("--index-path", type=Path, default=None, help="Path to built index.")
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="JSONL file with fields: query and optional expected_answer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/eval_report.jsonl"),
        help="Where to write evaluation output JSONL.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Retriever top-k override.")
    parser.add_argument(
        "--retriever",
        type=str,
        default=None,
        choices=available_retrievers(),
        help="Retriever backend plugin. If omitted, load from index metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    index_path = args.index_path or settings.index_path
    top_k = args.top_k or settings.top_k

    retriever = Retriever(index_path=index_path, plugin=args.retriever)
    retriever.load()
    graph = build_rag_graph(retriever)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    reciprocal_rank_total = 0.0
    with args.queries.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as sink:
        for line in source:
            if not line.strip():
                continue
            payload = json.loads(line)
            query = payload["query"]
            expected_answer = payload.get("expected_answer", "")
            result = graph.invoke({"query": query, "top_k": top_k})
            retrieved_docs = result.get("retrieved_docs", [])
            reciprocal_rank = reciprocal_rank_from_docs(retrieved_docs, expected_answer)
            reciprocal_rank_total += reciprocal_rank
            row_metrics = dict(result.get("metrics", {}))
            row_metrics["reciprocal_rank"] = reciprocal_rank
            output_row = {
                "query": query,
                "expected_answer": expected_answer,
                "answer": result.get("answer", ""),
                "metrics": row_metrics,
            }
            sink.write(json.dumps(output_row, ensure_ascii=True) + "\n")
            rows_written += 1

    print(f"Wrote {rows_written} evaluation rows to {args.output}")
    if rows_written > 0:
        mrr = reciprocal_rank_total / rows_written
        print(f"MRR: {mrr:.6f}")


if __name__ == "__main__":
    main()
