import argparse
from pathlib import Path

from src.config import load_settings
from src.datasets.hotpot_hf_loader import load_hotpot_hf
from src.datasets.hotpot_loader import load_hotpot
from src.processing.chunker import chunk_records
from src.retrieval.retriever import Retriever, available_retrievers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local RAG index with a pluggable retriever.")
    parser.add_argument("--hotpot", type=Path, help="Path to HotpotQA JSON file.")
    parser.add_argument(
        "--hotpot-hf-config",
        type=str,
        choices=["distractor", "fullwiki"],
        help="Load HotpotQA directly from Hugging Face by config.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split label.")
    parser.add_argument("--index-path", type=Path, default=None, help="Output index file path.")
    parser.add_argument(
        "--retriever",
        type=str,
        default="tfidf",
        choices=available_retrievers(),
        help="Retriever backend plugin to build the index with.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    index_path = args.index_path or settings.index_path

    records = []
    if args.hotpot:
        records.extend(load_hotpot(args.hotpot, split=args.split))
    if args.hotpot_hf_config:
        records.extend(load_hotpot_hf(config=args.hotpot_hf_config, split=args.split))

    if not records:
        raise SystemExit("No datasets provided. Use --hotpot or --hotpot-hf-config.")

    chunks = chunk_records(
        records,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    retriever = Retriever(index_path=index_path, plugin=args.retriever)
    retriever.build(chunks)
    retriever.save()

    print(f"Built index at: {index_path}")
    print(f"Retriever backend: {retriever.plugin_name}")
    print(f"Records processed: {len(records)}")
    print(f"Chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()
