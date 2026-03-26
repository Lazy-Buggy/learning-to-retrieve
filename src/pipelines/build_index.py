import argparse
from pathlib import Path

from src.config import load_settings
from src.datasets.hotpot_loader import load_hotpot
from src.datasets.nq_loader import load_natural_questions
from src.datasets.quality_loader import load_quality
from src.processing.chunker import chunk_records
from src.retrieval.retriever import Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local TF-IDF RAG index.")
    parser.add_argument("--quality", type=Path, help="Path to QuALITY JSONL file.")
    parser.add_argument("--nq", type=Path, help="Path to Natural Questions JSONL file.")
    parser.add_argument("--hotpot", type=Path, help="Path to HotpotQA JSON file.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split label.")
    parser.add_argument("--index-path", type=Path, default=None, help="Output index file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    index_path = args.index_path or settings.index_path

    records = []
    if args.quality:
        records.extend(load_quality(args.quality, split=args.split))
    if args.nq:
        records.extend(load_natural_questions(args.nq, split=args.split))
    if args.hotpot:
        records.extend(load_hotpot(args.hotpot, split=args.split))

    if not records:
        raise SystemExit("No datasets provided. Use --quality, --nq, or --hotpot.")

    chunks = chunk_records(
        records,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    retriever = Retriever(index_path=index_path)
    retriever.build(chunks)
    retriever.save()

    print(f"Built index at: {index_path}")
    print(f"Records processed: {len(records)}")
    print(f"Chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()
