from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.datasets.schema import DocumentChunk, NormalizedQARecord


def chunk_records(
    records: list[NormalizedQARecord], chunk_size: int = 800, chunk_overlap: int = 100
) -> list[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[DocumentChunk] = []
    for record in records:
        parts = splitter.split_text(record.document)
        for idx, text in enumerate(parts):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{record.example_id}_chunk_{idx}",
                    text=text,
                    metadata={
                        "dataset": record.dataset,
                        "split": record.split,
                        "example_id": record.example_id,
                        **record.metadata,
                    },
                )
            )
    return chunks
