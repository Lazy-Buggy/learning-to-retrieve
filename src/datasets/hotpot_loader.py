import json
from pathlib import Path

from src.datasets.schema import NormalizedQARecord


def _build_document(context: dict) -> str:
    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    sections: list[str] = []
    for title, sent_group in zip(titles, sentences, strict=False):
        joined = " ".join(sent_group)
        sections.append(f"{title}: {joined}".strip())
    return "\n".join(sections)


def load_hotpot(path: Path, split: str) -> list[NormalizedQARecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[NormalizedQARecord] = []
    for item in payload:
        records.append(
            NormalizedQARecord(
                dataset="hotpot_qa",
                split=split,
                example_id=str(item.get("id", "")),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                document=_build_document(item.get("context", {})),
                metadata={
                    "question_type": item.get("type", ""),
                    "difficulty": item.get("level", ""),
                },
                supporting_facts=[
                    {"title": t, "sent_id": sid}
                    for t, sid in zip(
                        item.get("supporting_facts", {}).get("title", []),
                        item.get("supporting_facts", {}).get("sent_id", []),
                        strict=False,
                    )
                ],
            )
        )
    return records
