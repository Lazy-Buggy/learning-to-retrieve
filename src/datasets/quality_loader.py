import json
from pathlib import Path

from src.datasets.schema import NormalizedQARecord


def load_quality(path: Path, split: str) -> list[NormalizedQARecord]:
    records: list[NormalizedQARecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            article_id = str(raw.get("article_id", "unknown"))
            article = raw.get("article", "")
            title = raw.get("title", "")
            questions = raw.get("questions", [])
            for idx, item in enumerate(questions):
                options = item.get("options", [])
                gold_label = int(item.get("gold_label", 1))
                option_index = max(0, min(len(options) - 1, gold_label - 1)) if options else 0
                answer = options[option_index] if options else ""
                records.append(
                    NormalizedQARecord(
                        dataset="quality",
                        split=split,
                        example_id=f"{article_id}_{idx}",
                        question=item.get("question", ""),
                        answer=answer,
                        document=article,
                        metadata={
                            "title": title,
                            "article_id": article_id,
                            "difficulty": item.get("difficult", 0),
                        },
                        supporting_facts=[],
                    )
                )
    return records
