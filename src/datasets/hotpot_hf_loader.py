from src.datasets.schema import NormalizedQARecord


def records_from_hf_rows(rows, split: str) -> list[NormalizedQARecord]:
    records: list[NormalizedQARecord] = []
    for item in rows:
        context = item.get("context", {})
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        sections: list[str] = []
        for title, sent_group in zip(titles, sentences, strict=False):
            joined = " ".join(sent_group)
            sections.append(f"{title}: {joined}".strip())

        supporting = item.get("supporting_facts", {})
        supporting_facts = [
            {"title": t, "sent_id": sid}
            for t, sid in zip(
                supporting.get("title", []),
                supporting.get("sent_id", []),
                strict=False,
            )
        ]

        records.append(
            NormalizedQARecord(
                dataset="hotpot_qa",
                split=split,
                example_id=str(item.get("id", "")),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                document="\n".join(sections),
                metadata={
                    "question_type": item.get("type", ""),
                    "difficulty": item.get("level", ""),
                    "source": "huggingface",
                },
                supporting_facts=supporting_facts,
            )
        )
    return records


def load_hotpot_hf(config: str, split: str) -> list[NormalizedQARecord]:
    from datasets import load_dataset

    dataset = load_dataset("hotpotqa/hotpot_qa", config, split=split)
    return records_from_hf_rows(dataset, split=split)
