from src.datasets.schema import NormalizedQARecord, NornalizedContextSentence


def context_sentences_from_hf_rows(rows, split:str) -> list[NornalizedContextSentence]:
    contexts : list[NornalizedContextSentence] = []
    title_set = set()
    for item in rows:
        context = item.get("context", {})
        context_titles = context.get("title", [])
        context_sentences = context.get("sentences", [])
        for title, sentences in zip(context_titles, context_sentences):
            if title not in title_set:
                for i, sent in enumerate(sentences):
                    contexts.append(
                        NornalizedContextSentence(
                            dataset="hotpot_qa",
                            split=split,
                            example_id=str(item.get("id", "")),
                            sent_id=i,
                            title=title,
                            text=sent
                        )
                    )
                title_set.add(title)
    return contexts


def qa_records_from_hf_rows(rows, split: str) -> list[NormalizedQARecord]:
    records: list[NormalizedQARecord] = []

    for item in rows:

        context = item.get("context", {})
        context_titles = context.get("title", [])
        context_sentences = context.get("sentences", [])

        supporting = item.get("supporting_facts", {})
        supporting_titles = supporting.get("title", [])
        supporting_sent_ids = supporting.get("sent_id", [])

        supporting_facts = []

        for title, sent_id in zip(supporting_titles, supporting_sent_ids):
            if title in context_titles:
                idx = context_titles.index(title)
                sentences = context_sentences[idx]
                if sent_id < len(sentences):
                    sent_text = sentences[sent_id]
                else:
                    sent_text = ""
                    # print(f"{title} {sent_id} not found")
                supporting_facts.append(
                    {'title': title,
                     'sent_id': sent_id,
                     'text': sent_text
                     }
                )
            else:
                print(f"{title} not found")

        records.append(
            NormalizedQARecord(
                dataset="hotpot_qa",
                split=split,
                example_id=str(item.get("id", "")),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                metadata={
                    "question_type": item.get("type", ""),
                    "difficulty": item.get("level", ""),
                    "source": "huggingface",
                },
                supporting_facts=supporting_facts,
            )
        )
    return records


def load_hotpot_qa_records_hf(config: str, split: str) -> list[NormalizedQARecord]:
    from datasets import load_dataset

    dataset = load_dataset("hotpotqa/hotpot_qa", config, split=split)
    return qa_records_from_hf_rows(dataset, split=split)


def load_hotpot_contexts_hf(config: str, split: str) -> list[NornalizedContextSentence]:
    from datasets import load_dataset

    dataset = load_dataset("hotpotqa/hotpot_qa", config, split=split)
    return context_sentences_from_hf_rows(dataset, split=split)


if __name__ == '__main__':
    # records = load_hotpot_qa_records_hf('distractor', split='validation')
    # print(records[0])
    contexts = load_hotpot_contexts_hf('distractor', split='validation')
    print(contexts[0])
