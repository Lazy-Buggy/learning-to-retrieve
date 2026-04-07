from src.datasets.hotpot_hf_loader import context_sentences_from_hf_rows, qa_records_from_hf_rows

def test_hotpot_hf_row_conversion():
    rows = [
        {
            "id": "row1",
            "question": "q",
            "answer": "a",
            "type": "bridge",
            "level": "easy",
            "context": {"title": ["T"], "sentences": [["S1", "S2"]]},
            "supporting_facts": {"title": ["T"], "sent_id": [0]},
        }
    ]
    records = qa_records_from_hf_rows(rows, split="validation")
    assert len(records) == 1
    first = records[0]
    assert first.dataset == "hotpot_qa"
    assert first.metadata["source"] == "huggingface"
    assert first.example_id == "row1"
    assert first.question == "q"
    assert first.answer == "a"
    assert first.supporting_facts[0]["title"] == "T"
    assert first.supporting_facts[0]["text"] == "S1"

    contexts = context_sentences_from_hf_rows(rows, split="train")
    assert len(contexts) == 2
    first_context = contexts[0]
    assert first_context.text == "S1"
    assert first_context.title == "T"
    assert first_context.dataset == "hotpot_qa"
