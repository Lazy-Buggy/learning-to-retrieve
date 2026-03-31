from pathlib import Path

from src.datasets.hotpot_hf_loader import qa_records_from_hf_rows, context_sentences_from_hf_rows


FIXTURES = Path(__file__).parent / "fixtures"



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

    contexts = context_sentences_from_hf_rows(rows, split="train")
    assert len(contexts) == 2
    first = records[0]
    assert first.text == "S1"
    assert first.title == "T"
