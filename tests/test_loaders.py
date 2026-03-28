from pathlib import Path

from src.datasets.hotpot_hf_loader import records_from_hf_rows
from src.datasets.hotpot_loader import load_hotpot


FIXTURES = Path(__file__).parent / "fixtures"


def test_hotpot_loader_merges_context():
    records = load_hotpot(FIXTURES / "hotpot_sample.json", split="train")
    assert len(records) == 1
    first = records[0]
    assert first.dataset == "hotpot_qa"
    assert "Arthur's Magazine" in first.document
    assert first.metadata["question_type"] == "comparison"


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
    records = records_from_hf_rows(rows, split="train")
    assert len(records) == 1
    first = records[0]
    assert first.dataset == "hotpot_qa"
    assert first.metadata["source"] == "huggingface"
    assert "T: S1 S2" in first.document
