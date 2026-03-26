from pathlib import Path

from src.datasets.hotpot_loader import load_hotpot
from src.datasets.nq_loader import load_natural_questions
from src.datasets.quality_loader import load_quality


FIXTURES = Path(__file__).parent / "fixtures"


def test_quality_loader_normalizes_questions():
    records = load_quality(FIXTURES / "quality_sample.jsonl", split="train")
    assert len(records) == 1
    first = records[0]
    assert first.dataset == "quality"
    assert first.question == "What is alpha?"
    assert first.answer == "A"
    assert "Alpha beta gamma" in first.document


def test_nq_loader_extracts_short_answer():
    records = load_natural_questions(FIXTURES / "nq_sample.jsonl", split="train")
    assert len(records) == 1
    first = records[0]
    assert first.dataset == "natural_questions"
    assert first.answer == "Larry Page and Sergey Brin"
    assert "Google was founded by" in first.document


def test_hotpot_loader_merges_context():
    records = load_hotpot(FIXTURES / "hotpot_sample.json", split="train")
    assert len(records) == 1
    first = records[0]
    assert first.dataset == "hotpot_qa"
    assert "Arthur's Magazine" in first.document
    assert first.metadata["question_type"] == "comparison"
