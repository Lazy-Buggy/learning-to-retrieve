from src.eval.metrics import reciprocal_rank_from_docs


def test_reciprocal_rank_first_hit():
    docs = [
        {"text": "Larry Page and Sergey Brin founded Google."},
        {"text": "HotpotQA provides supporting facts."},
    ]
    assert reciprocal_rank_from_docs(docs, "Larry Page and Sergey Brin") == 1.0


def test_reciprocal_rank_second_hit():
    docs = [
        {"text": "HotpotQA provides supporting facts."},
        {"text": "Larry Page and Sergey Brin founded Google."},
    ]
    assert reciprocal_rank_from_docs(docs, "Larry Page and Sergey Brin") == 0.5


def test_reciprocal_rank_no_hit_or_missing_answer():
    docs = [{"text": "HotpotQA provides supporting facts."}]
    assert reciprocal_rank_from_docs(docs, "Larry Page and Sergey Brin") == 0.0
    assert reciprocal_rank_from_docs(docs, "") == 0.0
