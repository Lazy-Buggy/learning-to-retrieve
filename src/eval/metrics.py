def reciprocal_rank_from_docs(retrieved_docs: list[dict], expected_answer: str) -> float:
    expected = expected_answer.strip().lower()
    if not expected:
        return 0.0

    for idx, doc in enumerate(retrieved_docs, start=1):
        text = str(doc.get("text", "")).lower()
        if expected in text:
            return 1.0 / idx
    return 0.0


def evaluate_result(query: str, retrieved_count: int, answer: str) -> dict:
    return {
        "query_chars": len(query),
        "retrieved_count": retrieved_count,
        "answer_nonempty": bool(answer.strip()),
    }
