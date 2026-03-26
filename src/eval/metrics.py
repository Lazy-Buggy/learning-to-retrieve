def evaluate_result(query: str, retrieved_count: int, answer: str) -> dict:
    return {
        "query_chars": len(query),
        "retrieved_count": retrieved_count,
        "answer_nonempty": bool(answer.strip()),
    }
