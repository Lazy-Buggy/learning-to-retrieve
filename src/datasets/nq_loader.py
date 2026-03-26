import json
from pathlib import Path

from src.datasets.schema import NormalizedQARecord


def _document_from_tokens(tokens: list[dict]) -> list[str]:
    values: list[str] = []
    for token in tokens:
        if token.get("html_token"):
            continue
        text = token.get("token", "").strip()
        if text:
            values.append(text)
    return values


def _extract_short_answer(annotation: dict, raw_tokens: list[dict]) -> str:
    short_answers = annotation.get("short_answers", [])
    if short_answers:
        first = short_answers[0]
        start = int(first.get("start_token", 0))
        end = int(first.get("end_token", start))
        if end > start and end <= len(raw_tokens):
            values = [
                token.get("token", "").strip()
                for token in raw_tokens[start:end]
                if not token.get("html_token") and token.get("token", "").strip()
            ]
            return " ".join(values)
    yes_no = annotation.get("yes_no_answer", "NONE")
    if yes_no in {"YES", "NO"}:
        return yes_no
    return ""


def load_natural_questions(path: Path, split: str) -> list[NormalizedQARecord]:
    records: list[NormalizedQARecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if "document_tokens" not in raw:
                raise ValueError(
                    "Natural Questions record is missing 'document_tokens'. "
                    "This appears to be NQ-Open format, which does not include source context for RAG indexing. "
                    "Use original NQ records with document tokens."
                )
            raw_tokens = raw.get("document_tokens", [])
            tokens = _document_from_tokens(raw_tokens)
            annotations = raw.get("annotations", [])
            answer = ""
            if annotations:
                answer = _extract_short_answer(annotations[0], raw_tokens)
            records.append(
                NormalizedQARecord(
                    dataset="natural_questions",
                    split=split,
                    example_id=str(raw.get("example_id", "")),
                    question=raw.get("question_text", ""),
                    answer=answer,
                    document=" ".join(tokens),
                    metadata={"document_url": raw.get("document_url", "")},
                    supporting_facts=[],
                )
            )
    return records
