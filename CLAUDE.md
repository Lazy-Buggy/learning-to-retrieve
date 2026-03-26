# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_loaders.py -v

# Run a single test by name
pytest tests/test_graph_smoke.py::test_graph_smoke_returns_answer_and_metrics -v

# Build TF-IDF index from HotpotQA (Hugging Face)
python -m src.pipelines.build_index \
  --hotpot-hf-config distractor \
  --split train

# Run evaluation against a JSONL query file
python -m src.pipelines.run_eval \
  --queries data/processed/sample_queries.jsonl \
  --output data/processed/eval_report.jsonl
```

Configuration is via environment variables (see `.env.example`). Copy to `.env` and adjust; all settings have working defaults.

## Architecture

The pipeline has five stages that flow from raw data to evaluated answers:

**1. Load & Normalize** (`src/datasets/`)
Hotpot loaders (`hotpot_loader`, `hotpot_hf_loader`) parse local JSON or Hugging Face dataset records and emit `NormalizedQARecord` objects defined in `schema.py`. This is the only format used downstream—loaders are the only place format-specific logic lives.

**2. Chunk** (`src/processing/chunker.py`)
`chunk_records()` wraps LangChain's `RecursiveCharacterTextSplitter` and produces `DocumentChunk` objects. Chunk metadata carries `dataset`, `split`, `example_id`, and any loader-specific metadata forward.

**3. Index** (`src/indexing/vector_store.py`)
`TfidfVectorStore` fits a scikit-learn `TfidfVectorizer` over all chunk texts and persists the vectorizer + sparse matrix + chunk list as a single pickle file. `Retriever` in `src/retrieval/retriever.py` is a thin wrapper used by the graph and pipelines.

**4. RAG Graph** (`src/graph/rag_graph.py`)
A four-node LangGraph state machine: `retrieve → assemble_context → generate → score`. The `generate` node is intentionally deterministic for this baseline—it prefixes the first line of retrieved context rather than calling an LLM. This is the integration point for Milestone 2 LLM work.

**5. Evaluate** (`src/eval/metrics.py`, `src/pipelines/run_eval.py`)
`evaluate_result()` records query character length, retrieval count, and whether the answer is non-empty. `run_eval.py` also computes reciprocal rank per query (using `expected_answer` against retrieved text) and prints aggregate MRR.

## Known Limitations (Milestone 1 Baseline)

- **Answer generation is not LLM-backed.** `answer_node` in `rag_graph.py` returns the first line of retrieved context with a fixed prefix. Intended to be replaced in Milestone 2.
- **Evaluation metrics are still baseline-level.** `metrics.py` includes reciprocal rank and aggregate MRR, but still has no F1, BLEU, or exact-match scoring.
- **`OPENAI_API_KEY` in `.env.example` is unused.** Placeholder for future LLM integration.
- **Index uses pickle.** Adequate for local development; not safe for untrusted index files.
