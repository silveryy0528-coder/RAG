# RAG

A small retrieval-augmented generation project built around SentenceTransformers, FAISS, and a lightweight query pipeline.

## Overview

The repository does four main things:

- ingest PDFs into text chunks,
- embed chunks with a SentenceTransformer model,
- build and query a FAISS index,
- generate answers from retrieved context with an OpenAI-compatible client.

## Installation

Recommended workflow:

1. Create a virtual environment.

   ```bash
   python -m venv .venv
   . .venv/bin/activate
   # or on Windows: .\.venv\Scripts\activate
   ```

2. Install PyTorch first for the selected platform.

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

   See https://pytorch.org/get-started/locally/ for alternatives.

3. Install runtime dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. Install the project in editable mode for packaging work.

   ```bash
   pip install -e .
   ```

## Quick start

### 1. Build the index

```bash
python -m scripts.build_index --raw-dir data/raw --processed-dir data/processed
```

### 2. Start an interactive chat loop

```bash
python -m scripts.rag_chat --processed-dir data/processed
```

Then type one question at a time. Use `exit` or `quit` to stop.

### 3. Run a single query

```bash
python -m scripts.rag_chat "What is the title of the thesis?" --processed-dir data/processed
```

### 4. Evaluate a QA dataset

```bash
python -m scripts.evaluate --dataset data/evaluation/qa_dataset.json --processed-dir data/processed
```

## Environment variables

- `OPENAI_API_KEY`: required for LLM generation.
- `HF_TOKEN`: optional Hugging Face token for faster model downloads.
- `HF_HUB_DISABLE_SYMLINKS_WARNING=1`: recommended on Windows in CI.

## Project layout

- `rag/`: core library code (embedding, retrieval, prompting, evaluation)
- `scripts/`: top-level CLI entry points
- `data/raw/`: raw PDF inputs
- `data/processed/`: generated chunks/index files
- `results/`: profiling, query, and offline evaluation outputs
- `tests/`: pytest coverage for the library and scripts

## Packaging notes

The project is now installable with `pip install -e .` and exposes console commands:

- `rag-chat`
- `rag-build`
- `rag-evaluate`

## CI notes

- The GitHub Actions workflow should cache pip and Hugging Face downloads.
- If the workflow uses a GPU image, keep the PyTorch install aligned with it.

## License

This project is currently provided for educational use without a dedicated license file.
