RAG — simple Retrieval-Augmented Generation demo

Description

A minimal RAG example that: reads PDF files, chunks text, embeds chunks with a SentenceTransformer model, builds a normalized FAISS index, and allows querying the index to generate answers using an LLM.

Installation

1. Create and activate a virtual environment (recommended):

   python -m venv .venv
   .\.venv\Scripts\activate

2. Install PyTorch first (sentence-transformers needs it). Follow instructions for your OS/GPU at https://pytorch.org/get-started/locally/ (example CPU install):

   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

3. Install the minimal runtime dependencies:

   pip install -r requirements.txt

Note: On Windows, faiss may be tricky to install — using faiss-cpu from PyPI is recommended when available.

Usage

1. Build the index from PDFs (reads data/raw and writes to data/processed):

   python scripts\build_index.py --raw-dir data\raw --out-dir data\processed

2. Query the index interactively from the command line:

   set OPENAI_API_KEY=your_key_here
   python scripts\query_index.py --index-dir data\processed --question "What are the main topics of the thesis?"

Environment variables

- OPENAI_API_KEY: required to call OpenAI-compatible LLM clients.
- HF_TOKEN: (optional) Hugging Face token to speed up model downloads and increase rate limits.
- HF_HUB_DISABLE_SYMLINKS_WARNING: set to 1 to silence Windows symlink warnings in CI.

Project layout

- rag/: core modules (text splitting, embedding, faiss index, llm wrapper, utils)
- scripts/: top-level scripts: build_index.py and query_index.py
- data/raw/: put PDF files to ingest
- data/processed/: outputs: chunks, embeddings, and faiss index
- tests/: pytest unit tests (use stubs in tests/conftest.py so tests run without heavy deps)

CI notes

- The GitHub Actions workflow sets HF_TOKEN and OPENAI_API_KEY via repository secrets; ensure these are configured in repository settings.
- The workflow also caches the Hugging Face cache and pip to speed CI.

License

This project is an educational example. No license specified.
