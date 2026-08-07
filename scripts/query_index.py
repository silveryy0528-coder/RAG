"""Query a built FAISS index and generate a RAG answer.

Workflow:
1. Embed the user question.
2. Load the FAISS index and chunk metadata saved in ``data/processed``.
3. Retrieve top-k chunks for the question.
4. Build a RAG prompt from the retrieved context and the question.
5. Generate an answer using the configured LLM client.

This is the minimal RAG runtime pipeline. Optional improvements include
- reranking retrieved chunks, filtering by score, or performing multi-hop retrieval
- grounding/verification steps after generation
- streaming responses or token-level decoding control
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np

from rag.embedding import load_embedder
from rag.retriever import retrieve_top_k
from rag.prompting import build_rag_prompt
from rag.llm import load_openai_client, generate_answer


DEFAULT_K = 3
DEFAULT_DEVICE = "cuda"
DEFAULT_MODEL = "gpt-4.1-mini"


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_chunks(processed_dir: Path) -> List[object]:
    chunk_file = processed_dir / "chunks.pkl"
    if not chunk_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunk_file}")
    with chunk_file.open("rb") as fh:
        chunks = pickle.load(fh)
    return chunks


def load_faiss_index(processed_dir: Path) -> faiss.Index:
    index_file = processed_dir / "faiss.index"
    if not index_file.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_file}")
    index = faiss.read_index(str(index_file))
    return index


def build_context_from_results(results: List[dict]) -> str:
    # Concatenate top results into a single context block. Add simple separators.
    parts = []
    for i, r in enumerate(results, start=1):
        text = r.get("text") or r.get("content") or ""
        meta = []
        if r.get("doc_id"):
            meta.append(f"doc:{r.get('doc_id')}")
        if r.get("page"):
            meta.append(f"page:{r.get('page')}")
        header = f"[chunk {i} {'|'.join(meta)}]" if meta else f"[chunk {i}]"
        parts.append(header + "\n" + text)
    return "\n\n".join(parts)


def run(
    question: str,
    processed_dir: Path,
    k: int,
    device: str,
    model_name: str,
    temperature: float,
    api_key: str | None,
) -> str:
    processed_dir = processed_dir.expanduser()

    # Load resources
    chunks = load_chunks(processed_dir)
    index = load_faiss_index(processed_dir)

    # Prepare embedder and client
    embedder = load_embedder(device=device)

    # Embed the question and normalize the vector to match index preprocessing
    q_vec = embedder.encode([question], device=device, convert_to_numpy=True, normalize_embeddings=False)
    # If the index was built with normalized vectors (as in build_index.py) we must
    # normalize the query vector before searching to get cosine-like behavior.
    try:
        faiss.normalize_L2(q_vec)
    except Exception:
        # If faiss normalization fails for any reason, continue with raw vector
        pass

    # Search the index
    scores, indices = index.search(q_vec, k)

    # Map search results to chunk dicts (similar to rag.retriever.retrieve_top_k)
    results = []
    for score_row, idx_row in zip(scores, indices):
        for score, idx in zip(score_row, idx_row):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            results.append(
                {
                    "id": getattr(chunk, "metadata", {}).get("chunk_id") if hasattr(chunk, "metadata") else getattr(chunk, "id", None),
                    "text": getattr(chunk, "text", None) or chunk.get("text") if isinstance(chunk, dict) else getattr(chunk, "text", ""),
                    "doc_id": getattr(chunk, "metadata", {}).get("doc_id") if hasattr(chunk, "metadata") else chunk.get("doc_id", "UNKNOWN"),
                    "score": float(score),
                    "page": getattr(chunk, "metadata", {}).get("page") if hasattr(chunk, "metadata") else chunk.get("page"),
                }
            )

    if not results:
        raise RuntimeError("No retrieval results — check that index and chunks match")

    context = build_context_from_results(results)

    prompt = build_rag_prompt(question, context)

    # Create client with helpful error messaging if credentials are missing
    try:
        # Try to import the OpenAI-specific error class for nicer exception handling
        try:
            from openai import OpenAIError  # type: ignore
        except Exception:
            OpenAIError = Exception  # fallback if openai package not installed

        client = load_openai_client(api_key=api_key)
    except OpenAIError as exc:  # pragma: no cover - runtime credential issue
        raise RuntimeError(
            "Failed to create OpenAI client: missing or invalid credentials. "
            "Provide an API key via the --api-key option, the OPENAI_API_KEY environment variable, "
            "or configure a fake/local client for testing."
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to create OpenAI client: {exc}. Provide OPENAI_API_KEY or --api-key."
        ) from exc

    try:
        answer = generate_answer(client, prompt, model_name=model_name, temperature=temperature)
    except Exception as exc:  # pragma: no cover - runtime generation error
        raise RuntimeError(
            f"Model generation failed: {exc}. Check that the client supports chat.completions.create and that model credentials are valid."
        ) from exc

    return answer


def parse_args() -> argparse.Namespace:
    project_root = resolve_project_root()
    default_processed = project_root / "data" / "processed"

    parser = argparse.ArgumentParser(description="Query the FAISS index and generate a RAG answer.")
    parser.add_argument("question", type=str, nargs="?", help="Question to ask (if omitted, read from stdin)")
    parser.add_argument("--processed-dir", type=Path, default=default_processed, help="Directory with chunks.pkl and faiss.index")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of top chunks to retrieve")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Embedder device (cpu or cuda)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name to use for generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the LLM")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the OpenAI-like client (can also be provided via env or client default)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question = args.question
    if not question:
        question = input("Question: ")

    answer = run(
        question=question,
        processed_dir=args.processed_dir,
        k=args.k,
        device=args.device,
        model_name=args.model,
        temperature=args.temperature,
        api_key=args.api_key,
    )

    print("\n===== Answer =====\n")
    print(answer)


if __name__ == "__main__":
    main()
