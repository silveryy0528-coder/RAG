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

from rag.embedding import load_embedder
from rag.retriever import retrieve_top_k
from rag.prompting import build_rag_prompt
from rag.llm import load_openai_client, generate_answer
from rag.utils import (
    build_context_from_results,
    show_top_k_results,
    evaluate_generated_answer,
    save_results_json,
)


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


def run(
    question: str,
    processed_dir: Path,
    k: int,
    device: str,
    model_name: str,
    temperature: float,
    api_key: str | None,
    show_top_k: bool = False,
    evaluate_answer_flag: bool = False,
    eval_model: str | None = None,
    output_json: Path | None = None,
    use_metadata_reranking: bool = False,
) -> tuple[str, dict | None]:
    processed_dir = processed_dir.expanduser()

    # Load resources
    chunks = load_chunks(processed_dir)
    index = load_faiss_index(processed_dir)

    # Prepare embedder and client
    embedder = load_embedder(device=device)

    results = retrieve_top_k(
        question,
        chunks,
        embedder,
        index,
        k=k,
        use_metadata_reranking=use_metadata_reranking,
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

    # Optionally show retrieved chunks
    if show_top_k:
        show_top_k_results(results)

    try:
        answer = generate_answer(client, prompt, model_name=model_name, temperature=temperature)
    except Exception as exc:  # pragma: no cover - runtime generation error
        raise RuntimeError(
            f"Model generation failed: {exc}. Check that the client supports chat.completions.create and that model credentials are valid."
        ) from exc

    # Optionally evaluate the generated answer with the LLM evaluator and grounding score
    evaluation_result = None
    if evaluate_answer_flag:
        try:
            eval_model_name = eval_model or (model_name if model_name else "gpt-3.5-turbo")
            evaluation_result = evaluate_generated_answer(answer, question, results, client, eval_model_name, temperature)
        except Exception as exc:  # pragma: no cover - runtime eval error
            print(f"Evaluation failed: {exc}")

    # Optionally save JSON output
    if output_json is not None:
        out_path = Path(output_json)
        if out_path.exists() and out_path.is_dir():
            save_dir = out_path
        else:
            # Treat provided path as file path -> use parent directory
            save_dir = out_path if str(out_path).endswith(('/', '\\')) else out_path.parent
        saved_path = save_results_json(results, answer, evaluation_result, save_dir, prefix="query")
        print(f"Saved JSON results to {saved_path}")

    return answer, evaluation_result



def parse_args() -> argparse.Namespace:
    project_root = resolve_project_root()
    default_processed = project_root / "data" / "processed"

    parser = argparse.ArgumentParser(description="Query the FAISS index and generate a RAG answer.")
    parser.add_argument("question", type=str, nargs="?", help="Question to ask (if omitted, read from stdin). (str)")
    parser.add_argument("--processed-dir", type=Path, default=default_processed, help="Directory with chunks.pkl and faiss.index. (Path)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of top chunks to retrieve. (int)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Embedder device (cpu or cuda). (str)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name to use for generation. (str)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the LLM. (float)")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the OpenAI-like client (can also be provided via env or client default). (str)")
    parser.add_argument("--show-top-k", action="store_true", help="Print the top-k retrieved chunks. (flag)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the generated answer using the LLM evaluator and grounding score. (flag)")
    parser.add_argument("--eval-model", type=str, default="gpt-3.5-turbo", help="Smaller/cheaper model to use for evaluation (defaults to gpt-3.5-turbo). (str)")
    parser.add_argument("--output-json", type=Path, default=None, help="Directory or file path to save JSON results (if omitted and --evaluate is set, saves to ./results/). (Path or file)")
    parser.add_argument("--use-metadata-reranking", action="store_true", help="Rerank the retrieved chunks using section metadata. (flag)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question = args.question
    if not question:
        question = input("Question: ")

    output_json = args.output_json
    if output_json is None and args.evaluate:
        output_json = Path("results")

    answer, evaluation_result = run(
        question=question,
        processed_dir=args.processed_dir,
        k=args.k,
        device=args.device,
        model_name=args.model,
        temperature=args.temperature,
        api_key=args.api_key,
        show_top_k=args.show_top_k,
        evaluate_answer_flag=args.evaluate,
        eval_model=args.eval_model,
        output_json=output_json,
        use_metadata_reranking=args.use_metadata_reranking,
    )

    print("\n===== Answer =====\n")
    print(answer)

    # Print evaluation summary after the answer (if present)
    if evaluation_result is not None:
        print("\n--- Evaluation summary ---\n")
        try:
            grounding = evaluation_result.get("grounding_score")
            print(f"Grounding score: {grounding:.3f}")
        except Exception:
            pass
        print("LLM evaluation:\n")
        print(evaluation_result.get("llm_evaluation"))


if __name__ == "__main__":
    main()
