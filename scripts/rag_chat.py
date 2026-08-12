"""Interactive RAG chat loop.

Start-up
--------
Resources (embedder, FAISS index, chunks) are loaded once.

Usage
-----
::

    python -m scripts.rag_chat                        # interactive loop
    python -m scripts.rag_chat "one-shot question"    # single question then exit

Type ``exit`` or ``quit`` (or press Ctrl-C / Ctrl-D) to leave the loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.engine import RAGEngine

DEFAULT_K = 3
DEFAULT_DEVICE = "cuda"
DEFAULT_MODEL = "gpt-4.1-mini"


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_query_results_dir(project_root: Path) -> Path:
    return project_root / "results" / "query"


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
    """Load resources and answer a single question.

    This is a convenience wrapper around :class:`rag.engine.RAGEngine` for
    one-shot CLI use.  For interactive or batch use, construct
    ``RAGEngine.from_processed_dir()`` directly so the embedder and index are
    loaded only once.
    """
    engine = RAGEngine.from_processed_dir(processed_dir, device=device)
    return engine.query(
        question,
        k=k,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        show_top_k=show_top_k,
        evaluate_answer_flag=evaluate_answer_flag,
        eval_model=eval_model,
        output_json=output_json,
        use_metadata_reranking=use_metadata_reranking,
    )


def parse_args() -> argparse.Namespace:
    project_root = resolve_project_root()
    default_processed = project_root / "data" / "processed"

    parser = argparse.ArgumentParser(
        description="Query the FAISS index and generate a RAG answer."
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        help="Question to ask (if omitted, read from stdin). (str)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed,
        help="Directory with chunks.pkl and faiss.index. (Path)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Number of top chunks to retrieve. (int)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Embedder device (cpu or cuda). (str)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name to use for generation. (str)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the LLM. (float)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the OpenAI-like client (can also be provided via env or client default). (str)",
    )
    parser.add_argument(
        "--show-top-k",
        action="store_true",
        help="Print the top-k retrieved chunks. (flag)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the generated answer using the LLM evaluator and grounding score. (flag)",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default="gpt-3.5-turbo",
        help="Smaller/cheaper model to use for evaluation (defaults to gpt-3.5-turbo). (str)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Directory or file path to save JSON results (if omitted and --evaluate is set, saves to results/query/). (Path or file)",
    )
    parser.add_argument(
        "--use-metadata-reranking",
        action="store_true",
        help="Rerank the retrieved chunks using section metadata. (flag)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()

    output_json = args.output_json
    if output_json is None and args.evaluate:
        output_json = default_query_results_dir(project_root)

    print("Loading resources…")
    engine = RAGEngine.from_processed_dir(args.processed_dir, device=args.device)
    print("Ready. Type 'exit' or 'quit' to stop.\n")

    def _answer(question: str) -> None:
        answer, evaluation_result = engine.query(
            question,
            k=args.k,
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
        if evaluation_result is not None:
            print("\n--- Evaluation summary ---\n")
            try:
                grounding = evaluation_result.get("grounding_score")
                print(f"Grounding score: {grounding:.3f}")
            except Exception:
                pass
            print("LLM evaluation:\n")
            print(evaluation_result.get("llm_evaluation"))
        print()

    # One-shot mode: question supplied on the command line.
    if args.question:
        _answer(args.question)
        return

    # Interactive loop.
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        _answer(question)


if __name__ == "__main__":
    main()
