"""Interactive RAG chat loop.

Start-up
--------
Resources (embedder, FAISS index, chunks) are loaded once.

Usage
-----
::

    python -m scripts.rag_chat                                  # interactive loop
    python -m scripts.rag_chat --question "one-shot question"   # single question
    python -m scripts.rag_chat --config configs/rag_chat.yaml   # YAML config

Type ``exit`` or ``quit`` (or press Ctrl-C / Ctrl-D) to leave the loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rag.engine import RAGEngine
from scripts.cli_config import load_config

DEFAULT_K = 3
DEFAULT_DEVICE = "cuda"
DEFAULT_MODEL = "gpt-4.1-mini"

_BOOL_KEYS: set[str] = {"show_top_k", "evaluate", "use_metadata_reranking"}
_PATH_KEYS: set[str] = {"processed_dir", "output_json"}
_INT_KEYS: set[str] = {"k"}
_NUMBER_KEYS: set[str] = {"temperature"}
_ALLOWED_KEYS: set[str] = {
    "question",
    "processed_dir",
    "k",
    "device",
    "model",
    "temperature",
    "api_key",
    "show_top_k",
    "evaluate",
    "eval_model",
    "output_json",
    "use_metadata_reranking",
}


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = resolve_project_root()
    default_processed = project_root / "data" / "processed"

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_defaults: dict[str, Any] = {}
    if pre_args.config is not None:
        config_defaults = load_config(
            pre_args.config.expanduser(),
            allowed=_ALLOWED_KEYS,
            bool_keys=_BOOL_KEYS,
            path_keys=_PATH_KEYS,
            int_keys=_INT_KEYS,
            number_keys=_NUMBER_KEYS,
        )

    parser = argparse.ArgumentParser(
        description="Interactive chat with a FAISS-backed RAG index."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file with defaults for CLI options. (Path)",
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--question",
        dest="question_flag",
        type=str,
        default=None,
        help="Single-shot question to answer once and exit. Omit to start the interactive loop. (str)",
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
        help="API key for the OpenAI-like client. (str)",
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
        help="Smaller/cheaper model to use for evaluation. (str)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Directory or file path to save JSON results. (Path)",
    )
    parser.add_argument(
        "--use-metadata-reranking",
        action="store_true",
        help="Rerank the retrieved chunks using section metadata. (flag)",
    )
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)
    if args.question_flag is not None:
        args.question = args.question_flag
    return args


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()

    output_json = args.output_json
    if output_json is None and args.evaluate:
        output_json = default_query_results_dir(project_root)

    print("Loading resources\u2026")
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
