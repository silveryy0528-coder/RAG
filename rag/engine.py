"""RAGEngine: single-load, multi-query RAG runtime.

Load the embedder, FAISS index, and chunks once at construction time.
Call ``query()`` repeatedly without reloading any resources.

Example
-------
>>> engine = RAGEngine.from_processed_dir(Path("data/processed"), device="cpu")
>>> answer, eval_result = engine.query("What is the title of the thesis?")
>>> answer2, _ = engine.query("Who supervised the work?")
"""

from __future__ import annotations

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


class RAGEngine:
    """Holds all loaded resources and answers repeated queries efficiently.

    Parameters
    ----------
    embedder :
        A loaded SentenceTransformer (or compatible) embedder.
    index :
        A loaded FAISS index.
    chunks :
        The list of chunk dicts that correspond positionally to the FAISS index.
    device :
        Device string forwarded to embed calls (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self, embedder, index: faiss.Index, chunks: List, device: str = "cuda"
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.chunks = chunks
        self.device = device

    @classmethod
    def from_processed_dir(
        cls,
        processed_dir: Path,
        *,
        device: str = "cpu",
        model_name: str = "all-MiniLM-L6-v2",
    ) -> "RAGEngine":
        """Load embedder, FAISS index, and chunks from *processed_dir*.

        Parameters
        ----------
        processed_dir :
            Directory that contains ``chunks.pkl`` and ``faiss.index``.
        device :
            Device for the embedder (``"cpu"`` or ``"cuda"``).
        model_name :
            Sentence-transformer model name.
        """
        processed_dir = processed_dir.expanduser()

        chunk_file = processed_dir / "chunks.pkl"
        if not chunk_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunk_file}")
        with chunk_file.open("rb") as fh:
            chunks = pickle.load(fh)

        index_file = processed_dir / "faiss.index"
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        index = faiss.read_index(str(index_file))

        embedder = load_embedder(model_name=model_name, device=device)

        return cls(embedder=embedder, index=index, chunks=chunks, device=device)

    def query(
        self,
        question: str,
        *,
        k: int = 3,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        api_key: str | None = None,
        show_top_k: bool = False,
        evaluate_answer_flag: bool = False,
        eval_model: str | None = None,
        output_json: Path | None = None,
        use_metadata_reranking: bool = False,
    ) -> tuple[str, dict | None]:
        """Embed *question*, retrieve, and generate an answer.

        No resources are reloaded between calls.

        Parameters
        ----------
        question :
            The user question.
        k :
            Number of chunks to retrieve.
        model_name :
            LLM model name for generation.
        temperature :
            Sampling temperature.
        api_key :
            Optional OpenAI API key (falls back to ``OPENAI_API_KEY`` env var).
        show_top_k :
            Print retrieved chunks to stdout when ``True``.
        evaluate_answer_flag :
            Run the LLM evaluator after generation when ``True``.
        eval_model :
            Model used for evaluation (defaults to *model_name*).
        output_json :
            Directory to write a JSON result file, or ``None`` to skip.
        use_metadata_reranking :
            Apply metadata-based reranking to a wider candidate pool.

        Returns
        -------
        tuple[str, dict | None]
            ``(answer, evaluation_result)`` where *evaluation_result* is
            ``None`` when *evaluate_answer_flag* is ``False``.
        """
        results = retrieve_top_k(
            question,
            self.chunks,
            self.embedder,
            self.index,
            k=k,
            use_metadata_reranking=use_metadata_reranking,
        )

        if not results:
            raise RuntimeError(
                "No retrieval results — check that index and chunks match"
            )

        if show_top_k:
            show_top_k_results(results)

        context = build_context_from_results(results)
        prompt = build_rag_prompt(question, context)

        try:
            from openai import OpenAIError  # type: ignore
        except Exception:
            OpenAIError = Exception

        try:
            client = load_openai_client(api_key=api_key)
        except OpenAIError as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to create OpenAI client: missing or invalid credentials. "
                "Provide an API key via the api_key argument or the OPENAI_API_KEY "
                "environment variable."
            ) from exc
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to create OpenAI client: {exc}.") from exc

        try:
            answer = generate_answer(
                client, prompt, model_name=model_name, temperature=temperature
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Model generation failed: {exc}.") from exc

        evaluation_result = None
        if evaluate_answer_flag:
            try:
                eval_model_name = eval_model or model_name
                evaluation_result = evaluate_generated_answer(
                    answer, question, results, client, eval_model_name, temperature
                )
            except Exception as exc:  # pragma: no cover
                print(f"Evaluation failed: {exc}")

        if output_json is not None:
            out_path = Path(output_json)
            save_dir = (
                out_path
                if out_path.is_dir() or str(out_path).endswith(("/", "\\"))
                else out_path.parent
            )
            saved_path = save_results_json(
                results, answer, evaluation_result, save_dir, prefix="query"
            )
            print(f"Saved JSON results to {saved_path}")

        return answer, evaluation_result
