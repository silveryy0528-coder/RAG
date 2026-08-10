"""Small runtime utilities used by scripts.

Includes helpers to display retrieved chunks, evaluate generated answers using
existing evaluating.* helpers, and to save a structured JSON result file.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Sequence

from rag.runtime_evaluation import compute_grounding_score, evaluate_answer_with_llm


def build_context_from_results(results: Sequence[Dict[str, Any]]) -> str:
    """Format retrieved results as a single context block.

    The helper adds lightweight metadata headers so downstream prompts and
    debugging output can trace where each piece of context came from.
    """
    parts: List[str] = []
    for i, result in enumerate(results, start=1):
        text = result.get("text") or result.get("content") or ""
        meta = []
        if result.get("doc_id"):
            meta.append(f"doc={result.get('doc_id')}")
        if result.get("page") is not None:
            meta.append(f"page={result.get('page')}")
        if result.get("section"):
            meta.append(f"section={result.get('section')}")

        header = f"[chunk {i} {'|'.join(meta)}]" if meta else f"[chunk {i}]"
        parts.append(header)
        parts.append(text)
    return "\n\n".join(parts)


def show_top_k_results(results: List[Dict[str, Any]]) -> None:
    """Print the retrieved top-k chunks with metadata and scores.

    Parameters
    ----------
    results : list[dict]
        Retrieval results as returned by the search step. Each item should
        contain at least 'text', 'score', and optional 'doc_id'/'page'.
    """
    print("\n--- Top retrieved chunks ---\n")
    for i, r in enumerate(results, start=1):
        doc = r.get("doc_id", "UNKNOWN")
        page = r.get("page", "?")
        section = r.get("section") or ""
        score = r.get("score")
        text = (r.get("text") or "").strip().replace("\n", " ")
        section_suffix = f" section={section}" if section else ""
        print(f"{i}. doc={doc} page={page}{section_suffix} score={score:.4f}")
        print(f"   {text[:400]}\n")


def evaluate_generated_answer(
    answer: str,
    question: str,
    results: List[Dict[str, Any]],
    client: Any,
    model_name: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate the generated answer using the LLM evaluator and grounding score.

    Returns a dictionary containing the grounding score and the raw LLM
    evaluation response.
    """
    context = "\n\n".join(
        (f"[chunk {i+1}]\n" + (r.get("text") or "")) for i, r in enumerate(results)
    )

    grounding = compute_grounding_score(answer, context)

    llm_eval = None
    try:
        llm_eval = evaluate_answer_with_llm(
            client,
            question,
            answer,
            context,
            model_name=model_name,
            temperature=temperature,
        )
    except Exception as exc:
        llm_eval = f"LLM evaluation failed: {exc}"

    return {"grounding_score": grounding, "llm_evaluation": llm_eval}


def save_results_json(
    results: List[Dict[str, Any]],
    answer: str,
    evaluation: Dict[str, Any] | None,
    out_dir: Path,
    prefix: str = "query",
) -> Path:
    """Save a structured JSON file containing retrievals, answer and evaluation.

    The function creates out_dir if necessary and writes a timestamped file.

    Returns the path to the written file.
    """
    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "retrievals": results,
        "answer": answer,
        "evaluation": evaluation,
    }

    fname = f"{prefix}-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path = out_dir / fname
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    return out_path
