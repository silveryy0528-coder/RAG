"""Small runtime utilities used by scripts.

Includes helpers to display retrieved chunks, evaluate generated answers using
existing evaluating.* helpers, and to save a structured JSON result file.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Any

from rag import evaluating


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
        score = r.get("score")
        text = (r.get("text") or "").strip().replace("\n", " ")
        print(f"{i}. doc={doc} page={page} score={score:.4f}")
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

    grounding = evaluating.compute_grounding_score(answer, context)

    llm_eval = None
    try:
        llm_eval = evaluating.evaluate_answer_with_llm(client, question, answer, context, model_name=model_name, temperature=temperature)
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
