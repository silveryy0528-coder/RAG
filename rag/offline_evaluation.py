"""Offline evaluation helpers for QA dataset metrics.

This module provides convenience functions and a batch evaluation harness for
measuring retrieval and QA quality with a small JSON dataset. It is intended
for offline (non-interactive) runs that produce aggregate metrics and per-
example details suitable for debugging.

Key features
- Load a simple QA dataset (JSON list of {question, ground_truth_answer, type}).
- Compute per-example metrics (exact match, token F1, negative-answer handling).
- Measure retrieval grounding by computing token-overlap between the reference
  answer and the concatenated retrieved top-k chunks (a simple proxy for whether
  the retrieval step returned supporting evidence).
- Optionally generate answers with an LLM and evaluate end-to-end.

The dataset format is very simple and works well as a starting point. For more
robust evaluation you can enrich examples with acceptable answer variants or
explicit relevance labels for chunks/documents.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rag.embedding import load_embedder
from rag.llm import generate_answer, load_openai_client
from rag.prompting import build_rag_prompt
from rag.retriever import retrieve_top_k
from rag.runtime_evaluation import compute_grounding_score
from rag.utils import build_context_from_results

NOT_FOUND_PHRASES = [
    "not found",
    "no answer",
    "not available",
    "no information",
    "cannot find",
    "not present",
    "not provided",
]


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_qa_dataset(dataset_path: Path | str) -> List[Dict[str, str]]:
    """Load a QA dataset from JSON."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"QA dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as fh:
        dataset = json.load(fh)

    if not isinstance(dataset, list):
        raise ValueError("QA dataset must be a JSON list of examples.")

    return [dict(item) for item in dataset]


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return " ".join(re.findall(r"\w+", (text or "").lower()))


def tokenize(text: str) -> List[str]:
    """Split normalized text into tokens."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def exact_match(prediction: str, reference: str) -> bool:
    """Return True when tokenized prediction matches the reference."""
    return tokenize(prediction) == tokenize(reference)


def f1_score(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = defaultdict(int)
    for token in ref_tokens:
        common[token] += 1

    matched = 0
    for token in pred_tokens:
        if common[token] > 0:
            matched += 1
            common[token] -= 1

    if matched == 0:
        return 0.0

    precision = matched / len(pred_tokens)
    recall = matched / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def is_not_found(text: str) -> bool:
    """Heuristic to detect negative answers like ``Not found``."""
    normalized = normalize_text(text)
    return any(phrase in normalized for phrase in NOT_FOUND_PHRASES)


def _chunk_to_dict(chunk: Any) -> Dict[str, Any]:
    if isinstance(chunk, dict):
        metadata = chunk.get("metadata") or {}
        return {
            "id": chunk.get("id"),
            "text": chunk.get("text", ""),
            "doc_id": chunk.get("doc_id", metadata.get("doc_id", "UNKNOWN")),
            "page": chunk.get("page", metadata.get("page")),
            "section": metadata.get("section"),
            "metadata": metadata,
        }

    metadata = getattr(chunk, "metadata", {}) or {}
    return {
        "id": metadata.get("chunk_id"),
        "text": getattr(chunk, "text", "") or "",
        "doc_id": metadata.get("doc_id", "UNKNOWN"),
        "page": metadata.get("page"),
        "section": metadata.get("section"),
        "metadata": metadata,
    }


def chunks_to_dicts(chunks: Iterable[Any]) -> List[Dict[str, Any]]:
    """Convert chunk objects into plain dictionaries."""
    return [_chunk_to_dict(chunk) for chunk in chunks]


def compute_retrieval_overlap(
    reference: str, results: Sequence[Dict[str, Any]]
) -> float:
    """Compute a grounding-like score between a reference and retrieved context."""
    context = build_context_from_results(results)
    return compute_grounding_score(reference, context)


def load_chunks(processed_dir: Path) -> List[Dict[str, Any]]:
    """Load saved chunks from the processed directory."""
    import pickle

    chunk_file = Path(processed_dir) / "chunks.pkl"
    if not chunk_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunk_file}")
    with chunk_file.open("rb") as fh:
        raw_chunks = pickle.load(fh)
    return chunks_to_dicts(raw_chunks)


def load_faiss_index(processed_dir: Path):
    """Load a FAISS index file from the processed directory."""
    import faiss

    index_file = Path(processed_dir) / "faiss.index"
    if not index_file.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_file}")
    return faiss.read_index(str(index_file))


def _load_client(api_key: Optional[str]) -> Any:
    if api_key is None:
        raise ValueError("API key is required to generate answers.")
    return load_openai_client(api_key=api_key)


def evaluate_sample(
    question: str,
    reference: str,
    results: Sequence[Dict[str, Any]],
    prediction: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single QA example."""
    retrieval_overlap = compute_retrieval_overlap(reference, results)
    retrieval_hit = retrieval_overlap >= 0.5 if not is_not_found(reference) else None
    prediction = prediction or ""
    is_negative_ref = is_not_found(reference)
    is_negative_pred = is_not_found(prediction)

    example: Dict[str, Any] = {
        "question": question,
        "reference": reference,
        "prediction": prediction,
        "exact_match": exact_match(prediction, reference) if prediction else None,
        "f1": f1_score(prediction, reference) if prediction else None,
        "is_negative_reference": is_negative_ref,
        "is_negative_prediction": is_negative_pred if prediction else None,
        "negative_correct": (
            is_negative_ref and is_negative_pred if prediction else None
        ),
        "retrieval_overlap": retrieval_overlap,
        "retrieval_hit": retrieval_hit,
        "top_k": len(results),
        "retrieved_text": build_context_from_results(results),
    }
    return example


def summarize_metrics(examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-example metrics into a summary dict."""
    total = len(examples)
    exact_matches = [
        ex["exact_match"] for ex in examples if ex.get("exact_match") is not None
    ]
    f1_scores = [ex["f1"] for ex in examples if ex.get("f1") is not None]
    negative_examples = [ex for ex in examples if ex.get("is_negative_reference")]
    negative_correct = [
        ex for ex in negative_examples if ex.get("negative_correct") is True
    ]
    retrieval_hits = [
        ex["retrieval_hit"] for ex in examples if ex.get("retrieval_hit") is not None
    ]
    retrieval_overlaps = [
        ex["retrieval_overlap"]
        for ex in examples
        if ex.get("retrieval_overlap") is not None
    ]

    summary: Dict[str, Any] = {
        "count": total,
        "exact_match": (
            sum(exact_matches) / len(exact_matches) if exact_matches else None
        ),
        "average_f1": sum(f1_scores) / len(f1_scores) if f1_scores else None,
        "negative_accuracy": (
            len(negative_correct) / len(negative_examples)
            if negative_examples
            else None
        ),
        "retrieval_hit_rate": (
            sum(1 for hit in retrieval_hits if hit) / len(retrieval_hits)
            if retrieval_hits
            else None
        ),
        "average_retrieval_overlap": (
            sum(retrieval_overlaps) / len(retrieval_overlaps)
            if retrieval_overlaps
            else None
        ),
        "by_type": {},
    }

    type_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        example_type = ex.get("type") or "unknown"
        type_buckets[example_type].append(ex)

    for example_type, bucket in type_buckets.items():
        hits = [
            ex["retrieval_hit"] for ex in bucket if ex.get("retrieval_hit") is not None
        ]
        overlaps = [
            ex["retrieval_overlap"]
            for ex in bucket
            if ex.get("retrieval_overlap") is not None
        ]
        exacts = [
            ex["exact_match"] for ex in bucket if ex.get("exact_match") is not None
        ]
        f1s = [ex["f1"] for ex in bucket if ex.get("f1") is not None]
        neg = [ex for ex in bucket if ex.get("is_negative_reference")]
        neg_correct = [ex for ex in neg if ex.get("negative_correct") is True]

        summary["by_type"][example_type] = {
            "count": len(bucket),
            "exact_match": sum(exacts) / len(exacts) if exacts else None,
            "average_f1": sum(f1s) / len(f1s) if f1s else None,
            "retrieval_hit_rate": (
                sum(1 for hit in hits if hit) / len(hits) if hits else None
            ),
            "average_retrieval_overlap": (
                sum(overlaps) / len(overlaps) if overlaps else None
            ),
            "negative_accuracy": len(neg_correct) / len(neg) if neg else None,
        }

    return summary


def run_offline_evaluation(
    dataset_path: Path,
    processed_dir: Path,
    k: int = 3,
    device: str = "cuda",
    generate_answers: bool = False,
    api_key: Optional[str] = None,
    model_name: str = "gpt-4.1-mini",
    temperature: float = 0.0,
    use_metadata_reranking: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run a batch offline evaluation over a QA dataset."""
    dataset = load_qa_dataset(dataset_path)
    chunks = load_chunks(processed_dir)
    index = load_faiss_index(processed_dir)
    embedder = load_embedder(device=device)
    client = _load_client(api_key) if generate_answers else None

    details: List[Dict[str, Any]] = []
    for example in dataset:
        question = example.get("question", "")
        reference = example.get("ground_truth_answer", "")
        result_rows = retrieve_top_k(
            question,
            chunks,
            embedder,
            index,
            k,
            use_metadata_reranking=use_metadata_reranking,
        )
        prediction = None
        if generate_answers:
            prompt = build_rag_prompt(question, build_context_from_results(result_rows))
            prediction = generate_answer(
                client, prompt, model_name=model_name, temperature=temperature
            )

        row = evaluate_sample(question, reference, result_rows, prediction)
        row["type"] = example.get("type")
        details.append(row)

    summary = summarize_metrics(details)
    return summary, details
