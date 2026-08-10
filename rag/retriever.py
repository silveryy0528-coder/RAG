"""Retriever helpers that query a FAISS index and format results.

This module contains functions to embed a query, run a FAISS search and
return a list of formatted result dictionaries.
"""

import re
from typing import List, Dict, Any

from rag.embedding import embed_text
from rag.ingestion.filtering import SPECIAL_SECTIONS


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _extract_query_terms(question: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (question or "").lower())
        if token not in _STOPWORDS and len(token) > 2
    }


def _chunk_to_result(chunk: Any, score: float, question: str) -> Dict[str, Any]:
    if isinstance(chunk, dict):
        text = chunk.get("text")
        metadata = chunk.get("metadata") or {}
        chunk_id = chunk.get("id")
        doc_id = chunk.get("doc_id", metadata.get("doc_id", "UNKNOWN"))
        page = chunk.get("page", metadata.get("page"))
        section = metadata.get("section")
    else:
        text = getattr(chunk, "text", None)
        metadata = getattr(chunk, "metadata", {}) or {}
        chunk_id = getattr(chunk, "id", metadata.get("chunk_id"))
        doc_id = metadata.get("doc_id", "UNKNOWN")
        page = metadata.get("page")
        section = metadata.get("section")

    metadata_boost = _metadata_boost(question, section)
    result = {
        "id": chunk_id,
        "text": text,
        "doc_id": doc_id,
        "score": float(score) + metadata_boost,
        "page": page,
        "section": section,
        "metadata": metadata,
    }
    return result


def _metadata_boost(question: str, section: str | None) -> float:
    if not section:
        return 0.0

    section_name = str(section).strip().lower()
    query_terms = _extract_query_terms(question)
    if not query_terms:
        return 0.0

    if section_name in {"structural"}:
        return -0.35

    if section_name in {"body"}:
        return 0.0

    if section_name in {"summary", "acknowledgements", "about the author", "samenvatting", "stellingen", "copyright", "references"}:
        if any(term in section_name for term in query_terms):
            return 0.1
        return -0.8

    if section_name in SPECIAL_SECTIONS:
        if any(term in section_name for term in query_terms):
            return 0.15
        return -0.6

    if any(term in section_name for term in query_terms):
        return 0.15

    return 0.0


def retrieve_top_k(
    question: str,
    chunks: List[Dict[str, Any]],
    embedder,
    faiss_index,
    k: int = 3,
    candidate_multiplier: int = 3,
) -> List[Dict[str, Any]]:
    """Retrieve top-k matching chunks for a question.

    Parameters
    ----------
    question : str
        Query string to embed and search for.
    chunks : list of dict
        List of chunk dictionaries. Each chunk is expected to contain at least
        ``id``, ``text`` and ``page`` keys. ``doc_id`` is optional.
    embedder : object
        Embedder object exposing an ``encode`` method compatible with
        ``sentence_transformers.SentenceTransformer``.
    faiss_index : object
        FAISS index instance exposing a ``search(query_vec, k)`` method.
    k : int, optional
        Number of top results to return. Default is 3.
    candidate_multiplier : int, optional
        How many candidates to collect before reranking. A larger value gives the
        metadata-based reranker more room to recover better matches.

    Returns
    -------
    list of dict
        A list of result dictionaries containing keys ``id``, ``text``,
        ``doc_id``, ``score``, ``page`` and ``section``.
    """
    query_vec = embed_text(embedder, texts=[question])
    search_k = max(k * candidate_multiplier, k)
    scores, indices = retrieve_top_k_raw(query_vec, faiss_index, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        # Guard against invalid indices (e.g., -1)
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append(_chunk_to_result(chunk, score, question))

    ranked_results = sorted(results, key=lambda item: item["score"], reverse=True)
    return ranked_results[:k]


def retrieve_top_k_raw(query_vec, faiss_index, k: int = 3):
    """Search FAISS index using a precomputed query vector.

    Parameters
    ----------
    query_vec : numpy.ndarray
        Query vector or a batch of query vectors shaped (1, D).
    faiss_index : object
        FAISS index exposing a ``search`` method.
    k : int, optional
        Number of nearest neighbors to return.

    Returns
    -------
    tuple
        (scores, indices) returned by the index's search method.
    """
    return faiss_index.search(query_vec, k)
