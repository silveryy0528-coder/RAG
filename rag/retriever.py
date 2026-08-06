"""Retriever helpers that query a FAISS index and format results.

This module contains functions to embed a query, run a FAISS search and
return a list of formatted result dictionaries.
"""

from typing import List, Dict, Any
from rag.embedding import embed_text


def retrieve_top_k(
    question: str, chunks: List[Dict[str, Any]], embedder, faiss_index, k: int = 3
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

    Returns
    -------
    list of dict
        A list of result dictionaries containing keys ``id``, ``text``,
        ``doc_id``, ``score`` and ``page``.
    """
    query_vec = embed_text(embedder, texts=[question])
    scores, indices = retrieve_top_k_raw(query_vec, faiss_index, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        # Guard against invalid indices (e.g., -1)
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        result = {
            "id": chunk.get("id"),
            "text": chunk.get("text"),
            "doc_id": chunk.get("doc_id", "UNKNOWN"),
            "score": float(score),
            "page": chunk.get("page"),
        }
        results.append(result)
    return results


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
