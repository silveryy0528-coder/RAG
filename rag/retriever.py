"""Retriever helpers that query a FAISS index and format results.

This module contains functions to embed a query, run a FAISS search and
return a list of formatted result dictionaries.
"""

import re
from typing import List, Dict, Any

from rag.embedding import embed_text

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
    """Return normalized non-stopword terms from a question."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (question or "").lower())
        if token not in _STOPWORDS and len(token) > 2
    }


def _normalize_token(token: str) -> str:
    """Normalize a token for coarse singular/plural matching."""
    normalized = re.sub(r"[^a-z0-9]+", "", (token or "").lower())
    if not normalized:
        return ""
    if normalized.endswith("ies") and len(normalized) > 4:
        return normalized[:-3] + "y"
    if normalized.endswith("es") and len(normalized) > 4:
        return normalized[:-2]
    if normalized.endswith("s") and len(normalized) > 3:
        return normalized[:-1]
    return normalized


def _chunk_fields(chunk: Any) -> Dict[str, Any]:
    """Extract the fields the retriever needs from a chunk object."""
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

    return {
        "text": text,
        "metadata": metadata,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "page": page,
        "section": section,
    }


def _chunk_to_result(
    chunk: Any,
    score: float,
    question: str,
    use_metadata_reranking: bool = False,
) -> Dict[str, Any]:
    """Convert a chunk into the public retrieval result shape."""
    fields = _chunk_fields(chunk)
    text = fields["text"]
    metadata = fields["metadata"]
    chunk_id = fields["chunk_id"]
    doc_id = fields["doc_id"]
    page = fields["page"]
    section = fields["section"]

    metadata_boost = (
        _metadata_boost(question, section, text) if use_metadata_reranking else 0.0
    )
    result = {
        "id": chunk_id,
        "text": text,
        "doc_id": doc_id,
        "page": page,
        "score": float(score) + metadata_boost,
        "section": section,
        "metadata": metadata,
    }
    return result


def _metadata_boost(question: str, section: str | None, text: str | None) -> float:
    """Compute a small score bonus for direct query overlap."""
    query_terms = _extract_query_terms(question)
    if not query_terms:
        return 0.0

    normalized_query_terms = {_normalize_token(term) for term in query_terms}

    if section:
        section_name = str(section).strip().lower()
        if section_name not in {"structural", "body"}:
            normalized_section_tokens = {
                _normalize_token(token)
                for token in re.findall(r"[a-z0-9]+", section_name)
            }
            explicit_section_matches = [
                term
                for term in normalized_query_terms
                if term and term in normalized_section_tokens
            ]
            if explicit_section_matches:
                return 0.04

    if text:
        normalized_text_tokens = {
            _normalize_token(token)
            for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        }
        explicit_text_matches = [
            term
            for term in normalized_query_terms
            if term and term in normalized_text_tokens
        ]
        if explicit_text_matches:
            return 0.02

    return 0.0


def _intent_match_candidates(
    question: str, chunks: List[Dict[str, Any]]
) -> List[tuple[int, Dict[str, Any]]]:
    """Find chunks that explicitly match the query intent."""
    query_terms = _extract_query_terms(question)
    normalized_query_terms = {_normalize_token(term) for term in query_terms}
    if not normalized_query_terms:
        return []

    intent_terms = set(normalized_query_terms)
    if "publication" in normalized_query_terms or "publicat" in normalized_query_terms:
        intent_terms = {"publication", "publications", "publicat"}
    elif "title" in normalized_query_terms or "thesis" in normalized_query_terms:
        intent_terms = {"title", "thesis", "dissertation", "proefschrift"}

    candidates: List[tuple[int, Dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        fields = _chunk_fields(chunk)
        text = fields["text"] or ""
        section = fields["section"] or ""
        normalized_section_tokens = {
            _normalize_token(token)
            for token in re.findall(r"[a-z0-9]+", str(section).lower())
        }
        normalized_text_tokens = {
            _normalize_token(token)
            for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        }
        if any(
            term
            for term in intent_terms
            if term and term in normalized_section_tokens | normalized_text_tokens
        ):
            candidates.append((index, chunk))

    return candidates


def retrieve_top_k(
    question: str,
    chunks: List[Dict[str, Any]],
    embedder,
    faiss_index,
    k: int = 3,
    candidate_multiplier: int = 3,
    use_metadata_reranking: bool = False,
) -> List[Dict[str, Any]]:
    """Retrieve top-k matching chunks for a question."""
    query_vec = embed_text(embedder, texts=[question])
    search_k = max(k * candidate_multiplier, k)
    if use_metadata_reranking:
        search_k = max(search_k, k * 8)
    scores, indices = retrieve_top_k_raw(query_vec, faiss_index, search_k)

    results = []
    seen_indices = set()
    if use_metadata_reranking:
        intent_matches = _intent_match_candidates(question, chunks)
        for chunk_index, chunk in intent_matches[: max(k * 2, k)]:
            result = _chunk_to_result(
                chunk, 0.0, question, use_metadata_reranking=False
            )
            result["score"] += 3.0
            if chunk_index in seen_indices:
                continue
            seen_indices.add(chunk_index)
            results.append(result)

    for score, idx in zip(scores[0], indices[0]):
        # Guard against invalid indices (e.g., -1)
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        result = _chunk_to_result(
            chunk,
            score,
            question,
            use_metadata_reranking=use_metadata_reranking,
        )
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        results.append(result)

    if use_metadata_reranking:
        ranked_results = sorted(results, key=lambda item: item["score"], reverse=True)
        return ranked_results[:k]

    return results[:k]


def retrieve_top_k_raw(query_vec, faiss_index, k: int = 3):
    """Search FAISS with a precomputed query vector."""
    return faiss_index.search(query_vec, k)
