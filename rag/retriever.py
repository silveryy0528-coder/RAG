"""Retriever helpers that query a FAISS index and format results.

This module contains functions to embed a query, run a FAISS search and
return a list of formatted result dictionaries.

Retrieval design
----------------
Without reranking: FAISS returns the top-k results directly; no extra work.

With reranking (``use_metadata_reranking=True``):
  1. FAISS retrieves a wider candidate pool (k * 8).
  2. Both an intent scan and a metadata boost are applied to those candidates only.
  3. The final top-k are selected from the reranked candidate pool.

The intent scan and metadata boost are scoped to the FAISS candidate pool on
purpose: widening that pool is the mechanism that lets hard-to-embed pages
(title page, publications page) surface; once they surface, limiting the scan
to those candidates is consistent and avoids scanning the entire corpus.
"""

import re
from typing import List, Dict, Any

from rag.embedding import embed_text

# General retrieval helpers
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

# Thesis-specific query terms used to bias retrieval toward the dissertation.
_TECHNICAL_QUERY_TERMS = {
    "haadf",
    "eds",
    "stem",
    "tomography",
    "fusion",
    "cross",
    "modal",
    "bimodal",
    "multichannel",
    "reconstruction",
    "regularization",
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
        _metadata_boost(question, doc_id, section, text)
        if use_metadata_reranking
        else 0.0
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


def _is_technical_query(normalized_query_terms: set[str]) -> bool:
    """Return True when the query looks like a thesis-technical question."""
    return any(term in _TECHNICAL_QUERY_TERMS for term in normalized_query_terms)


def _is_cv_doc(doc_id: str | None) -> bool:
    """Return True when the chunk comes from the CV document."""
    return bool(doc_id) and str(doc_id).lower().startswith("cv")


def _metadata_boost(
    question: str, doc_id: str | None, section: str | None, text: str | None
) -> float:
    """Compute a small score bonus for direct query overlap."""
    query_terms = _extract_query_terms(question)
    if not query_terms:
        return 0.0

    normalized_query_terms = {_normalize_token(term) for term in query_terms}
    technical_query = _is_technical_query(normalized_query_terms)

    # Thesis-specific: downrank CV chunks for technical thesis questions.
    if technical_query and _is_cv_doc(doc_id):
        return -0.25

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
                return 0.08 if technical_query else 0.04

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
            return 0.18 if technical_query else 0.02

    return 0.0


def _intent_match_candidates(
    question: str,
    candidates: List[tuple[int, Any]],
) -> List[tuple[int, Any, int]]:
    """Find candidates that explicitly match the query intent.

    Parameters
    ----------
    question:
        The user question.
    candidates:
        List of ``(chunk_index, chunk)`` pairs from the FAISS candidate pool.
        ``chunk_index`` is the position in the original full chunk list and is
        preserved in the output so ``seen_indices`` in the caller stays correct.

    Returns
    -------
    List of ``(chunk_index, chunk, match_count)`` for candidates that matched.
    """
    query_terms = _extract_query_terms(question)
    normalized_query_terms = {_normalize_token(term) for term in query_terms}
    if not normalized_query_terms:
        return []

    # Thesis-specific intent shortcuts for titles and publication pages.
    intent_terms = set(normalized_query_terms)
    if "publication" in normalized_query_terms or "publicat" in normalized_query_terms:
        intent_terms = {"publication", "publications", "publicat"}
    elif "title" in normalized_query_terms or "thesis" in normalized_query_terms:
        intent_terms = {"title", "thesis", "dissertation", "proefschrift"}

    matched: List[tuple[int, Any, int]] = []
    for chunk_index, chunk in candidates:
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
        matched_terms = {
            term
            for term in intent_terms
            if term
            and (term in normalized_section_tokens or term in normalized_text_tokens)
        }
        if matched_terms:
            matched.append((chunk_index, chunk, len(matched_terms)))

    return matched


def retrieve_top_k(
    question: str,
    chunks: List[Dict[str, Any]],
    embedder,
    faiss_index,
    k: int = 3,
    use_metadata_reranking: bool = False,
) -> List[Dict[str, Any]]:
    """Retrieve top-k matching chunks for a question.

    Without reranking FAISS is asked for exactly ``k`` results and they are
    returned in score order — no extra work.

    With reranking (``use_metadata_reranking=True``) a wider candidate pool
    (``k * 8``) is fetched so that hard-to-embed pages can surface.  The intent
    scan and metadata boost are then applied to *those candidates only*, which
    is consistent with the purpose of widening the pool and avoids scanning the
    full chunk list on every query.
    """
    query_vec = embed_text(embedder, texts=[question])

    search_k = k * 8 if use_metadata_reranking else k
    scores, indices = retrieve_top_k_raw(query_vec, faiss_index, search_k)

    # Build the FAISS candidate pool as (chunk_index, chunk, faiss_score) triples.
    faiss_candidates: List[tuple[int, Any, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        faiss_candidates.append((int(idx), chunks[idx], float(score)))

    if not use_metadata_reranking:
        return [
            _chunk_to_result(chunk, score, question)
            for _, chunk, score in faiss_candidates[:k]
        ]

    # --- Reranking path ---
    # Intent scan: check only the FAISS candidates, not the full corpus.
    intent_pool = [(idx, chunk) for idx, chunk, _ in faiss_candidates]
    intent_matches = _intent_match_candidates(question, intent_pool)

    results = []
    seen_indices: set[int] = set()

    # Insert intent-matched candidates first with a large score boost.
    for chunk_index, chunk, boost in sorted(
        intent_matches, key=lambda item: item[2], reverse=True
    )[: max(k * 2, k)]:
        if chunk_index in seen_indices:
            continue
        seen_indices.add(chunk_index)
        result = _chunk_to_result(chunk, 0.0, question, use_metadata_reranking=False)
        result["score"] += 3.0 + boost
        results.append(result)

    # Fill remaining slots from the FAISS pool with metadata boost applied.
    for chunk_index, chunk, score in faiss_candidates:
        if chunk_index in seen_indices:
            continue
        seen_indices.add(chunk_index)
        results.append(
            _chunk_to_result(chunk, score, question, use_metadata_reranking=True)
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)[:k]


def retrieve_top_k_raw(query_vec, faiss_index, k: int = 3):
    """Search FAISS with a precomputed query vector."""
    return faiss_index.search(query_vec, k)
