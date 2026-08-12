import pytest

from rag.retriever import _intent_match_candidates, retrieve_top_k, retrieve_top_k_raw


class FakeIndex:
    def __init__(self, scores, indices):
        self._scores = scores
        self._indices = indices

    def search(self, query_vec, k):
        return self._scores, self._indices


class FakeEmbedder:
    def encode(
        self, texts, device=None, convert_to_numpy=None, normalize_embeddings=None
    ):
        # return a single vector batch
        return [[0.5, 0.6, 0.7]]


def test_retrieve_top_k_WHEN_index_matches_chunks_THEN_formats_results():
    chunks = [
        {"id": "c0", "text": "first", "page": 1, "doc_id": "d0"},
        {"id": "c1", "text": "second", "page": 2},
    ]

    # pretend the index returns two neighbors with scores
    scores = [[0.9, 0.1]]
    indices = [[0, 1]]
    idx = FakeIndex(scores, indices)
    embedder = FakeEmbedder()

    results = retrieve_top_k("query", chunks, embedder, idx, k=2)

    assert len(results) == 2
    assert results[0]["id"] == "c0"
    assert results[0]["score"] == pytest.approx(0.9)
    assert results[1]["id"] == "c1"
    assert results[1]["doc_id"] == "UNKNOWN"


def test_retrieve_top_k_WHEN_no_reranking_THEN_asks_faiss_for_exactly_k():
    """Without reranking, search_k == k so FAISS is not over-fetched."""
    chunks = [{"id": f"c{i}", "text": f"text {i}", "page": i, "doc_id": "d0"} for i in range(10)]

    requested_ks = []

    class TrackingIndex:
        def search(self, query_vec, k):
            requested_ks.append(k)
            return [[1.0 - i * 0.1 for i in range(k)]], [list(range(k))]

    results = retrieve_top_k("query", chunks, FakeEmbedder(), TrackingIndex(), k=3)

    assert requested_ks == [3], f"Expected FAISS to be called with k=3, got {requested_ks}"
    assert len(results) == 3


def test_retrieve_top_k_WHEN_section_label_matches_query_THEN_gives_small_boost():
    chunks = [
        {
            "id": "special",
            "text": "list of publications",
            "page": 2,
            "doc_id": "d0",
            "metadata": {"section": "list of publications"},
        },
        {
            "id": "body",
            "text": "main body",
            "page": 3,
            "doc_id": "d0",
            "metadata": {"section": "body"},
        },
    ]

    idx = FakeIndex([[0.9, 0.89]], [[0, 1]])
    embedder = FakeEmbedder()

    results = retrieve_top_k(
        "What publications resulted from this PhD work?",
        chunks,
        embedder,
        idx,
        k=2,
        use_metadata_reranking=True,
    )

    assert results[0]["id"] == "special"
    assert results[0]["section"] == "list of publications"
    assert results[1]["id"] == "body"


def test_retrieve_top_k_WHEN_chunk_text_matches_query_THEN_gives_small_boost():
    chunks = [
        {
            "id": "publications",
            "text": "This page lists publications from the thesis.",
            "page": 2,
            "doc_id": "d0",
            "metadata": {"section": "body"},
        },
        {
            "id": "body",
            "text": "Main body content",
            "page": 3,
            "doc_id": "d0",
            "metadata": {"section": "body"},
        },
    ]

    idx = FakeIndex([[0.9, 0.89]], [[0, 1]])
    embedder = FakeEmbedder()

    results = retrieve_top_k(
        "What publications resulted from this PhD work?",
        chunks,
        embedder,
        idx,
        k=2,
        use_metadata_reranking=True,
    )

    assert results[0]["id"] == "publications"
    assert results[1]["id"] == "body"


def test_retrieve_top_k_WHEN_technical_query_matches_thesis_THEN_prefers_thesis_over_cv():
    chunks = [
        {
            "id": "cv",
            "text": "Data scientist with broad imaging experience.",
            "page": 1,
            "doc_id": "CV_YanGuo.pdf",
            "metadata": {"section": "body"},
        },
        {
            "id": "thesis",
            "text": "HAADF-STEM and EDS cross-modal fusion are discussed here.",
            "page": 2,
            "doc_id": "Final_thesis_Yan.pdf",
            "metadata": {"section": "body"},
        },
    ]

    idx = FakeIndex([[0.95, 0.94]], [[0, 1]])
    embedder = FakeEmbedder()

    results = retrieve_top_k(
        "What limitations or challenges of the HAADF-EDS cross-modal fusion framework are discussed in the thesis?",
        chunks,
        embedder,
        idx,
        k=2,
        use_metadata_reranking=True,
    )

    assert results[0]["id"] == "thesis"
    assert results[1]["id"] == "cv"


def test_retrieve_top_k_WHEN_reranking_is_disabled_THEN_preserves_faiss_order():
    chunks = [
        {"id": "special", "text": "special", "page": 2, "doc_id": "d0", "metadata": {"section": "propositions"}},
        {"id": "body", "text": "body", "page": 3, "doc_id": "d0", "metadata": {"section": "body"}},
    ]

    idx = FakeIndex([[0.8, 0.9]], [[0, 1]])
    embedder = FakeEmbedder()

    results = retrieve_top_k("What is the main contribution?", chunks, embedder, idx, k=2)

    assert results[0]["id"] == "special"
    assert results[1]["id"] == "body"


def test_retrieve_top_k_raw_WHEN_called_THEN_returns_search_output():
    idx = FakeIndex([[0.1]], [[2]])
    query_vec = [[0.2, 0.3]]
    scores, indices = retrieve_top_k_raw(query_vec, idx, k=1)
    assert scores == [[0.1]]
    assert indices == [[2]]


def test_intent_match_candidates_WHEN_called_with_candidate_pairs_THEN_matches_correctly():
    candidates = [
        (7, {"text": "Image quality assessment and image fusion for electron tomography", "metadata": {}}),
        (12, {"text": "Chapter 1: Introduction to this thesis and dissertation", "metadata": {}}),
        (42, {"text": "unrelated content about cooking", "metadata": {}}),
    ]

    matches = _intent_match_candidates("What is the title of the thesis", candidates)

    matched_indices = [idx for idx, _, _ in matches]
    # chunk 12 contains "thesis" and "dissertation" — should match
    assert 12 in matched_indices
    # chunk 7 has no title/thesis/dissertation/proefschrift terms — no match
    assert 7 not in matched_indices
    # chunk 42 has no matching terms
    assert 42 not in matched_indices


def test_retrieve_top_k_WHEN_reranking_THEN_intent_scan_limited_to_faiss_candidates():
    """Intent scan must not touch chunks outside the FAISS candidate pool."""
    # chunk 99 contains "thesis" but is NOT in the FAISS results.
    # It must not appear in the output even though it would match the intent scan
    # if all 100 chunks were scanned.
    n = 100
    chunks = [
        {"id": f"c{i}", "text": "unrelated content", "page": i, "doc_id": "d0", "metadata": {}}
        for i in range(n)
    ]
    chunks[99] = {
        "id": "c99",
        "text": "title of the thesis proefschrift",
        "page": 99,
        "doc_id": "d0",
        "metadata": {},
    }
    # FAISS returns only indices 0..9 — chunk 99 is NOT in the candidate pool.
    faiss_scores = [[1.0 - i * 0.01 for i in range(10)]]
    faiss_indices = [[i for i in range(10)]]
    idx = FakeIndex(faiss_scores, faiss_indices)

    results = retrieve_top_k(
        "What is the title of the thesis",
        chunks,
        FakeEmbedder(),
        idx,
        k=3,
        use_metadata_reranking=True,
    )

    result_ids = [r["id"] for r in results]
    assert "c99" not in result_ids, "Intent scan should not reach chunks outside FAISS candidates"
