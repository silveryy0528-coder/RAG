import pytest

from rag.retriever import retrieve_top_k, retrieve_top_k_raw


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


def test_retrieve_top_k_WHEN_special_section_is_ranked_THEN_demotes_it():
    chunks = [
        {
            "id": "special",
            "text": "propositions",
            "page": 2,
            "doc_id": "d0",
            "metadata": {"section": "propositions"},
        },
        {
            "id": "body",
            "text": "main body",
            "page": 3,
            "doc_id": "d0",
            "metadata": {"section": "body"},
        },
    ]

    idx = FakeIndex([[0.8, 0.9]], [[0, 1]])
    embedder = FakeEmbedder()

    results = retrieve_top_k(
        "What is the main contribution?",
        chunks,
        embedder,
        idx,
        k=2,
        use_metadata_reranking=True,
    )

    assert results[0]["id"] == "body"
    assert results[0]["section"] == "body"
    assert results[1]["id"] == "special"


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
