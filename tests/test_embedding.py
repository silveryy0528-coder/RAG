from sentence_transformers import SentenceTransformer

import rag.embedding as embedding


def test_load_embedder_returns_sentence_transformer_instance():
    embedder = embedding.load_embedder("my-model", device="cpu")

    assert isinstance(embedder, SentenceTransformer)
    assert embedder.model_name == "my-model"
    assert embedder.device == "cpu"


def test_load_embedder_uses_default_parameters():
    embedder = embedding.load_embedder()

    assert isinstance(embedder, SentenceTransformer)
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.device == "cuda"


def test_embed_text_calls_encode_with_expected_options():
    embedder = SentenceTransformer("my-model", device="cpu")
    texts = ["hello", "world"]

    result = embedding.embed_text(embedder, texts, device="cpu")

    assert result == [[0.1, 0.2, 0.3]]
    assert len(embedder.encode_calls) == 1
    call = embedder.encode_calls[0]
    assert call["texts"] == texts
    assert call["device"] == "cpu"
    assert call["convert_to_numpy"] is True
    assert call["normalize_embeddings"] is True
