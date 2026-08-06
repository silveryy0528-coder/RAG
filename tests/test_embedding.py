import sys
import types

# Stub the dependencies used by rag.embedding so tests can run without installing
# the real transformers and sentence_transformers packages.
fake_transformers_logging = types.ModuleType("transformers.logging")

def set_verbosity_error():
    set_verbosity_error.called = True

fake_transformers_logging.set_verbosity_error = set_verbosity_error
fake_transformers = types.ModuleType("transformers")
fake_transformers.logging = fake_transformers_logging
sys.modules["transformers"] = fake_transformers
sys.modules["transformers.logging"] = fake_transformers_logging


class StubSentenceTransformer:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device
        self.encode_calls = []

    def encode(self, texts, device=None, convert_to_numpy=None, normalize_embeddings=None):
        self.encode_calls.append(
            {
                "texts": texts,
                "device": device,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return [[0.1, 0.2, 0.3]]


fake_sentence_transformers = types.ModuleType("sentence_transformers")
fake_sentence_transformers.SentenceTransformer = StubSentenceTransformer
sys.modules["sentence_transformers"] = fake_sentence_transformers

import rag.embedding as embedding


def test_load_embedder_returns_sentence_transformer_instance():
    embedder = embedding.load_embedder("my-model", device="cpu")

    assert isinstance(embedder, StubSentenceTransformer)
    assert embedder.model_name == "my-model"
    assert embedder.device == "cpu"


def test_load_embedder_uses_default_parameters():
    embedder = embedding.load_embedder()

    assert isinstance(embedder, StubSentenceTransformer)
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.device == "cuda"


def test_embed_text_calls_encode_with_expected_options():
    embedder = StubSentenceTransformer("my-model", device="cpu")
    texts = ["hello", "world"]

    result = embedding.embed_text(embedder, texts, device="cpu")

    assert result == [[0.1, 0.2, 0.3]]
    assert len(embedder.encode_calls) == 1
    call = embedder.encode_calls[0]
    assert call["texts"] == texts
    assert call["device"] == "cpu"
    assert call["convert_to_numpy"] is True
    assert call["normalize_embeddings"] is True
