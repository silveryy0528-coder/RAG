import sys
import types


class StubSentenceSplitter:
    def __init__(self, chunk_size=None, chunk_overlap=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        return [
            {
                "text": text,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }
        ]


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


# Stub llama_index so rag.chunking can import SentenceSplitter at test time.
fake_node_parser = types.ModuleType("llama_index.core.node_parser")
fake_node_parser.SentenceSplitter = StubSentenceSplitter

fake_core = types.ModuleType("llama_index.core")
fake_core.node_parser = fake_node_parser

fake_llama = types.ModuleType("llama_index")
fake_llama.core = fake_core

sys.modules["llama_index"] = fake_llama
sys.modules["llama_index.core"] = fake_core
sys.modules["llama_index.core.node_parser"] = fake_node_parser


# Stub transformers and sentence_transformers so rag.embedding can import them.
fake_transformers_logging = types.ModuleType("transformers.logging")


def set_verbosity_error():
    pass


fake_transformers_logging.set_verbosity_error = set_verbosity_error
fake_transformers = types.ModuleType("transformers")
fake_transformers.logging = fake_transformers_logging
sys.modules["transformers"] = fake_transformers
sys.modules["transformers.logging"] = fake_transformers_logging

fake_sentence_transformers = types.ModuleType("sentence_transformers")
fake_sentence_transformers.SentenceTransformer = StubSentenceTransformer
sys.modules["sentence_transformers"] = fake_sentence_transformers
