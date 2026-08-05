import sys
import types
from types import SimpleNamespace

import rag.chunking as chunking


class FakeSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        # store constructor args so tests can assert on them
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.last_text = None

    def split_text(self, text):
        # record input and return a predictable structure
        self.last_text = text
        return [
            {
                "text": text,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }
        ]


def test_sentence_splitter_passes_arguments_and_returns_nodes(monkeypatch):
    # Install a fake llama_index.core.node_parser module so the lazy import
    # inside sentence_splitter resolves to our FakeSplitter.
    fake = types.ModuleType("llama_index.core.node_parser")
    fake.SentenceSplitter = FakeSplitter
    monkeypatch.setitem(sys.modules, "llama_index", types.ModuleType("llama_index"))
    monkeypatch.setitem(sys.modules, "llama_index.core", types.ModuleType("llama_index.core"))
    monkeypatch.setitem(sys.modules, "llama_index.core.node_parser", fake)

    full_text = "hello world"
    nodes = chunking.sentence_splitter(full_text, chunk_size=123, chunk_overlap=45)

    assert isinstance(nodes, list)
    assert nodes[0]["text"] == full_text
    assert nodes[0]["chunk_size"] == 123
    assert nodes[0]["chunk_overlap"] == 45


def test_chunk_text_joins_documents_and_uses_settings(monkeypatch):
    # Install a fake llama_index.core.node_parser module so the lazy import
    # inside sentence_splitter resolves to our FakeSplitter.
    fake = types.ModuleType("llama_index.core.node_parser")
    fake.SentenceSplitter = FakeSplitter
    monkeypatch.setitem(sys.modules, "llama_index", types.ModuleType("llama_index"))
    monkeypatch.setitem(sys.modules, "llama_index.core", types.ModuleType("llama_index.core"))
    monkeypatch.setitem(sys.modules, "llama_index.core.node_parser", fake)

    docs = [SimpleNamespace(text="first"), SimpleNamespace(text="second")]
    settings = chunking.ChunkingSentenceConfig(chunk_size=10, chunk_overlap=2)

    nodes = chunking.chunk_text(docs, settings)

    # chunk_text should join document texts with a newline
    assert nodes[0]["text"] == "first\nsecond"
    assert nodes[0]["chunk_size"] == 10
    assert nodes[0]["chunk_overlap"] == 2


def test_config_dataclass_defaults():
    cfg = chunking.ChunkingConfig()
    assert cfg.chunk_size == 500

    scfg = chunking.ChunkingSentenceConfig()
    assert scfg.chunk_size == 500
    assert scfg.chunk_overlap == 50
