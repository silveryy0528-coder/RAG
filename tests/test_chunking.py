import sys
import types
from types import SimpleNamespace

# Create a minimal stub package 'llama_index.core.node_parser' so importing
# rag.chunking does not attempt to import the real llama_index package.
# This must be done before importing rag.chunking.
fake_node_parser = types.ModuleType("llama_index.core.node_parser")

class _StubSentenceSplitter:
    def __init__(self, chunk_size=None, chunk_overlap=None):
        # store values so tests or the module under test can inspect them
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        # Return a simple predictable result; tests will monkeypatch later with a richer FakeSplitter
        return [{"text": text, "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}]

# expose SentenceSplitter in the stub module
fake_node_parser.SentenceSplitter = _StubSentenceSplitter

# Build intermediate modules and insert into sys.modules so nested imports succeed
fake_core = types.ModuleType("llama_index.core")
fake_core.node_parser = fake_node_parser
fake_llama = types.ModuleType("llama_index")
fake_llama.core = fake_core

# Register them in sys.modules so 'from llama_index.core.node_parser import SentenceSplitter'
# works when rag.chunking is imported.
sys.modules["llama_index"] = fake_llama
sys.modules["llama_index.core"] = fake_core
sys.modules["llama_index.core.node_parser"] = fake_node_parser

# Now safe to import the module under test
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
    # Replace the SentenceSplitter used in the module with a fake one
    monkeypatch.setattr(chunking, "SentenceSplitter", FakeSplitter)

    full_text = "hello world"
    nodes = chunking.sentence_splitter(full_text, chunk_size=123, chunk_overlap=45)

    assert isinstance(nodes, list)
    assert nodes[0]["text"] == full_text
    assert nodes[0]["chunk_size"] == 123
    assert nodes[0]["chunk_overlap"] == 45


def test_chunk_text_joins_documents_and_uses_settings(monkeypatch):
    # Patch the splitter to capture the joined text and settings
    monkeypatch.setattr(chunking, "SentenceSplitter", FakeSplitter)

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
