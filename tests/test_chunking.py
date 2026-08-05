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
