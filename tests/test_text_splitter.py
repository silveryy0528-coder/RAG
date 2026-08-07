from types import SimpleNamespace

import rag.text_splitter as chunking


class FakeSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.last_text = None

    def split_text(self, text):
        self.last_text = text
        return [
            {
                "text": text,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }
        ]


def test_sentence_splitter_WHEN_called_with_chunk_parameters_THEN_forwards_arguments_and_returns_nodes(
    monkeypatch,
):
    monkeypatch.setattr(chunking, "SentenceSplitter", FakeSplitter)

    full_text = "hello world"
    nodes = chunking.sentence_splitter(full_text, chunk_size=123, chunk_overlap=45)

    assert isinstance(nodes, list)
    assert nodes[0]["text"] == full_text
    assert nodes[0]["chunk_size"] == 123
    assert nodes[0]["chunk_overlap"] == 45


def test_chunk_text_WHEN_documents_joined_and_settings_provided_THEN_uses_settings_for_splitter(
    monkeypatch,
):
    monkeypatch.setattr(chunking, "SentenceSplitter", FakeSplitter)

    docs = [SimpleNamespace(text="first"), SimpleNamespace(text="second")]
    settings = chunking.ChunkingSentenceConfig(chunk_size=10, chunk_overlap=2)

    nodes = chunking.chunk_text(docs, settings)

    assert nodes[0]["text"] == "first\nsecond"
    assert nodes[0]["chunk_size"] == 10
    assert nodes[0]["chunk_overlap"] == 2


def test_ChunkingConfig_WHEN_default_constructed_THEN_uses_expected_defaults():
    cfg = chunking.ChunkingConfig()
    assert cfg.chunk_size == 500

    scfg = chunking.ChunkingSentenceConfig()
    assert scfg.chunk_size == 500
    assert scfg.chunk_overlap == 50
