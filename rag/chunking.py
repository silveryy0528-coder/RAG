# %%
from dataclasses import dataclass
from llama_index.core.node_parser import SentenceSplitter


# %%
@dataclass
class ChunkingConfig:
    chunk_size: int = 500


@dataclass
class ChunkingSentenceConfig(ChunkingConfig):
    chunk_overlap: int = 50


def sentence_splitter(full_text, chunk_size, chunk_overlap):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.split_text(full_text)
    return nodes


def chunk_text(documents, settings):
    full_text = "\n".join(doc.text for doc in documents)
    nodes = sentence_splitter(full_text, settings.chunk_size, settings.chunk_overlap)
    return nodes


if __name__ == "__main__":
    pass
