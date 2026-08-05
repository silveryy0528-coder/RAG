"""Utilities for splitting text into sentence-based chunks.

This module provides small dataclasses for chunking configuration and two
helper functions that wrap the SentenceSplitter from the llama_index project.

The import of llama_index is optional for tooling/CI that does not install
that package; a pylint disable is used so linters do not fail in minimal
environments.
"""
from dataclasses import dataclass
# The llama_index package is an optional heavy dependency used at runtime.
# CI / linters may not have it available; silence static import checks here.
from llama_index.core.node_parser import SentenceSplitter  # pylint: disable=import-error


# %%
@dataclass
class ChunkingConfig:
    """Configuration for text chunking behavior.

    Attributes:
        chunk_size: Maximum number of characters (approx) per chunk. Defaults to 500.
    """
    chunk_size: int = 500


@dataclass
class ChunkingSentenceConfig(ChunkingConfig):
    """Chunking configuration specialized for sentence-based splitting.

    Attributes:
        chunk_overlap: Number of characters (approx) to overlap between chunks.
    """
    chunk_overlap: int = 50


def sentence_splitter(full_text, chunk_size, chunk_overlap):
    """Split the provided full_text into sentence-based nodes.

    Wraps the SentenceSplitter from llama_index.core.node_parser and returns
    whatever structure that class produces. The function purposefully does
    not interpret the returned nodes; it simply forwards arguments and returns
    the result so callers can remain library-agnostic.

    Args:
        full_text: The complete text to split into chunks.
        chunk_size: Approximate maximum size for each chunk.
        chunk_overlap: Approximate overlap between adjacent chunks.

    Returns:
        The result of SentenceSplitter(...).split_text(full_text).
    """
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.split_text(full_text)
    return nodes


def chunk_text(documents, settings):
    """Join document texts and split them into nodes using provided settings.

    The function concatenates the .text attribute from each document with a
    newline separator, then delegates to sentence_splitter using values from
    the provided settings dataclass.

    Args:
        documents: Iterable of objects that have a .text attribute.
        settings: ChunkingConfig or ChunkingSentenceConfig instance.

    Returns:
        The nodes produced by sentence_splitter.
    """
    full_text = "\n".join(doc.text for doc in documents)
    nodes = sentence_splitter(full_text, settings.chunk_size, settings.chunk_overlap)
    return nodes


if __name__ == "__main__":
    pass
