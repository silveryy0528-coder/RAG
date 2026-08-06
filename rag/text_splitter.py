"""Utilities for splitting text into sentence-based chunks.

This module wraps the SentenceSplitter implementation from the
``llama_index`` project and exposes small configuration dataclasses and
convenience helpers to split text or collections of documents into chunks.

Note
----
This module imports ``SentenceSplitter`` from ``llama_index.core.node_parser``
and therefore requires that package at runtime. Tests may stub or monkeypatch
that dependency as needed.
"""

from dataclasses import dataclass
from llama_index.core.node_parser import SentenceSplitter


@dataclass
class ChunkingConfig:
    """Configuration for text chunking.

    Parameters
    ----------
    chunk_size : int, optional
        Approximate maximum number of characters per chunk. Default is 500.
    """

    chunk_size: int = 500


@dataclass
class ChunkingSentenceConfig(ChunkingConfig):
    """Chunking settings specialized for sentence-based splitting.

    Parameters
    ----------
    chunk_overlap : int, optional
        Approximate number of characters to overlap between adjacent chunks.
    """

    chunk_overlap: int = 50


def sentence_splitter(full_text, chunk_size, chunk_overlap):
    """Split full text into sentence-based nodes.

    This function constructs a ``SentenceSplitter`` with the provided
    ``chunk_size`` and ``chunk_overlap`` and returns the nodes produced by
    its ``split_text`` method.

    Parameters
    ----------
    full_text : str
        The complete text to split into chunks.
    chunk_size : int
        Approximate maximum size for each chunk.
    chunk_overlap : int
        Approximate overlap between adjacent chunks.

    Returns
    -------
    list
        The nodes produced by :meth:`SentenceSplitter.split_text`.
    """

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.split_text(full_text)
    return nodes


def chunk_text(documents, settings):
    """Join document texts and split them into nodes using ``settings``.

    The function concatenates the ``.text`` attribute from each document with
    a newline separator and delegates to :func:`sentence_splitter` using
    values from the provided ``settings`` dataclass.

    Parameters
    ----------
    documents : iterable
        Iterable of objects that expose a ``.text`` attribute.
    settings : ChunkingConfig or ChunkingSentenceConfig
        Chunking configuration containing ``chunk_size`` and ``chunk_overlap``.

    Returns
    -------
    list
        The nodes produced by :func:`sentence_splitter`.
    """

    full_text = "\n".join(doc.text for doc in documents)
    nodes = sentence_splitter(full_text, settings.chunk_size, settings.chunk_overlap)
    return nodes


if __name__ == "__main__":
    pass
