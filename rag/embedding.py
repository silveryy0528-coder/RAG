"""Embedding utilities for the RAG project.

This module configures Hugging Face transformers logging to reduce noise and
provides a small wrapper around ``sentence_transformers.SentenceTransformer``
for loading a sentence embedder and encoding text into vector embeddings.
"""

import os
import logging
from transformers import logging as hf_logging
from sentence_transformers import SentenceTransformer

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()


def load_embedder(model_name="all-MiniLM-L6-v2", device="cuda"):
    """Load a sentence embedder.

    Parameters
    ----------
    model_name : str, optional
        The name of the sentence-transformers model to load. Default is
        ``"all-MiniLM-L6-v2"``.
    device : str, optional
        The device on which to load the model, such as ``"cuda"`` or
        ``"cpu"``. Default is ``"cuda"``.

    Returns
    -------
    SentenceTransformer
        The loaded sentence transformer embedder.
    """
    return SentenceTransformer(model_name, device=device)


def embed_text(embedder, texts, device="cuda"):
    """Encode text strings into vector embeddings.

    Parameters
    ----------
    embedder : SentenceTransformer
        The sentence transformer embedder instance.
    texts : list[str] | str
        One or more text strings to encode.
    device : str, optional
        The device on which to encode the text. Default is ``"cuda"``.

    Returns
    -------
    numpy.ndarray
        Normalized vector embeddings for the input texts.
    """
    embeddings = embedder.encode(
        texts, device=device, convert_to_numpy=True, normalize_embeddings=True
    )
    return embeddings
