"""Embedding utilities for the RAG project.

This module configures Hugging Face transformers logging to reduce noise and
provides a small wrapper around ``sentence_transformers.SentenceTransformer``
for loading a sentence embedder and encoding text into vector embeddings.
"""

import os
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def _configure_transformers_logging():
    try:
        from transformers import logging as hf_logging
    except ImportError:
        return

    hf_logging.set_verbosity_error()


_configure_transformers_logging()


def load_embedder(model_name="all-MiniLM-L6-v2", device="cuda"):
    """Load a sentence embedder."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence_transformers is required to load an embedder. "
            "Install it or provide a compatible fake class for testing."
        ) from exc

    return SentenceTransformer(model_name, device=device)


def embed_text(embedder, texts, device="cuda"):
    """Encode text strings into vector embeddings."""
    embeddings = embedder.encode(
        texts, device=device, convert_to_numpy=True, normalize_embeddings=True
    )
    return embeddings
