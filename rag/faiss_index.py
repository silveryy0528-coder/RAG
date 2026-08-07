"""FAISS index builders and helpers.

This module provides small dataclasses to describe FAISS index settings and
helper functions to build an index for given embeddings.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class FaissConfig:
    """Base class for FAISS configuration objects."""


@dataclass
class FaissFlatConfig(FaissConfig):
    """Configuration selecting a flat (exact) L2 index."""


@dataclass
class FaissIvfConfig(FaissConfig):
    """Configuration for an IVF flat index.

    Parameters
    ----------
    nlist : int
        Number of inverted lists (centroids) for the IVF index.
    """

    nlist: int = 40


@dataclass
class FaissIvfpqConfig(FaissConfig):
    """Configuration for an IVFPQ index.

    Parameters
    ----------
    nlist : int
        Number of inverted lists.
    m : int
        Number of subvectors for PQ.
    nbits : int
        Number of bits per subvector code.
    """

    nlist: int = 40
    m: int = 8
    nbits: int = 8


def _build_faiss_flat(dim: int):
    import faiss

    return faiss.IndexFlatL2(dim)


def _build_faiss_ivf(embeddings: Any, dim: int, nlist: int):
    import faiss

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    # IVF indexes require training
    index.train(embeddings)  # pylint: disable=no-value-for-parameter
    return index


def _build_faiss_ivfpq(
    embeddings: Any, dim: int, nlist: int, m: int, nbits: int
):
    import faiss

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    # IVFPQ requires training as well
    index.train(embeddings)  # pylint: disable=no-value-for-parameter
    return index


def build_faiss_index(embeddings: Any, settings: FaissConfig):
    """Build and populate a FAISS index according to ``settings``.

    Parameters
    ----------
    embeddings : numpy.ndarray
        2-D array of shape (N, D) containing the float32 embeddings.
    settings : FaissConfig
        One of the config dataclasses defined in this module.

    Returns
    -------
    faiss.Index
        A FAISS index instance populated with the provided embeddings.

    Raises
    ------
    TypeError
        If ``settings`` is not a FaissConfig instance.
    ValueError
        If ``embeddings`` is not a 2-D numpy array.
    """
    if not isinstance(settings, FaissConfig):
        raise TypeError("Wrong FAISS settings provided.")

    if not hasattr(embeddings, "shape") or len(embeddings.shape) != 2:
        raise ValueError("Embeddings must be a 2-D numpy array")

    dim = int(embeddings.shape[1])

    if isinstance(settings, FaissFlatConfig):
        index = _build_faiss_flat(dim)
    elif isinstance(settings, FaissIvfConfig):
        index = _build_faiss_ivf(embeddings, dim, settings.nlist)
    elif isinstance(settings, FaissIvfpqConfig):
        index = _build_faiss_ivfpq(
            embeddings, dim, settings.nlist, settings.m, settings.nbits
        )
    else:
        raise TypeError("Unsupported FAISS settings type")

    # Add embeddings to the index
    index.add(embeddings)
    return index
