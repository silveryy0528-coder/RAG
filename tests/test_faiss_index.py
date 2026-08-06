import numpy as np
import pytest

from rag.faiss_index import (
    build_faiss_index,
    FaissFlatConfig,
    FaissIvfConfig,
    FaissIvfpqConfig,
)


def test_build_flat_index_adds_embeddings():
    embeddings = np.random.rand(5, 3).astype("float32")
    cfg = FaissFlatConfig()

    index = build_faiss_index(embeddings, cfg)

    # Our fake index stores added embeddings in .added
    assert hasattr(index, "added")
    assert index.added.shape == embeddings.shape


def test_build_raises_with_wrong_settings():
    embeddings = np.zeros((2, 4), dtype="float32")
    with pytest.raises(TypeError):
        build_faiss_index(embeddings, settings="not-a-config")


def test_build_ivf_and_ivfpq_train_and_add():
    embeddings = np.random.rand(10, 8).astype("float32")

    ivf_cfg = FaissIvfConfig(nlist=10)
    idx_ivf = build_faiss_index(embeddings, ivf_cfg)
    assert idx_ivf.trained is True
    assert idx_ivf.added.shape == embeddings.shape

    ivfpq_cfg = FaissIvfpqConfig(nlist=8, m=4, nbits=8)
    idx_ivfpq = build_faiss_index(embeddings, ivfpq_cfg)
    assert idx_ivfpq.trained is True
    assert idx_ivfpq.added.shape == embeddings.shape
