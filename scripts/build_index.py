"""Build document chunks, embeddings and a FAISS index from raw PDF files.

This CLI reads PDF files from ``data/raw`` by default, chunks them, embeds the
chunks, normalizes the embedding vectors with FAISS, and writes the processed
artifacts into ``data/processed``. It supports both direct CLI flags and YAML
config files via ``--config``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import faiss
import numpy as np

from rag.embedding import embed_text, load_embedder
from rag.faiss_index import build_faiss_index, FaissFlatConfig
from rag.ingestion import chunk_multiple_documents
from rag.ingestion.pipeline import IngestOptions
from rag.text_splitter import ChunkingSentenceConfig
from scripts.cli_config import load_config

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50
_PATH_KEYS: set[str] = {"raw_dir", "processed_dir"}
_INT_KEYS: set[str] = {"chunk_size", "chunk_overlap"}
_ALLOWED_KEYS: set[str] = {
    "raw_dir",
    "processed_dir",
    "device",
    "chunk_size",
    "chunk_overlap",
}


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_pdf_files(raw_dir: Path) -> list[Path]:
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")
    return pdf_files


def save_chunks(chunks: list, output_path: Path) -> None:
    with output_path.open("wb") as file:
        pickle.dump(chunks, file)


def save_embeddings(embeddings: np.ndarray, output_path: Path) -> None:
    np.save(str(output_path), embeddings)


def save_faiss_index(index: faiss.Index, output_path: Path) -> None:
    faiss.write_index(index, str(output_path))


def build_index(
    raw_dir: Path,
    processed_dir: Path,
    device: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    raw_dir = raw_dir.expanduser()
    processed_dir = processed_dir.expanduser()
    processed_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = discover_pdf_files(raw_dir)
    print(f"Found {len(pdf_files)} PDF file(s) in {raw_dir}")

    chunk_settings = ChunkingSentenceConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    options = IngestOptions(chunk_settings=chunk_settings)
    chunks = chunk_multiple_documents(pdf_files, options=options)
    print(f"Created {len(chunks)} chunks")

    texts = [chunk.text for chunk in chunks]
    embedder = load_embedder(device=device)
    embeddings = embed_text(embedder, texts, device=device)
    faiss.normalize_L2(embeddings)

    index = build_faiss_index(embeddings, FaissFlatConfig())
    print(f"Built FAISS index with {index.ntotal} vectors")

    chunk_file = processed_dir / "chunks.pkl"
    embeddings_file = processed_dir / "embeddings.npy"
    index_file = processed_dir / "faiss.index"

    save_chunks(chunks, chunk_file)
    save_embeddings(embeddings, embeddings_file)
    save_faiss_index(index, index_file)

    print(f"Saved chunks to {chunk_file}")
    print(f"Saved embeddings to {embeddings_file}")
    print(f"Saved FAISS index to {index_file}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = resolve_project_root()
    default_raw = project_root / "data" / "raw"
    default_processed = project_root / "data" / "processed"

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_defaults = {}
    if pre_args.config is not None:
        config_defaults = load_config(
            pre_args.config.expanduser(),
            allowed=_ALLOWED_KEYS,
            bool_keys=set(),
            path_keys=_PATH_KEYS,
            int_keys=_INT_KEYS,
        )

    parser = argparse.ArgumentParser(
        description="Build chunks, embeddings and a normalized FAISS index from raw PDFs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file with defaults for CLI options. (Path)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw,
        help="Directory containing raw PDF files. (Path)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed,
        help="Directory where chunks, embeddings and index are written. (Path)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Embedding device, e.g. 'cuda' or 'cpu'. (str)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Approximate maximum characters per text chunk. (int)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Approximate overlapping characters between chunks. (int)",
    )
    parser.set_defaults(**config_defaults)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    build_index(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        device=args.device,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
