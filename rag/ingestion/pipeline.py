"""Ingestion pipeline helpers: grouping pages and turning them into chunks.

This module orchestrates PDF pages (produced by pdf_io.read_pdf_file) into
chunk objects suitable for indexing. It delegates the actual text splitting
primitive to :mod:`rag.text_splitter` (imported lazily when available) so the
pipeline remains independent from heavy third-party runtime dependencies.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional
import logging

from .filtering import clean_text
from .pdf_io import read_pdf_file

# Try to import the low-level text splitter at module import time. If the
# optional dependency (llama_index) is not present we fall back to a simple
# joiner later. ImportError is expected in tests that inject fakes.
try:
    from rag.text_splitter import chunk_text as _chunk_text, ChunkingSentenceConfig  # type: ignore
except ImportError:
    _chunk_text = None
    ChunkingSentenceConfig = None

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Small container representing a text chunk with metadata.

    Attributes
    ----------
    text : str
        Chunk text content.
    metadata : dict
        Chunk metadata such as section, doc_id and chunk_id.
    """

    text: str
    metadata: dict


def group_pages_by_section(pages: Iterable[Any]) -> Dict[str, List[Any]]:
    """Group pages by section."""
    sections: Dict[str, List[Any]] = {}
    for page in pages:
        meta = getattr(page, "metadata", {})
        section_name = meta.get("section")
        sections.setdefault(section_name, []).append(page)
    return sections


@dataclass
class IngestOptions:
    """Container for optional ingestion parameters.

    Using a single options object keeps the public function signature small and
    easier to maintain while still allowing callers to inject test doubles.
    """

    chunk_settings: Any = None
    fitz_module: Any = None
    document_factory: Callable[[str, dict], Any] = None
    chunk_text_fn: Optional[Callable[[Iterable[Any], Any], List[str]]] = None


def chunk_single_document(
    pdf_file: str,
    options: Optional[IngestOptions] = None,
    chunk_id_offset: int = 0,
) -> List[Chunk]:
    """Read a PDF and produce chunks for a single document."""
    options = options or IngestOptions()

    # Resolve chunk_text function and default settings. Prefer an injected
    # chunk_text_fn, otherwise use the installed low-level splitter when present
    # or fall back to a simple joiner for environments without the splitter.
    chunk_text_fn = options.chunk_text_fn or _chunk_text
    chunk_settings = options.chunk_settings
    if chunk_text_fn is None:

        def _fallback_chunk_text(pages, _settings=None):
            return ["\n".join(getattr(p, "text", "") for p in pages)]

        chunk_text_fn = _fallback_chunk_text
    else:
        if chunk_settings is None and ChunkingSentenceConfig is not None:
            chunk_settings = ChunkingSentenceConfig()

    # Read pages (pdf_io handles fitz injection)
    pages = read_pdf_file(
        pdf_file,
        fitz_module=options.fitz_module,
        document_factory=options.document_factory,
    )
    if not pages:
        return []

    first_meta = getattr(pages[0], "metadata", {})
    doc_id = first_meta.get("doc_id")

    sections = group_pages_by_section(pages)

    all_chunks: List[Chunk] = []
    chunk_id = chunk_id_offset
    for section_name, section_pages in sections.items():
        logger.info(
            "Processing %s - Section: %s with %d pages",
            doc_id,
            section_name,
            len(section_pages),
        )
        nodes = chunk_text_fn(section_pages, chunk_settings)
        for node in nodes:
            node_text = clean_text(node)
            chunk = Chunk(
                text=node_text,
                metadata={
                    "section": section_name,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                },
            )
            chunk_id += 1
            all_chunks.append(chunk)

    return all_chunks


def chunk_multiple_documents(
    pdf_files: Iterable[str],
    options: Optional[IngestOptions] = None,
) -> List[Chunk]:
    """Process multiple PDF files and concatenate produced chunks."""
    all_chunks: List[Chunk] = []
    id_offset = 0

    for pdf_file in pdf_files:
        chunks = chunk_single_document(
            pdf_file, options=options, chunk_id_offset=id_offset
        )
        id_offset += len(chunks)
        all_chunks.extend(chunks)

    return all_chunks
