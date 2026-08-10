"""Ingestion package.

Public API mirrors the previous rag.ingestion module while splitting
responsibilities into pdf_io, filtering and chunking submodules.
"""

from .pdf_io import read_pdf_file, Page  # noqa: F401
from .filtering import (
    clean_text,
    is_bad_page,
    extract_section_name,
    SPECIAL_SECTIONS,
    EXCLUDED_SECTIONS,
    Margin,
)  # noqa: F401
from .pipeline import (
    chunk_single_document,
    chunk_multiple_documents,
    Chunk,
)  # noqa: F401

__all__ = [
    "read_pdf_file",
    "Page",
    "clean_text",
    "is_bad_page",
    "extract_section_name",
    "SPECIAL_SECTIONS",
    "EXCLUDED_SECTIONS",
    "Margin",
    "chunk_single_document",
    "chunk_multiple_documents",
    "Chunk",
]
