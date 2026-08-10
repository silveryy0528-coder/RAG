"""PDF ingestion utilities.

This module reads a PDF file, filters unwanted pages, normalizes page text,
and produces page-like document objects for further ingestion.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional
import os

from .filtering import (
    clean_text,
    is_bad_page,
    extract_section_name,
    Margin,
    EXCLUDED_SECTIONS,
)


@dataclass
class Page:
    text: str
    metadata: dict


def _default_document_factory(text: str, metadata: dict) -> Page:
    return Page(text=text, metadata=metadata)


def read_pdf_file(
    pdf_file: str,
    fitz_module: Any = None,
    margin: Margin = Margin(),
    excluded_sections: Optional[Iterable[str]] = None,
    document_factory: Callable[[str, dict], Any] = None,
) -> List[Any]:
    """Open a PDF and return per-page document objects."""
    if fitz_module is None:
        try:
            import fitz as _fitz  # type: ignore

            fitz_module = _fitz
        except Exception as exc:  # pragma: no cover - environment specific
            raise ImportError(
                "fitz (PyMuPDF) is required for read_pdf_file unless a fitz_module is provided"
            ) from exc

    if document_factory is None:
        document_factory = _default_document_factory

    excluded_sections = set(excluded_sections or EXCLUDED_SECTIONS)
    doc = fitz_module.open(pdf_file)
    doc_id = os.path.basename(pdf_file)

    documents = []
    for i, page in enumerate(doc):
        try:
            raw_text = page.get_text("text") or ""
        except Exception:
            raw_text = ""

        if is_bad_page(raw_text):
            continue

        section_name = extract_section_name(page)
        if section_name in excluded_sections:
            continue

        if "CV" in doc_id and section_name == "structural":
            section_name = "body"

        rect = getattr(page, "rect", None)
        if rect is not None:
            try:
                content_rect = fitz_module.Rect(
                    rect.x0 + margin.left,
                    rect.y0 + margin.top,
                    rect.x1 - margin.right,
                    rect.y1 - margin.bottom,
                )
                page_text = page.get_text("text", clip=content_rect) or ""
            except TypeError:
                page_text = page.get_text("text") or ""
        else:
            try:
                page_text = page.get_text("text") or ""
            except Exception:
                page_text = ""

        page_text = clean_text(page_text)
        metadata = {"page": i + 1, "doc_id": doc_id, "section": section_name}
        document = document_factory(page_text, metadata)
        documents.append(document)

    return documents
