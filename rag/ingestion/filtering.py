"""Filtering helpers for PDF ingestion.

This module contains heuristics for normalizing page text, identifying pages
that should be skipped, and extracting section labels from page headers.
"""

import re
from dataclasses import dataclass
from typing import Iterable, Optional

SPECIAL_SECTIONS = [
    "list of publications",
    "summary",
    "propositions",
    "acknowledgements",
    "about the author",
    "samenvatting",
    "stellingen",
    "copyright",
    "references",
]

EXCLUDED_SECTIONS = [
    "samenvatting",
    "stellingen",
    "copyright",
    "references",
]


@dataclass
class Margin:
    top: int = 50
    bottom: int = 50
    left: int = 50
    right: int = 50


def clean_text(text: str) -> str:
    """Normalize page text."""
    text = (text or "").replace("\xa0", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r" +", " ", text)
    return text


def is_content_page(text: str) -> bool:
    text = (text or "").strip()
    return ". . . . ." in text


def is_empty_page(text: str) -> bool:
    return len((text or "").strip()) == 0


def is_bad_page(text: str) -> bool:
    return is_empty_page(text) or is_content_page(text)


def extract_header(page, header_height: int = 40) -> str:
    """Extract header text from a page-like object."""
    rect = getattr(page, "rect", None)
    if rect is None:
        try:
            return (page.get_text("text") or "").strip().lower()
        except Exception:
            return ""

    try:
        header_rect = type(rect)(rect.x0, rect.y0, rect.x1, rect.y0 + header_height)
    except Exception:
        header_rect = rect

    try:
        header_text = page.get_text("text", clip=header_rect)
    except TypeError:
        header_text = page.get_text("text")
    except Exception:
        header_text = ""

    return (header_text or "").strip().lower()


def extract_section_name(page, special_sections: Optional[Iterable[str]] = None) -> str:
    """Determine a coarse section label for a page."""
    special_sections = list(special_sections or SPECIAL_SECTIONS)
    header = extract_header(page)
    for section_name in special_sections:
        if section_name in header:
            return section_name

    if len(header.strip()) == 0:
        try:
            text = (page.get_text("text") or "").lower()
        except Exception:
            text = ""
        for section_name in special_sections:
            if section_name in text:
                return section_name
        return "structural"

    return "body"
