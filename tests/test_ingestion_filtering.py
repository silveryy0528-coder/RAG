from types import SimpleNamespace

import rag.ingestion.filtering as filtering


class Rect:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1


class FakePage:
    def __init__(self, header_text: str = "", full_text: str = ""):
        self.rect = Rect(0, 0, 100, 200)
        self._header_text = header_text
        self._full_text = full_text

    def get_text(self, kind, clip=None):
        # ignore clip in this fake and return header or full text depending on call
        if clip is not None:
            return self._header_text
        return self._full_text


def test_clean_text_WHEN_called_THEN_replaces_nonbreaking_spaces_tabs_and_hyphen_newlines():
    raw = "Hello\xa0world\tand\nnew-lines -\nbroken"
    cleaned = filtering.clean_text(raw)
    assert "\xa0" not in cleaned
    assert "\t" not in cleaned
    assert "-\n" not in cleaned


def test_is_bad_page_WHEN_page_is_empty_or_toc_THEN_returns_true():
    assert filtering.is_bad_page("")
    assert filtering.is_bad_page("   ")
    assert filtering.is_bad_page("1 . . . . . 2")


def test_extract_section_name_WHEN_header_contains_special_section_THEN_returns_that_section():
    p = FakePage(header_text="Summary of contributions", full_text="")
    name = filtering.extract_section_name(p)
    assert name == "summary"


def test_extract_section_name_WHEN_header_empty_and_full_text_contains_special_section_THEN_returns_that_section():
    p = FakePage(
        header_text="", full_text="This thesis contains propositions and other text"
    )
    name = filtering.extract_section_name(p)
    assert name == "propositions"


def test_extract_section_name_WHEN_header_contains_non_special_text_THEN_returns_body():
    p = FakePage(header_text="Chapter 1: Introduction", full_text="Intro text")
    name = filtering.extract_section_name(p)
    assert name == "body"
