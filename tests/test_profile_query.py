"""Tests for scripts/profile_query.py."""

from __future__ import annotations

from scripts.profile_query import parse_args


def test_parse_args_WHEN_called_THEN_uses_expected_defaults():
    args = parse_args([])

    assert args.question == "What is the title of the thesis"
    assert args.device == "cpu"
    assert args.k == 3
    assert args.use_metadata_reranking is True
