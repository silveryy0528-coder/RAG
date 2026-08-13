"""Tests for scripts/build_index.py argument parsing."""

from __future__ import annotations

import importlib.util

import pytest

from scripts.build_index import parse_args


def test_parse_args_WHEN_no_args_THEN_uses_defaults():
    args = parse_args([])

    assert args.device == "cuda"
    assert args.chunk_size == 400
    assert args.chunk_overlap == 50


def test_parse_args_WHEN_cli_flags_provided_THEN_parsed_correctly():
    args = parse_args(
        [
            "--device",
            "cpu",
            "--chunk-size",
            "256",
            "--chunk-overlap",
            "32",
        ]
    )

    assert args.device == "cpu"
    assert args.chunk_size == 256
    assert args.chunk_overlap == 32


def test_parse_args_WHEN_config_file_provided_THEN_loads_yaml_defaults(tmp_path):
    if importlib.util.find_spec("yaml") is None:
        pytest.skip("PyYAML is not installed in this environment")

    config_path = tmp_path / "build_index.yaml"
    config_path.write_text(
        "device: cpu\n"
        "chunk_size: 256\n"
        "chunk_overlap: 32\n",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path)])

    assert args.device == "cpu"
    assert args.chunk_size == 256
    assert args.chunk_overlap == 32


def test_parse_args_WHEN_config_and_cli_value_provided_THEN_cli_overrides(tmp_path):
    if importlib.util.find_spec("yaml") is None:
        pytest.skip("PyYAML is not installed in this environment")

    config_path = tmp_path / "build_index.yaml"
    config_path.write_text("device: cpu\nchunk_size: 256\n", encoding="utf-8")

    args = parse_args(
        ["--config", str(config_path), "--device", "cuda", "--chunk-size", "400"]
    )

    assert args.device == "cuda"
    assert args.chunk_size == 400
