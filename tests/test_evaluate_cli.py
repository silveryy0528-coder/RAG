"""Tests for scripts/evaluate.py argument parsing."""

from __future__ import annotations

import importlib.util

import pytest

from scripts.evaluate import parse_args


def test_parse_args_WHEN_no_args_THEN_uses_defaults():
    args = parse_args([])

    assert args.k == 3
    assert args.device == "cuda"
    assert args.generate is False
    assert args.use_metadata_reranking is False
    assert args.debug_failures == 0


def test_parse_args_WHEN_cli_flags_provided_THEN_parsed_correctly():
    args = parse_args([
        "--device", "cpu",
        "--k", "5",
        "--generate",
        "--api-key", "sk-test",
        "--use-metadata-reranking",
    ])

    assert args.device == "cpu"
    assert args.k == 5
    assert args.generate is True
    assert args.api_key == "sk-test"
    assert args.use_metadata_reranking is True


def test_parse_args_WHEN_config_file_provided_THEN_loads_yaml_defaults(tmp_path):
    if importlib.util.find_spec("yaml") is None:
        pytest.skip("PyYAML is not installed in this environment")

    config_path = tmp_path / "evaluate.yaml"
    config_path.write_text(
        "k: 7\n"
        "device: cpu\n"
        "use_metadata_reranking: true\n"
        "debug_failures: 3\n",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path)])

    assert args.k == 7
    assert args.device == "cpu"
    assert args.use_metadata_reranking is True
    assert args.debug_failures == 3


def test_parse_args_WHEN_config_and_cli_value_provided_THEN_cli_overrides(tmp_path):
    if importlib.util.find_spec("yaml") is None:
        pytest.skip("PyYAML is not installed in this environment")

    config_path = tmp_path / "evaluate.yaml"
    config_path.write_text("k: 7\ndevice: cpu\n", encoding="utf-8")

    args = parse_args(["--config", str(config_path), "--device", "cuda", "--k", "3"])

    assert args.device == "cuda"
    assert args.k == 3


def test_parse_args_WHEN_unknown_config_key_THEN_raises(tmp_path):
    if importlib.util.find_spec("yaml") is None:
        pytest.skip("PyYAML is not installed in this environment")

    config_path = tmp_path / "evaluate.yaml"
    config_path.write_text("nonexistent_key: oops\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config keys"):
        parse_args(["--config", str(config_path)])
