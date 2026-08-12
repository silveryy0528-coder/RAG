import sys
import importlib.util

import pytest
from scripts.rag_chat import parse_args


def test_parse_args_WHEN_api_key_only_THEN_starts_interactive_loop(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag_chat.py",
            "--device",
            "cuda",
            "--show-top-k",
            "--api-key",
            "secret-key",
        ],
    )

    args = parse_args()

    assert args.question is None
    assert args.question_flag is None
    assert args.api_key == "secret-key"
    assert args.show_top_k is True


def test_parse_args_WHEN_question_flag_provided_THEN_runs_one_shot(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag_chat.py",
            "--question",
            "What is the title of the thesis?",
            "--device",
            "cpu",
        ],
    )

    args = parse_args()

    assert args.question == "What is the title of the thesis?"
    assert args.question_flag == "What is the title of the thesis?"


def test_parse_args_WHEN_config_file_provided_THEN_loads_yaml_defaults(tmp_path):
    if importlib.util.find_spec("yaml") is None:
        pytest.skip("PyYAML is not installed in this environment")

    config_path = tmp_path / "rag_chat.yaml"
    config_path.write_text(
        "k: 5\n"
        "device: cpu\n"
        "show_top_k: true\n"
        "use_metadata_reranking: true\n"
        "question: What is the title of the thesis?\n",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path)])

    assert args.k == 5
    assert args.device == "cpu"
    assert args.show_top_k is True
    assert args.use_metadata_reranking is True
    assert args.question == "What is the title of the thesis?"


def test_parse_args_WHEN_config_and_cli_value_provided_THEN_cli_overrides(tmp_path):
    if importlib.util.find_spec("yaml") is None:
        pytest.skip("PyYAML is not installed in this environment")

    config_path = tmp_path / "rag_chat.yaml"
    config_path.write_text("device: cpu\nk: 5\n", encoding="utf-8")

    args = parse_args(["--config", str(config_path), "--device", "cuda", "--k", "3"])

    assert args.device == "cuda"
    assert args.k == 3
