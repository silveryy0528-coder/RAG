"""Lightweight LLM client helpers.

This module provides a tiny wrapper around an OpenAI-style client and a
backward-compatible helper for building RAG prompts. The module deliberately
keeps model invocation logic thin and delegates prompt construction to the
``rag.prompting`` module.
"""

from typing import Any


def load_openai_client(api_key=None):
    """Instantiate an OpenAI client wrapper."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai is required to create a client. "
            "Install it or provide a compatible fake module for testing."
        ) from exc

    return OpenAI(api_key=api_key)


def generate_answer(
    client: Any,
    prompt: str,
    model_name: str = "gpt-4.1-mini",
    temperature: float = 1.0,
) -> str:
    """Generate an answer from the provided client."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return normalize_answer(content)


def normalize_answer(text: str) -> str:
    """Normalize generated answers for QA-style evaluation."""
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = cleaned.replace("\n", " ")
    cleaned = " ".join(cleaned.split())

    for prefix in [
        "Answer:",
        "The answer is",
        "It is",
        "The thesis is",
        "The main contribution is",
    ]:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
            break

    if cleaned.lower().startswith("not found"):
        return "Not found"

    return cleaned
