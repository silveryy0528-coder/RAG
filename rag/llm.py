"""Lightweight LLM client helpers.

This module provides a tiny wrapper around an OpenAI-style client and a
backward-compatible helper for building RAG prompts. The module deliberately
keeps model invocation logic thin and delegates prompt construction to the
``rag.prompting`` module.
"""

from typing import Any


def load_openai_client(api_key=None):
    """Instantiate an OpenAI client wrapper.

    Parameters
    ----------
    api_key : str, optional
        API key used to initialize the client.

    Returns
    -------
    object
        An OpenAI client instance.
    """
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
    """Generate an answer from the provided client using a prompt.

    Parameters
    ----------
    client : object
        The model client exposing chat.completions.create(...).
    prompt : str
        The prompt text to send to the model.
    model_name : str, optional
        The model identifier to use.
    temperature : float, optional
        Sampling temperature.

    Returns
    -------
    str
        Text content of the first choice returned by the model.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return normalize_answer(content)


def normalize_answer(text: str) -> str:
    """Normalize generated answers for QA-style evaluation.

    The goal is to make answers more concise and comparable to the ground-truth
    dataset. The function strips surrounding whitespace, removes common leading
    phrases such as "The answer is" or "It is", and collapses repeated
    whitespace. It keeps the content mostly intact while making the output more
    extractive and less verbose.
    """
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = cleaned.replace("\n", " ")
    cleaned = " ".join(cleaned.split())

    for prefix in ["Answer:", "The answer is", "It is", "The thesis is", "The main contribution is"]:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
            break

    if cleaned.lower().startswith("not found"):
        return "Not found"

    return cleaned
