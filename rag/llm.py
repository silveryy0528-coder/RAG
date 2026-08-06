"""Lightweight LLM client helpers.

This module provides a tiny wrapper around an OpenAI-style client and a
backward-compatible helper for building RAG prompts. The module deliberately
keeps model invocation logic thin and delegates prompt construction to the
``rag.prompting`` module.
"""

from typing import Any
from openai import OpenAI


def load_openai_client(api_key=None):
    """Instantiate an OpenAI client wrapper.

    Parameters
    ----------
    api_key : str, optional
        API key used to initialize the client.

    Returns
    -------
    OpenAI
        An OpenAI client instance.
    """
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
    return response.choices[0].message.content
