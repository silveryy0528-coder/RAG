"""Evaluation helpers using an LLM and small heuristics.

This module exposes a grounding score function (token-overlap) and two
LLM-backed evaluators that ask a model to judge answer correctness and
retrieval relevance. Prompt templates live in :mod:`rag.prompting`.
"""

import re
from typing import Any

from rag.llm import generate_answer
from rag.prompting import build_evaluation_prompt, build_relevance_prompt


def compute_grounding_score(answer: str, context: str) -> float:
    """Compute a simple grounding score based on token overlap."""
    answer_tokens = re.findall(r"\w+", (answer or "").lower())
    context_tokens = set(re.findall(r"\w+", (context or "").lower()))

    if not answer_tokens:
        return 0.0

    overlap = [t for t in answer_tokens if t in context_tokens]
    return len(overlap) / len(answer_tokens)


def evaluate_answer_with_llm(
    client: Any,
    question: str,
    answer: str,
    context: str,
    model_name: str = "gpt-4.1-mini",
    temperature: float = 0.0,
) -> str:
    """Ask an LLM whether an answer is supported by context."""
    prompt = build_evaluation_prompt(question, answer, context)
    return generate_answer(client, prompt, model_name, temperature)


def evaluate_chunk_relevance_with_llm(
    client: Any,
    question: str,
    chunk: str,
    model_name: str = "gpt-4.1-mini",
    temperature: float = 0.0,
) -> str:
    """Ask an LLM whether a retrieved chunk is relevant."""
    prompt = build_relevance_prompt(question, chunk)
    return generate_answer(client, prompt, model_name, temperature)
