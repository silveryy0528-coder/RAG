"""Prompt templates and rendering utilities.

This module centralizes prompt templates and provides a small PromptRenderer
that renders templates using Python str.format. It validates that required
placeholders are present in the provided parameters.
"""
from typing import Mapping


class PromptRenderingError(ValueError):
    """Raised when rendering fails due to missing parameters."""


def render_template(template: str, params: Mapping[str, object]) -> str:
    """Render a template using ``params``.

    Parameters
    ----------
    template : str
        Template string using Python ``str.format`` placeholders, e.g.
        "Answer: {answer}".
    params : Mapping[str, object]
        Mapping of placeholder names to values.

    Returns
    -------
    str
        The rendered string.

    Raises
    ------
    PromptRenderingError
        If a placeholder in the template is not provided in ``params``.
    """
    try:
        return template.format(**params)
    except KeyError as e:
        raise PromptRenderingError(f"Missing parameter for template: {e}") from e


# Example: a simple RAG prompt template
RAG_PROMPT = (
    "Answer the question using ONLY the context below.\n"
    "If the answer is not in the context, say \"Not found\".\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n"
)


def build_rag_prompt(question: str, context: str) -> str:
    """Build a RAG prompt by rendering the RAG_PROMPT template.

    Parameters
    ----------
    question : str
        The user question.
    context : str
        The contextual text used to answer the question.

    Returns
    -------
    str
        The rendered prompt string ready to send to an LLM.
    """
    return render_template(RAG_PROMPT, {"question": question, "context": context})
