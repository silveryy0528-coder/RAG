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
    'If the answer is not in the context, say "Not found".\n\n'
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


# Evaluation prompt templates
EVALUATION_PROMPT = (
    "You are evaluating a QA system.\n\n"
    "Question:\n{question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:\n{answer}\n\n"
    "Evaluation rules:\n"
    "1. If the context contains the answer, the answer must match it.\n"
    '2. If the context does NOT contain the answer, the correct response is "Not found".\n'
    '3. If the answer says "Not found" and the context indeed lacks the information, this is CORRECT.\n'
    "4. If the answer contains information not in the context, it is INCORRECT.\n\n"
    "Is the answer correct?\n\n"
    "Return in this format:\n"
    "Result: YES or NO\n"
    "Reason: <short explanation>\n"
)

RELEVANCE_PROMPT = (
    "Given the question and retrieved chunk\n\n"
    "Question:\n{question}\n\n"
    "Retrieved chunk:\n{chunk}\n\n"
    "Is this chunk relevant for answering the question?\n"
    "ONLY answer YES or NO\n"
)


def build_evaluation_prompt(question: str, answer: str, context: str) -> str:
    """Render the evaluation prompt for answer correctness.

    Parameters
    ----------
    question : str
        The user question.
    answer : str
        The candidate answer to evaluate.
    context : str
        The context used to produce the answer.

    Returns
    -------
    str
        The rendered evaluation prompt.
    """
    return render_template(
        EVALUATION_PROMPT, {"question": question, "answer": answer, "context": context}
    )


def build_relevance_prompt(question: str, chunk: str) -> str:
    """Render the prompt asking whether a retrieved chunk is relevant.

    Parameters
    ----------
    question : str
        The user question.
    chunk : str
        The retrieved chunk text.

    Returns
    -------
    str
        The rendered relevance prompt.
    """
    return render_template(RELEVANCE_PROMPT, {"question": question, "chunk": chunk})
