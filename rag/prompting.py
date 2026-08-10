"""Prompt templates and rendering utilities.

This module centralizes prompt templates and provides a small PromptRenderer
that renders templates using Python str.format. It validates that required
placeholders are present in the provided parameters.
"""

from typing import Mapping


class PromptRenderingError(ValueError):
    """Raised when rendering fails due to missing parameters."""


def render_template(template: str, params: Mapping[str, object]) -> str:
    """Render a template using ``params``."""
    try:
        return template.format(**params)
    except KeyError as e:
        raise PromptRenderingError(f"Missing parameter for template: {e}") from e


# Example: a simple RAG prompt template
RAG_PROMPT = (
    "You are answering a factual question from the provided context.\n"
    "Answer using ONLY the information in the context.\n"
    "If the answer is not supported by the context, say 'Not found'.\n"
    "Prefer a short, direct answer.\n"
    "For questions about a whole thesis or a broad topic, summarize the main point across the relevant chunks.\n"
    "For questions about special pages such as publications, list, or appendix pages, use the exact information present there.\n"
    "Do not write a long paragraph unless the question explicitly asks for explanation.\n"
    "Return the answer as a concise phrase or sentence.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n"
)


def build_rag_prompt(question: str, context: str) -> str:
    """Build a RAG prompt."""
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
    """Build the evaluation prompt."""
    return render_template(
        EVALUATION_PROMPT, {"question": question, "answer": answer, "context": context}
    )


def build_relevance_prompt(question: str, chunk: str) -> str:
    """Build the relevance prompt."""
    return render_template(RELEVANCE_PROMPT, {"question": question, "chunk": chunk})
