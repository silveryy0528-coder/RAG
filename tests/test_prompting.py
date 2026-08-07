import pytest

from rag.prompting import (
    render_template,
    build_rag_prompt,
    PromptRenderingError,
    RAG_PROMPT,
)


def test_render_template_WHEN_valid_parameters_provided_THEN_fills_placeholders():
    tpl = "Hello {name}, you have {n} messages"
    out = render_template(tpl, {"name": "Alice", "n": 5})
    assert "Alice" in out
    assert "5" in out


def test_render_template_WHEN_parameter_missing_THEN_raises_prompt_rendering_error():
    with pytest.raises(PromptRenderingError):
        render_template("Hi {user}", {})


def test_build_rag_prompt_WHEN_question_and_context_provided_THEN_uses_the_rag_template():
    q = "What is X?"
    c = "Context here"
    prompt = build_rag_prompt(q, c)
    assert q in prompt
    assert c in prompt
    assert "Context:" in RAG_PROMPT
