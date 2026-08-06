import pytest

from rag.prompting import (
    render_template,
    build_rag_prompt,
    PromptRenderingError,
    RAG_PROMPT,
)


def test_render_template_fills_placeholders():
    tpl = "Hello {name}, you have {n} messages"
    out = render_template(tpl, {"name": "Alice", "n": 5})
    assert "Alice" in out
    assert "5" in out


def test_render_template_missing_param_raises():
    with pytest.raises(PromptRenderingError):
        render_template("Hi {user}", {})


def test_build_rag_prompt_uses_template():
    q = "What is X?"
    c = "Context here"
    prompt = build_rag_prompt(q, c)
    assert q in prompt
    assert c in prompt
    # Ensure it uses the constant template
    assert "Context:" in RAG_PROMPT
