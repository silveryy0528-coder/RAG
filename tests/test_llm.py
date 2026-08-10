import rag.llm as llm


def test_generate_answer_WHEN_client_returns_response_THEN_forwards_prompt_and_returns_content(
    fake_client,
):
    fake_client.set_response("The answer")
    out = llm.generate_answer(
        fake_client, "prompt text", model_name="m", temperature=0.5
    )
    assert out == "The answer"


def test_normalize_answer_WHEN_model_returns_verbose_prefix_THEN_strips_prefix():
    assert llm.normalize_answer("The answer is automatic parameter selection") == "automatic parameter selection"


def test_normalize_answer_WHEN_model_returns_not_found_THEN_returns_canonical_form():
    assert llm.normalize_answer("The answer is not found") == "Not found"
