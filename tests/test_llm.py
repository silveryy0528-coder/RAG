import rag.llm as llm


def test_generate_answer_WHEN_client_returns_response_THEN_forwards_prompt_and_returns_content(
    fake_client,
):
    fake_client.set_response("The answer")
    out = llm.generate_answer(
        fake_client, "prompt text", model_name="m", temperature=0.5
    )
    assert out == "The answer"
