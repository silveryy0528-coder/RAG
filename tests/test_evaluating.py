import rag.evaluating as evaluating


def test_compute_grounding_score_WHEN_answer_and_context_overlap_THEN_returns_overlap_fraction():
    score = evaluating.compute_grounding_score("Hello world", "Hello there")
    assert score == 0.5


def test_evaluate_answer_with_llm_WHEN_model_called_THEN_builds_prompt_and_returns_response(
    fake_client,
):
    fake_client.set_response("Result: YES\nReason: present in context")

    res = evaluating.evaluate_answer_with_llm(fake_client, "Q", "A", "C")
    assert isinstance(res, str)
    last = fake_client.chat.completions.create_calls[-1]
    messages = last["messages"]
    assert any("Question:" in m.get("content", "") for m in messages)
    assert any("Context:" in m.get("content", "") for m in messages)
    assert any("Answer:" in m.get("content", "") for m in messages)


def test_evaluate_chunk_relevance_with_llm_WHEN_model_called_THEN_builds_relevance_prompt_and_returns_response(
    fake_client,
):
    fake_client.set_response("YES")

    res = evaluating.evaluate_chunk_relevance_with_llm(fake_client, "Q", "chunk text")
    assert res == "YES"
    last = fake_client.chat.completions.create_calls[-1]
    messages = last["messages"]
    assert any("Retrieved chunk:" in m.get("content", "") for m in messages)
