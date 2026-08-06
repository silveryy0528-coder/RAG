import rag.evaluating as evaluating


def test_compute_grounding_score():
    score = evaluating.compute_grounding_score("Hello world", "Hello there")
    assert score == 0.5


def test_evaluate_answer_with_llm_builds_and_sends_prompt(fake_client):
    fake_client.set_response("Result: YES\nReason: present in context")

    res = evaluating.evaluate_answer_with_llm(fake_client, "Q", "A", "C")
    assert isinstance(res, str)
    # inspect the recorded create call to ensure prompt contains expected parts
    last = fake_client.chat.completions.create_calls[-1]
    messages = last["messages"]
    assert any("Question:" in m.get("content", "") for m in messages)
    assert any("Context:" in m.get("content", "") for m in messages)
    assert any("Answer:" in m.get("content", "") for m in messages)


def test_evaluate_chunk_relevance_with_llm_builds_and_sends_prompt(fake_client):
    fake_client.set_response("YES")

    res = evaluating.evaluate_chunk_relevance_with_llm(fake_client, "Q", "chunk text")
    assert res == "YES"
    last = fake_client.chat.completions.create_calls[-1]
    messages = last["messages"]
    assert any("Retrieved chunk:" in m.get("content", "") for m in messages)
