import types

import rag.evaluating as evaluating


class FakeClient:
    def __init__(self):
        self.chat = types.SimpleNamespace()
        self.chat.completions = types.SimpleNamespace()
        self.chat.completions.create_calls = []

    def set_response(self, content: str):
        def create(model=None, messages=None, temperature=None):
            resp = types.SimpleNamespace()
            choice = types.SimpleNamespace()
            choice.message = types.SimpleNamespace()
            choice.message.content = content
            resp.choices = [choice]
            # record call arguments for inspection
            create.last_call = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            self.chat.completions.create_calls.append(create.last_call)
            return resp

        self.chat.completions.create = create


def test_compute_grounding_score():
    score = evaluating.compute_grounding_score("Hello world", "Hello there")
    assert score == 0.5


def test_evaluate_answer_with_llm_builds_and_sends_prompt():
    client = FakeClient()
    client.set_response("Result: YES\nReason: present in context")

    res = evaluating.evaluate_answer_with_llm(client, "Q", "A", "C")
    assert isinstance(res, str)
    # inspect the recorded create call to ensure prompt contains expected parts
    last = client.chat.completions.create_calls[-1]
    messages = last["messages"]
    assert any("Question:" in m.get("content", "") for m in messages)
    assert any("Context:" in m.get("content", "") for m in messages)
    assert any("Answer:" in m.get("content", "") for m in messages)


def test_evaluate_chunk_relevance_with_llm_builds_and_sends_prompt():
    client = FakeClient()
    client.set_response("YES")

    res = evaluating.evaluate_chunk_relevance_with_llm(client, "Q", "chunk text")
    assert res == "YES"
    last = client.chat.completions.create_calls[-1]
    messages = last["messages"]
    assert any("Retrieved chunk:" in m.get("content", "") for m in messages)
