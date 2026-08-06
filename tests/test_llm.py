import types

import rag.llm as llm


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
            create.calls = {"model": model, "messages": messages, "temperature": temperature}
            return resp

        self.chat.completions.create = create


def test_generate_answer_uses_client():
    client = FakeClient()
    client.set_response("The answer")
    out = llm.generate_answer(client, "prompt text", model_name="m", temperature=0.5)
    assert out == "The answer"
