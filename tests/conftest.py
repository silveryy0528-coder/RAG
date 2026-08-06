import sys
import types


class StubSentenceSplitter:
    def __init__(self, chunk_size=None, chunk_overlap=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        return [
            {
                "text": text,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }
        ]


class StubSentenceTransformer:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device
        self.encode_calls = []

    def encode(
        self, texts, device=None, convert_to_numpy=None, normalize_embeddings=None
    ):
        self.encode_calls.append(
            {
                "texts": texts,
                "device": device,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return [[0.1, 0.2, 0.3]]


# Stub llama_index so rag.chunking can import SentenceSplitter at test time.
fake_node_parser = types.ModuleType("llama_index.core.node_parser")
fake_node_parser.SentenceSplitter = StubSentenceSplitter

fake_core = types.ModuleType("llama_index.core")
fake_core.node_parser = fake_node_parser

fake_llama = types.ModuleType("llama_index")
fake_llama.core = fake_core

sys.modules["llama_index"] = fake_llama
sys.modules["llama_index.core"] = fake_core
sys.modules["llama_index.core.node_parser"] = fake_node_parser


# Stub transformers and sentence_transformers so rag.embedding can import them.
fake_transformers_logging = types.ModuleType("transformers.logging")


def set_verbosity_error():
    pass


fake_transformers_logging.set_verbosity_error = set_verbosity_error
fake_transformers = types.ModuleType("transformers")
fake_transformers.logging = fake_transformers_logging
sys.modules["transformers"] = fake_transformers
sys.modules["transformers.logging"] = fake_transformers_logging

fake_sentence_transformers = types.ModuleType("sentence_transformers")
fake_sentence_transformers.SentenceTransformer = StubSentenceTransformer
sys.modules["sentence_transformers"] = fake_sentence_transformers

# Stub faiss so indexing code can import it in tests without the real package.
fake_faiss = types.ModuleType("faiss")

# Metric constant
fake_faiss.METRIC_L2 = 0

class _FakeIndexBase:
    def __init__(self, *args, **kwargs):
        self.trained = False
        self.added = None

    def train(self, embeddings):
        self.trained = True

    def add(self, embeddings):
        self.added = embeddings

    def search(self, query_vec, k):
        # Default: return zeros
        return [[0.0] * k], [[0] * k]


class IndexFlatL2(_FakeIndexBase):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim


class IndexIVFFlat(_FakeIndexBase):
    def __init__(self, quantizer, dim, nlist, metric):
        super().__init__()
        self.quantizer = quantizer
        self.dim = dim
        self.nlist = nlist
        self.metric = metric


class IndexIVFPQ(_FakeIndexBase):
    def __init__(self, quantizer, dim, nlist, m, nbits):
        super().__init__()
        self.quantizer = quantizer
        self.dim = dim
        self.nlist = nlist
        self.m = m
        self.nbits = nbits


fake_faiss.IndexFlatL2 = IndexFlatL2
fake_faiss.IndexIVFFlat = IndexIVFFlat
fake_faiss.IndexIVFPQ = IndexIVFPQ

sys.modules["faiss"] = fake_faiss

# Stub openai (OpenAI client) so rag.llm can import it in tests.
fake_openai = types.ModuleType("openai")

class _FakeChatCompletions:
    def __init__(self):
        self.create_calls = []

    def create(self, model=None, messages=None, temperature=None):
        # simulate a response object with choices[0].message.content
        resp = types.SimpleNamespace()
        choice = types.SimpleNamespace()
        choice.message = types.SimpleNamespace()
        # return the concatenation of messages content for visibility
        content = " ".join(m.get("content", "") for m in (messages or []))
        choice.message.content = content
        resp.choices = [choice]
        self.create_calls.append({"model": model, "messages": messages, "temperature": temperature})
        return resp

class _FakeOpenAI:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.chat = types.SimpleNamespace()
        self.chat.completions = _FakeChatCompletions()

fake_openai.OpenAI = _FakeOpenAI
sys.modules["openai"] = fake_openai