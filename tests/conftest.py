import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeSentenceTransformer:
    """Deterministic stand-in for SentenceTransformer (no model download)."""

    DIM = 4

    def __init__(self, model_name="fake", device=None):
        self.model_name = model_name
        self.device = device
        self.max_seq_length = 0
        self.tokenizer = type("Tok", (), {"model_max_length": 0})()
        self.halved = False

    def half(self):
        self.halved = True
        return self

    def _first_module(self):
        raise RuntimeError("no underlying module")

    def get_embedding_dimension(self):
        return self.DIM

    def encode(self, text, normalize_embeddings=True):
        if isinstance(text, str):
            return self._vec(text)
        return np.stack([self._vec(t) for t in text]) if text else np.empty((0, self.DIM), dtype=np.float32)

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.DIM, dtype=np.float32)
        for i, ch in enumerate(text):
            vec[i % self.DIM] += (ord(ch) % 17) + 1
        norm = float(np.linalg.norm(vec))
        if norm:
            vec /= norm
        return vec


@pytest.fixture
def fake_st(monkeypatch):
    import embedder

    monkeypatch.setattr(embedder, "SentenceTransformer", FakeSentenceTransformer)
    return FakeSentenceTransformer


class FakeEncoder:
    """Stand-in for EmbeddingModel (deterministic, no model download)."""

    def __init__(self, model_name="fake", device=None, float_type="fp32"):
        self.model_name = model_name
        self.device = device
        self.float_type = float_type
        self.dim = FakeSentenceTransformer.DIM
        self.query_prefix = ""
        self.passage_prefix = ""
        self._st = FakeSentenceTransformer(model_name, device)

    def embed(self, text):
        return self._st.encode(text)

    def embed_many(self, texts):
        return self._st.encode(list(texts))

    def as_passages(self, texts):
        return [self.passage_prefix + t for t in texts] if self.passage_prefix else list(texts)

    def embed_passage(self, text):
        return self.embed(self.passage_prefix + text if self.passage_prefix else text)

    def embed_query(self, query):
        return self.embed(self.query_prefix + query if self.query_prefix else query)


@pytest.fixture
def fake_encoder(monkeypatch):
    import mcp_server

    monkeypatch.setattr(mcp_server, "EmbeddingModel", FakeEncoder)
    return FakeEncoder


@pytest.fixture
def app(fake_encoder, tmp_path):
    import mcp_server

    return mcp_server.EmbedderApp("proj", "fake-model", data_dir=str(tmp_path))
