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
