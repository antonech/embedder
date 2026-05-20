import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wraps a SentenceTransformer model for producing embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts."""
        return self.model.encode(texts, normalize_embeddings=True)


class VectorStore:
    """In-memory vector store with cosine similarity search."""

    def __init__(self):
        self.vectors: list[np.ndarray] = []
        self.texts: list[str] = []

    def add(self, vec: np.ndarray, text: str) -> None:
        """Add one vector-text pair."""
        self.vectors.append(vec)
        self.texts.append(text)

    def add_many(self, vecs: np.ndarray, texts: list[str]) -> None:
        """Add multiple vector-text pairs."""
        self.vectors.extend(vecs)
        self.texts.extend(texts)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return top_k nearest items as [{text, score}]."""
        if not self.vectors:
            return []
        scores = np.dot(np.stack(self.vectors), query_vec)
        top_idxs = np.argsort(scores)[-top_k:][::-1]
        return [
            {"text": self.texts[i], "score": float(scores[i])}
            for i in top_idxs
        ]

    def __len__(self) -> int:
        return len(self.vectors)


class StorageIO:
    """Save/load vectors, texts and dimension to/from .npz files."""

    @staticmethod
    def save(path: str, vectors: list[np.ndarray], texts: list[str], dim: int) -> None:
        """Persist vectors, texts and dimension to a compressed .npz file."""
        np.savez_compressed(
            path,
            dim=np.array(dim),
            vectors=np.stack(vectors) if vectors else np.array([]),
            texts=np.array(texts, dtype=object),
        )

    @staticmethod
    def load(path: str) -> tuple[list[np.ndarray], list[str], int]:
        """Load vectors, texts and dimension from a .npz file."""
        data = np.load(path, allow_pickle=True)
        vecs = data["vectors"]
        vectors = [vecs[i] for i in range(len(vecs))]
        texts = list(data["texts"])
        dim = int(data["dim"])
        return vectors, texts, dim
