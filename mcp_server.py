import json
import argparse
from mcp.server.fastmcp import FastMCP
from embedder import EmbeddingModel, StorageIO, VectorStore


class EmbedderApp:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = EmbeddingModel(model_name)
        self.store = VectorStore()

    def init(self, data_path: str) -> str:
        vecs, texts, dim = StorageIO.load(data_path)
        self.store = VectorStore()
        self.store.vectors = vecs
        self.store.texts = texts
        return f"Loaded {len(self.store)} vectors, dim={dim}"

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        qv = self.encoder.embed(query)
        return self.store.search(qv, top_k=top_k)

    def embed(self, text: str) -> list[float]:
        return self.encoder.embed(text).tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return self.encoder.embed_many(texts).tolist()

    def add_document(self, text: str) -> str:
        vec = self.encoder.embed(text)
        self.store.add(vec, text)
        return f"Added, total vectors: {len(self.store)}"

    def add_documents(self, texts: list[str]) -> str:
        vecs = self.encoder.embed_many(texts)
        self.store.add_many(vecs, texts)
        return f"Added {len(texts)} docs, total vectors: {len(self.store)}"

    def save(self, path: str) -> str:
        dim = self.encoder.dim
        StorageIO.save(path, self.store.vectors, self.store.texts, dim)
        return f"Saved {len(self.store)} vectors to {path}"

    def info(self) -> str:
        n = len(self.store)
        sample = self.store.texts[:3] if n > 0 else []
        return json.dumps({"vectors": n, "sample_texts": sample}, ensure_ascii=False)


app: EmbedderApp | None = None
mcp = FastMCP("embedder")


@mcp.tool()
def search(query: str, top_k: int = 5) -> str:
    """Search documents by semantic similarity."""
    if app is None:
        return "Error: server not initialized"
    return json.dumps(app.search(query, top_k), ensure_ascii=False, default=str)


@mcp.tool()
def embed(text: str) -> str:
    """Embed a single text into a vector."""
    if app is None:
        return "Error: server not initialized"
    return json.dumps(app.embed(text))


@mcp.tool()
def embed_many(texts: list[str]) -> str:
    """Embed multiple texts into vectors."""
    if app is None:
        return "Error: server not initialized"
    return json.dumps(app.embed_many(texts))


@mcp.tool()
def store_info() -> str:
    """Return store statistics and sample texts."""
    if app is None:
        return "Error: server not initialized"
    return app.info()


@mcp.tool()
def init_store(data_path: str) -> str:
    """Load vectors from a saved .npz file into the store."""
    if app is None:
        return "Error: server not initialized"
    return app.init(data_path)


@mcp.tool()
def add_document(text: str) -> str:
    """Add a single document: embed it and store."""
    if app is None:
        return "Error: server not initialized"
    return app.add_document(text)


@mcp.tool()
def add_documents(texts: list[str]) -> str:
    """Add multiple documents at once."""
    if app is None:
        return "Error: server not initialized"
    return app.add_documents(texts)


@mcp.tool()
def save_store(path: str) -> str:
    """Save current store to a .npz file."""
    if app is None:
        return "Error: server not initialized"
    return app.save(path)


def main():
    global app

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--data", default="")
    args = parser.parse_args()

    app = EmbedderApp(args.model)
    if args.data:
        app.init(args.data)

    mcp.run_stdio_async()


if __name__ == "__main__":
    main()
