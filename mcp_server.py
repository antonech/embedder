import os
import json
import asyncio
import argparse
from mcp.server.fastmcp import FastMCP
from embedder import EmbeddingModel, StorageIO, VectorStore
from tree_search import TreeIndex


class EmbedderApp:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", data_dir: str = "data", device: str | None = None):
        self.encoder = EmbeddingModel(model_name, device=device)
        self.store = VectorStore()
        self.data_dir = data_dir

    def init(self, data_path: str) -> str:
        vecs, texts, dim = StorageIO.load(data_path)
        self.store = VectorStore()
        self.store.vectors = vecs
        self.store.texts = texts
        self._delta_count = 0
        self.data_dir = os.path.dirname(data_path)
        if hasattr(self, "_tree"):
            del self._tree
        return f"Loaded {len(self.store)} vectors, dim={dim}"

    def load_delta(self, data_path: str) -> str:
        if not os.path.exists(data_path):
            return "No delta file"
        vecs, texts, dim = StorageIO.load(data_path)
        self.store.vectors.extend(vecs)
        self.store.texts.extend(texts)
        self._delta_count = len(vecs)
        return f"Loaded {len(vecs)} delta vectors"

    def clear_delta(self) -> str:
        if self._delta_count < 1 or not hasattr(self, '_delta_count'):
            return "No delta to clear"
        self.store.vectors = self.store.vectors[:-self._delta_count]
        self.store.texts = self.store.texts[:-self._delta_count]
        self._delta_count = 0
        return "Delta cleared"

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
        delta = getattr(self, '_delta_count', 0)
        return json.dumps({"vectors": n, "delta": delta, "sample_texts": sample}, ensure_ascii=False)

    def tree_search(self, query: str, top_k: int = 5) -> str:
        if not self.store.vectors:
            return json.dumps({"error": "Store not loaded"})
        if not hasattr(self, "_tree"):
            self._tree = TreeIndex(data_dir=self.data_dir)
        qv = self.encoder.embed(query)
        return json.dumps(self._tree.search_with_context(self.store, qv, top_k), ensure_ascii=False, default=str)


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


@mcp.tool()
def tree_search(query: str, top_k: int = 5) -> str:
    """Search with AST context (children/parent/siblings)."""
    if app is None:
        return "Error: server not initialized"
    return app.tree_search(query, top_k)


@mcp.tool()
def load_delta(data_path: str) -> str:
    """Load delta vectors on top of the existing store."""
    if app is None:
        return "Error: server not initialized"
    return app.load_delta(data_path)


@mcp.tool()
def clear_delta() -> str:
    """Remove delta layer from the store."""
    if app is None:
        return "Error: server not initialized"
    return app.clear_delta()


async def main():
    global app

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--root", default=None)
    args = parser.parse_args()

    model_name = args.model
    device = None
    cfg = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)

    if model_name is None:
        model_name = cfg.get("model_name", "all-MiniLM-L6-v2")
        device = cfg.get("device")

    if args.data_dir:
        data_dir = args.data_dir
    elif args.data:
        data_dir = os.path.dirname(args.data) or "data"
    elif cfg.get("embedding_store"):
        store_root = os.path.expandvars(os.path.expanduser(cfg["embedding_store"]))
        data_dir = store_root
        if args.root:
            data_dir = os.path.join(data_dir, os.path.basename(os.path.abspath(args.root)))
    else:
        data_dir = "data"

    app = EmbedderApp(model_name, data_dir=data_dir, device=device)

    data_path = args.data or os.path.join(data_dir, "enriched_vectors.npz")
    if os.path.exists(data_path):
        print(f"Loading store from {data_path}...")
        app.init(data_path)
    elif args.data:
        print(f"Data file not found: {data_path}")

    delta_path = os.path.join(data_dir, "delta.npz")
    if os.path.exists(delta_path):
        print(f"Loading delta from {delta_path}...")
        app.load_delta(delta_path)

    tree_delta = os.path.join(data_dir, "delta_tree_index.json")
    if os.path.exists(tree_delta):
        print(f"Tree delta found at {tree_delta}")

    await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
