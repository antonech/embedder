import sys
import argparse
from mcp.server.fastmcp import FastMCP
from embedder import EmbeddingModel, StorageIO, VectorStore


encoder: EmbeddingModel | None = None
store: VectorStore | None = None


mcp = FastMCP("embedder")


@mcp.tool()
def search(query: str, top_k: int = 5) -> str:
    """Search documents by semantic similarity to the query."""
    if encoder is None:
        return "Error: model not loaded (server not initialized)"
    qv = encoder.embed(query)
    if store is not None and len(store) > 0:
        results = store.search(qv, top_k=top_k)
    else:
        results = []
    return str(results)


@mcp.tool()
def embed(text: str) -> str:
    """Embed a single text into a vector."""
    if encoder is None:
        return "Error: model not loaded (server not initialized)"
    vec = encoder.embed(text)
    return str(vec.tolist())


@mcp.tool()
def embed_many(texts: list[str]) -> str:
    """Embed multiple texts into vectors."""
    if encoder is None:
        return "Error: model not loaded (server not initialized)"
    vecs = encoder.embed_many(texts)
    return str(vecs.tolist())


@mcp.tool()
def store_info() -> str:
    """Return number of stored vectors and sample texts."""
    if store is None:
        return "No data loaded"
    n = len(store)
    sample = store.texts[:3] if n > 0 else []
    return f"Vectors: {n}, sample texts: {sample}"


def main():
    global encoder, store

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--data", default="", help="Path to saved .npz file to preload")
    args = parser.parse_args()

    encoder = EmbeddingModel(args.model)
    if args.data:
        vecs, texts, dim = StorageIO.load(args.data)
        store = VectorStore()
        store.vectors = vecs
        store.texts = texts

    mcp.run_stdio_async()


if __name__ == "__main__":
    main()
