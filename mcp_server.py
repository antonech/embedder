import os
import json
import asyncio
import argparse
import re
from mcp.server.fastmcp import FastMCP
from embedder import EmbeddingModel, StorageIO, VectorStore
from tree_search import TreeIndex


class EmbedderApp:
    def __init__(self, project_name: str, model_name: str = "all-MiniLM-L6-v2", data_dir: str = "data", device: str | None = None):
        self.project_name = project_name
        self.model_name = model_name
        # Handle CUDA fallback
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    device = "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.data_dir = data_dir
        self.encoder = EmbeddingModel(model_name, device=device)
        self.store = VectorStore()
        self._bm25 = None

    def _tokenize(self, text: str) -> list[str]:
        words = re.split(r'[^a-zA-Z0-9]+', text)
        result = []
        for w in words:
            if not w:
                continue
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|\d+', w)
            for p in parts:
                p = p.lower()
                if len(p) >= 2:
                    result.append(p)
        return result

    def _build_bm25(self):
        if len(self.store) == 0:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi
        corpus = [self._tokenize(t) for t in self.store.texts]
        self._bm25 = BM25Okapi(corpus)

    def init(self, data_path: str) -> str:
        vecs, texts, dim = StorageIO.load(data_path)
        self.store = VectorStore()
        self.store.vectors = vecs
        self.store.texts = texts
        self._delta_count = 0
        self.data_dir = os.path.dirname(data_path)
        if hasattr(self, "_tree"):
            del self._tree
        self._build_bm25()
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

    def _get_tree(self):
        if not hasattr(self, "_tree"):
            self._tree = TreeIndex(data_dir=self.data_dir)
        return self._tree

    def _annotate(self, hits: list[dict]) -> list[dict]:
        return self._get_tree().annotate(hits)

    @staticmethod
    def _format(hits: list[dict], fmt: str = "json") -> str:
        if fmt == "json":
            return json.dumps(hits, ensure_ascii=False, default=str)
        lines = []
        for i, h in enumerate(hits, 1):
            ctx = h.get("context")
            if ctx:
                parent = ctx.get("parent")
                parent_str = f"{parent['name']} ({parent['type']})" if parent else "—"
                children_str = ", ".join(c["name"] for c in ctx.get("children", [])[:5])
                if len(ctx.get("children", [])) > 5:
                    children_str += "..."
                siblings_str = ", ".join(s["name"] for s in ctx.get("siblings", [])[:5])
                if len(ctx.get("siblings", [])) > 5:
                    siblings_str += "..."
            lines.append(f"### {i}. [{h['score']:.3f}] {h['text'][:200]}")
            if ctx:
                lines.append(f"  **Parent:** {parent_str}")
                if children_str:
                    lines.append(f"  **Children:** {children_str}")
                if siblings_str:
                    lines.append(f"  **Siblings:** {siblings_str}")
            lines.append("")
        return "\n".join(lines).strip()

    def search(self, query: str, top_k: int = 5, fmt: str = "json") -> str:
        qv = self.encoder.embed(query)
        hits = self.store.search(qv, top_k=top_k)
        return self._format(self._annotate(hits), fmt)

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5, fmt: str = "json") -> str:
        if self._bm25 is None:
            return self.search(query, top_k=top_k, fmt=fmt)
        qv = self.encoder.embed(query)
        emb_hits = self.store.search(qv, top_k=top_k * 3)
        tokenized = self._tokenize(query)
        n = len(self.store)
        bm25_scores = self._bm25.get_scores(tokenized)
        bm25_top = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 3]
        emb_idx = {h["idx"] for h in emb_hits}

        if alpha >= 1.0:
            hits = [dict(**h, method="embed") for h in self.store.search(qv, top_k=top_k)]
            return self._format(self._annotate(hits), fmt)
        if alpha <= 0.0:
            hits = [{"text": self.store.texts[i], "score": float(bm25_scores[i]), "idx": i, "method": "bm25"}
                    for i in bm25_top[:top_k]]
            return self._format(self._annotate(hits), fmt)

        K = 30
        def rrf(doc_id):
            emb_rank = next((r for r, h in enumerate(emb_hits) if h["idx"] == doc_id), None)
            bm25_rank = next((r for r, i in enumerate(bm25_top) if i == doc_id), None)
            s = 0
            if emb_rank is not None:
                s += alpha / (K + emb_rank)
            if bm25_rank is not None:
                s += (1 - alpha) / (K + bm25_rank)
            return s

        candidates = set(h["idx"] for h in emb_hits) | set(bm25_top)
        scored = [(doc_id, rrf(doc_id)) for doc_id in candidates]
        scored.sort(key=lambda x: -x[1])
        hits = [
            {"text": self.store.texts[idx], "score": round(s, 4), "idx": idx,
             "method": "hybrid"}
            for idx, s in scored[:top_k]
        ]
        return self._format(self._annotate(hits), fmt)

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
        return self.search(query, top_k=top_k, fmt="json")

    def tree_hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> str:
        return self.hybrid_search(query, top_k=top_k, alpha=alpha, fmt="json")


projects: dict[str, EmbedderApp] = {}
mcp = FastMCP("embedder")


@mcp.tool()
def search(query: str, project: str, top_k: int = 5, fmt: str = "markdown") -> str:
    """Search documents by semantic similarity, includes AST context (children/parent/siblings)."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.search(query, top_k=top_k, fmt=fmt)


@mcp.tool()
def hybrid_search(query: str, project: str, top_k: int = 5, alpha: float = 0.5, fmt: str = "markdown") -> str:
    """Search by hybrid BM25 + embedding (alpha=1: pure embed, alpha=0: pure BM25), includes AST context."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.hybrid_search(query, top_k=top_k, alpha=alpha, fmt=fmt)


@mcp.tool()
def embed(text: str, project: str) -> str:
    """Embed a single text into a vector."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return json.dumps(target_app.embed(text))


@mcp.tool()
def embed_many(texts: list[str], project: str) -> str:
    """Embed multiple texts into vectors."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return json.dumps(target_app.embed_many(texts))


@mcp.tool()
def store_info(project: str) -> str:
    """Return store statistics and sample texts."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.info()


@mcp.tool()
def init_store(data_path: str, project: str) -> str:
    """Load vectors from a saved .npz file into the store."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.init(data_path)


@mcp.tool()
def tree_search(query: str, project: str, top_k: int = 5, fmt: str = "json") -> str:
    """Alias for search (json format)."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.tree_search(query, top_k)


@mcp.tool()
def tree_hybrid_search(query: str, project: str, top_k: int = 5, alpha: float = 0.5, fmt: str = "json") -> str:
    """Alias for hybrid_search (json format)."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.tree_hybrid_search(query, top_k, alpha)


@mcp.tool()
def add_document(text: str, project: str) -> str:
    """Add a single document: embed it and store."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.add_document(text)


@mcp.tool()
def add_documents(texts: list[str], project: str) -> str:
    """Add multiple documents at once."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.add_documents(texts)


@mcp.tool()
def save_store(path: str, project: str) -> str:
    """Save current store to a .npz file."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.save(path)


@mcp.tool()
def load_delta(data_path: str, project: str) -> str:
    """Load delta vectors on top of the existing store."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.load_delta(data_path)


@mcp.tool()
def clear_delta(project: str) -> str:
    """Remove delta layer from the store."""
    global projects
    if not projects:
        return "Error: server not initialized"
    target_app = projects.get(project)
    if target_app is None:
        return f"Error: project '{project}' not found"
    return target_app.clear_delta()


async def main():
    global projects

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Project name (default: basename of workdir)")
    args = parser.parse_args()

    model_name = "all-MiniLM-L6-v2"
    device = None
    data_dir = "data"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model_name = cfg.get("model_name", model_name)
        device = cfg.get("device")
        store_root = cfg.get("embedding_store", "")
        if store_root:
            store_root = os.path.expandvars(os.path.expanduser(store_root))

    project_name = args.project or os.path.basename(os.getcwd())

    if store_root:
        data_dir = os.path.join(store_root, project_name)

    if project_name not in projects:
        projects[project_name] = EmbedderApp(project_name, model_name, data_dir=data_dir, device=device)

    app = projects[project_name]

    data_path = os.path.join(data_dir, "enriched_vectors.npz")
    if os.path.exists(data_path):
        app.init(data_path)

    delta_path = os.path.join(data_dir, "delta.npz")
    if os.path.exists(delta_path):
        app.load_delta(delta_path)

    await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
